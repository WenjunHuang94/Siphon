"""CPU-only tests for paired data validation and frozen-feature cache plans."""

import base64
from contextlib import nullcontext, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from siphon.inference import TASK_PROMPTS
from siphon.prepare_training_data import _encode_pair, _load_encoders, build_plan, create_cache, load_pairs, make_parser, read_image_size


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lO0AAAAASUVORK5CYII="
)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.input = self.data / "marked.png"
        self.target = self.data / "clean.png"
        self.input.write_bytes(PNG)
        self.target.write_bytes(PNG)
        self.manifest = self.data / "train.jsonl"
        self.write_rows({"input": "marked.png", "target": "clean.png", "task": "box_edit"})

    def write_rows(self, *rows):
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def plan(self, *extra):
        return build_plan(make_parser().parse_args([
            "--manifest", str(self.manifest), "--cache-dir", str(self.root / "cache"), *extra,
        ]))

    def test_pair_paths_are_relative_to_jsonl_and_generic_task_prompt_is_used(self):
        pair = load_pairs(self.manifest)[0]
        self.assertEqual(pair.input, self.input)
        self.assertEqual(pair.target, self.target)
        self.assertEqual(pair.prompt, TASK_PROMPTS["box_edit"])
        self.assertEqual(pair.input_size, (1, 1))
        self.assertEqual(pair.target_size, (1, 1))

    def test_legacy_target_and_reference_fields_are_not_reversed(self):
        prompt = "Follow the visual pointer and text to edit the image"
        self.write_rows({"img_path": "clean.png", "ref_img_path": "marked.png", "prompt": prompt, "width": 1, "height": 1})
        pair = load_pairs(self.manifest)[0]
        self.assertEqual(pair.input, self.input)
        self.assertEqual(pair.target, self.target)
        self.assertEqual(pair.prompt, prompt)

    def test_missing_targets_bad_images_and_same_source_target_fail(self):
        self.target.unlink()
        with self.assertRaisesRegex(ValueError, "Image does not exist"):
            load_pairs(self.manifest)
        self.target.write_text("not an image")
        with self.assertRaisesRegex(ValueError, "PNG, JPEG, or WebP"):
            load_pairs(self.manifest)
        self.write_rows({"input": "marked.png", "target": "marked.png", "task": "text_edit"})
        with self.assertRaisesRegex(ValueError, "different files"):
            load_pairs(self.manifest)

    def test_invalid_rows_tasks_and_prompts_fail_with_source_line(self):
        self.write_rows({"input": "marked.png", "target": "clean.png", "task": "unknown"})
        with self.assertRaisesRegex(ValueError, "train.jsonl:1:.*unknown task"):
            load_pairs(self.manifest)
        self.write_rows({"input": "marked.png", "target": "clean.png"})
        with self.assertRaisesRegex(ValueError, "nonempty generic prompt"):
            load_pairs(self.manifest)

    def test_safe_unique_ids_and_limit(self):
        row = {"id": "one", "input": "marked.png", "target": "clean.png", "task": "text_edit"}
        self.write_rows(row, row)
        self.assertEqual(len(load_pairs(self.manifest, limit=1)), 1)
        with self.assertRaisesRegex(ValueError, "duplicate sample ID"):
            load_pairs(self.manifest)
        row["id"] = "../escape"
        self.write_rows(row)
        with self.assertRaisesRegex(ValueError, "unsafe sample ID"):
            load_pairs(self.manifest)
        with self.assertRaisesRegex(ValueError, "positive"):
            load_pairs(self.manifest, limit=0)

    def test_output_shapes_and_unpacked_scaled_cache_contract(self):
        plan = self.plan("--resolution", "512", "--revision", "base-commit")
        self.assertEqual((plan["width"], plan["height"]), (512, 512))
        self.assertEqual(plan["base_revision"], "base-commit")
        self.assertEqual(plan["latent_layout"], "CHW")
        self.assertTrue(plan["latents_scaled"])
        self.assertTrue(plan["has_null_prompt_embeds"])
        self.assertEqual(plan["tensor_keys"], ["target_latents", "ref_latents", "prompt_embeds", "null_prompt_embeds"])
        self.assertEqual(plan["cache_manifest"], str(self.root / "cache/manifest.jsonl"))
        self.assertFalse((self.root / "cache").exists())
        plan = self.plan("--width", "768", "--height", "512")
        self.assertEqual((plan["width"], plan["height"]), (768, 512))

    def test_mismatched_original_geometry_is_valid_with_explicit_paired_resize(self):
        png = bytearray(PNG)
        png[16:24] = struct.pack(">II", 123, 80)
        self.target.write_bytes(png)
        plan = self.plan()
        self.assertEqual(plan["pairs"][0]["target_size"], [123, 80])
        self.assertEqual((plan["width"], plan["height"]), (512, 512))

    def test_partial_or_incompatible_training_geometry_fails(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            self.plan("--width", "512")
        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            self.plan("--resolution", "513")

    def test_existing_cache_requires_overwrite_and_source_manifest_is_protected(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "manifest.jsonl").write_text("old cache")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.plan()
        self.plan("--overwrite")
        (cache / "manifest.jsonl").unlink()
        (cache / "manifest.jsonl").symlink_to(self.manifest)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite source"):
            self.plan("--overwrite")

    def test_jpeg_header_dimensions_are_available_without_pillow(self):
        jpeg = self.data / "sample.jpg"
        jpeg.write_bytes(b"\xff\xd8\xff\xe0\x00\x04AB\xff\xc0\x00\x08\x08" + struct.pack(">HH", 240, 320) + b"\x01\xff\xd9")
        self.assertEqual(read_image_size(jpeg), (320, 240))

    def test_help_and_dry_run_do_not_import_model_dependencies(self):
        guard = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'PIL', 'numpy', 'diffusers', 'peft', 'accelerate', 'safetensors'}:
        raise AssertionError('unexpected dependency import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['siphon.prepare_training_data'] + sys.argv[1:]
runpy.run_module('siphon.prepare_training_data', run_name='__main__')
'''
        args = [sys.executable, "-c", guard, "--manifest", str(self.manifest), "--cache-dir", str(self.root / "cache"), "--dry-run"]
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["count"], 1)
        self.assertFalse((self.root / "cache").exists())
        result = subprocess.run([sys.executable, "-c", guard, "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_feature_loader_never_loads_the_denoising_transformer(self):
        factory = MagicMock()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16")
        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": SimpleNamespace(LongCatImageEditPipeline=factory)}):
            pipe, _ = _load_encoders(self.plan("--revision", "base-commit", "--local-files-only"))
        options = factory.from_pretrained.call_args.kwargs
        self.assertIsNone(options["transformer"])
        self.assertEqual(options["revision"], "base-commit")
        self.assertTrue(options["local_files_only"])
        pipe.text_encoder.requires_grad_.assert_called_once_with(False)
        pipe.vae.requires_grad_.assert_called_once_with(False)
        pipe.enable_model_cpu_offload.assert_called_once_with()

    def test_cache_encodes_real_empty_prompt_and_moves_vae_inputs_to_execution_device(self):
        class Tensor:
            def __init__(self, shape):
                self.shape = shape
                self.ndim = len(shape)
                self.moves = []

            def __getitem__(self, index):
                return Tensor(self.shape[1:])

            def detach(self):
                return self

            def contiguous(self):
                return self

            def to(self, *args, **kwargs):
                self.moves.append((args, kwargs))
                return self

        pixels = [Tensor((1, 3, 512, 512)), Tensor((1, 3, 512, 512))]
        pipe = SimpleNamespace(
            _execution_device="cuda:0",
            encode_prompt=MagicMock(side_effect=[(Tensor((1, 768, 3584)), None), (Tensor((1, 769, 3584)), None)]),
            image_processor=SimpleNamespace(preprocess=MagicMock(side_effect=pixels)),
            _encode_vae_image=MagicMock(side_effect=[Tensor((1, 16, 64, 64)), Tensor((1, 16, 64, 64))]),
        )
        fake_image = MagicMock()
        fake_torch = SimpleNamespace(inference_mode=nullcontext, bfloat16="bf16")
        with patch.dict(sys.modules, {"PIL": SimpleNamespace(Image=fake_image)}):
            result = _encode_pair(pipe, fake_torch, {"input": "marked.png", "target": "clean.png", "prompt": "generic"}, 512, 512)
        for tensor in pixels:
            self.assertEqual(tensor.moves, [((), {"device": "cuda:0", "dtype": "bf16"})])
        self.assertEqual([call.kwargs["prompt"] for call in pipe.encode_prompt.call_args_list], ["generic", ""])
        self.assertEqual(result["prompt_embeds"].shape, (768, 3584))
        self.assertEqual(result["null_prompt_embeds"].shape, (769, 3584))
        self.assertEqual(result["target_latents"].shape, (16, 64, 64))

    def test_cache_manifest_records_portable_paths_hashes_and_real_null_feature_contract(self):
        plan = self.plan()
        pipe = SimpleNamespace(
            vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)),
            vae_scale_factor=8, maybe_free_model_hooks=MagicMock(),
        )
        tensors = {
            "target_latents": SimpleNamespace(shape=(16, 64, 64)),
            "ref_latents": SimpleNamespace(shape=(16, 64, 64)),
            "prompt_embeds": SimpleNamespace(shape=(768, 3584)),
            "null_prompt_embeds": SimpleNamespace(shape=(768, 3584)),
        }

        def save(tensors, filename, metadata):
            Path(filename).write_bytes(json.dumps(metadata).encode())

        fake_safe = SimpleNamespace(save_file=save)
        with patch.dict(sys.modules, {"safetensors": SimpleNamespace(torch=fake_safe), "safetensors.torch": fake_safe}), patch("siphon.prepare_training_data._load_encoders", return_value=(pipe, object())), patch("siphon.prepare_training_data._encode_pair", return_value=tensors):
            self.assertEqual(create_cache(plan), 0)
        cache_dir = Path(plan["cache_dir"])
        row = json.loads((cache_dir / "manifest.jsonl").read_text())
        self.assertFalse(Path(row["cache_path"]).is_absolute())
        self.assertFalse(Path(row["source"]["input"]).is_absolute())
        self.assertEqual(row["source"]["input_sha256"], hashlib.sha256(PNG).hexdigest())
        self.assertEqual(row["tensor_shapes"]["target_latents"], [16, 64, 64])
        self.assertIn("null_prompt_embeds", row["tensor_shapes"])
        self.assertEqual(row["cache_sha256"], hashlib.sha256((cache_dir / row["cache_path"]).read_bytes()).hexdigest())
        self.assertTrue(row["latents_scaled"])
        self.assertEqual(json.loads((cache_dir / "dataset_metadata.json").read_text())["status"], "complete")
        pipe.maybe_free_model_hooks.assert_called_once_with()

    def test_encoder_failure_is_recorded_and_returns_nonzero(self):
        fake_safe = SimpleNamespace(save_file=MagicMock())
        plan = self.plan()
        with patch.dict(sys.modules, {"safetensors": SimpleNamespace(torch=fake_safe), "safetensors.torch": fake_safe}), patch("siphon.prepare_training_data._load_encoders", side_effect=RuntimeError("CUDA unavailable")), redirect_stderr(io.StringIO()):
            self.assertEqual(create_cache(plan), 1)
        report = json.loads((Path(plan["cache_dir"]) / "dataset_metadata.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("CUDA unavailable", report["error"])


if __name__ == "__main__":
    unittest.main()
