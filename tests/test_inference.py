"""CPU-only validation of the portable inference interface."""

import base64
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from siphon.inference import TASK_PROMPTS, _load_pipeline, build_plan, execute_plan, make_parser


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lO0AAAAASUVORK5CYII="
)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "marked.png"
        self.input.write_bytes(PNG)
        self.output = self.root / "generated.png"
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (self.adapter / "adapter_model.safetensors").write_bytes(b"test weight placeholder")

    def plan(self, *extra):
        return build_plan(make_parser().parse_args([
            "--lora", str(self.adapter), "--input", str(self.input),
            "--task", "arrow_edit", "--output", str(self.output), *extra,
        ]))

    def manifest(self, **changes):
        directory = self.root / "examples"
        directory.mkdir(exist_ok=True)
        (directory / "inputs").mkdir(exist_ok=True)
        (directory / "inputs" / "a.png").write_bytes(PNG)
        (directory / "inputs" / "b.png").write_bytes(PNG)
        data = {
            "schema_version": 1,
            "defaults": {"seed": 7, "steps": 12, "guidance_scale": 3.5},
            "examples": [
                {"id": "a", "type": "III", "task": "arrow_edit", "input": "inputs/a.png", "reference_output": "references/a.png"},
                {"id": "b", "type": "IV", "task": "box_edit", "input": "inputs/b.png", "reference_output": "references/b.png"},
            ],
        }
        data.update(changes)
        path = directory / "examples.json"
        path.write_text(json.dumps(data))
        return path

    def batch(self, manifest, *extra):
        return build_plan(make_parser().parse_args([
            "--base-only", "--examples", str(manifest),
            "--output-dir", str(self.root / "outputs"), *extra,
        ]))

    def test_task_prompts_keep_sample_specific_text_in_image(self):
        self.assertEqual(TASK_PROMPTS["arrow_edit"], "Follow the visual pointer and text to edit the image")
        self.assertEqual(TASK_PROMPTS["box_edit"], "Modify the image at the annotated location according to the text instruction")
        self.assertEqual(set(TASK_PROMPTS), {"text_to_image", "text_edit", "arrow_edit", "box_edit"})

    def test_manifest_paths_are_relative_to_manifest_not_cwd(self):
        manifest = self.manifest()
        plan = self.batch(manifest)
        self.assertEqual(plan.jobs[0].input, manifest.parent / "inputs/a.png")
        self.assertEqual(plan.jobs[0].reference_output, manifest.parent / "references/a.png")
        self.assertEqual((plan.seed, plan.steps, plan.guidance_scale), (7, 12, 3.5))
        self.assertFalse((self.root / "outputs").exists())

    def test_repeat_case_selection_and_cli_override(self):
        plan = self.batch(self.manifest(), "--case", "b", "--case", "a", "--seed", "43", "--steps", "50", "--guidance", "4.5")
        self.assertEqual([job.id for job in plan.jobs], ["a", "b"])
        self.assertEqual((plan.seed, plan.steps, plan.guidance_scale), (43, 50, 4.5))
        self.assertEqual([job.id for job in self.batch(self.manifest(), "--case", "b").jobs], ["b"])
        with self.assertRaisesRegex(ValueError, "Unknown --case"):
            self.batch(self.manifest(), "--case", "typo")

    def test_missing_input_and_invalid_image_fail_before_loading(self):
        self.input.unlink()
        with self.assertRaisesRegex(ValueError, "Input image does not exist"):
            self.plan()
        self.input.write_text("not an image")
        with self.assertRaisesRegex(ValueError, "supported image header"):
            self.plan()

    def test_adapter_is_required_and_empty_adapter_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            make_parser().parse_args(["--dry-run"])
        with self.assertRaisesRegex(ValueError, "empty"):
            build_plan(make_parser().parse_args(["--lora", "", "--input", str(self.input), "--task", "arrow_edit", "--output", str(self.output)]))
        (self.adapter / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "No adapter_model"):
            self.plan()

    def test_local_base_config_and_remote_ids(self):
        base = self.root / "base"
        base.mkdir()
        with self.assertRaisesRegex(ValueError, "model_index.json"):
            self.plan("--base-model", str(base))
        (base / "model_index.json").write_text("{}")
        (base / "transformer").mkdir()
        (base / "transformer/config.json").write_text("{}")
        self.assertEqual(self.plan("--base-model", str(base)).base_model, str(base))
        plan = build_plan(make_parser().parse_args(["--lora", "example/siphon-adapter", "--input", str(self.input), "--task", "box_edit", "--output", str(self.output)]))
        self.assertEqual(plan.lora, "example/siphon-adapter")

    def test_revision_is_optional_and_recorded_in_the_plan(self):
        self.assertIsNone(self.plan().base_revision)
        plan = self.plan("--revision", "reproducible-base-commit")
        self.assertEqual(plan.base_revision, "reproducible-base-commit")
        self.assertEqual(plan.to_dict()["base_revision"], "reproducible-base-commit")

    def test_revision_selects_both_base_components_but_not_the_adapter(self):
        transformer_factory, pipeline_factory, adapter_factory = MagicMock(), MagicMock(), MagicMock()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16")
        fake_diffusers = SimpleNamespace(
            LongCatImageEditPipeline=pipeline_factory,
            LongCatImageTransformer2DModel=transformer_factory,
        )
        plan = self.plan("--revision", "base-commit", "--local-files-only")
        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers, "peft": SimpleNamespace(PeftModel=adapter_factory)}):
            pipe, _ = _load_pipeline(plan)
        self.assertEqual(transformer_factory.from_pretrained.call_args.kwargs["revision"], "base-commit")
        self.assertEqual(pipeline_factory.from_pretrained.call_args.kwargs["revision"], "base-commit")
        self.assertNotIn("revision", adapter_factory.from_pretrained.call_args.kwargs)
        self.assertIs(adapter_factory.from_pretrained.call_args.args[0], transformer_factory.from_pretrained.return_value)
        self.assertIs(pipeline_factory.from_pretrained.call_args.kwargs["transformer"], adapter_factory.from_pretrained.return_value.merge_and_unload.return_value)
        pipe.enable_model_cpu_offload.assert_called_once_with()
        transformer_factory.from_pretrained.return_value.to.assert_not_called()

    def test_generated_files_require_overwrite_and_inputs_are_never_overwritten(self):
        self.output.write_bytes(PNG)
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.plan()
        self.assertEqual(self.plan("--overwrite").jobs[0].output, self.output)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.plan("--output", str(self.input), "--overwrite")
        self.output.unlink()
        self.output.with_suffix(".json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.plan()

    def test_reference_outputs_are_protected_even_with_overwrite(self):
        manifest = self.manifest()
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.batch(manifest, "--output-dir", str(manifest.parent / "references"), "--overwrite")

    def test_symlinks_cannot_bypass_reference_protection(self):
        manifest = self.manifest()
        reference = manifest.parent / "references/a.png"
        reference.parent.mkdir()
        reference.write_bytes(PNG)
        output_dir = self.root / "outputs"
        output_dir.mkdir()
        (output_dir / "a.png").symlink_to(reference)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.batch(manifest, "--overwrite")

    def test_adjacent_metadata_cannot_overwrite_adapter_configuration(self):
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.plan("--output", str(self.adapter / "adapter_config.png"), "--overwrite")

    def test_unsafe_case_ids_duplicate_ids_and_nonfinite_guidance(self):
        manifest = self.manifest()
        data = json.loads(manifest.read_text())
        data["examples"][0]["id"] = "../escape"
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.batch(manifest)
        manifest = self.manifest()
        data = json.loads(manifest.read_text())
        data["examples"][1]["id"] = "a"
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.batch(manifest)
        with self.assertRaisesRegex(ValueError, "finite positive"):
            self.plan("--guidance", "nan")

    def test_dry_run_and_help_do_not_import_model_dependencies_or_write_outputs(self):
        guard = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'PIL', 'numpy', 'diffusers', 'peft', 'accelerate'}:
        raise AssertionError('unexpected dependency import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['siphon.inference'] + sys.argv[1:]
runpy.run_module('siphon.inference', run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", guard, "--base-only", "--revision", "pinned-base-commit", "--dry-run", "--input", str(self.input), "--task", "text_edit", "--output", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["jobs"][0]["task"], "text_edit")
        self.assertEqual(data["base_revision"], "pinned-base-commit")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".json").exists())
        result = subprocess.run([sys.executable, "-c", guard, "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)

    def test_model_load_failure_is_recorded_and_returns_nonzero(self):
        plan = self.plan()
        with patch("siphon.inference._load_pipeline", side_effect=RuntimeError("no CUDA")), redirect_stderr(io.StringIO()):
            self.assertEqual(execute_plan(plan), 1)
        report = json.loads(plan.jobs[0].metadata.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phase"], "model_load")
        self.assertIn("no CUDA", report["error"])
        self.assertFalse(self.output.exists())

    def test_batch_records_failed_case_and_continues_remaining_cases(self):
        plan = self.batch(self.manifest())
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda: "test CUDA"))

        def run_job(pipe, torch, request, job):
            if job.id == "a":
                raise RuntimeError("deliberate decoding failure")
            job.output.parent.mkdir(parents=True, exist_ok=True)
            job.output.write_bytes(PNG)
            return {"input_size": [1, 1], "output_size": [1, 1], "inference_seconds": 0.01}

        with patch("siphon.inference._load_pipeline", return_value=(object(), fake_torch)), patch("siphon.inference._run_job", side_effect=run_job), redirect_stderr(io.StringIO()):
            self.assertEqual(execute_plan(plan), 1)
        report = json.loads(plan.run_metadata.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual([record["status"] for record in report["results"]], ["failed", "complete"])
        self.assertIn("decoding failure", json.loads(plan.jobs[0].metadata.read_text())["error"])
        self.assertTrue(plan.jobs[1].output.exists())


if __name__ == "__main__":
    unittest.main()
