"""Dependency-free training planning checks, plus an optional tiny CPU objective test."""

import base64
from contextlib import redirect_stderr
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
from unittest.mock import patch

from siphon.train import (
    BatchCursor, CacheEntry, DEFAULTS, _loss, _pack_latents, _position_ids, _save_checkpoint,
    build_plan, checkpoint_steps, make_parser, warmup_factor,
)


def write_cache(path, *, null_shape=(6, 8), prompt_shape=(6, 8), latent_shape=(4, 4, 4)):
    header = {}
    offset = 0
    tensors = {"target_latents": latent_shape, "ref_latents": latent_shape,
               "prompt_embeds": prompt_shape}
    if null_shape:
        tensors["null_prompt_embeds"] = null_shape
    for name, shape in tensors.items():
        size = 4
        for dim in shape:
            size *= dim
        header[name] = {"shape": list(shape), "dtype": "F32", "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


class TrainingPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        write_cache(self.cache / "a.safetensors")
        self.manifest = self.cache / "manifest.jsonl"
        self.manifest.write_text(json.dumps({"id": "a", "cache_path": "a.safetensors"}) + "\n")
        self.output = self.root / "training"

    def plan(self, *args):
        return build_plan(make_parser().parse_args([
            "--cache-manifest", str(self.manifest), "--output-dir", str(self.output), *args,
        ]))

    def adapter(self, path):
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text('{"peft_type": "LORA"}')
        (path / "adapter_model.safetensors").write_bytes(b"test adapter")

    def test_stage2_is_7600_updates_with_a_new_optimizer(self):
        adapter = self.root / "stage1" / "checkpoints-2400"
        self.adapter(adapter)
        plan = self.plan("--init-lora", str(adapter), "--initial-step", "2400", "--max-steps", "10000")
        self.assertEqual(plan.start_step, 2400)
        self.assertEqual(plan.to_dict()["new_optimizer_updates"], 7600)
        self.assertEqual(plan.to_dict()["optimizer_policy"], "new")
        self.assertEqual(checkpoint_steps(2400, 10000, 200)[-1], 10000)
        self.assertEqual(checkpoint_steps(2400, 2402, 200), [2402])
        self.assertEqual(checkpoint_steps(0, 2, 1), [1, 2])

    def test_initial_step_requires_weights_and_maximum_has_no_extra_update(self):
        with self.assertRaisesRegex(ValueError, "initial_step"):
            self.plan("--initial-step", "2400")
        with self.assertRaisesRegex(ValueError, "max_train_steps"):
            self.plan("--max-steps", "0")
        with self.assertRaisesRegex(ValueError, "greater than"):
            self.plan("--max-steps", "1", "--initial-step", "1", "--init-lora", "author/adapter")

    def test_config_paths_and_cli_override_have_different_bases(self):
        configs = self.root / "configs"
        configs.mkdir()
        config = configs / "stage1.json"
        config.write_text(json.dumps({
            "cache_manifest": "../cache/manifest.jsonl", "output_dir": "../from-config",
            "null_text_ratio": 0.1, "max_train_steps": 8,
        }))
        plan = build_plan(make_parser().parse_args(["--config", str(config), "--output-dir", str(self.output), "--max-steps", "2"]))
        self.assertEqual(plan.output_dir, self.output)
        self.assertEqual(plan.manifest, self.manifest)
        self.assertEqual(plan.settings["max_train_steps"], 2)

    def test_cache_shapes_are_inspected_and_null_features_are_required(self):
        plan = self.plan()
        self.assertEqual(plan.entries[0].shapes, ((4, 4, 4), (4, 4, 4), (6, 8), (6, 8)))
        write_cache(self.cache / "a.safetensors", null_shape=None)
        with self.assertRaisesRegex(ValueError, "null_prompt_embeds"):
            self.plan()
        config = self.root / "zero.json"
        config.write_text('{"null_text_ratio": 0}')
        self.assertEqual(len(self.plan("--config", str(config)).entries), 1)
        write_cache(self.cache / "a.safetensors", latent_shape=(4, 3, 4))
        with self.assertRaisesRegex(ValueError, "even"):
            self.plan("--config", str(config))

    def test_different_null_lengths_are_not_silently_padded(self):
        write_cache(self.cache / "a.safetensors", null_shape=(3, 8))
        self.assertEqual(self.plan().settings["batch_size"], 1)
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            self.plan("--batch-size", "2")

    def test_invalid_ratio_and_unknown_config_fail_before_training(self):
        config = self.root / "bad.json"
        for value in (-0.1, 1.1, True, "0.1"):
            config.write_text(json.dumps({"null_text_ratio": value}))
            with self.assertRaisesRegex(ValueError, "null_text_ratio"):
                self.plan("--config", str(config))
        config.write_text('{"null_text_ratio": 0, "typo": 3}')
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.plan("--config", str(config))

    def test_resume_requires_full_state_and_matching_cache_settings_and_adapter(self):
        original = self.plan("--max-steps", "2")
        checkpoint = self.root / "previous" / "checkpoints-1"
        self.adapter(checkpoint)
        private = checkpoint / "training_state"
        private.mkdir()
        metadata = {
            "settings": original.settings, "cache_fingerprint": original.cache_fingerprint,
            "global_step": 1,
            "adapter_sha256": hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest(),
        }
        (private / "metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "state.pt"):
            self.plan("--resume", str(checkpoint), "--max-steps", "2")
        (private / "state.pt").write_bytes(b"private state fixture")
        plan = self.plan("--resume", str(checkpoint), "--max-steps", "2", "--init-lora", "old/config-initialization")
        self.assertEqual(plan.start_step, 1)
        self.assertIsNone(plan.init_lora)  # Resume overrides a phase config's initialization adapter.
        self.assertEqual(plan.to_dict()["optimizer_policy"], "restore")
        self.assertEqual(plan.to_dict()["new_optimizer_updates"], 1)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.plan("--resume", str(checkpoint), "--max-steps", "2", "--learning-rate", "0.2")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "adapter hash"):
            self.plan("--resume", str(checkpoint), "--max-steps", "2")

    def test_output_cannot_contain_training_data_or_existing_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "cache manifest"):
            self.plan("--output-dir", str(self.cache))
        (self.output / "checkpoints-1").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already has checkpoints"):
            self.plan()

    def test_resume_cursor_returns_exact_remaining_samples_across_epoch_boundaries(self):
        shape = ((4, 4, 4), (4, 4, 4), (5, 8))
        other_shape = ((4, 8, 4), (4, 8, 4), (7, 8))
        entries = tuple(CacheEntry(str(i), self.root / str(i), shape if i % 2 else other_shape) for i in range(7))
        cursor = BatchCursor(entries, 2, 43)
        for _ in range(3):
            batch = cursor.next()
            self.assertEqual(len({entries[i].shapes for i in batch}), 1)
        state = cursor.state_dict()
        expected = [cursor.next() for _ in range(12)]
        restored = BatchCursor(entries, 2, 43)
        restored.load_state_dict(state)
        self.assertEqual([restored.next() for _ in range(12)], expected)

    def test_stage_warmup_restarts_but_resume_keeps_the_counter(self):
        self.assertEqual(warmup_factor(0, 250), 1 / 250)
        self.assertEqual(warmup_factor(249, 250), 1)
        self.assertEqual(warmup_factor(400, 250), 1)
        self.assertEqual(warmup_factor(0, 0), 1)

    def test_checkpoint_export_explicitly_disables_embedding_hub_probe(self):
        plan = self.plan("--max-steps", "2")
        saved = {}

        class Adapter:
            def save_pretrained(self, directory, **options):
                saved.update(options)
                (directory / "adapter_model.safetensors").write_bytes(b"adapter weights")
                (directory / "adapter_config.json").write_text('{"peft_type":"LORA"}')

        def save_state(state, path):
            saved["state"] = state
            path.write_bytes(b"private resume state")

        torch = SimpleNamespace(
            save=save_state, get_rng_state=lambda: b"CPU RNG",
            cuda=SimpleNamespace(get_rng_state=lambda: b"CUDA RNG"),
        )
        checkpoint = _save_checkpoint(torch, plan, Adapter(),
                                      SimpleNamespace(state_dict=lambda: {"momentum": "test"}),
                                      BatchCursor(plan.entries, 1, 43), 1, 1)
        self.assertIs(saved["save_embedding_layers"], False)
        self.assertIs(saved["safe_serialization"], True)
        self.assertTrue((checkpoint / "adapter_config.json").is_file())
        self.assertTrue((checkpoint / "training_state/state.pt").is_file())
        self.assertEqual(saved["state"]["scheduler"]["completed_updates"], 1)

    def test_dry_run_has_no_tensor_dependencies_and_creates_no_outputs(self):
        guard = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'PIL', 'numpy', 'diffusers', 'peft', 'accelerate', 'safetensors'}:
        raise AssertionError('unexpected import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['siphon.train'] + sys.argv[1:]
runpy.run_module('siphon.train', run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", guard, "--dry-run", "--cache-manifest", str(self.manifest),
                                 "--output-dir", str(self.output), "--max-steps", "2"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["new_optimizer_updates"], 2)
        self.assertFalse(self.output.exists())


class TinyCPUObjectiveTest(unittest.TestCase):
    def test_peft_explicit_false_does_not_look_up_remote_base_embedding_config(self):
        try:
            import torch
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("PyTorch/PEFT are optional for the dependency-free test suite")
        model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)),
                               LoraConfig(r=2, lora_alpha=2, target_modules=["0"]))
        model.peft_config["default"].base_model_name_or_path = "test-public/base-model"
        with tempfile.TemporaryDirectory() as temporary:
            with patch("peft.utils.save_and_load.check_file_exists_on_hf_hub",
                       side_effect=AssertionError("unexpected Hub lookup")) as probe:
                model.save_pretrained(temporary, safe_serialization=True, save_embedding_layers=False)
                probe.assert_not_called()
            self.assertTrue((Path(temporary) / "adapter_model.safetensors").is_file())

    def test_cached_objective_backpropagates_through_only_the_trainable_layer(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional for the dependency-free test suite")

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Linear(16, 16, bias=False)
                self.frozen.requires_grad_(False)
                self.adapter = torch.nn.Linear(16, 16, bias=False)
                self.last_shapes = None

            def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, **kwargs):
                self.last_shapes = (hidden_states.shape, img_ids.shape, txt_ids.shape)
                return (self.frozen(hidden_states) + self.adapter(hidden_states),)

        torch.manual_seed(43)
        model = TinyModel()
        target = torch.randn(1, 4, 4, 6)
        batch = {"target_latents": target, "ref_latents": torch.randn_like(target),
                 "prompt_embeds": torch.randn(1, 7, 8)}
        loss = _loss(torch, model, SimpleNamespace(), batch, dynamic_shift=False)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(model.frozen.weight.grad)
        self.assertGreater(model.adapter.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.last_shapes, ((1, 12, 16), (12, 3), (7, 3)))
        text_ids, image_ids = _position_ids(torch, target, target, 7)
        self.assertEqual(image_ids[:, 0].tolist(), [1.] * 6 + [2.] * 6)
        self.assertEqual(image_ids[0].tolist(), [1., 7., 7.])
        self.assertEqual(text_ids[-1].tolist(), [0., 6., 6.])


if __name__ == "__main__":
    unittest.main()
