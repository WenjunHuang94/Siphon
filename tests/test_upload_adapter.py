"""Prevent publishing the wrong tensor or accidentally copying training states."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("siphon_upload_tool", ROOT / "tools/upload_adapter.py")
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class AdapterReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.weight = b"test adapter placeholder; tensor loading is not required by the exporter"
        self.sha = hashlib.sha256(self.weight).hexdigest()
        (self.checkpoint / "adapter_model.safetensors").write_bytes(self.weight)
        self.config = {"peft_type": "LORA", "r": 64, "lora_alpha": 8,
                       "base_model_name_or_path": None,
                       "auto_mapping": {"parent_library": "longcat_image.models.longcat_image_dit"}}
        (self.checkpoint / "adapter_config.json").write_text(json.dumps(self.config))
        private = self.checkpoint / "pytorch_model"
        private.mkdir()
        (private / "optim_states.pt").write_bytes(b"do not publish")

    def test_prepare_copies_only_adapter_and_public_metadata_without_changing_source(self):
        destination = self.root / "release"
        upload.prepare(self.checkpoint, destination, expected_sha=self.sha)
        self.assertEqual((destination / "adapter_model.safetensors").read_bytes(), self.weight)
        self.assertEqual({path.name for path in destination.iterdir()},
                         {"adapter_model.safetensors", "adapter_config.json", "README.md", "SHA256SUMS", "release.json"})
        self.assertEqual(json.loads((self.checkpoint / "adapter_config.json").read_text()), self.config)
        normalized = json.loads((destination / "adapter_config.json").read_text())
        self.assertEqual(normalized["base_model_name_or_path"], upload.BASE_MODEL)
        self.assertNotIn("longcat_image.models", normalized["auto_mapping"]["parent_library"])
        self.assertIsNone(json.loads((destination / "release.json").read_text())["weight_license"])

    def test_hash_mismatch_and_nonempty_staging_fail_before_copying(self):
        destination = self.root / "release"
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            upload.prepare(self.checkpoint, destination)
        self.assertFalse(destination.exists())
        destination.mkdir()
        (destination / "unrelated.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty staging"):
            upload.prepare(self.checkpoint, destination, expected_sha=self.sha)
        self.assertEqual((destination / "unrelated.txt").read_text(), "keep")

    def test_dry_run_writes_nothing_and_does_not_need_hub_auth(self):
        destination = self.root / "release"
        with redirect_stdout(io.StringIO()) as output:
            result = upload.main(["--checkpoint", str(self.checkpoint), "--stage-dir", str(destination),
                                  "--expected-sha256", self.sha, "--dry-run"])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(output.getvalue())["upload"])
        self.assertFalse(destination.exists())

    def test_upload_requires_explicit_target_and_weight_license(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            upload.main(["--checkpoint", str(self.checkpoint), "--upload"])


if __name__ == "__main__":
    unittest.main()
