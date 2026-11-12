"""Keep the portable demo manifest and its recorded assets consistent."""

import hashlib
import json
from pathlib import Path
import re
import struct
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/examples.json"


class RecordedExamplesTests(unittest.TestCase):
    def test_each_type_has_two_cases_and_asset_hashes_and_dimensions_match(self):
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        cases = data["examples"]
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case["id"] for case in cases}), 8)
        for category in ("I", "II", "III", "IV"):
            self.assertEqual(sum(case["type"] == category for case in cases), 2)
        for case in cases:
            for field, digest_key, size_key in (
                ("input", "input_sha256", "input_size"),
                ("reference_output", "reference_output_sha256", "reference_output_size"),
            ):
                with self.subTest(case=case["id"], field=field):
                    path = MANIFEST.parent / case[field]
                    self.assertFalse(Path(case[field]).is_absolute())
                    contents = path.read_bytes()
                    self.assertEqual(hashlib.sha256(contents).hexdigest(), case[digest_key])
                    self.assertEqual(contents[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertEqual(list(struct.unpack(">II", contents[16:24])), case[size_key])

    def test_public_manifest_has_provenance_and_no_server_paths(self):
        raw = MANIFEST.read_text(encoding="utf-8")
        data = json.loads(raw)
        self.assertNotIn("/home/", raw)
        self.assertNotIn("/opt/", raw)
        self.assertEqual(len(data["provenance"]["adapter"]["sha256"]), 64)
        self.assertEqual(len(data["provenance"]["base_revision"]), 40)
        self.assertTrue(all(case["observation"] for case in data["examples"]))

    def test_documentation_local_file_links_exist(self):
        for relative in ("README.md", "README_zh-CN.md", "examples/README.md", "THIRD_PARTY_NOTICES.md"):
            path = ROOT / relative
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)]+)\)", text):
                target = target.split("#", 1)[0]
                if not target or re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target):
                    continue
                with self.subTest(document=relative, target=target):
                    self.assertTrue((path.parent / unquote(target)).exists())


if __name__ == "__main__":
    unittest.main()
