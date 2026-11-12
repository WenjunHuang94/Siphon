"""Prepare the tested 10000 PEFT adapter; upload only with explicit --upload.

Authentication uses `hf auth login` or HF_TOKEN. Never put tokens in arguments.
The source checkpoint is never modified. Optimizer and base-model states are
excluded, even if the source checkpoint contains them.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA256 = "0ed9694868d2552d243140caf242ef38e0d69f6673a3eec36e4ddc53ed459535"
BASE_MODEL = "meituan-longcat/LongCat-Image-Edit"
BASE_REVISION = "7b54ef423aa7854be7861600024be5c56ab7875a"
FILES = ("adapter_model.safetensors", "adapter_config.json", "README.md", "SHA256SUMS", "release.json", "LICENSE")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(checkpoint, expected_sha=WEIGHT_SHA256):
    checkpoint = Path(checkpoint).resolve()
    weight = checkpoint / "adapter_model.safetensors"
    config = checkpoint / "adapter_config.json"
    if not weight.is_file() or not config.is_file():
        raise ValueError("A standard PEFT adapter_model.safetensors and adapter_config.json are required.")
    digest = sha256(weight)
    if digest != expected_sha:
        raise ValueError("Adapter hash does not match the requested 10000 release; refusing a different checkpoint.")
    data = json.loads(config.read_text(encoding="utf-8"))
    if data.get("peft_type") != "LORA":
        raise ValueError("Expected a PEFT LORA adapter configuration.")
    return checkpoint, data, digest, weight.stat().st_size


def prepare(checkpoint, destination, *, repo_id=None, weight_license=None, expected_sha=WEIGHT_SHA256):
    checkpoint, config, digest, size = inspect_checkpoint(checkpoint, expected_sha)
    destination = Path(destination).resolve()
    if destination == checkpoint:
        raise ValueError("Staging directory must differ from the source checkpoint.")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use a new, empty staging directory to avoid mixing release files.")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint / "adapter_model.safetensors", destination / "adapter_model.safetensors")
    # Normalize metadata only. The adapter tensors remain byte-for-byte identical.
    config["base_model_name_or_path"] = BASE_MODEL
    config["revision"] = BASE_REVISION
    config["auto_mapping"] = {
        "base_model_class": "LongCatImageTransformer2DModel",
        "parent_library": "diffusers.models.transformers.transformer_longcat_image",
    }
    (destination / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    template = (ROOT / "model_cards/longcat-10000.md").read_text(encoding="utf-8")
    card = template.replace("{{ADAPTER_REPO_ID}}", repo_id or "YOUR_HF_ACCOUNT/Siphon-LongCat-LoRA")
    card = card.replace("{{LICENSE_METADATA}}", f"license: {weight_license}" if weight_license else "# Weight license must be selected before public upload.")
    card = card.replace("{{LICENSE_NOTE}}", f"Weight license: `{weight_license}`." if weight_license else "Draft package: the weight license has not yet been selected. Public upload requires an explicit license choice.")
    (destination / "README.md").write_text(card, encoding="utf-8")
    if weight_license == "apache-2.0":
        shutil.copy2(ROOT / "LICENSE", destination / "LICENSE")
    release = {
        "checkpoint": "stage2-final-0225/checkpoints-10000",
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "adapter_sha256": digest, "adapter_bytes": size,
        "repo_id": repo_id, "weight_license": weight_license,
        "metadata_normalized": True,
        "excluded": ["optimizer states", "full model states", "cached latents", "base weights", "duplicate transformer adapter"],
    }
    (destination / "release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    hashes = [f"{sha256(destination / name)}  {name}" for name in FILES
              if name != "SHA256SUMS" and (destination / name).is_file()]
    (destination / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    if sha256(destination / "adapter_model.safetensors") != digest:
        raise RuntimeError("Staged adapter failed its checksum check.")
    return release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stage-dir", type=Path, default=Path("outputs/hub-adapter"))
    parser.add_argument("--repo-id", help="Your actual Hugging Face namespace/model-name")
    parser.add_argument("--license", dest="weight_license", help="Explicit model-weight license ID, e.g. apache-2.0")
    parser.add_argument("--expected-sha256", default=WEIGHT_SHA256)
    parser.add_argument("--dry-run", action="store_true", help="Inspect without copying files or accessing the Hub")
    parser.add_argument("--upload", action="store_true", help="Create/upload the model repository after staging and authentication")
    args = parser.parse_args(argv)
    try:
        if args.repo_id and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo_id):
            raise ValueError("repo-id must be namespace/model-name.")
        if args.weight_license and not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", args.weight_license):
            raise ValueError("Use a valid lowercase model-license ID.")
        if args.upload and (args.dry_run or not args.repo_id or not args.weight_license):
            raise ValueError("Upload requires --repo-id and an explicit --license; cannot combine with --dry-run.")
        _, _, digest, size = inspect_checkpoint(args.checkpoint, args.expected_sha256)
        if args.dry_run:
            print(json.dumps({"adapter_sha256": digest, "adapter_bytes": size,
                              "stage_dir": str(args.stage_dir), "upload": False,
                              "files": [name for name in FILES if name != "LICENSE" or args.weight_license == "apache-2.0"]}, indent=2))
            return 0
        if args.upload:
            from huggingface_hub import HfApi, get_token
            token = get_token()
            if not token:
                raise ValueError("Not logged in to Hugging Face. Run `hf auth login` in your terminal; do not send tokens in chat.")
            api = HfApi(token=token)
            profile = api.whoami()
            print(f"Authenticated Hub account: {profile.get('name')}; target: {args.repo_id}", flush=True)
        release = prepare(args.checkpoint, args.stage_dir, repo_id=args.repo_id,
                          weight_license=args.weight_license, expected_sha=args.expected_sha256)
        print(json.dumps(release, indent=2))
        if args.upload:
            api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
            commit = api.upload_folder(
                repo_id=args.repo_id, repo_type="model", folder_path=str(args.stage_dir),
                allow_patterns=list(FILES), commit_message="Release Siphon LongCat LoRA checkpoint 10000",
            )
            print(f"Uploaded: https://huggingface.co/{args.repo_id}\nCommit: {commit}")
        else:
            print("Prepared locally only. No Hub upload was attempted.")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
