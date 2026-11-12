# SPDX-License-Identifier: Apache-2.0
"""Validate image pairs and cache frozen LongCat features for LoRA training.

The reference path resizes both images to the same geometry (512 x 512 by
default). It uses Diffusers' own image-conditioned prompt encoder and scaled VAE
mode latents, without loading the denoising transformer. See TRAINING.md for the
scope of this single-GPU reference implementation and dataset requirements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
from typing import Any

from .inference import TASK_PROMPTS


CACHE_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class Pair:
    id: str
    input: Path
    target: Path
    prompt: str
    task: str | None
    input_size: tuple[int, int]
    target_size: tuple[int, int]
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "input": str(self.input), "target": str(self.target),
            "task": self.task, "prompt": self.prompt, "line": self.line,
            "input_size": list(self.input_size), "target_size": list(self.target_size),
        }


def read_image_size(path: Path) -> tuple[int, int]:
    """Read PNG/JPEG/WebP dimensions using only stdlib; pixel decode is later."""
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    with path.open("rb") as stream:
        header = stream.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            size = struct.unpack(">II", header[16:24])
        elif header.startswith(b"\xff\xd8"):
            stream.seek(2)
            size = None
            while True:
                byte = stream.read(1)
                if not byte:
                    break
                if byte != b"\xff":
                    continue
                marker = stream.read(1)
                while marker == b"\xff":
                    marker = stream.read(1)
                if not marker or marker in (b"\xda", b"\xd9"):
                    break
                if marker in (b"\xd8", b"\x01") or b"\xd0" <= marker <= b"\xd7":
                    continue
                length_bytes = stream.read(2)
                if len(length_bytes) != 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]
                if length < 2:
                    break
                if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    sof = stream.read(5)
                    if len(sof) == 5:
                        height, width = struct.unpack(">HH", sof[1:])
                        size = (width, height)
                    break
                stream.seek(length - 2, 1)
            if size is None:
                raise ValueError(f"Cannot read JPEG dimensions: {path}")
        elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            if header[12:16] == b"VP8X" and len(header) >= 30:
                size = (int.from_bytes(header[24:27], "little") + 1, int.from_bytes(header[27:30], "little") + 1)
            elif header[12:16] == b"VP8L" and len(header) >= 25 and header[20] == 0x2F:
                bits = int.from_bytes(header[21:25], "little")
                size = ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
            elif header[12:16] == b"VP8 " and len(header) >= 30 and header[23:26] == b"\x9d\x01\x2a":
                width, height = struct.unpack("<HH", header[26:30])
                size = (width & 0x3FFF, height & 0x3FFF)
            else:
                raise ValueError(f"Cannot read WebP dimensions: {path}")
        else:
            raise ValueError(f"Expected a PNG, JPEG, or WebP image: {path}")
    if any(dimension <= 0 for dimension in size):
        raise ValueError(f"Invalid image geometry {size}: {path}")
    return size


def _path(value: Any, parent: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty image path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def load_pairs(manifest: Path, limit: int | None = None) -> list[Pair]:
    """Accept portable input/target/task rows and legacy ref_img_path/img_path."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    manifest = manifest.expanduser().resolve()
    pairs = []
    seen = set()
    with manifest.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each JSONL row must be an object")
                case_id = row.get("id", f"sample_{number:06d}")
                if not isinstance(case_id, str) or not _SAFE_ID.fullmatch(case_id):
                    raise ValueError(f"unsafe sample ID: {case_id!r}")
                if case_id in seen:
                    raise ValueError(f"duplicate sample ID: {case_id}")
                if "input" in row or "target" in row:
                    input_path = _path(row.get("input"), manifest.parent, "input")
                    target_path = _path(row.get("target"), manifest.parent, "target")
                else:
                    input_path = _path(row.get("ref_img_path"), manifest.parent, "ref_img_path (marked input)")
                    target_path = _path(row.get("img_path"), manifest.parent, "img_path (clean target)")
                if input_path == target_path:
                    raise ValueError("marked input and clean target must be different files")
                task = row.get("task")
                if task is not None and task not in TASK_PROMPTS:
                    raise ValueError(f"unknown task: {task!r}")
                prompt = row.get("prompt", TASK_PROMPTS.get(task))
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("supply a known task or a nonempty generic prompt")
                for dimension in ("width", "height"):
                    if dimension in row and (type(row[dimension]) is not int or row[dimension] < 1):
                        raise ValueError(f"legacy {dimension} must be a positive integer")
                pair = Pair(case_id, input_path, target_path, prompt, task,
                            read_image_size(input_path), read_image_size(target_path), number)
            except (ValueError, OSError) as exc:
                raise ValueError(f"{manifest.name}:{number}: {exc}") from exc
            pairs.append(pair)
            seen.add(case_id)
            if limit is not None and len(pairs) >= limit:
                break
    if not pairs:
        raise ValueError("The paired training manifest is empty")
    return pairs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="Paired JSONL; image paths resolve relative to this file")
    parser.add_argument("--cache-dir", type=Path, help="Write manifest.jsonl, metadata, and per-pair safetensors here")
    parser.add_argument("--base-model", default="meituan-longcat/LongCat-Image-Edit")
    parser.add_argument("--revision", help="Optional base snapshot revision")
    parser.add_argument("--resolution", type=int, default=512, help="Square paired resize; default 512")
    parser.add_argument("--width", type=int, help="Explicit paired width; requires --height")
    parser.add_argument("--height", type=int, help="Explicit paired height; requires --width")
    parser.add_argument("--limit", type=int, help="Only validate/cache the first N nonempty records")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and image headers/geometry without model dependencies")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated cache files; never source images")
    return parser


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be supplied together")
    width, height = (args.width, args.height) if args.width is not None else (args.resolution, args.resolution)
    if any(dimension < 16 or dimension % 16 for dimension in (width, height)):
        raise ValueError("Training width and height must be positive multiples of 16")
    if not args.base_model or not args.base_model.strip():
        raise ValueError("--base-model must not be empty")
    if args.cache_dir is None and not args.dry_run:
        raise ValueError("--cache-dir is required to create features; use --dry-run for validation only")
    pairs = load_pairs(args.manifest, args.limit)
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    source_paths = {args.manifest.expanduser().resolve()}
    source_paths.update(path for pair in pairs for path in (pair.input, pair.target))
    if cache_dir:
        destinations = [cache_dir / "manifest.jsonl", cache_dir / "dataset_metadata.json"]
        destinations.extend(cache_dir / "tensors" / f"{pair.id}.safetensors" for pair in pairs)
        for path in destinations:
            if path.resolve() in source_paths:
                raise ValueError(f"Refusing to overwrite source data: {path}")
            if path.exists() and (not args.overwrite or not path.is_file()):
                raise ValueError(f"Cache file already exists: {path}; use --overwrite")
            if path.parent.exists() and not path.parent.is_dir():
                raise ValueError(f"Cache parent is not a directory: {path.parent}")
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source_manifest": str(args.manifest.expanduser().resolve()),
        "base_model": args.base_model, "base_revision": args.revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "cache_manifest": str(cache_dir / "manifest.jsonl") if cache_dir else None,
        "width": width, "height": height, "resize": "paired_resize_no_crop",
        "dtype": "bfloat16", "latent_layout": "CHW", "latents_scaled": True,
        "prompt_layout": "TD", "has_null_prompt_embeds": True,
        "tensor_keys": ["target_latents", "ref_latents", "prompt_embeds", "null_prompt_embeds"],
        "local_files_only": args.local_files_only,
        "count": len(pairs), "pairs": [pair.to_dict() for pair in pairs],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_encoders(plan: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from diffusers import LongCatImageEditPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature caching; --dry-run is CPU-only")
    options = {"torch_dtype": torch.bfloat16, "local_files_only": plan["local_files_only"], "transformer": None}
    if plan["base_revision"] is not None:
        options["revision"] = plan["base_revision"]
    pipe = LongCatImageEditPipeline.from_pretrained(plan["base_model"], **options)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.enable_model_cpu_offload()
    return pipe, torch


def _encode_pair(pipe: Any, torch: Any, record: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    from PIL import Image

    with Image.open(record["input"]) as image:
        marked = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    with Image.open(record["target"]) as image:
        target = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(prompt=record["prompt"], image=marked, num_images_per_prompt=1)
        prompt_embeds = prompt_embeds[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        null_embeds, _ = pipe.encode_prompt(prompt="", image=marked, num_images_per_prompt=1)
        null_embeds = null_embeds[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        # _encode_vae_image uses latent_dist.mode(), then (latent - shift) * scale,
        # exactly as the image-edit pipeline's reference-image encoding does.
        # AutoencoderKL.encode() is not a regular module forward; move its input
        # explicitly even when Accelerate offload hooks move the VAE itself.
        device = pipe._execution_device
        ref_pixels = pipe.image_processor.preprocess(marked, height=height, width=width).to(device=device, dtype=torch.bfloat16)
        target_pixels = pipe.image_processor.preprocess(target, height=height, width=width).to(device=device, dtype=torch.bfloat16)
        ref_latents = pipe._encode_vae_image(ref_pixels, generator=None)[0].detach().to("cpu").contiguous()
        target_latents = pipe._encode_vae_image(target_pixels, generator=None)[0].detach().to("cpu").contiguous()
    if target_latents.shape != ref_latents.shape or target_latents.ndim != 3:
        raise ValueError("Expected matching unpacked target/reference CHW latents")
    if any(size % 2 for size in target_latents.shape[-2:]):
        raise ValueError("Latent height/width must be even for LongCat 2x2 packing")
    if prompt_embeds.ndim != 2 or null_embeds.ndim != 2:
        raise ValueError("Expected prompt embeddings without a batch dimension")
    return {"target_latents": target_latents, "ref_latents": ref_latents,
            "prompt_embeds": prompt_embeds, "null_prompt_embeds": null_embeds}


def create_cache(plan: dict[str, Any], *, overwrite: bool = False) -> int:
    from safetensors.torch import save_file

    cache_dir = Path(plan["cache_dir"])
    metadata = {key: value for key, value in plan.items() if key != "pairs"}
    metadata.update({"status": "running", "completed": 0, "source_manifest_sha256": _sha256(Path(plan["source_manifest"]))})
    started = time.perf_counter()
    try:
        pipe, torch = _load_encoders(plan)
        metadata["vae_scaling_factor"] = pipe.vae.config.scaling_factor
        metadata["vae_shift_factor"] = pipe.vae.config.shift_factor
        metadata["vae_scale_factor"] = pipe.vae_scale_factor
        metadata["versions"] = {}
        for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "Pillow"):
            try:
                metadata["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        (cache_dir / "tensors").mkdir(parents=True, exist_ok=True)
        with (cache_dir / "manifest.jsonl").open("w" if overwrite else "x", encoding="utf-8") as manifest:
            for index, record in enumerate(plan["pairs"], 1):
                print(f"[{index}/{plan['count']}] Caching {record['id']}", flush=True)
                input_hash = _sha256(Path(record["input"]))
                target_hash = _sha256(Path(record["target"]))
                tensors = _encode_pair(pipe, torch, record, plan["width"], plan["height"])
                relative = f"tensors/{record['id']}.safetensors"
                output = cache_dir / relative
                if output.exists() and not overwrite:
                    raise FileExistsError(f"Cache file already exists: {output}")
                tensor_metadata = {
                    "cache_schema_version": str(CACHE_SCHEMA_VERSION),
                    "latent_layout": "CHW", "latents_scaled": "true", "prompt_layout": "TD",
                    "input_sha256": input_hash, "target_sha256": target_hash,
                    "prompt": record["prompt"], "null_prompt": "",
                }
                save_file(tensors, str(output), metadata=tensor_metadata)
                entry = {
                    "id": record["id"], "cache_path": relative,
                    "cache_sha256": _sha256(output),
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "task": record["task"], "prompt": record["prompt"],
                    "width": plan["width"], "height": plan["height"],
                    "latent_layout": "CHW", "latents_scaled": True,
                    "prompt_layout": "TD", "has_null_prompt_embeds": True,
                    "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
                    "source": {
                        "manifest": os.path.relpath(plan["source_manifest"], cache_dir),
                        "line": record["line"],
                        "input": os.path.relpath(record["input"], cache_dir),
                        "target": os.path.relpath(record["target"], cache_dir),
                        "input_size": record["input_size"], "target_size": record["target_size"],
                        "input_sha256": input_hash, "target_sha256": target_hash,
                    },
                }
                manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                manifest.flush()
                metadata["completed"] = index
                pipe.maybe_free_model_hooks()
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(metadata["error"], file=sys.stderr, flush=True)
    metadata["seconds"] = round(time.perf_counter() - started, 3)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return int(metadata["status"] != "complete")


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        return create_cache(plan, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
