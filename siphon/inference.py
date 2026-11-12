# SPDX-License-Identifier: Apache-2.0
"""Portable single-image and example-manifest inference.

Adapted from the authors' LongCat-based research inference scripts. See
THIRD_PARTY_NOTICES.md for the upstream implementation and attribution.

Planning, --help, and --dry-run require only the Python standard library. Model
dependencies are imported only after the entire request has been validated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any


TASK_PROMPTS = {
    "text_to_image": "Generate a realistic image based on the text description in the image",
    "text_edit": "Edit the image according to the text instruction in the image",
    "arrow_edit": "Follow the visual pointer and text to edit the image",
    "box_edit": "Modify the image at the annotated location according to the text instruction",
}
TASK_TYPES = {
    "text_to_image": "I", "text_edit": "II", "arrow_edit": "III", "box_edit": "IV",
}
DEFAULTS = {"seed": 43, "steps": 50, "guidance_scale": 4.5}
_HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_OUTPUT_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}


@dataclass(frozen=True)
class Job:
    id: str
    task: str
    input: Path
    output: Path
    reference_output: Path | None = None

    @property
    def metadata(self) -> Path:
        return self.output.with_suffix(".json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "type": TASK_TYPES[self.task], "task": self.task,
            "prompt": TASK_PROMPTS[self.task], "input": str(self.input),
            "output": str(self.output), "metadata": str(self.metadata),
            "reference_output": str(self.reference_output) if self.reference_output else None,
        }


@dataclass(frozen=True)
class Plan:
    base_model: str
    lora: str | None
    jobs: tuple[Job, ...]
    seed: int
    steps: int
    guidance_scale: float
    offload: str
    local_files_only: bool
    overwrite: bool
    manifest: Path | None = None
    run_metadata: Path | None = None
    base_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model, "base_revision": self.base_revision,
            "lora": self.lora,
            "base_only": self.lora is None, "dtype": "bfloat16",
            "settings": {
                "seed": self.seed, "steps": self.steps,
                "guidance_scale": self.guidance_scale, "negative_prompt": "",
                "num_images_per_prompt": 1, "offload": self.offload,
                "local_files_only": self.local_files_only,
            },
            "examples": str(self.manifest) if self.manifest else None,
            "run_metadata": str(self.run_metadata) if self.run_metadata else None,
            "jobs": [job.to_dict() for job in self.jobs],
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-model", default="meituan-longcat/LongCat-Image-Edit")
    parser.add_argument("--revision", help="Optional Hugging Face base-model revision; does not select an adapter revision")
    adapter = parser.add_mutually_exclusive_group(required=True)
    adapter.add_argument("--lora", help="PEFT adapter directory or Hugging Face repository ID")
    adapter.add_argument("--base-only", action="store_true", help="Explicitly run the unadapted base model")
    parser.add_argument("--input", type=Path, help="A complete marked RGB image; annotations stay in the pixels")
    parser.add_argument("--task", choices=TASK_PROMPTS)
    parser.add_argument("--output", type=Path, help="Single-image output (.png, .jpg, or .webp)")
    parser.add_argument("--examples", type=Path, help="Version 1 example manifest")
    parser.add_argument("--output-dir", type=Path, help="Batch output directory")
    parser.add_argument("--case", action="append", default=[], help="Select a manifest case ID; repeat for multiple cases")
    parser.add_argument("--seed", type=int, default=None, help="Default: 43, or manifest defaults.seed")
    parser.add_argument("--steps", type=int, default=None, help="Default: 50, or manifest defaults.steps")
    parser.add_argument("--guidance", type=float, default=None, help="Default: 4.5, or manifest defaults.guidance_scale")
    parser.add_argument("--offload", choices=("model", "none"), default="model")
    parser.add_argument("--local-files-only", action="store_true", help="Disable model and adapter downloads")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print planned jobs without importing model dependencies")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing generated outputs and their metadata; never inputs or references")
    return parser


def _json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _model_source(value: str, *, adapter: bool) -> str:
    if not value or not value.strip():
        raise ValueError("An empty model/LoRA path is not allowed; use --base-only explicitly for the base model")
    path = Path(value).expanduser()
    if not path.exists() and _HF_ID.fullmatch(value) and not value.startswith((".", "~", "/")):
        return value  # No network access during planning, even for remote IDs.
    if not path.is_dir():
        raise ValueError(f"Model/LoRA directory does not exist: {path}")
    path = path.resolve()
    configs = ["adapter_config.json"] if adapter else ["model_index.json", "transformer/config.json"]
    for name in configs:
        _json_object(path / name)
    if adapter and not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"No adapter_model.safetensors or adapter_model.bin in {path}")
    return str(path)


def _image_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Input image does not exist: {path}")
    with path.open("rb") as stream:
        header = stream.read(16)
    supported = (
        header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8\xff")
        or header.startswith((b"GIF87a", b"GIF89a", b"BM", b"II*\x00", b"MM\x00*"))
        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
    )
    if not supported:
        raise ValueError(f"Input does not have a supported image header: {path}")


def _relative_path(value: Any, parent: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest {field} must be a nonempty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def build_plan(args: argparse.Namespace) -> Plan:
    """Validate the full request without importing image, tensor, or model packages."""
    base_model = _model_source(args.base_model, adapter=False)
    lora = _model_source(args.lora, adapter=True) if args.lora is not None else None
    if lora is None and not args.base_only:
        raise ValueError("Supply --lora or explicitly choose --base-only")
    jobs = []
    settings = dict(DEFAULTS)
    protected: set[Path] = set()
    for source, names in (
        (base_model, ("model_index.json", "transformer/config.json")),
        (lora, ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")),
    ):
        if source is not None and Path(source).is_dir():
            protected.update((Path(source) / name).resolve() for name in names)
    manifest = None
    run_metadata = None
    if args.examples is not None:
        if args.input is not None or args.task is not None or args.output is not None:
            raise ValueError("--examples cannot be combined with --input, --task, or --output")
        if args.output_dir is None:
            raise ValueError("--examples requires --output-dir")
        manifest = args.examples.expanduser().resolve()
        data = _json_object(manifest)
        if data.get("schema_version") != 1:
            raise ValueError("Manifest schema_version must be 1")
        defaults = data.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ValueError("Manifest defaults must be an object")
        settings.update({key: defaults[key] for key in DEFAULTS if key in defaults})
        entries = data.get("examples")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Manifest examples must be a nonempty list")
        by_id = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each manifest example must be an object")
            case_id, task = entry.get("id"), entry.get("task")
            if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
                raise ValueError(f"Unsafe or missing example ID: {case_id!r}")
            if case_id in by_id:
                raise ValueError(f"Duplicate example ID: {case_id}")
            if task not in TASK_PROMPTS or entry.get("type") != TASK_TYPES[task]:
                raise ValueError(f"Example {case_id}: task/type must be one of {TASK_TYPES}")
            input_path = _relative_path(entry.get("input"), manifest.parent, "input")
            reference = entry.get("reference_output")
            reference_path = _relative_path(reference, manifest.parent, "reference_output") if reference is not None else None
            protected.add(input_path)
            if reference_path:
                protected.add(reference_path)
            by_id[case_id] = (task, input_path, reference_path)
        unknown = set(args.case) - by_id.keys()
        if unknown:
            raise ValueError(f"Unknown --case ID(s): {', '.join(sorted(unknown))}")
        output_dir = args.output_dir.expanduser().resolve()
        run_metadata = output_dir / "run_metadata.json"
        for case_id, (task, input_path, reference) in by_id.items():
            if not args.case or case_id in args.case:
                jobs.append(Job(case_id, task, input_path, output_dir / f"{case_id}.png", reference))
    else:
        if args.output_dir is not None or args.case:
            raise ValueError("--output-dir and --case require --examples")
        if args.input is None or args.task is None or args.output is None:
            raise ValueError("Single-image inference requires --input, --task, and --output")
        input_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        protected.add(input_path)
        jobs.append(Job(output_path.stem, args.task, input_path, output_path))
    if args.seed is not None:
        settings["seed"] = args.seed
    if args.steps is not None:
        settings["steps"] = args.steps
    if args.guidance is not None:
        settings["guidance_scale"] = args.guidance
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1")
    if type(settings["steps"]) is not int or settings["steps"] < 1:
        raise ValueError("steps must be a positive integer")
    guidance = settings["guidance_scale"]
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not math.isfinite(guidance) or guidance <= 0:
        raise ValueError("guidance_scale must be a finite positive number")
    destinations = []
    for job in jobs:
        _image_file(job.input)
        if job.output.suffix.lower() not in _OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output extension: {job.output}")
        destinations.extend((job.output, job.metadata))
    if run_metadata:
        destinations.append(run_metadata)
    if manifest:
        protected.add(manifest)
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("Output and metadata paths collide; rename the case or output")
    for path in destinations:
        if path.resolve() in protected:
            raise ValueError(f"Refusing to overwrite an input, reference output, manifest, or model configuration: {path}")
        if path.exists() and (not args.overwrite or not path.is_file()):
            raise ValueError(f"Output already exists: {path}; use --overwrite for generated files")
        if path.parent.exists() and not path.parent.is_dir():
            raise ValueError(f"Output parent is not a directory: {path.parent}")
    return Plan(base_model, lora, tuple(jobs), settings["seed"], settings["steps"], float(guidance),
                args.offload, args.local_files_only, args.overwrite, manifest, run_metadata,
                args.revision)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(plan: Plan) -> dict[str, Any]:
    hashes: dict[str, Any] = {
        "implementation": _sha256(Path(__file__)),
        "inputs": {job.id: _sha256(job.input) for job in plan.jobs},
    }
    if plan.manifest:
        hashes["manifest"] = _sha256(plan.manifest)
    for label, value, names in (
        ("base_model", plan.base_model, ("model_index.json", "transformer/config.json")),
        ("lora", plan.lora, ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")),
    ):
        if value is not None and Path(value).is_dir():
            hashes[label] = {name: _sha256(Path(value) / name) for name in names if (Path(value) / name).is_file()}
    return hashes


def _write_json(path: Path, data: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _load_pipeline(plan: Plan) -> tuple[Any, Any]:
    import torch
    from diffusers import LongCatImageEditPipeline, LongCatImageTransformer2DModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference; --dry-run works without a GPU")
    options = {"torch_dtype": torch.bfloat16, "local_files_only": plan.local_files_only}
    if plan.base_revision is not None:
        options["revision"] = plan.base_revision
    # Load and merge on CPU first. This avoids a full transformer allocation on
    # CUDA before model offloading has been configured.
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        plan.base_model, subfolder="transformer", use_safetensors=True, **options,
    )
    if plan.lora is not None:
        from peft import PeftModel
        transformer = PeftModel.from_pretrained(
            transformer, plan.lora, local_files_only=plan.local_files_only,
        ).merge_and_unload()
    pipe = LongCatImageEditPipeline.from_pretrained(plan.base_model, transformer=transformer, **options)
    if plan.offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe, torch


def _run_job(pipe: Any, torch: Any, plan: Plan, job: Job) -> dict[str, Any]:
    from PIL import Image

    with Image.open(job.input) as source:
        image = source.convert("RGB")
    # Reset for every case: skipping/reordering cases does not change their seed.
    generator = torch.Generator("cpu").manual_seed(plan.seed)
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            image=image, prompt=TASK_PROMPTS[job.task], negative_prompt="",
            guidance_scale=plan.guidance_scale, num_inference_steps=plan.steps,
            num_images_per_prompt=1, generator=generator,
        ).images[0]
    elapsed = time.perf_counter() - started
    job.output.parent.mkdir(parents=True, exist_ok=True)
    with job.output.open("wb" if plan.overwrite else "xb") as stream:
        result.save(stream, format=_OUTPUT_FORMATS[job.output.suffix.lower()])
    return {"input_size": list(image.size), "output_size": list(result.size), "inference_seconds": round(elapsed, 3)}


def execute_plan(plan: Plan) -> int:
    """Run all selected cases; persist failures and return nonzero if any fail."""
    report = plan.to_dict()
    report.update({"status": "running", "source_sha256": _source_hashes(plan), "python": platform.python_version()})
    results = []
    started = time.perf_counter()
    try:
        pipe, torch = _load_pipeline(plan)
    except Exception as exc:
        report.update({"status": "failed", "phase": "model_load", "error": f"{type(exc).__name__}: {exc}"})
        _write_json(plan.run_metadata or plan.jobs[0].metadata, report, overwrite=plan.overwrite)
        print(report["error"], file=sys.stderr)
        return 1
    report["model_load_seconds"] = round(time.perf_counter() - started, 3)
    report["device"] = torch.cuda.get_device_name()
    report["versions"] = {}
    for package in ("torch", "diffusers", "transformers", "peft", "accelerate", "Pillow", "numpy"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    for index, job in enumerate(plan.jobs, 1):
        print(f"[{index}/{len(plan.jobs)}] {job.id}: {job.input} -> {job.output}", flush=True)
        record = {key: value for key, value in report.items() if key != "jobs"}
        record.update(job.to_dict())
        try:
            record.update(_run_job(pipe, torch, plan, job))
            record["status"] = "complete"
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{job.id}: {record['error']}", file=sys.stderr, flush=True)
        try:
            _write_json(job.metadata, record, overwrite=plan.overwrite)
        except OSError as exc:
            record.update({"status": "failed", "metadata_error": f"{type(exc).__name__}: {exc}"})
            print(f"{job.id}: cannot write metadata: {exc}", file=sys.stderr)
        results.append(record)
    report.update({"results": results, "total_seconds": round(time.perf_counter() - started, 3)})
    failed = any(record["status"] != "complete" for record in results)
    report["status"] = "failed" if failed else "complete"
    if plan.run_metadata:
        _write_json(plan.run_metadata, report, overwrite=plan.overwrite)
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
            return 0
        return execute_plan(plan)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
