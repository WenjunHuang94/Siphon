# SPDX-License-Identifier: Apache-2.0
"""Single-GPU reference LoRA training from frozen-encoder caches.

This is a portable reference implementation, not a claim of exact reproduction
of every paper experiment. --help and --dry-run use only the standard library.
An initialization adapter starts a NEW optimizer; --resume restores a private
training-state checkpoint as well as the adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import struct
import sys
import time
from typing import Any

from .inference import _json_object, _model_source


TARGET_MODULES = (
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
)
DEFAULTS = {
    "base_model": "meituan-longcat/LongCat-Image-Edit", "revision": None,
    "seed": 43, "resolution": 512, "lora_rank": 64, "lora_alpha": 8,
    "learning_rate": 2e-4, "batch_size": 1, "gradient_accumulation_steps": 8,
    "mixed_precision": "bf16", "gradient_checkpointing": True,
    "max_train_steps": 2500, "initial_step": 0, "lr_warmup_steps": 100,
    "checkpointing_steps": 200, "weight_decay": 0.01,
    "max_grad_norm": 1.0, "use_dynamic_shifting": True, "null_text_ratio": 0.1,
    "local_files_only": False,
}
_CONFIG_KEYS = set(DEFAULTS) | {"cache_manifest", "output_dir", "init_lora", "resume"}
_RESUME_KEYS = (
    "base_model", "revision", "seed", "resolution", "lora_rank", "lora_alpha",
    "learning_rate", "batch_size", "gradient_accumulation_steps", "mixed_precision",
    "gradient_checkpointing", "initial_step", "lr_warmup_steps", "weight_decay",
    "max_grad_norm", "use_dynamic_shifting", "null_text_ratio",
)


@dataclass(frozen=True)
class CacheEntry:
    id: str
    path: Path
    shapes: tuple[tuple[int, ...], ...]
    sha256: str | None = None


@dataclass(frozen=True)
class TrainPlan:
    settings: dict[str, Any]
    manifest: Path
    entries: tuple[CacheEntry, ...]
    cache_fingerprint: str
    output_dir: Path
    init_lora: str | None
    resume: Path | None
    start_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": "single_gpu_cached_latent_reference",
            "settings": self.settings, "cache_manifest": str(self.manifest),
            "cache_fingerprint": self.cache_fingerprint, "num_examples": len(self.entries),
            "num_shape_buckets": len({entry.shapes for entry in self.entries}),
            "output_dir": str(self.output_dir), "init_lora": self.init_lora,
            "resume": str(self.resume) if self.resume else None,
            "start_step": self.start_step,
            "new_optimizer_updates": self.settings["max_train_steps"] - self.start_step,
            "target_modules": list(TARGET_MODULES),
            "optimizer_policy": "restore" if self.resume else "new",
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, help="JSON configuration; paths are relative to this file")
    parser.add_argument("--base-model")
    parser.add_argument("--revision")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--init-lora", help="Adapter weights only; optimizer and scheduler start afresh")
    parser.add_argument("--resume", type=Path, help="Checkpoint directory with adapter + training_state/state.pt")
    parser.add_argument("--max-steps", dest="max_train_steps", type=int)
    parser.add_argument("--initial-step", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpointing-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and cache headers without importing model packages")
    return parser


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_header(path: Path) -> dict[str, Any]:
    """Read only a safetensors header, including on a dependency-free dry run."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= 16 * 1024 * 1024:
                raise ValueError("invalid safetensors header length")
            data = json.loads(stream.read(length))
        if not isinstance(data, dict):
            raise ValueError("header must be an object")
        return data
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot inspect cache {path}: {exc}") from exc


def read_cache_manifest(path: Path) -> tuple[tuple[CacheEntry, ...], str]:
    entries = []
    seen_ids, seen_paths = set(), set()
    fingerprint = hashlib.sha256(path.read_bytes())
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                value = record["cache_path"]
                if not isinstance(value, str) or not value:
                    raise ValueError("cache_path must be a nonempty string")
                cache = Path(value).expanduser()
                cache = (cache if cache.is_absolute() else path.parent / cache).resolve()
                case_id = record.get("id", str(line_number))
                if not isinstance(case_id, str) or not case_id:
                    raise ValueError("id must be a nonempty string")
                if case_id in seen_ids or cache in seen_paths:
                    raise ValueError("duplicate cache id or path")
                header = _tensor_header(cache)
                shapes = []
                for key, rank in (("target_latents", 3), ("ref_latents", 3), ("prompt_embeds", 2)):
                    tensor = header.get(key, {})
                    shape = tensor.get("shape")
                    if (not isinstance(shape, list) or len(shape) != rank
                            or any(type(size) is not int or size < 1 for size in shape)
                            or tensor.get("dtype") not in ("F32", "F16", "BF16")):
                        raise ValueError(f"{key} must be a floating rank-{rank} tensor with positive dimensions")
                    if rank == 3 and (shape[1] % 2 or shape[2] % 2):
                        raise ValueError(f"{key} spatial dimensions must be even for 2x2 latent packing")
                    shapes.append(tuple(shape))
                if shapes[0][0] != shapes[1][0]:
                    raise ValueError("target/ref latent channel counts differ")
                if "null_prompt_embeds" in header:
                    tensor = header["null_prompt_embeds"]
                    shape = tensor.get("shape")
                    if (not isinstance(shape, list) or len(shape) != 2
                            or any(type(size) is not int or size < 1 for size in shape)
                            or shape[-1] != shapes[2][-1]
                            or tensor.get("dtype") not in ("F32", "F16", "BF16")):
                        raise ValueError("null_prompt_embeds must be a floating [T_null,D] tensor with the same D as prompt_embeds")
                    shapes.append(tuple(shape))
                # Producers record content hashes, allowing a large-cache dry
                # run to read headers only. Each used file is verified before
                # loading its tensors during actual training.
                content_hash = record.get("cache_sha256")
                if content_hash is not None and (not isinstance(content_hash, str) or len(content_hash) != 64
                                                 or any(char not in "0123456789abcdef" for char in content_hash)):
                    raise ValueError("cache_sha256 must be a lowercase SHA256 hex digest")
                if content_hash is None:
                    content_hash = _file_hash(cache)
                fingerprint.update(case_id.encode())
                fingerprint.update(content_hash.encode())
                fingerprint.update(json.dumps(header, sort_keys=True).encode())
                entries.append(CacheEntry(case_id, cache, tuple(shapes), content_hash))
                seen_ids.add(case_id)
                seen_paths.add(cache)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not entries:
        raise ValueError("Cache manifest contains no samples")
    return tuple(entries), fingerprint.hexdigest()


def _path(value: Any, parent: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Expected a nonempty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def _source(value: str, parent: Path, *, adapter: bool) -> str:
    if not isinstance(value, str):
        raise ValueError("Model/adapter source must be a string")
    candidate = _path(value, parent)
    if candidate.exists() or value.startswith((".", "/", "~")):
        value = str(candidate)
    return _model_source(value, adapter=adapter)


def build_plan(args: argparse.Namespace) -> TrainPlan:
    config_parent = Path.cwd()
    values = dict(DEFAULTS)
    if args.config is not None:
        config = args.config.expanduser().resolve()
        supplied = _json_object(config)
        unknown = set(supplied) - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown training config key(s): {', '.join(sorted(unknown))}")
        values.update(supplied)
        config_parent = config.parent
    parents = {key: config_parent for key in _CONFIG_KEYS}
    for key in _CONFIG_KEYS:
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
            parents[key] = Path.cwd()
    values["base_model"] = _source(values["base_model"], parents["base_model"], adapter=False)
    revision = values["revision"]
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision must be a nonempty string or null")
    for key in ("seed", "resolution", "lora_rank", "lora_alpha", "batch_size",
                "gradient_accumulation_steps", "max_train_steps", "initial_step",
                "lr_warmup_steps", "checkpointing_steps"):
        value = values[key]
        minimum = 0 if key in ("seed", "initial_step", "lr_warmup_steps") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if values["seed"] >= 2**63:
        raise ValueError("seed must be less than 2**63")
    if values["resolution"] % 16:
        raise ValueError("resolution must be divisible by 16")
    for key in ("learning_rate", "weight_decay", "max_grad_norm"):
        value = values[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key != "weight_decay" and value == 0)):
            raise ValueError(f"{key} must be a finite {'nonnegative' if key == 'weight_decay' else 'positive'} number")
    ratio = values["null_text_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("null_text_ratio must be a finite number between 0 and 1")
    for key in ("gradient_checkpointing", "use_dynamic_shifting", "local_files_only"):
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if values["mixed_precision"] not in ("bf16", "no"):
        raise ValueError("mixed_precision must be 'bf16' or 'no' (float32)")
    if "cache_manifest" not in values or "output_dir" not in values:
        raise ValueError("Supply cache_manifest and output_dir in JSON or on the command line")
    manifest = _path(values.pop("cache_manifest"), parents["cache_manifest"])
    output_dir = _path(values.pop("output_dir"), parents["output_dir"])
    entries, fingerprint = read_cache_manifest(manifest)
    if ratio > 0:
        if any(len(entry.shapes) != 4 for entry in entries):
            raise ValueError("null_text_ratio > 0 requires cached null_prompt_embeds; rebuild the cache or explicitly set null_text_ratio to 0")
        if values["batch_size"] > 1 and any(entry.shapes[2] != entry.shapes[3] for entry in entries):
            raise ValueError("When positive/null prompt lengths differ, use batch_size=1; embeddings are never padded without a mask")
    resume_value, init_value = values.pop("resume", None), values.pop("init_lora", None)
    resume = _path(resume_value, parents["resume"]) if resume_value else None
    init_lora = _source(init_value, parents["init_lora"], adapter=True) if init_value and not resume else None
    if init_value == "":
        raise ValueError("init_lora must not be empty")
    start = values["initial_step"]
    if start > 0 and not init_lora and not resume:
        raise ValueError("initial_step > 0 requires init_lora or a full resume checkpoint")
    if resume:
        _model_source(str(resume), adapter=True)
        metadata = _json_object(resume / "training_state" / "metadata.json")
        if not (resume / "training_state" / "state.pt").is_file():
            raise ValueError("Full resume requires training_state/state.pt, not just adapter weights")
        if metadata.get("cache_fingerprint") != fingerprint:
            raise ValueError("Resume cache fingerprint does not match current manifest and tensor files")
        prior = metadata.get("settings", {})
        mismatches = [key for key in _RESUME_KEYS if prior.get(key) != values[key]]
        if mismatches:
            raise ValueError(f"Resume config differs in: {', '.join(mismatches)}")
        start = metadata.get("global_step")
        if type(start) is not int or start < values["initial_step"]:
            raise ValueError("Resume global_step is missing or invalid")
        # Adapter weight identity is checked separately from the private state.
        if metadata.get("adapter_sha256") != _file_hash(resume / "adapter_model.safetensors"):
            raise ValueError("Resume adapter hash differs from saved training state")
    if values["max_train_steps"] <= start:
        raise ValueError("max_train_steps must be greater than the starting/global resume step")
    protected = [manifest, *(entry.path for entry in entries)]
    if any(path == output_dir or output_dir in path.parents for path in protected):
        raise ValueError("Training output directory must not contain the cache manifest or cache tensors")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("output_dir is not a directory")
        checkpoints = list(output_dir.glob("checkpoints-*"))
        if checkpoints and not resume:
            raise ValueError("output_dir already has checkpoints; choose a fresh directory or --resume")
        if any((output_dir / f"checkpoints-{step}").exists() for step in checkpoint_steps(start, values["max_train_steps"], values["checkpointing_steps"])):
            raise ValueError("A future checkpoint already exists; choose a new output directory")
    return TrainPlan(values, manifest, entries, fingerprint, output_dir, init_lora, resume, start)


def checkpoint_steps(start: int, maximum: int, interval: int) -> list[int]:
    """Scheduled global update counts, always including the final update exactly once."""
    steps = list(range((start // interval + 1) * interval, maximum + 1, interval))
    if not steps or steps[-1] != maximum:
        steps.append(maximum)
    return steps


def warmup_factor(completed: int, warmup: int) -> float:
    """LR multiplier for the NEXT update; a new phase has its own warmup."""
    return min(1.0, (completed + 1) / warmup) if warmup else 1.0


class BatchCursor:
    """Deterministic shape buckets with an explicitly checkpointable cursor."""

    def __init__(self, entries: tuple[CacheEntry, ...], batch_size: int, seed: int):
        self.entries, self.batch_size, self.seed = entries, batch_size, seed
        self.epoch, self.index = 0, 0
        self.batches = self._make_batches()

    def _make_batches(self) -> list[list[int]]:
        groups: dict[Any, list[int]] = {}
        for index, entry in enumerate(self.entries):
            groups.setdefault(entry.shapes, []).append(index)
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for shape in sorted(groups):
            group = groups[shape]
            rng.shuffle(group)
            batches.extend(group[offset:offset + self.batch_size] for offset in range(0, len(group), self.batch_size))
        rng.shuffle(batches)
        return batches

    def next(self) -> list[int]:
        if self.index >= len(self.batches):
            self.epoch += 1
            self.index = 0
            self.batches = self._make_batches()
        batch = self.batches[self.index]
        self.index += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "index": self.index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        epoch, index = state.get("epoch"), state.get("index")
        if type(epoch) is not int or epoch < 0 or type(index) is not int or index < 0:
            raise ValueError("Invalid saved batch cursor")
        self.epoch = epoch
        self.batches = self._make_batches()
        if index > len(self.batches):
            raise ValueError("Saved batch cursor is outside the current dataset")
        self.index = index


def _pack_latents(latents: Any) -> Any:
    batch, channels, height, width = latents.shape
    return (latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, (height // 2) * (width // 2), channels * 4))


def _position_ids(torch: Any, target: Any, reference: Any, text_length: int) -> tuple[Any, Any]:
    device = target.device
    text = torch.zeros(text_length, 3, device=device, dtype=torch.float64)
    text[:, 1] = text[:, 2] = torch.arange(text_length, device=device)
    ids = []
    for modality, tensor in ((1, target), (2, reference)):
        height, width = tensor.shape[-2] // 2, tensor.shape[-1] // 2
        image = torch.zeros(height, width, 3, device=device, dtype=torch.float64)
        image[..., 0] = modality
        image[..., 1] = torch.arange(height, device=device)[:, None] + text_length
        image[..., 2] = torch.arange(width, device=device)[None, :] + text_length
        ids.append(image.reshape(-1, 3))
    return text, torch.cat(ids, dim=0)


def _load_microbatch(torch: Any, entries: tuple[CacheEntry, ...], indexes: list[int], device: Any, dtype: Any,
                     null_text_ratio: float = 0.0) -> dict[str, Any]:
    from safetensors.torch import load_file

    samples = []
    for index in indexes:
        entry = entries[index]
        if entry.sha256 is not None and _file_hash(entry.path) != entry.sha256:
            raise ValueError(f"Cached tensor hash mismatch: {entry.path}")
        samples.append(load_file(str(entry.path), device="cpu"))
    for sample in samples:
        if null_text_ratio and random.random() < null_text_ratio:
            sample["prompt_embeds"] = sample["null_prompt_embeds"]
    return {key: torch.stack([sample[key] for sample in samples]).to(device=device, dtype=dtype)
            for key in ("target_latents", "ref_latents", "prompt_embeds")}


def _loss(torch: Any, model: Any, scheduler: Any, batch: dict[str, Any], *, dynamic_shift: bool) -> Any:
    target, reference, prompt = (batch[key] for key in ("target_latents", "ref_latents", "prompt_embeds"))
    # Flow matching: z_t=(1-sigma)*z + sigma*noise, target velocity=noise-z.
    sigma = torch.randn(target.shape[0], device=target.device, dtype=torch.float32).sigmoid()
    if dynamic_shift:
        config = scheduler.config
        seq_len = (target.shape[-2] // 2) * (target.shape[-1] // 2)
        mu = (config.base_shift + (seq_len - config.base_image_seq_len)
              * (config.max_shift - config.base_shift)
              / (config.max_image_seq_len - config.base_image_seq_len))
        sigma = scheduler.time_shift(mu, 1.0, sigma)
    blend = sigma.to(target.dtype).reshape(-1, 1, 1, 1)
    noise = torch.randn_like(target)
    noisy = (1 - blend) * target + blend * noise
    packed = _pack_latents(noisy)
    text_ids, image_ids = _position_ids(torch, target, reference, prompt.shape[1])
    prediction = model(
        hidden_states=torch.cat((packed, _pack_latents(reference)), dim=1),
        encoder_hidden_states=prompt, timestep=sigma,
        img_ids=image_ids, txt_ids=text_ids, guidance=None, return_dict=False,
    )[0][:, :packed.shape[1]]
    velocity = _pack_latents(noise - target)
    return (prediction.float() - velocity.float()).square().mean()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(torch: Any, plan: TrainPlan, model: Any, optimizer: Any,
                     cursor: BatchCursor, step: int, updates: int) -> Path:
    checkpoint = plan.output_dir / f"checkpoints-{step}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    # Our adapters target only attention/FFN projections, never embeddings.
    # PEFT's default "auto" probes the Hub for the base vocabulary config even
    # for a Diffusers model; False keeps checkpoint export genuinely local.
    model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
    private = checkpoint / "training_state"
    private.mkdir()
    state = {
        "schema_version": 1, "global_step": step, "optimizer_updates": updates,
        "scheduler": {"kind": "constant_with_warmup", "completed_updates": updates,
                      "warmup_steps": plan.settings["lr_warmup_steps"]},
        "optimizer": optimizer.state_dict(), "cursor": cursor.state_dict(),
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    temporary = private / "state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(private / "state.pt")
    _write_json(private / "metadata.json", {
        "schema_version": 1, "global_step": step, "optimizer_updates": updates,
        "cache_fingerprint": plan.cache_fingerprint, "settings": plan.settings,
        "adapter_sha256": _file_hash(checkpoint / "adapter_model.safetensors"),
        "initialization_adapter": plan.init_lora,
        "state_policy": "Private optimizer/RNG/cursor state for trusted local resume; do not publish this directory as model weights.",
    })
    return checkpoint


def execute_plan(plan: TrainPlan) -> int:
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, LongCatImageTransformer2DModel
    from peft import LoraConfig, PeftModel, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("Single-GPU CUDA training is required; --dry-run works without a GPU")
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This reference trainer is single-process; do not launch it with distributed accelerate/torchrun")
    cfg = plan.settings
    dtype = torch.bfloat16 if cfg["mixed_precision"] == "bf16" else torch.float32
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 training requires a GPU with bfloat16 support")
    device = torch.device("cuda")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    options = {"local_files_only": cfg["local_files_only"]}
    if cfg["revision"] is not None:
        options["revision"] = cfg["revision"]
    model = LongCatImageTransformer2DModel.from_pretrained(
        cfg["base_model"], subfolder="transformer", torch_dtype=dtype, use_safetensors=True, **options,
    )
    model.requires_grad_(False)
    adapter = str(plan.resume) if plan.resume else plan.init_lora
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True,
                                          local_files_only=cfg["local_files_only"])
        loaded = model.peft_config["default"]
        if loaded.r != cfg["lora_rank"] or loaded.lora_alpha != cfg["lora_alpha"] or set(loaded.target_modules) != set(TARGET_MODULES):
            raise ValueError("Initialization/resume adapter rank, alpha or target_modules do not match the training config")
    else:
        model = get_peft_model(model, LoraConfig(
            r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], target_modules=list(TARGET_MODULES),
            init_lora_weights="gaussian", use_dora=False, use_rslora=False,
        ))
    # Frozen backbone stays bf16; adapters and their AdamW states stay float32.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if cfg["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    model.to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=cfg["weight_decay"])
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(cfg["base_model"], subfolder="scheduler", **options)
    expected_channels = model.config.in_channels // 4
    expected_prompt_dim = model.config.joint_attention_dim
    for entry in plan.entries:
        if entry.shapes[0][0] != expected_channels or entry.shapes[2][-1] != expected_prompt_dim:
            raise ValueError(f"Cache {entry.id} channel/prompt dimensions do not match the base model")
    cursor = BatchCursor(plan.entries, cfg["batch_size"], cfg["seed"])
    updates = 0
    if plan.resume:
        state = torch.load(plan.resume / "training_state" / "state.pt", map_location="cpu", weights_only=True)
        if state.get("global_step") != plan.start_step:
            raise ValueError("Private state global_step differs from its metadata")
        optimizer.load_state_dict(state["optimizer"])
        cursor.load_state_dict(state["cursor"])
        updates = state["optimizer_updates"]
        if (type(updates) is not int or updates < 0
                or state.get("scheduler") != {"kind": "constant_with_warmup", "completed_updates": updates,
                                               "warmup_steps": cfg["lr_warmup_steps"]}):
            raise ValueError("Private optimizer/scheduler progress is invalid")
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    report = plan.to_dict()
    report.update({"status": "running", "trainable_parameters": sum(p.numel() for p in trainable),
                   "versions": {}, "started_at_unix": time.time()})
    for name in ("torch", "diffusers", "transformers", "peft", "safetensors"):
        report["versions"][name] = importlib.metadata.version(name)
    _write_json(plan.output_dir / "training_run.json", report)
    save_at = set(checkpoint_steps(plan.start_step, cfg["max_train_steps"], cfg["checkpointing_steps"]))
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    try:
        # A bounded loop prevents the historical extra update after max_steps.
        for step in range(plan.start_step + 1, cfg["max_train_steps"] + 1):
            microbatches = [cursor.next() for _ in range(cfg["gradient_accumulation_steps"])]
            sample_count = sum(len(indexes) for indexes in microbatches)
            loss_value = 0.0
            for indexes in microbatches:
                batch = _load_microbatch(torch, plan.entries, indexes, device, dtype, cfg["null_text_ratio"])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    loss = _loss(torch, model, scheduler, batch, dynamic_shift=cfg["use_dynamic_shifting"])
                    weighted = loss * len(indexes) / sample_count
                if not torch.isfinite(weighted):
                    raise RuntimeError(f"Non-finite loss at global step {step}")
                weighted.backward()
                loss_value += weighted.detach().float().item()
                del batch, loss, weighted
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at global step {step}")
            lr = cfg["learning_rate"] * warmup_factor(updates, cfg["lr_warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            print(f"step={step}/{cfg['max_train_steps']} loss={loss_value:.6f} lr={lr:.3g} grad_norm={float(grad_norm):.4g}", flush=True)
            report.update({"global_step": step, "optimizer_updates": updates, "last_loss": loss_value})
            if step in save_at:
                checkpoint = _save_checkpoint(torch, plan, model, optimizer, cursor, step, updates)
                report["latest_checkpoint"] = str(checkpoint)
                _write_json(plan.output_dir / "training_run.json", report)
        report["status"] = "complete"
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        report["training_seconds"] = round(time.perf_counter() - started, 3)
        _write_json(plan.output_dir / "training_run.json", report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
            return 0
        return execute_plan(plan)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Training error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
