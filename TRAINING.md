# LongCat LoRA training reference

[中文说明](TRAINING_zh-CN.md) · [Inference and examples](README.md#inference)

This single-GPU reference trains a LongCat-Image-Edit LoRA from paired marked inputs and clean targets. It includes pair validation, an encoder/VAE cache builder, and a cached-latent trainer. It does not require the original research workspace, DeepSpeed, W&B, or a VLM API during training.

The released inference examples use the historical `stage2/10000` adapter. The portable trainer below preserves its task prompts, LoRA target layers, and main optimization settings, but is a reference implementation rather than a claim that the historical checkpoint or all paper results can be regenerated exactly.

## 1. Install and prepare pairs

Use the same Python/CUDA environment as inference:

```bash
python -m pip install -r requirements-train.txt
```

A training manifest is JSONL, with one pair per line:

```json
{"input":"marked/example.png","target":"clean/example.png","task":"arrow_edit"}
```

Paths resolve relative to the manifest file. `input` is the complete image containing its visual instructions; `target` is the clean, edited supervision image. Valid tasks are `text_to_image`, `text_edit`, `arrow_edit`, and `box_edit`. The task chooses the same generic text prompt used at inference; the specific edit remains in the image. The loader also accepts the old research keys `ref_img_path` (input), `img_path` (target), and `prompt`.

Build a Type I manifest for Stage 1 and a mixed Type I–IV manifest for Stage 2. Use geometrically aligned pairs. The cache builder resizes both complete images to the selected training dimensions and records their original sizes and hashes; it does not use a separate box mask. The default is 512 × 512; use `--width` and `--height` for another supported size.

Tiny manifests under `training/examples/` reuse the displayed input/output pairs only to illustrate the format and exercise the pipeline. They are not the original training set or an independent evaluation split. Do not use those eight examples to claim model quality.

## 2. Cache frozen conditioning features and latents

For your own manifests at `data/stage1.jsonl` and `data/stage2.jsonl`:

```bash
python -m siphon.prepare_training_data \
  --manifest data/stage1.jsonl \
  --base-model meituan-longcat/LongCat-Image-Edit \
  --revision 7b54ef423aa7854be7861600024be5c56ab7875a \
  --cache-dir caches/stage1 --resolution 512

python -m siphon.prepare_training_data \
  --manifest data/stage2.jsonl \
  --base-model meituan-longcat/LongCat-Image-Edit \
  --revision 7b54ef423aa7854be7861600024be5c56ab7875a \
  --cache-dir caches/stage2 --resolution 512
```

Use a downloaded snapshot directory with `--local-files-only` for offline use. `--dry-run` validates and plans the pairs without importing model dependencies. `--limit` restricts the number of pairs for a smoke test. The cache contains per-pair safetensors, `manifest.jsonl`, and dataset metadata. It is derived training data, not a model release; rebuild it if the base revision, input pairs, prompt, or image dimensions change.

## 3. Train the two stages

```bash
python -m siphon.train --config configs/stage1.json
python -m siphon.train --config configs/stage2.json
```

Config paths resolve relative to the config file; CLI path overrides resolve relative to the current working directory. Override `--base-model`, `--cache-manifest`, and `--output-dir` as needed. `--dry-run` checks the training request before model loading.

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Data | Type I rendered-text pairs | Mixed Type I–IV pairs |
| LoRA initialization | New adapter | Stage 1 `checkpoints-2400` |
| Cumulative step interval | 0 → 2500 | 2400 → 10000 |
| Learning rate | `2e-4` | `1e-4` |
| Warmup updates | 100 | 250 |
| LoRA rank / alpha | 64 / 8 | 64 / 8 |
| Microbatch × accumulation | 1 × 8 | 1 × 8 |

Stage 1 saves every 200 updates and at its final update, so `checkpoints-2400` remains available for Stage 2 even though Stage 1 ends at 2500. Stage 2 performs 7600 new optimizer updates when initialized at step 2400; it is not an additional 10000-update stage. The reference microbatch and accumulation give an effective batch of 8 on one GPU. This differs from the historical physical batch of 8 and does not promise bitwise matching.

The backbone is frozen. LoRA targets include attention projections and feed-forward projections, matching the released adapter. The trainer uses bf16, gradient checkpointing, AdamW, gradient accumulation, warmup, and gradient clipping. A `null_text_ratio` of 0.1 selects cached empty-prompt conditioning during training. The empty text input still accompanies the marked image; it does not remove the visual instruction.

## 4. Initialization, resume, and exported adapters

`init_lora`/`--init-lora` loads adapter weights with a new optimizer and schedule; `initial_step` only defines the cumulative step label. Use this for the transition between stages.

`--resume` restores a checkpoint's private training state, including optimizer, scheduler, random-number state, and step counter:

```bash
python -m siphon.train --config configs/stage2.json \
  --resume outputs/training/stage2/checkpoints-4000
```

Resume with the same base model, cached data, and compatible configuration. Resume takes precedence over the Stage 2 initialization setting.

Each checkpoint contains a standard PEFT adapter and a separate `training_state/` directory. Inference needs `adapter_config.json` and `adapter_model.safetensors`; publish those with a model card, not optimizer state or cached data. The default Stage 2 final adapter is under `outputs/training/stage2/checkpoints-10000`.

```bash
python -m siphon.inference \
  --lora outputs/training/stage2/checkpoints-10000 \
  --examples examples/examples.json --case box_double_bass \
  --output-dir outputs/after_training
```

## Minimal pipeline smoke test

The included two-pair manifest can exercise caching, a short optimization run, and export. It is too small to train a useful editor:

```bash
python -m siphon.prepare_training_data \
  --manifest training/examples/stage1.jsonl \
  --revision 7b54ef423aa7854be7861600024be5c56ab7875a \
  --cache-dir caches/smoke --resolution 512 --limit 1

python -m siphon.train --config configs/stage1.json \
  --cache-manifest caches/smoke/manifest.jsonl \
  --output-dir outputs/training/smoke \
  --max-steps 2 --batch-size 1 --gradient-accumulation-steps 1
```

This exports `outputs/training/smoke/checkpoints-2`, which can be passed to the inference command above instead of the full Stage 2 adapter. Two updates are only an integration check, not a replacement for the pretrained `10000` adapter used in the gallery.

## Historical recipe and scope

Saved configs and logs associate the published gallery adapter with 4000 Stage 1 pairs at 512 resolution, a selected Stage 1 step 2400, and 20000 mixed Stage 2 pairs trained to cumulative step 10000. The old implementation encoded VAE/text features online, used multiple generic prompt templates, and injected LoRA into attention and feed-forward layers.

The portable cache builder uses deterministic VAE mode latents and encodes the marked image at the configured cache size. The historical trainer sampled VAE latents each step and used a smaller image for its visual text encoder. Cached conditioning, deterministic latents, microbatch accumulation, and implementation changes therefore prevent a claim of exact historical-checkpoint reproduction.

This release does not contain the full original training dataset, the teacher/VLM data-synthesis engine, the complete paper-scale cached-data pipeline, or the Qwen-backbone trainer. Pair preparation reads existing marked-input/clean-target pairs; it does not synthesize ground truth. See [VALIDATION.md](VALIDATION.md) for checks actually performed on the released code.
