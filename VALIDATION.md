# Validation of the LongCat inference and training release

Checks performed on 2026-11-12:

- 52 CPU unit tests passed with the installed ML stack: inference planning, task prompts, adapter selection, manifest paths, output protection, asset integrity, training-data caching, training configuration and resume checks, offline adapter saving, export allowlists, and local documentation links. Without PyTorch/PEFT, the two integration tests for adapter saving are skipped; the other checks use the standard library.
- The package built and installed successfully, including the `siphon-infer` console entry point.
- All eight selected examples passed dry-run validation without importing model dependencies.
- A real CUDA run through the refactored CLI completed the `box_double_bass` case with the same research adapter used for the recorded gallery.
- A one-pair CUDA cache/train/resume/inference check completed through the new training reference. This checks the implementation, not the historical training run or paper reproduction.

## GPU check

The actual run used an NVIDIA A100 80 GB, model CPU offload, bf16, 50 steps, guidance 4.5, seed 43, and a CPU generator. The local base snapshot was `7b54ef423aa7854be7861600024be5c56ab7875a`; the PEFT adapter hash is recorded in [examples/examples.json](examples/examples.json).

The output was a valid 1040 × 1024 PNG. Visual inspection confirmed a double bass in the marked kitchen region and removal of the box and instruction text. Model loading took approximately 18.7 seconds; generation and saving took approximately 133.8 seconds. These timings are observations on this machine, not performance guarantees.

Runtime versions were PyTorch `2.6.0+cu124`, Diffusers `0.39.0.dev0` at commit `e39aecff57ed14d1018529c3de6ec3c34fadb559`, Transformers `4.51.3`, PEFT `0.17.1`, Accelerate `1.14.0`, and Pillow `12.3.0`. The test used an isolated project environment together with the server's preinstalled CUDA/PyTorch stack. A full installation from an empty machine was not tested.

The CLI implementation SHA256 at the time of this check was `2c8079e83b6c9d1d78324f614651ff69d818e98d2793201c27466549edc059e6`.

## Training pipeline check

On the same A100 and software stack:

1. Cached one 512 × 512 pair from `training/examples/stage1.jsonl` with the frozen encoder and VAE, without loading the diffusion transformer. This took approximately 19.7 seconds. Target and reference latents had shape `[16, 64, 64]`; positive and empty-text embeddings each had shape `[838, 3584]`.
2. Loaded the base transformer with the selected `10000` adapter and ran two optimizer updates with batch size 1 and gradient accumulation 1. All 94,371,840 LoRA parameters were trainable. The observed losses were approximately 0.464 and 0.363. Both updates exported standard PEFT adapters and separate private training states; the update loop took approximately 8.1 seconds.
3. Resumed from the first checkpoint, including optimizer, scheduler, random states, and data cursor, and completed the second update. The reported final loss matched the continuous run. This checks resume continuity; GPU kernels did not produce bit-identical exported weights.
4. Loaded the resumed adapter through `siphon.inference` and generated `box_double_bass` with 8 steps. It produced a valid 1040 × 1024 PNG with a double bass and removed the marks. Total runtime was approximately 50.5 seconds, including approximately 10.1 seconds for model loading. The short run was visibly less refined than the recorded 50-step example.

These tiny runs use the release examples only to exercise the code. They do not train a new useful editor, validate generalization, or recreate the historical two-stage training. The original `10000` tensors were unchanged, and the smoke-test adapters are not part of the model release.

## Published adapter check

The selected adapter was published under Apache-2.0 at [huangwenjun1994/Siphon-LongCat-LoRA](https://huggingface.co/huangwenjun1994/Siphon-LongCat-LoRA), revision `3eebc6f1c1a37fd390d90f6b354b4821186f71a1`. The package contains the standard adapter tensor/config, model card, license, checksums, and release metadata. Optimizer states, full-base weights, and the smoke-test adapters are excluded.

The public model API was checked without authentication. Both the config and tensor file were then freshly downloaded with `token=False`. The downloaded tensor was 188,794,136 bytes and matched the original `10000` SHA256:

```text
0ed9694868d2552d243140caf242ef38e0d69f6673a3eec36e4ddc53ed459535
```

With implicit authentication disabled, the inference CLI then loaded `--lora huangwenjun1994/Siphon-LongCat-LoRA` using the anonymous download cache and the recorded local base snapshot. An 8-step `box_double_bass` run completed on the A100: approximately 16.2 seconds for loading and 54.9 seconds total, producing a valid 1040 × 1024 PNG. Visual inspection confirmed a double bass and removal of the box/text. This was a loading and generation smoke test; the gallery retains its original 50-step outputs. A transient adapter-config lookup failure on the first attempt was resolved by fetching the public config before retrying.

## Repeat the checks

From the repository root:

```bash
python -m unittest discover -s tests -v

python -m siphon.inference --base-only \
  --examples examples/examples.json --output-dir outputs/dry-run --dry-run

# Supply local copies of the recorded base snapshot and adapter.
python -m siphon.inference \
  --base-model "$BASE_MODEL" --lora "$SIPHON_LORA" \
  --examples examples/examples.json --case box_double_bass \
  --output-dir outputs/validation --local-files-only
```

The CPU tests do not replace GPU generation checks, and generating a file does not establish successful editing. The gallery's per-case observations retain the instruction-following and preservation limitations found during visual inspection.
