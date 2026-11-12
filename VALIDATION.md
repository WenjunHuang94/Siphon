# Validation of the initial LongCat inference release

Checks performed on 2026-11-12:

- 19 standard-library CPU unit tests passed: CLI planning, task prompts, explicit adapter selection, manifest paths, output/reference protection, base revision forwarding, failure reporting, asset hashes/dimensions, and local documentation links.
- The package built and installed successfully, including the `siphon-infer` console entry point.
- All eight selected examples passed dry-run validation without importing model dependencies.
- A real CUDA run through the refactored CLI completed the `box_double_bass` case with the same research adapter used for the recorded gallery.

## GPU check

The actual run used an NVIDIA A100 80 GB, model CPU offload, bf16, 50 steps, guidance 4.5, seed 43, and a CPU generator. The local base snapshot was `7b54ef423aa7854be7861600024be5c56ab7875a`; the PEFT adapter hash is recorded in [examples/examples.json](examples/examples.json).

The output was a valid 1040 × 1024 PNG. Visual inspection confirmed a double bass in the marked kitchen region and removal of the box and instruction text. Model loading took approximately 18.7 seconds; generation and saving took approximately 133.8 seconds. These timings are observations on this machine, not performance guarantees.

Runtime versions were PyTorch `2.6.0+cu124`, Diffusers `0.39.0.dev0` at commit `e39aecff57ed14d1018529c3de6ec3c34fadb559`, Transformers `4.51.3`, PEFT `0.17.1`, Accelerate `1.14.0`, and Pillow `12.3.0`. The test used an isolated project environment together with the server's preinstalled CUDA/PyTorch stack. A full installation from an empty machine was not tested.

The CLI implementation SHA256 at the time of this check was `2c8079e83b6c9d1d78324f614651ff69d818e98d2793201c27466549edc059e6`.

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
