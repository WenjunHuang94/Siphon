---
base_model: meituan-longcat/LongCat-Image-Edit
library_name: peft
pipeline_tag: image-to-image
{{LICENSE_METADATA}}
tags:
- siphon
- longcat-image
- lora
- diffusers
---

# Siphon LongCat LoRA — checkpoint 10000

This adapter is the `stage2-final-0225/checkpoints-10000` LoRA used for the eight recorded examples in [WenjunHuang94/Siphon](https://github.com/WenjunHuang94/Siphon).
It adds visual-instruction editing to LongCat-Image-Edit: marked-image text, arrows, and boxes supply the case-specific edit; the text channel uses a generic task template.

This is an adapter, not a complete editor. Download the LongCat-Image-Edit base separately. Use the explicit Diffusers + PEFT loading flow in the code repository; an arbitrary Transformers AutoPEFT loader is not a supported entry point for this Diffusers backbone.

## Use

After installing the code repository:

```bash
python -m siphon.inference \
  --base-model meituan-longcat/LongCat-Image-Edit \
  --revision 7b54ef423aa7854be7861600024be5c56ab7875a \
  --lora {{ADAPTER_REPO_ID}} \
  --examples examples/examples.json --case box_double_bass \
  --output-dir outputs/box_example
```

Reference generation settings are 50 steps, guidance scale 4.5, seed 43, bf16 and model CPU offload. Full details and image-specific limitations are in the code repository's example manifest.

## Identity and training provenance

- Base: `meituan-longcat/LongCat-Image-Edit`, revision `7b54ef423aa7854be7861600024be5c56ab7875a`.
- LoRA rank 64, alpha 8; attention and feed-forward projection layers.
- Adapter SHA256: `0ed9694868d2552d243140caf242ef38e0d69f6673a3eec36e4ddc53ed459535`.
- Tensor file: 188,794,136 bytes (about 180 MiB).
- Historical Stage 1: 4000 rendered-text pairs, 512 resolution, learning rate 2e-4; selected step 2400.
- Historical Stage 2: 20000 mixed pairs, learning rate 1e-4; cumulative step 2400 to 10000.

The portable cached-data trainer now provided in the code repository is a reference implementation. It is not a claim that the historical adapter, the full paper-scale training recipe, or all paper results are reproduced exactly. Public metadata is normalized to the public base-model ID and Diffusers class; the adapter tensor bytes are unchanged. No full-base or optimizer states are included.

## Observed limitations

The gallery is selected, not a success-rate estimate. Earlier arrow examples failed to replace a basketball and a remote control. In the selected flower example, other flowers also change color. Faces, hands, textures and background details can be redrawn. Rendered-text generation does not consistently satisfy exact lettering or all requested details. Successful marker removal does not imply successful semantic editing or pixel-exact preservation.

## License and attribution

{{LICENSE_NOTE}}

The base model and source images retain their own terms. The Apache-2.0 source-code license does not by itself select a license for these adapter weights.

Paper: *Internalize External Competence for Visual Instruction Editing*, Siphon authors, NeurIPS 2026. See the code repository for citation and author information.
