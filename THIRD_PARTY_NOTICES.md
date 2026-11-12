# Third-party notices and asset provenance

### LongCat-Image

The inference loading flow is adapted from the LongCat-Image editing examples and the authors' research scripts based on that project:

- Upstream repository: https://github.com/meituan-longcat/LongCat-Image
- Base model: https://huggingface.co/meituan-longcat/LongCat-Image-Edit
- Upstream source-code license: Apache License, Version 2.0. A copy is included in [LICENSE](LICENSE).

The release changes replace machine-specific paths and fixed task lists with a portable command-line entry point, explicit model/adapter arguments, and a manifest-driven example runner. The published inference path uses the Diffusers implementation and a standard PEFT adapter; the original `longcat_image` training package is not bundled.

### Runtime libraries

Inference uses PyTorch, Torchvision, Hugging Face Diffusers, Transformers, PEFT, Accelerate, Safetensors, and Pillow. These are installed as dependencies and retain their respective licenses. See package metadata and the linked projects for their notices. Diffusers is pinned to the recorded source revision in the package dependency specification.

### Paper, model weights, and image assets

The Apache-2.0 license in this repository applies to the released code. The model owner separately licenses the released [Siphon adapter weights](https://huggingface.co/huangwenjun1994/Siphon-LongCat-LoRA) under [Apache-2.0](https://huggingface.co/huangwenjun1994/Siphon-LongCat-LoRA/blob/main/LICENSE). The paper, paper figures, example images, and base-model weights retain their own terms.

The paper and the original project-page figures are retained under `docs/` as part of the accompanying research publication. Recorded demonstration pairs under `examples/` were assembled separately during code-release preparation. Their logical research-source paths, asset hashes, adapter hash, settings, and observations are recorded in [examples/examples.json](examples/examples.json).

The existing research workspace does not yet contain complete original-owner attribution and redistribution records for all source images used by those examples. Logical source paths and generated-output hashes provide run provenance, not an independent grant of image rights. This release does not describe those assets as Apache-2.0-licensed. See [examples/README.md](examples/README.md#provenance) for the current scope of those records.

The base model is obtained from its upstream model repository. The public Siphon adapter contains the selected `10000` checkpoint only; see [WEIGHTS.md](WEIGHTS.md) for its identity, download commands, and package contents.
