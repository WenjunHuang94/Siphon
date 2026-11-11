# Siphon

## Internalize External Competence for Visual Instruction Editing

**NeurIPS 2026**

[Project page](https://wenjunhuang94.github.io/Siphon/) | [Paper (PDF)](docs/Siphon.pdf) | [OpenReview](https://openreview.net/forum?id=mKfQzVpQIM) | [NeurIPS](https://neurips.cc/virtual/2026/loc/sydney/poster/149907)

**Wenjun Huang\***, **Rui Zhao\***, Suyang Hou, Jiahao Tang, Wenjia Wang, Zhuobai Dong, **Alex Jinpeng Wang†**, **Hu Jian Guo†**

\* Equal contribution. † Co-corresponding authors.

Sun Yat-sen University; National University of Singapore; Central South University; Wuhan University; Shenzhen Research Institute of Sun Yat-Sen University.

> Your image is the instruction. Siphon teaches image editors to follow short text labels, boxes, and arrows drawn directly on an image, without an external VLM planner at inference time.

![The Siphon framework](docs/assets/pipeline.webp)

### Overview

Siphon first uses a VLM spatial planner and an image editor to synthesize pairs of marked inputs and clean editing targets. It then internalizes visual instruction-following competence through a text-as-image curriculum and lightweight LoRA fine-tuning. At inference time, the adapted editor reads the marked image directly.

The project page includes interactive input/output comparisons and the original qualitative figures from the paper. These are recorded paper results, not live inference.

### Release status

- [x] Camera-ready paper
- [x] Paper project website and qualitative results
- [ ] Training and inference code
- [ ] Model checkpoints
- [ ] Dataset and data synthesis pipeline

This initial release contains the paper and website. Research code will be added in a future update. Code licensing and dependency information will accompany that release.

### Citation

```bibtex
@inproceedings{huang2026siphon,
  title     = {Internalize External Competence for Visual Instruction Editing},
  author    = {Wenjun Huang and Rui Zhao and Suyang Hou and
               Jiahao Tang and Wenjia Wang and Zhuobai Dong and
               Alex Jinpeng Wang and Hu Jian Guo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=mKfQzVpQIM}
}
```

### Website development

The site is buildless HTML, CSS, and JavaScript in `docs/`. Preview it locally:

```bash
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. To publish using GitHub Pages, select **Settings > Pages > Deploy from a branch > main > /docs**.

The paper figures and example pairs are taken from the accompanying paper. The interactive examples show the marked input and Siphon + LongCat-Image output from the same columns of the box-guided results figure.

### Contact

- Wenjun Huang: huangwj98@mail2.sysu.edu.cn
- Alex Jinpeng Wang (co-corresponding): jinpengwang@csu.edu.cn
- Hu Jian Guo (co-corresponding): hujguo@mail.sysu.edu.cn
