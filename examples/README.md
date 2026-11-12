# Recorded inference examples

These eight selected pairs cover four visual instruction types with two examples each. They were generated during code-release preparation using LongCat-Image-Edit with the Siphon LoRA from the local `edit_lora_model-stage2-final-0225/checkpoints-10000` run. They are separate from the original paper figures in `../docs/`.

[Overview](../assets/examples_overview.jpg) · [Machine-readable manifest](examples.json) · [Inference instructions](../README.md#inference)

### Inputs, outputs, and observations

All cases use 50 steps, guidance scale 4.5, seed 43 with a CPU generator, an empty negative prompt, and one output per input. The corresponding adapter weights have SHA256 `0ed9694868d2552d243140caf242ef38e0d69f6673a3eec36e4ddc53ed459535`. The recorded base snapshot is `7b54ef423aa7854be7861600024be5c56ab7875a` of `meituan-longcat/LongCat-Image-Edit`.

| Type | Case | Intended behavior | Observed result and limits |
| --- | --- | --- | --- |
| I | [`text_rifle`](inputs/text_rifle.png) → [output](outputs/text_rifle.png) | Generate the scene described by the text: a black rifle on blue fabric. | The main scene is generated and the instruction paragraph disappears. Requested engraving is inaccurate and the silver muzzle detail is not clear. |
| I | [`text_botanical`](inputs/text_botanical.png) → [output](outputs/text_botanical.png) | Generate a botanical illustration with flowers, leaves, scientific labels, and a beige border. | The botanical illustration and border are present. Scientific labels and some flower/color-label requirements are missing. |
| II | [`text_bow_tie`](inputs/text_bow_tie.png) → [output](outputs/text_bow_tie.png) | Add a dark red bow tie to the man. | The bow tie is added and the instruction overlay is removed. Face, beard, suit, and shirt details are visibly redrawn. |
| II | [`text_bananas`](inputs/text_bananas.png) → [output](outputs/text_bananas.png) | Replace the fruit on the plate with bananas. | The replacement is made and the instruction overlay is removed. Bananas are large; furniture, blinds, and branches are also redrawn while the broad layout remains recognizable. |
| III | [`arrow_pink_flower`](inputs/arrow_pink_flower.png) → [output](outputs/arrow_pink_flower.png) | Change the arrow-indicated purple flower cluster to pink. | The target becomes pink and the arrow/text disappear. Other flowers also turn pink, and flower/leaf geometry changes; this is not an exclusively local edit. |
| III | [`arrow_denim_shirt`](inputs/arrow_denim_shirt.png) → [output](outputs/arrow_denim_shirt.png) | Replace the indicated red hoodie with a light blue denim button-up shirt. | A blue denim-style shirt with buttons and chest pockets is generated; arrow/text disappear. Pose and office layout remain recognizable, but face and scene details are redrawn. |
| IV | [`box_double_bass`](inputs/box_double_bass.png) → [output](outputs/box_double_bass.png) | Insert a double bass inside the red box in the kitchen. | The instrument is inserted and the box/text disappear. The kitchen layout is preserved broadly, with changes to cabinet and floor textures. |
| IV | [`box_toy_car`](inputs/box_toy_car.png) → [output](outputs/box_toy_car.png) | Insert a toy car in the red box beside the elephant. | An orange-red model convertible appears in the marked region; the box/text disappear. Elephant, rainbow, and trees retain their overall composition, with texture changes. |

### Side-by-side comparisons

Each comparison is a resized preview; the PNG inputs and outputs retain their recorded resolution. Output sizes may differ from input sizes because the pipeline selects supported dimensions.

| Type | Comparison | Generic text-channel prompt |
| --- | --- | --- |
| I | [Text-to-image](comparisons/type_I.jpg) | `Generate a realistic image based on the text description in the image` |
| II | [Text instruction editing](comparisons/type_II.jpg) | `Edit the image according to the text instruction in the image` |
| III | [Arrow instruction editing](comparisons/type_III.jpg) | `Follow the visual pointer and text to edit the image` |
| IV | [Box instruction editing](comparisons/type_IV.jpg) | `Modify the image at the annotated location according to the text instruction` |

The case-specific instruction is already rendered in the input image. These runs did not transcribe that instruction into a separate task-specific text prompt.

### Reproduce a case

Supply your adapter; the public checkpoint download is still pending:

```bash
python -m siphon.inference \
  --lora /path/to/siphon-adapter \
  --examples examples/examples.json \
  --case box_double_bass \
  --output-dir outputs
```

Run this from the repository root. Input paths in the manifest resolve relative to the manifest file, so the manifest can also be passed by its full path. The code defaults to the public LongCat-Image-Edit base model. To match the recorded artifacts, also use the recorded adapter hash, base revision, and software versions in [examples.json](examples.json). Seeds alone do not ensure identical images across software or hardware environments.

For the public base model, select the recorded snapshot with `--revision 7b54ef423aa7854be7861600024be5c56ab7875a`; alternatively pass the downloaded snapshot directory as `--base-model`.

Use `--dry-run` to inspect job selection without loading the model. For a baseline comparison, use `--base-only` without `--lora`, and save into a separate output directory; baseline outputs must not be labeled Siphon.

### Selection and observed failures

This is a selected qualitative gallery, not an independent test set or a success-rate measurement. The two arrow cases shown here were generated after an earlier pair failed: the basketball was not replaced by a frisbee, and the remote control was not replaced by a crystal ball, even though the annotation overlays disappeared. Those failed attempts are not copied into this gallery. Their behavior shows why annotation removal alone is insufficient to establish successful instruction following.

The included arrow cases complete the requested target changes, with the preservation limits documented above. None of these observations establishes pixel-exact preservation outside the edited region.

### Provenance

The release manifest records logical source paths from the research workspace, input/output asset hashes, the adapter hash, generation settings, and observations. Logical source names identify the files used in these runs; they do not establish original image ownership or dataset redistribution terms.

The paired inputs were supplied from the authors' existing research examples, and the outputs were generated in the recorded runs. Complete upstream attribution and redistribution records for the underlying source images have not yet been assembled. These assets and model weights are therefore not represented as covered by the repository's Apache-2.0 code license. The paper figures under `docs/` remain separately attributed to the accompanying paper.
