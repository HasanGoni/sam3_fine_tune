# UNVEIL recipe on a custom dataset: rules

Status: **proposed**, nothing implemented. Tracked in #18. Source: UNVEIL repo (`configs/unveil.yaml`,
`unveil/data/streams.py`, `unveil/train/{trainer,losses,calibrate,joint}.py`), paper arXiv 2611.25747.

## What the paper's recipe is made of

| Part | Paper | Custom dataset |
|---|---|---|
| Prompts | 8 task prompts (ids 0-7), then LVIS names | one prompt per `category_name`, ids `0..N-1` |
| Streams | 12 hard-coded in `TRUNK_STREAMS`, 2 images per GPU each | one stream per category, built from the dataset |
| Loss by kind | `region` BCE+IoU; `shadow`, `edge`, `skeleton` class-balanced BCE | `region` or `balanced`, declared per category |
| Presence | 1 present, 0 verified absent, otherwise skipped | same; negatives must be verified (rule 4) |
| Replay (LVIS) | positives + verified-absent categories | **off** (`per_gpu: 0`) |
| Joint presence module | salient + camouflaged, two convention vectors | **generalized** to a configured pair of categories, optional |
| Calibration | hard-coded SBU, ISTD, Trans10K, MSD | per category, from a held-out slice of our train split |

The trainer already takes `index=` and `prompts=` and a `per_gpu` override. It does not take custom streams, a custom
joint pair or custom calibration sets; those need code changes in our fork of the training code.

## Rules

1. **Prompts.** The prompt of a category is its `category_name`, exactly as in the COCO file (train and inference must
   match). Task prompt ids are `0..N-1` in category order; replay ids would start at `N`.
2. **One stream per category.** `per_gpu = max(1, 24 // N)` images per step. The paper's 24 images per step is the
   memory budget, so GPU memory does not grow with N. N = 12 gives 2 per stream, the paper's value.
3. **Loss kind is declared per category** in the recipe file (`kind: region | balanced`). There is no built-in guess.
   `region` = BCE + IoU. `balanced` = class-balanced BCE (+ IoU), for thin structures where background dominates.
4. **Negatives must be verified.** A category may take presence-0 images only from sources where it was exhaustively
   annotated. An image that merely has no annotation for it may hold an unlabeled defect, which teaches the presence
   decoder wrong answers. Each source in the merged dataset therefore lists the categories it annotates.
5. **Steps scale with the data.** `iterations = epochs x train_images / images_per_step`. The paper's 11,000 x 24 is
   about 264k samples. Keep the optimizer settings: lr 8e-5 (heads), 2.5e-4 (LoRA r16, alpha 32, ViT blocks 24-31),
   clip 0.1, warmup 300 then cosine.
6. **Batch layout.** `GPUs x accumulate = 8` reproduces the paper's batch. With one GPU, `accumulate: 8` costs about
   8x the time per step.
7. **Joint module (optional).** `joint: [category_a, category_b]` enables it for two categories that can share an
   image and need separate presence decisions. Without it the trainer skips the module and the convention vectors.
8. **Calibration uses train data only.** Presence thresholds are fitted on a held-out slice of the train split (a
   fixed fraction per category, never the test split). Binarization thresholds are fitted per category on train images.
9. **Fixed.** Input resolution 1008 (RoPE), SAM3 base weights, bf16 autocast, TF32 on in training, off in evaluation.

## Recipe file (sketch)

```yaml
data_root: /path/to/dataset            # train/{images,annotation_merged.json}
categories:
  matrix_chip:      {kind: region}
  thin_wire_panda:  {kind: balanced}
  eberhard_pin:     {kind: region, negatives_from: [source_a]}   # rule 4
joint: [thin_wire_HD, thick_wire_HD]   # optional
replay: false
epochs: 10                             # rule 5 derives iterations
```

## Open questions

- Which sources annotate which categories (rule 4)? This decides how many verified negatives each category gets.
- Does the `balanced` loss need a different weight than the paper's HED/RCF form for wires 2-5 px wide?
- Real step time and peak memory on the GB10 (measure with a few hundred steps before choosing `epochs`).
- Licence: SAM License and the non-commercial terms of UNVEIL's training sets, before any Acme use.
