# SAM3 Fine-Tuning

Fine-tune Meta's [SAM3](https://github.com/facebookresearch/sam3) ([paper](https://arxiv.org/abs/2511.16719)) for
text-prompted segmentation of your own categories. Pipeline: **images + masks → COCO dataset → fine-tune → evaluate**.
The text prompt is the COCO `category_name`; use the same string at training and inference.

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Build or extend a dataset (`sam3-dataset`)](#build-or-extend-a-dataset-sam3-dataset)
4. [Train and evaluate (`sam3-finetune`)](#train-and-evaluate-sam3-finetune)
5. [Inference and further evaluation](#inference-and-further-evaluation)
6. [SAMBA-UNet baseline](#samba-unet-baseline)
7. [Training on an LSF cluster](#training-on-an-lsf-cluster)
8. [Troubleshooting](#troubleshooting)
9. [Repository layout](#repository-layout)

---

## Installation

Requirements: Python 3.12, PyTorch 2.7, CUDA 12.6 (a GPU is strongly recommended; CPU works for debugging only).

```bash
git clone https://github.com/HasanGoni/sam3_fine_tune.git
cd sam3_fine_tune

uv venv --python 3.12
source .venv/bin/activate
uv sync
uv pip install -e . --no-deps   # once: installs the `sam3-dataset`, `sam3-finetune` and `sam3-infer` commands
```

Optional: `uv sync --extra sambaunet` for the SAMBA-UNet baseline. Check the install with `pytest tests`.

You also need two files on the machine that trains:

| File | What |
|---|---|
| `sam3.pt` | the downloaded SAM3 base weights |
| `bpe_simple_vocab_16e6.txt.gz` | tokenizer vocabulary, from the SAM3 release (not stored in this repo) |

## Quick start

```bash
# 1) images + binary masks -> COCO dataset (repeat once per category)
sam3-dataset images=./imgs masks=./masks category="bond wire" into=/path/to/dataset

# 2) train, then evaluate the best checkpoint on test/ automatically
sam3-finetune --data-root /path/to/dataset \
              --sam3-checkpoint /path/to/sam3.pt \
              --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz \
              --epochs 10
```

| Command | Also runs as | Purpose |
|---|---|---|
| `sam3-dataset` | `python -m sam3_fine_tune.dataset` | Hydra CLI: images + masks → COCO, train/test split, create or extend a category |
| `sam3-finetune` | `python -m sam3_fine_tune.finetune` | train SAM3, then evaluate the best checkpoint on `test/` |
| `sam3-infer` | `python -m sam3_fine_tune.inference` | base or fine-tuned SAM3 with one or several text prompts (each with its own threshold) on a folder of images |
| `sam3-infer-unveil` | `python -m sam3_fine_tune.infer_unveil` | UNVEIL on a folder of images or a COCO split, same arguments and output layout as `sam3-infer`, plus `gate=false` and `path=sam3` (needs the `unveil` package and a CUDA GPU) |
| `sam3-compare` | `python -m sam3_fine_tune.compare` | score SAM3 (base, fine-tuned, saved predictions) and UNVEIL columns on the prepared test split with pooled union-mask IoU per category; UNVEIL columns cut the map at the checkpoint's per-category `binarize` when present, else `threshold`, and the table says which |
| `sam3-train-unveil` | `python -m sam3_fine_tune.train_unveil` | UNVEIL recipe on a prepared root: `validate_only` checks `recipe.yaml`, the prepared view and the per-category streams, a trial run (`validate_only=false`) reports step time and peak memory (`model=unveil` is untested on the real model; `model=stub` is a CPU smoke run); training comes in a later PR |
| – | `python -m sam3_fine_tune.sambaunet.train` / `.evaluate` | Hydra CLIs for the SAMBA-UNet baseline |

---

## Build or extend a dataset (`sam3-dataset`)

Input: a folder of images and a folder of binary masks (0 = background, 255 = object) for **one** category. An image and
its mask are matched by file stem. Output layout:

```
dataset/
├── train/{images/, annotation.json, annotation_merged.json}
└── test/ {images/, annotation.json, annotation_merged.json}
```

```bash
sam3-dataset images=./imgs masks=./masks category="bond wire" into=/path/to/dataset test_fraction=0.2
```

Every key is a Hydra `key=value` override (defaults in `sam3_fine_tune/dataset/conf/dataset.yaml`):

| Key | Meaning |
|---|---|
| `images`, `masks` | input folders (png/jpg/tif/bmp). Unmatched files are reported; size mismatches abort before anything is written |
| `category` | category name = the SAM3 text prompt |
| `into` | dataset root; created if missing |
| `mode` | `auto` (default): `extend` if the category already exists, else `new`; or force `new` / `extend` |
| `test_fraction`, `seed` | reproducible train/test split of the new images (`0` = all to train) |
| `prefix` | prepended to copied image names to avoid collisions (default: slug of the category) |
| `dry_run=true` | print what would change, write nothing |

- Existing images, annotations and ids are never changed; a new category gets the same id in `train` and `test`.
- Re-running is safe: images already present (by prefixed name) are skipped.
- `annotation_merged.json` (the file the training configs read) is created if absent and receives the same additions if present.
- The result is validated; problems are printed and the exit code is non-zero.

Alternatives: `nbs/01_coco_format.ipynb` (masks → COCO) and `nbs/02_merge_dataset.ipynb` (`merge_datasets_multi`,
`append_dataset`) for notebook-driven or HPC-driven workflows.

---

## Train and evaluate (`sam3-finetune`)

One command copies `sam3/train/configs/full_train_config.yaml`, applies your flags, runs `sam3/train/train.py`, selects the
best-validation-AP checkpoint and scores it on `test/`. No scheduler is needed; it runs wherever you start it.

```bash
sam3-finetune \
    --data-root /path/to/dataset \
    --sam3-checkpoint /path/to/sam3.pt \
    --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz \
    --experiment-dir experiments/my_run \
    --epochs 10 --train-batch-size 2 --grad-accum 2 --category-chunk-size 9
```

Outputs in `<experiment-dir>/`: `config_used.yaml`, `checkpoints/`, `tensorboard/`, `logs/` and
`eval/multiclass_foreground_iou_summary.json` (per-category `overall_foreground_iou`, `pixel_precision`, `pixel_recall`,
plus a `meta` block with checkpoint, threshold, device and latency).

| Flag | Meaning |
|---|---|
| `--data-root`, `--sam3-checkpoint`, `--bpe-path` | required |
| `--experiment-dir` | output directory (default `experiments/sam3_<timestamp>`) |
| `--epochs`, `--train-batch-size`, `--val-batch-size`, `--grad-accum` | schedule; `num_gpus × train_batch_size × grad_accum` must divide the train set size |
| `--category-chunk-size` | GPU-memory lever (train and val); GPU memory grows with it, so check headroom before raising it |
| `--num-gpus`, `--accelerator auto\|cuda\|cpu` | hardware; `cpu` is for debugging only |
| `--base-config` | use another YAML than `full_train_config.yaml` as the template |
| `--skip-eval` / `--eval-only` | train only / evaluate an existing `--experiment-dir` |
| `--threshold`, `--max-eval-images` | evaluation confidence threshold (default 0.5) and an image cap for quick checks |

Notes:

- Full fine-tuning (all components trained) is the recommended mode; decoder-only converges later and scores lower.
  Use `full_train_config.yaml` for both — not `decoder_train_config.yaml`.
- A score of `0.0` after a very short run usually means the model's confidence is below `--threshold`; re-run with
  `--eval-only --threshold 0.1`.
- Re-using an `--experiment-dir` resumes from its `checkpoint.pt`; use a fresh directory for a clean run.
- `scratch.resolution` is fixed at 1008 (ViT RoPE is precomputed for it).
- Verified: unit tests and one end-to-end run (tiny synthetic dataset, real `sam3.pt`, 1 epoch, single GPU). Not yet
  verified: `bsub`, multi-GPU, and agreement with the notebook evaluation (issue #11).

To edit a config by hand instead, change `paths.pin_dir_root`, `paths.experiment_log_dir`, `paths.bpe_path` and
`trainer.model.checkpoint_path` in a copy of `full_train_config.yaml`, then run from `sam3/train/`:

```bash
python train.py -c configs/my_config.yaml --num-gpus 1 --num-nodes 1
```

---

## Inference and further evaluation

### Segment a folder with text prompts (`sam3-infer`)

```bash
# base SAM3
sam3-infer images=./imgs prompt="bond wire" out=./pred \
           sam3_checkpoint=/path/to/sam3.pt bpe_path=/path/to/bpe_simple_vocab_16e6.txt.gz
# fine-tuned model: add the checkpoint (the base sam3.pt is still needed to build the model)
sam3-infer ... checkpoint=experiments/run1/checkpoints/checkpoint.pt
```

Writes `out/masks/<name>.png` (0/255, union of all instances), `out/overlays/<name>.png` and `out/summary.json`
(per image: instances, foreground pixels, scores; plus prompt, threshold, device, checkpoint). A rerun overwrites the
same files. Options: `threshold=0.5`, `device=auto|cpu|cuda` (defaults in `sam3_fine_tune/inference/conf/infer.yaml`).
Also runs as `python -m sam3_fine_tune.inference`.

**Several prompts**, each with its own confidence threshold (entries without one use `threshold`):

```bash
sam3-infer images=./imgs out=./pred sam3_checkpoint=... bpe_path=... threshold=0.5 \
  'prompts=[{text: "bond wire", threshold: 0.4}, {text: pin}, {type: text, text: crack, threshold: 0.7}]'
```

**Threshold sweep:** give a prompt a list of thresholds to compare them side by side.

```bash
sam3-infer images=./imgs out=./pred sam3_checkpoint=... bpe_path=... \
  'prompts=[{text: "bond wire", threshold: [0.3, 0.5, 0.7]}]'
```

The model runs once per image and prompt, at the lowest threshold; the higher ones only re-filter the instances by score,
so a sweep costs almost nothing extra. Masks and overlays get one `thr_<t>/` folder per threshold
(`masks/[<prompt>/]thr_0.5/<name>.png`, `overlays/[<prompt>/]thr_0.5/<name>.png`), `summary.json` has a `sweep` list per
image (threshold, instances, pixels, scores), and a table of total instances vs threshold per prompt is printed. A
single number behaves as before; `overlays/<name>.png` shows every prompt at its lowest threshold.

The image backbone runs once per image and is reused for every prompt. With several prompts the masks go to
`out/masks/<prompt>/<name>.png`, the overlay shows one colour per prompt, and `summary.json` has a per-prompt entry
(instances, foreground pixels, scores, threshold used; a `sweep` list instead for swept prompts) under each image. Invalid config (empty list, duplicate texts,
threshold outside (0, 1]) fails before the model loads, naming the entry. Each entry is `{type, text, threshold}`;
`type: text` is the only kind today and the entry shape is the extension point for other prompt kinds (`PROMPT_TYPES`
in `sam3_fine_tune/inference/run.py`). A single `prompt="..."` still works; set one of `prompt` / `prompts`, not both.

**Score against a COCO split and compare models:**

```bash
# score one model on dataset/test (the prompt text must equal a category_name)
sam3-infer coco_split=dataset/test coco_ann=annotation_merged.json out=./pred \
  sam3_checkpoint=... bpe_path=... checkpoint=experiments/run1/checkpoints/checkpoint.pt \
  'prompts=[{text: "bond wire"}, {text: pin}]'
# base vs fine-tuned(s) on the same images: null is the base model
sam3-infer coco_split=dataset/test ... 'checkpoint=[null, experiments/run1/checkpoints/checkpoint.pt]'
```

`coco_split` replaces `images`: the split's images are segmented and scored against the rasterised COCO masks. Scores are
written to `out/multiclass_foreground_iou_summary.json` in the same layout as `sam3-finetune` and the SAMBA-UNet
evaluation (they share the scoring code), so the numbers match `sam3-finetune --eval-only` for the same checkpoint,
threshold and images (when every category has a prompt: `--eval-only` scores all categories, this scores the matching ones); use `coco_ann=annotation_merged.json` to match its annotation file. With a checkpoint list, each model
writes to `out/<model>/` (`base`, or `<experiment>_<checkpoint name>`) with its own summary, and a side-by-side table of
foreground IoU / precision / recall (%) per category is printed. Prompts without a matching category, and categories
without a prompt, are reported on stderr and under `coco` in `summary.json` (and in the score file's `meta`); only
matching pairs are scored, and no match at all is an error before the model loads. Scoring needs one threshold per
prompt (no sweeps).

### Notebooks

Further inference and evaluation steps in `nbs/`:

| Notebook | Use |
|---|---|
| `03_inference.decoder_inference` | load a fine-tuned checkpoint and segment one image with a text prompt |
| `04_single_eval_pipeline` / `05_multi_eval_pipeline` | foreground IoU, precision, recall, F1 per category |
| `06_visualization_pipeline` | predictions overlaid on ground truth |
| `07_img_to_mask` | run a checkpoint over a folder and save masks and overlays |
| `08_correct_incorrect_filter` | split images into correctly / incorrectly segmented by an IoU threshold |

Metrics: IoU = TP / (TP + FP + FN), precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2PR / (P + R).

---

## SAMBA-UNet baseline

A semantic-segmentation baseline (SAM ViT-B + Mamba U-Net, no text prompts) trained on the same COCO dataset. It writes the
same `multiclass_foreground_iou_summary.json`, so the two models compare key by key.

```bash
uv sync --extra sambaunet
python -m sam3_fine_tune.sambaunet.train data.root=/path/to/dataset trainer.epochs=100
python -m sam3_fine_tune.sambaunet.evaluate checkpoint=experiments/sambaunet/<run>/checkpoints/best.pt data.root=/path/to/dataset
```

Tutorial (synthetic data, CPU): `nbs/14_sambaunet_api_and_cli.ipynb`. Details: `sam3_fine_tune/sambaunet/README.md`.

---

## Training on an LSF cluster

Wrap the same training command in `bsub` (adapt queue, resources and project):

```bash
bsub -q gpu_compute -gpu "num=1:j_exclusive=yes" -n 8 \
  "bash -c 'source .venv/bin/activate && sam3-finetune --data-root ... --sam3-checkpoint ... --bpe-path ...'"
```

Single-GPU `gpu_compute` is preferred over multi-GPU: `train_batch_size` is per GPU, so more GPUs do not lower per-GPU memory.
For out-of-memory errors, reduce `--train-batch-size`, `--val-batch-size` or `--category-chunk-size`.
Runbooks for the Acme HPC workflow live in `.github/skills/` (start with `configure-repo`).

---

## Troubleshooting

**`Loss is nan`, or loss stuck at exactly `0.00000`**

1. Inspect `<experiment-dir>/dumps/anomalous_batches/*.pt` (written automatically, max 3 per run):
   ```bash
   python .github/scripts/inspect_anomalous_batch.py <experiment-dir>/dumps/anomalous_batches/<file>.pt
   ```
2. Fall back to `<experiment-dir>/checkpoints/checkpoint_val_pin_detection_coco_eval_segm_AP.pt`, the best-validation-AP
   checkpoint, which is kept permanently.
3. See `.github/skills/loss-debugging/SKILL.md` for which loss terms are NaN-guarded.

**Job killed with exit code 143 (`TERM_OWNER`)**: the cluster kills jobs whose GPU utilisation stays below ~5% for
5+ minutes. Check for a slow setup phase or high `Data Time` steps before assuming a bug.

**`gradient_accumulation_steps` assertion** (`Expected N batches, got M`): `num_gpus × train_batch_size × grad_accum`
must divide the training set size.

---

## Repository layout

```
sam3/                  vendored SAM3 model + training framework (train/train.py, train/configs/)
sam3_fine_tune/
├── dataset/           sam3-dataset
├── finetune/          sam3-finetune
├── sambaunet/         SAMBA-UNet baseline
├── preprocessing/     mask → COCO, merge helpers, DINOv3 diversity rebalancing
├── evaluation/        single / multi-class evaluation pipelines
└── inference/         sam3-infer, decoder inference helper
nbs/                   numbered notebooks (01 coco_format … 14 sambaunet_api_and_cli)
tests/                 pytest suite
scripts/               analysis and poster scripts
experiments/           per-run outputs
.github/skills/        HPC runbooks
```
