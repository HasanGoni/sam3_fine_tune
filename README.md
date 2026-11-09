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
uv pip install -e . --no-deps   # once: installs the `sam3-dataset` and `sam3-finetune` commands
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

Notebooks in `nbs/` (a `sam3-infer` command is planned, see issues #13–#17):

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
└── inference/         decoder inference helper
nbs/                   numbered notebooks (01 coco_format … 14 sambaunet_api_and_cli)
tests/                 pytest suite
scripts/               analysis and poster scripts
experiments/           per-run outputs
.github/skills/        HPC runbooks
```
