# SAM3 Fine-Tuning: Complete Documentation

This guide provides comprehensive documentation for fine-tuning SAM3 for segmentation tasks. It covers the complete pipeline from data preparation to evaluation, and can be followed either to **fine-tune on a new category** (full fine-tuning) or to **segment existing categories** already covered by a checkpoint you've trained with this workflow.

---

## Table of Contents

1. [What is SAM3?](#what-is-sam3)
2. [Two Ways to Use This Repository](#two-ways-to-use-this-repository)
3. [Overview: Full Fine-Tuning vs. Decoder-Only Fine-Tuning](#overview-full-fine-tuning-vs-decoder-only-fine-tuning)
4. [Getting Started](#getting-started)
5. [Repository Structure](#repository-structure)
6. [Pipeline Overview](#pipeline-overview)
7. [Step 1: Data Preparation](#step-1-data-preparation)
8. [Step 2: Convert to COCO Format](#step-2-convert-to-coco-format)
9. [Step 3: Merge Datasets (Multi-Class)](#step-3-merge-datasets-multi-class)
10. [Step 4: Training Configuration](#step-4-training-configuration)
11. [Step 5: Run Training](#step-5-run-training)
12. [Step 6: Inference](#step-6-inference)
13. [Step 7: Evaluation](#step-7-evaluation)
14. [SAMBA-UNet baseline](#samba-unet-baseline-comparison-with-sam3)
15. [Step 8: Visualization &amp; Analysis](#step-8-visualization--analysis)
16. [Future Direction of the Project](#future-direction-of-the-project)
17. [Training Commands Quick Reference](#training-commands-quick-reference)
18. [Troubleshooting: Training Crashes &amp; Anomaly Dumps](#troubleshooting-training-crashes--anomaly-dumps)

---

## What is SAM3?

[Paper](https://arxiv.org/abs/2511.16719) . [GitHub](https://github.com/facebookresearch/sam3)

SAM3 (Segment Anything Model 3) is Meta's open-vocabulary segmentation model that combines a powerful vision backbone (ViT), a language backbone (text encoder), and a transformer-based decoder to segment objects described by text prompts. Unlike classical segmentation models trained on fixed categories, SAM3 is designed to generalize to new object classes via natural-language queries.

SAM3 consists of:

- A **vision backbone** that extracts rich spatial features from images
- A **language backbone** that encodes free-form text prompts
- A **cross-attention encoder** that fuses visual and textual features
- A **decoder and segmentation head** that predicts instance masks

---

## Two Ways to Use This Repository

### Option 1: Use a Pre-Trained Checkpoint (Inference Only)

If your category is already covered by a checkpoint you (or a teammate) already trained, you can skip training entirely and go straight to inference and evaluation:

- [Step 6: Inference](#step-6-inference) — `03_inference.decoder_inference.ipynb`
- [Step 7: Evaluation](#step-7-evaluation) — `04_single_eval_pipeline.ipynb` / `05_multi_eval_pipeline.ipynb`
- [Step 8: Visualization &amp; Analysis](#step-8-visualization--analysis) — `06_visualization_pipeline.ipynb` / `07_img_to_mask.ipynb` / `08_correct_incorrect_filter.ipynb`

Which categories/prompts are already covered depends on which checkpoint you
have — see the `metadata.json` shipped alongside your fine-tuned checkpoint
for the exact category list, per-category train/test counts, and training
configuration.

> **Note on layout:** the text prompts and model weights produced by this
> workflow are only as good as the image layout they were trained on. Using
> a substantially different layout — particularly noisy or cluttered images
> — may result in poorer segmentation performance.

### Option 2: Fine-Tune on a New Category

If you want to train on a new category or a different domain, follow the full pipeline from [Step 1](#step-1-data-preparation) through [Step 7](#step-7-evaluation). Two training modes were tried (decoder and full), however, full finetuning has performed better and even converged earlier. Hence, it would be recommended to do full-finetuning.

**Fastest path (two commands, no scheduler needed):**

```bash
# 1) images + binary masks -> COCO dataset (new dataset, new category, or extend a category); repeat per category
sam3-dataset images=./imgs masks=./masks category="bond wire" into=/path/to/dataset
# 2) train, then evaluate the best checkpoint on test/ automatically
sam3-finetune --data-root /path/to/dataset --sam3-checkpoint /path/to/sam3.pt --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz --epochs 10
```

Details: [`sam3-dataset`](#shortcut-build-or-extend-a-dataset-with-one-command-sam3-dataset) and
[`sam3-finetune`](#shortcut-train-and-evaluate-with-one-command). The step-by-step notebook/HPC route below still works.

---

## Overview: Full Fine-Tuning vs. Decoder-Only Fine-Tuning

### Components Overview

| Component                        |   Decoder-Only   | Full Fine-Tuning | Description                  |
| -------------------------------- | :---------------: | :---------------: | ---------------------------- |
| `backbone.vision_backbone.*`   | **FROZEN** | **TRAINED** | ViT image encoder            |
| `backbone.language_backbone.*` | **FROZEN** | **TRAINED** | Text encoder                 |
| `transformer.encoder.*`        | **FROZEN** | **TRAINED** | Cross-attention encoder      |
| `transformer.decoder.*`        | **TRAINED** | **TRAINED** | Object query decoder         |
| `segmentation_head.*`          | **TRAINED** | **TRAINED** | Mask prediction head         |
| `geometry_encoder.*`           | **TRAINED** | **TRAINED** | Spatial/geometric embeddings |
| `dot_prod_scoring.*`           | **TRAINED** | **TRAINED** | Classification scores        |

---

## Getting Started

### Prerequisites

- Python 3.12
- PyTorch 2.7
- CUDA 12.6
- Downloaded SAM3 model weights (path provided below)

### Setup

```bash
git clone https://github.com/HasanGoni/sam3_fine_tune.git
cd sam3_fine_tune

uv venv --python 3.12
source .venv/bin/activate
uv sync
uv pip install -e . --no-deps   # once: creates the `sam3-dataset` and `sam3-finetune` commands (a plain `uv sync` also does)
```

Optional extras: `uv sync --extra sambaunet` for the SAMBA-UNet baseline. Run the tests with `pytest tests`.

### Command-line tools

| Command | Also runs as | Purpose |
|---|---|---|
| `sam3-dataset` | `python -m sam3_fine_tune.dataset` | Hydra CLI: images + masks -> COCO, train/test split, create or extend a category |
| `sam3-finetune` | `python -m sam3_fine_tune.finetune` | argparse CLI: train SAM3, then evaluate the best checkpoint on `test/` |
| - | `python -m sam3_fine_tune.sambaunet.train` / `.evaluate` | Hydra CLIs for the SAMBA-UNet baseline |

### Required paths:

You'll need, on whatever machine you run training/inference on:

- **SAM3 Base Checkpoint:** path to the downloaded `sam3.pt` weights.
- **BPE Tokenizer:** path to `asset/bpe_simple_vocab_16e6.txt.gz`.
- **Fine-Tuned Checkpoint(s):** wherever you save output from your own
  training runs (`paths.experiment_log_dir` in the training config).
- **Dataset:** the COCO-format dataset you built via
  [Step 1](#step-1-data-preparation)–[Step 3](#step-3-merge-datasets-multi-class).

Update these paths in `sam3/train/configs/full_train_config.yaml` (and in
the inference/evaluation notebooks) to point at your own copies — none of
this repo's example configs ship with real checkpoint paths.

---

## Repository Structure

### Added Files (correspondingly named Python scripts for notebooks)

```
sam3/
├── 01_coco_format.ipynb                       # Convert binary masks to COCO polygon format
├── 02_merge_dataset.ipynb                     # Merge multiple datasets for multi-class training
├── 03_inference.decoder_inference.ipynb       # Run inference with fine-tuned model (single image)
├── 04_single_eval_pipeline.ipynb              # Evaluate single-class models using Foreground IOU
├── 05_multi_eval_pipeline.ipynb               # Evaluate multi-class models per class/category
├── 06_visualization_pipeline.ipynb            # Visualize predictions vs. ground truth masks (Optional)
├── 07_img_to_mask.ipynb                       # Generate masks and overlays from a fine-tuned checkpoint (Optional)
├── 08_correct_incorrect_filter.ipynb          # Filter images into correct/incorrect predictions (Optional)
├── 14_sambaunet_api_and_cli.ipynb             # SAMBA-UNet Python API + CLI tutorial (synthetic data, CPU)
└── sam3/
    └── train/
        └── configs/
            ├── decoder_train_config.yaml      # Decoder-only fine-tuning config
            └── full_train_config.yaml         # Full fine-tuning config
```

Command-line tools and their tests:

```
sam3_fine_tune/
├── dataset/      # `sam3-dataset`: masks -> COCO, split, new/extend category (conf/dataset.yaml)
├── finetune/     # `sam3-finetune`: config patching, train.py launch, test-set evaluation
└── sambaunet/    # SAMBA-UNet baseline: model, Hydra train/evaluate CLIs
tests/            # pytest: dataset/, finetune/, sambaunet/, cpu_compat/
```

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FINE-TUNING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA PREPARATION                                            │
│     └── Organize images and binary masks                        │
│                    ↓                                            │
│  2. COCO CONVERSION (01_coco_format.ipynb | preprocessing/coco_format.py)     │
│     └── Convert masks to COCO polygon format                    │
│         shortcut: `sam3-dataset` (steps 2+3 in one command)     │
│                    ↓                                            │
│  3. [Optional] MERGE DATASETS (02_merge_dataset.ipynb)          │
│     └── Combine multiple classes into one dataset               │
│                    ↓                                            │
│  4. CONFIGURE TRAINING                                          │
│     └── decoder_train_config.yaml  OR  full_train_config.yaml   │
│                    ↓                                            │
│  5. RUN TRAINING (sam3/train/train.py)                          │
│     └── Train decoder-only or full fine-tuning                  │
│         shortcut: `sam3-finetune` (train + test-set eval)       │
│                    ↓                                            │
│  6. INFERENCE (03_inference.decoder_inference.ipynb)            │
│     └── Load checkpoint and run prediction                      │
│                    ↓                                            │
│  7. EVALUATION (04/05_eval_pipeline.ipynb)                      │
│     └── Compute IoU, Precision, Recall, F1                      │
│                    ↓                                            │
│  8. [Optional] VISUALIZATION & ANALYSIS                         │
│     └── 06_visualization_pipeline.ipynb                         │
│         07_img_to_mask.ipynb                                    │
│         08_correct_incorrect_filter.ipynb                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Copilot Skills

The repository includes VS Code Copilot skills under `.github/skills/` that
automate the most common HPC workflows. Use these instead of running
commands manually when possible.

| Skill | Purpose | When to use |
|---|---|---|
| `configure-repo` | Resolve the repo root, activate the project venv, and detect Python/Torch/CUDA versions on the HPC. | Run this **first** as a prerequisite for any HPC skill. |
| `coco-format-conversion` | Convert binary masks to COCO polygon annotations on the HPC CPU queue. | Step 2: after you have `images/` and `masks/`. |
| `merge-category-dataset` | Add a new category into an existing merged SAM3 dataset. | Step 3 Option B: extending an existing dataset. |
| `diverse-wire-dataset` | Build a category-rebalanced wire dataset with DINOv3 embedding diversity selection. | Building/rebalancing a wire-category dataset. |
| `finetune-merged-dataset` | Update the training config and submit a full fine-tuning job on the HPC GPU queue. | Step 5: training on a merged dataset. |
| `post-training-error-analysis` | Run error analysis and generate overlay posters after a training run finishes. | After training completes and predictions are available. |
| `hpc-gpu-submit` | Run an arbitrary GPU workload on the HPC. | Custom GPU scripts or inference at scale. |
| `hpc-cpu-submit` | Run an arbitrary CPU workload on the HPC. | Preprocessing, poster generation, data audits. |
| `coco-grid-poster` | Generate a grid poster of COCO category images with mask overlays. | Visual label-quality checks. |
| `training-watchdog` | Monitor a training experiment and resubmit the batch job if it disappears. | Long-running training jobs. |

Each skill file contains the full command sequence and decision logic.
Invoke them from Copilot Chat by name (e.g. `run finetune-merged-dataset`).

---

## Step 1: Data Preparation

### Required Directory Structure

The dataset should follow this structure:

```
your_dataset/
├── train/
│   ├── images/
│   │   ├── image001.png
│   │   ├── image002.png
│   │   └── ...
│   └── annotation.json      # Will be created in Step 2
└── test(validation)/
    ├── images/
    │   ├── image101.png
    │   ├── image102.png
    │   └── ...
    └── annotation.json      # Will be created in Step 2
```

### Binary Masks

To train polygon segmentations are required in coco format and hence binary masks (0 = background, 255 = object) are required to generate them in the next step. These can be organized as:

```
train/
├── images/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── masks/
    ├── image001.png    # Binary mask (0/255)
    ├── image002.png
    └── ...
```

**Important:** Image and mask filenames must match exactly.

---

## Step 2: Convert to COCO Format

SAM3 training requires COCO-format annotations with polygon segmentation.

### Using the sh file (For Automation)

go to "sam3_fine_tune/preprocessing" folder Use     `./coco_format.sh`. It will run the following python script to hpc node. Change your path based on image, mask and json filenames

```bash
python coco_format.py \
    --img_dir /path/to/images \
    --mask_dir /path/to/masks \
    --file_name "name_of_json_file" \
    --output_dir /path/to/output \
    --category_name "your_class" \
    --category_id id_number
```

## Step 3: Merge Datasets (Multi-Class)

For training on multiple object classes, merge individual datasets. Or merge your data with the existing one. The notebook provides two functions depending on your situation.

---

### Option A: Build a merged dataset from scratch — `merge_datasets_multi`

Use this when creating a new merged dataset for the first time. All image IDs and annotation IDs start from 1.

Open and run `02_merge_dataset.ipynb`:

1. Define a list of datasets, one entry per class:

   ```python
   datasets = [
       {"path": "/path/to/class_a_dataset", "category_name": "et_pin",          "prefix": "a"},
       {"path": "/path/to/class_b_dataset", "category_name": "eberhard_pin",    "prefix": "b"},
       {"path": "/path/to/class_c_dataset", "category_name": "wuxi_pin",        "prefix": "c"},
       {"path": "/path/to/class_d_dataset", "category_name": "war_crack",       "prefix": "d"},
       {"path": "/path/to/class_e_dataset", "category_name": "matrix_chip",     "prefix": "e"},
       {"path": "/path/to/class_f_dataset", "category_name": "thick wire HD",   "prefix": "f"},
       {"path": "/path/to/class_g_dataset", "category_name": "thin wire HD",    "prefix": "g"},
       {"path": "/path/to/class_h_dataset", "category_name": "thick wire panda","prefix": "h"},
       {"path": "/path/to/class_i_dataset", "category_name": "thin wire panda", "prefix": "i"},
       # add more classes here
   ]
   OUTPUT_PATH = "/path/to/merged_dataset"

   merge_datasets_multi(datasets, OUTPUT_PATH)
   ```

   - `path` — root of the single-class dataset (must contain `train/` and `test/` subdirs with `images/` and `annotation.json`)
   - `category_name` — text prompt / class name (must match what you use at training and inference)
   - `prefix` — short string prepended to image filenames to avoid name collisions across datasets

   Category IDs are assigned sequentially (1, 2, 3, …) based on list order.

---

### Option B: Add a category to an existing merged dataset — `append_dataset`

Use this when a merged dataset already exists (mine can be used) and you want to add one more category without rebuilding from scratch. The existing data is untouched — IDs continue from the current max.

```python
new_dataset = {
    "path": "/path/to/new_class_dataset",
    "category_name": "war_crack",
    "prefix": "d"
}
EXISTING_OUTPUT = "/path/to/existing/merged_dataset"

append_dataset(new_dataset, EXISTING_OUTPUT)
```

---

### Output Structure

```
merged_dataset/
├── train/
│   ├── images/
│   │   ├── a_image001.png    # From class A
│   │   ├── b_image001.png    # From class B
│   │   └── ...
│   └── annotation.json       # Merged annotations
└── test/
    ├── images/
    └── annotation.json
```

The merged `annotation.json` includes all categories:

```json
{
  "categories": [
    {"id": 1, "name": "et_pin"},
    {"id": 2, "name": "eberhard_pin"},
    {"id": 3, "name": "wuxi_pin"},
    {"id": 4, "name": "war_crack"},
    {"id": 5, "name": "matrix_chip"},
    {"id": 6, "name": "thick wire HD"},
    {"id": 7, "name": "thin wire HD"},
    {"id": 8, "name": "thick wire panda"},
    {"id": 9, "name": "thin wire panda"}
  ]
}
```

---

### Shortcut: build or extend a dataset with one command (`sam3-dataset`)

Steps 2-3 can be replaced by one local command (no `ssh`/`bsub`) when you have a folder of images and a folder of binary
masks for **one** category. It converts the masks to COCO polygons, splits them into train/test, and adds them to the
dataset in `into` - creating the dataset, a new category, or extending an existing category:

```bash
sam3-dataset images=./imgs masks=./masks category="bond wire" into=/path/to/dataset test_fraction=0.2
# equivalent: python -m sam3_fine_tune.dataset ...
```

It is Hydra-configured (`sam3_fine_tune/dataset/conf/dataset.yaml`), so every key is a `key=value` override:

| Key | Meaning |
|---|---|
| `images`, `masks` | folders; an image and its mask are matched by file stem (png/jpg/tif/bmp). Unmatched files are reported, size mismatches abort before anything is written |
| `category` | category name = the SAM3 text prompt |
| `into` | dataset root `<into>/{train,test}/{images,annotation.json}`; created if missing |
| `mode` | `auto` (default): `extend` if the name already exists in `into`, else `new`; or force `new` / `extend` |
| `test_fraction`, `seed` | reproducible split of the new images (`0` = all to train) |
| `prefix` | prepended to copied image names so they cannot collide (default: slug of the category) |
| `dry_run=true` | print what would change, write nothing |

Behaviour worth knowing:

* Existing images, annotations and ids are never changed; new ids continue after the current maximum, and a new
  category gets the **same** id in `train` and `test`.
* Re-running the same command is safe: images whose prefixed name is already present are skipped (reported as
  `skipped_existing`).
* `annotation_merged.json` (read by the training configs) is created as a copy if absent; if it already exists - for
  example consolidated with `merge_duplicate_photo_entries.py` - it receives the same additions instead of being replaced.
* The result is validated (images exist, ids unique, polygons non-empty, categories known); problems are printed and the
  exit code is non-zero.
* Add one category per call; repeat the command for more categories. The HPC `.github/scripts/run_append_*.sh` helpers and
  the notebook functions still work and are unchanged.

---

## Step 4: Training Configuration

### Configuration Files

Located in `sam3/train/configs/`:

| Config File                   | Use Case                                  |
| ----------------------------- | ----------------------------------------- |
| `decoder_train_config.yaml` | Decoder-only fine-tuning (encoder frozen) |
| `full_train_config.yaml`    | Full fine-tuning (all components trained) |

---

### Key Settings to Modify

#### 1. Paths (Required — both configs)

These three paths must be updated:

```yaml
paths:
  pin_dir_root: /path/to/your/merged_dataset
  experiment_log_dir: /path/to/sam3_fine_tune/experiments/my_experiment
  bpe_path: /path/to/sam3_fine_tune/asset/bpe_simple_vocab_16e6.txt.gz
```

#### 2. Model Checkpoint (Required — both configs)

```yaml
trainer:
  model:
    checkpoint_path: /path/to/sam3.pt           # Path to the downloaded SAM3 base weights
```

#### 3. Chunk Size

```yaml
category_chunk_size: 4
```

The above parameter controls how many categories the model evaluates simultaneously per forward pass. Setting it equal to the total number of categories ensures maximum negative pressure (the model must distinguish all categories at once), which improves training quality. But increasing it also requires more GPU memory.

#### 4. Text Prompt / Category Name

The text prompt used during training must match what you use at inference time. It is set in the dataset annotation JSON (via the `category_name` used in Step 2), not directly in the YAML. Ensure consistency between training and evaluation.

## Step 5: Run Training

The following example GPU cluster (LSF `bsub`) configurations were used for
training on our HPC — adapt the queue names, resource selectors, and project
tags to whatever scheduler/cluster you have available:

```bash
mgpu_() { bsub -Is -q gpu -gpu "num=3:j_exclusive=yes:gmem=20G" -R 'select[type==X64LIN] rusage[mem=200]' -n 4 "$1"; }

mgpu_ "python train.py -c configs/config_gpu_q.yaml --num-gpus 3 --num-nodes 1"
```

```bash
gpu_q_() { bsub -Is -q gpu_compute -gpu "num=1:j_exclusive=yes" -n 8 "$1";}

gpu_q_ "python train.py -c configs/config_gpu_q.yaml --num-gpus 1 --num-nodes 1"
```

> Note: To train, one must be in the sam3/train directory.

### Shortcut: train and evaluate with one command

After Steps 2-3 you have a COCO dataset `<root>/{train,test}/{images,annotation.json}`. One command then patches a copy
of `full_train_config.yaml`, trains, and scores the best checkpoint on `test/` (text prompt = `category_name`, ground
truth = the COCO masks). It needs no scheduler: it runs wherever you start it, so on a machine with a GPU (or the CPU) it
works without `bsub`.

```bash
python -m sam3_fine_tune.finetune \
    --data-root /path/to/dataset \
    --sam3-checkpoint /path/to/sam3.pt \
    --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz \
    --experiment-dir experiments/my_run \
    --epochs 10 --train-batch-size 2 --grad-accum 2 --category-chunk-size 9
```

The same command is installed as a console script (`sam3-finetune ...`; run `uv pip install -e . --no-deps` or `uv sync`
once to create it).

**What happens, in order**

1. The dataset layout is checked, and `annotation_merged.json` is created as a copy of `annotation.json` in `train/` and
   `test/` if missing (the shipped configs read that file).
2. `full_train_config.yaml` is copied, your flags are applied, and the result is saved as
   `<experiment-dir>/config_used.yaml`. A temporary `sam3/train/configs/_gen_<run>.yaml` is written for `train.py`
   (which loads configs by name from that folder) and deleted afterwards.
3. `sam3/train/train.py` runs with that config (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set unless you set it).
4. The best-validation-AP checkpoint (`checkpoints/val_pin_detection_coco_eval_segm_AP.pt`) is selected, or
   `checkpoints/checkpoint.pt` if no best one was kept, and evaluated on `test/`.

**Outputs** in `<experiment-dir>/`: `config_used.yaml`, `checkpoints/`, `tensorboard/`, `logs/`, and
`eval/multiclass_foreground_iou_summary.json`. The JSON has the same layout as the Step 7 multi-class evaluation
(`summary.categories.<name>.overall_foreground_iou`, `pixel_precision`, `pixel_recall`, `summary.overall`,
`per_category_details`) plus a `meta` block (checkpoint used, threshold, device, median latency per image), so it can be
compared key-by-key with the SAMBA-UNet baseline below.

**Options** (anything not set keeps the value in `full_train_config.yaml`)

| Flag | Meaning |
|---|---|
| `--data-root`, `--sam3-checkpoint`, `--bpe-path` | required: dataset, base `sam3.pt`, tokenizer vocabulary |
| `--experiment-dir` | output directory (default `experiments/sam3_<timestamp>`) |
| `--epochs`, `--train-batch-size`, `--val-batch-size`, `--grad-accum` | training schedule; `num_gpus x train_batch_size x grad_accum` must divide the train set size |
| `--category-chunk-size` | GPU-memory lever (train and val); see the "Training config gotchas" in `CLAUDE.md` before raising it |
| `--num-gpus`, `--accelerator auto\|cuda\|cpu` | hardware; `cpu` is for debugging only (minutes per step) |
| `--base-config` | use another YAML than `full_train_config.yaml` as the template |
| `--skip-eval` | train only |
| `--eval-only` | evaluate the existing `--experiment-dir` without training |
| `--threshold`, `--max-eval-images` | evaluation confidence threshold (default 0.5) and an image cap for quick checks |

**Good to know**

* The trainer validates every `val_epoch_freq` epochs; the CLI lowers it to at most `--epochs`, so short runs still
  validate once. The first epoch's validation is skipped by the base config.
* An evaluation score of `0.0` right after a very short or tiny-data run usually means the model's confidence is below
  `--threshold`, not that evaluation is broken. Re-run with `--eval-only --threshold 0.1` to see lower-confidence masks.
* Re-running with the same `--experiment-dir` resumes from its `checkpoint.pt` (the trainer's behaviour); use a fresh
  directory for a clean run.

**Verified so far:** unit tests (`pytest tests/finetune`) and one end-to-end smoke run (2-image synthetic dataset,
real `sam3.pt`, 1 epoch, single GPU) covering train, best-checkpoint selection and evaluation. **Not yet verified:** runs
on the HPC through `bsub`, multi-GPU (`--num-gpus > 1`), and a comparison with the notebook-based Step 7 numbers on the
same data (tracked in issue #11). To run it on the HPC, wrap the same command in `bsub` as in the examples above.

---

## Step 6: Inference

Use `03_inference.decoder_inference.ipynb` to run inference with your fine-tuned model.

1. Set checkpoint paths and image:

   ```python
   bpe_path = '/path/to/sam3_fine_tune/asset/bpe_simple_vocab_16e6.txt.gz'
   sam3_checkpoint = Path('/path/to/sam3.pt')
   finetuned_checkpoint = Path('/path/to/your_finetuned_checkpoint/model.pt')
   image_path = '/path/to/image.png'
   model = load_model(bpe_path, sam3_checkpoint, finetuned_checkpoint)
   run_inference(model, image_path, prompt="your category prompt")
   ```
2. Run the notebook to:

   - Observe the masks produced by model and its predicting probablity on a single image

---

## Step 7: Evaluation

### Single-Class Evaluation

Use `04_single_eval_pipeline.ipynb` or `single_eval_pipeline.py`:

1. **Configure paths in the notebook:**

   ```python
   TEST_IMAGES_PATH = Path('/path/to/test/images')
   GT_MASKS_PATH = Path('/path/to/test/masks')
   FINETUNED_CHECKPOINT = Path('/path/to/checkpoint.pt')
   TEXT_PROMPT = "pin"
   ```
2. **Run evaluation to compute:**

   - Foreground IoU
   - Precision
   - Recall
   - F1 Score
   - Per-image metrics

### Multi-Class Evaluation

Use `05_multi_eval_pipeline.ipynb` or `multi_eval_pipeline.py`:

1. **Configure category paths:**

   ```python
   PIN_CATEGORIES = {
       'pin': {
           'prompt': 'pin',
           'image_folder': BASE_PATH / 'images' / 'pin',
           'mask_folder': BASE_PATH / 'masks' / 'pin',
       },
       'eberhard pin': {
           'prompt': 'eberhard pin',
           'image_folder': BASE_PATH / 'images' / 'eberhard_pin',
           'mask_folder': BASE_PATH / 'masks' / 'eberhard_pin',
       },
   }
   ```
2. **Run evaluation to compute per-category metrics**

### Evaluation Metrics

| Metric                   | Description                                       |
| ------------------------ | ------------------------------------------------- |
| **Foreground IoU** | TP / (TP + FP + FN)                               |
| **Precision**      | TP / (TP + FP)                                    |
| **Recall**         | TP / (TP + FN)                                    |
| **F1 Score**       | 2 × (Precision × Recall) / (Precision + Recall) |

---

## SAMBA-UNet baseline (comparison with SAM3)

`sam3_fine_tune/sambaunet/` is a semantic-segmentation baseline (SAM ViT-B + Mamba U-Net, no text prompts) trained on
the same COCO dataset, with Hydra CLIs and an evaluation that writes the same `multiclass_foreground_iou_summary.json`:

```bash
uv sync --extra sambaunet
python -m sam3_fine_tune.sambaunet.train data.root=/path/to/dataset trainer.epochs=100
python -m sam3_fine_tune.sambaunet.evaluate checkpoint=experiments/sambaunet/<run>/checkpoints/best.pt data.root=/path/to/dataset
```

A runnable tutorial (Python API and CLI, synthetic data, CPU) is `nbs/14_sambaunet_api_and_cli.ipynb`; details are in
`sam3_fine_tune/sambaunet/README.md`. Compare the two models on the same `test/` split via
`summary.categories.<name>.overall_foreground_iou`, `pixel_precision` and `pixel_recall`.

---

## Step 8: Visualization & Analysis

Beyond the core training/evaluation pipeline, three utility notebooks are provided for deeper evaluation and analysis:

### `06_visualization_pipeline.ipynb`

Visualizes model predictions overlaid on the ground-truth masks. Useful for a qualitative review of results across the test set — helps identify systematic failure modes such as missed detections or false positives in noisy regions.

### `07_img_to_mask.ipynb`

Runs a fine-tuned checkpoint over a directory of images and saves the predicted binary masks and overlays. Use this to generate a mask dataset from model predictions — for example, to bootstrap annotation for additional categories.

### `08_correct_incorrect_filter.ipynb`

Filters test images into two groups — correctly segmented and incorrectly segmented — based on an IoU threshold. This is useful for:

- Curating hard negatives for re-training
- Understanding which image conditions (layout, noise, lighting) lead to failures
- Reporting per-condition performance breakdowns

---

## Future direction of the project

1. **Train on other categories:** Train on a few of the most common categories like bondwire, void
2. **Style Transfer:** Try to implement Style Transfer so that the existing work can be applied to other layouts

## Training Commands Quick Reference

```bash
# Local / any machine (no scheduler):
sam3-dataset images=./imgs masks=./masks category="bond wire" into=/path/to/dataset test_fraction=0.2
sam3-finetune --data-root /path/to/dataset --sam3-checkpoint sam3.pt --bpe-path bpe_simple_vocab_16e6.txt.gz --epochs 10

# HPC (LSF) examples:
mgpu_() { bsub -Is -q gpu -gpu "num=3:j_exclusive=yes:gmem=20G" -R 'select[type==X64LIN] rusage[mem=200]' -n 4 "$1"; }
mgpu_ "python train.py -c configs/config_gpu_q.yaml --num-gpus 3 --num-nodes 1"

gpu_q_() { bsub -Is -q gpu_compute -gpu "num=1:j_exclusive=yes" -n 8 "$1";}
gpu_q_ "python train.py -c configs/config_gpu_q.yaml --num-gpus 1 --num-nodes 1"
```

---

## Troubleshooting: Training Crashes &amp; Anomaly Dumps

> Added 2026-07-17, alongside an epoch-9 zero-loss collapse investigation —
> see `.github/skills/loss-debugging/SKILL.md` for the full investigation and
> guard status of every loss term.

If training crashes with `FloatingPointError: Loss is nan`, or you notice the
train loss is stuck at exactly `0.00000` for many consecutive steps, check
these two places **first**, both for users and for an agent picking this up:

1. **`<experiment_log_dir>/dumps/anomalous_batches/`** — automatically
   populated by `Trainer._maybe_dump_anomalous_batch()`
   (`sam3/train/trainer.py`) the moment the real training loss goes
   non-finite or exactly zero (rate-limited to 3 dumps per run). Each
   `.pt` file (named `<phase>_<key>_epoch<E>_step<S>_<n>.pt`) contains
   **both** the post-augmentation batch the model actually saw
   (`img_batch`, `find_targets`) **and**, if available, the
   pre-augmentation raw source (`raw_images`, `raw_boxes`) — so you can
   tell whether the augmentation pipeline produced a degenerate
   box/image, or it was already bad in the source annotation. Inspect a
   dump with:
   ```bash
   python .github/scripts/inspect_anomalous_batch.py \
       <experiment_log_dir>/dumps/anomalous_batches/train_pin_epoch9_step0_0.pt
   ```
2. **`<experiment_log_dir>/checkpoints/`** — besides the always-overwritten
   `checkpoint.pt` / periodic `checkpoint_<epoch>.pt`, a
   `checkpoint_val_pin_detection_coco_eval_segm_AP.pt` is kept permanently
   the first time validation AP improves (`trainer.checkpoint.save_best_meters`
   / `save_best_meter_subkeys` in `full_train_config.yaml`) — use this as
   the known-good fallback if a later epoch regresses/collapses and
   overwrites `checkpoint.pt` with a worse state.

Also worth knowing: this HPC cluster auto-kills a job if GPU utilization
stays below ~5% for more than ~5 minutes (`TERM_OWNER: job killed by owner.`
/ exit code `143` in the `.out` summary — easy to mistake for a manual
`bkill`). See `.github/skills/hpc-gpu-submit/SKILL.md` for known triggers
(slow setup phase, high `Data Time` steps, a brand-new `experiment_log_dir`).

---
