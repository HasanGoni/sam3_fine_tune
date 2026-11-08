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
14. [Step 8: Visualization &amp; Analysis](#step-8-visualization--analysis)
15. [Future Direction of the Project](#future-direction-of-the-project)
16. [Training Commands Quick Reference](#training-commands-quick-reference)
17. [Troubleshooting: Training Crashes &amp; Anomaly Dumps](#troubleshooting-training-crashes--anomaly-dumps)

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
```

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
└── sam3/
    └── train/
        └── configs/
            ├── decoder_train_config.yaml      # Decoder-only fine-tuning config
            └── full_train_config.yaml         # Full fine-tuning config
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
│                    ↓                                            │
│  3. [Optional] MERGE DATASETS (02_merge_dataset.ipynb)          │
│     └── Combine multiple classes into one dataset               │
│                    ↓                                            │
│  4. CONFIGURE TRAINING                                          │
│     └── decoder_train_config.yaml  OR  full_train_config.yaml   │
│                    ↓                                            │
│  5. RUN TRAINING (sam3/train/train.py)                          │
│     └── Train decoder-only or full fine-tuning                  │
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

After Steps 2-3 (a COCO dataset `<root>/{train,test}/{images,annotation.json}`), a single command patches a copy of
`full_train_config.yaml`, trains, then scores the best checkpoint on `test/` (prompt = `category_name`, ground truth =
the COCO masks):

```bash
python -m sam3_fine_tune.finetune \
    --data-root /path/to/dataset --sam3-checkpoint /path/to/sam3.pt --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz \
    --experiment-dir experiments/my_run --epochs 10 --train-batch-size 2 --grad-accum 2 --category-chunk-size 9
```

* Results: `<experiment-dir>/eval/multiclass_foreground_iou_summary.json` (same layout as Step 7, so it can be compared
  with SAMBA-UNet); the exact config is saved as `<experiment-dir>/config_used.yaml`.
* `--skip-eval` trains only; `--eval-only` scores an existing `--experiment-dir` (e.g. with `--threshold 0.3`).
* `--accelerator auto|cuda|cpu`, `--num-gpus N`. Unset options keep the values of `full_train_config.yaml`;
  see the CLAUDE.md notes on `category_chunk_size` and `gradient_accumulation_steps` before changing them.
* On the HPC wrap it in `bsub` as in the examples above.

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
