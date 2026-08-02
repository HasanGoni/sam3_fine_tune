---
name: post-training-error-analysis
description: 'After a SAM3 fine-tuning run completes, run thin-wire-panda error analysis and generate overlay posters for both test and training sets, saving all artifacts under the dataset directory.'
argument-hint: '[--experiment-dir <path>] [--dataset-dir <path>] [--category-name <name>]'
user-invocable: true
disable-model-invocation: false
---
# Post-Training Error Analysis

Run the standard post-training artifact generation for a completed SAM3
experiment: test-set bad-case analysis, overlay posters, and (optionally)
training-set error analysis once val-on-train predictions exist.

## When To Use

- A training job has finished successfully and you need the standard
  thin-wire-panda error-analysis package.
- You want to compare model predictions against ground truth visually.
- You want label-review overlays for the training set.

## Scope

This skill currently targets the v2 9-category diverse-wires experiment
(`9cat_diverse_wires_v2`). It is easy to generalize to other categories by
editing the helper scripts; for now it hard-codes `thin wire panda`
(category ID 9) because that is the lowest-performing class.

## Artifacts Produced

All outputs are saved **under the dataset directory**, not under the
experiment directory:

| Artifact | Path |
|---|---|
| Test bad-case summary + poster | `<dataset>/test/error_analysis/` |
| Test thin-wire-panda overlay poster | `<dataset>/test/error_analysis/thin_wire_panda_overlay_poster.jpg` |
| Training label overlay poster | `<dataset>/train/error_analysis/label_review/thin_wire_panda_train_overlay_poster.jpg` |
| Training error analysis (when preds exist) | `<dataset>/train/error_analysis/` |

## Prerequisites

- Training has completed and produced test predictions at
  `<experiment>/dumps/pins/coco_predictions_segm.json`.
- For the optional training-set analysis, val-on-train predictions must
  exist at `<experiment>_val_on_train/dumps/train_pins/coco_predictions_segm.json`.
  If they don't, this skill tells you how to generate them.
- `configure-repo` has been run (venv + CUDA module are known).

## Procedure

Run the master script on the HPC. It is a CPU workload but may take several
minutes because it renders many images, so submit it as a batch job rather
than running interactively:

```bash
ssh hpc_muc '
REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
LOGDIR=$REPO/experiments/9cat_diverse_wires_v2_2gpu/bsub_logs
mkdir -p "$LOGDIR"
cat > "$LOGDIR/run_post_training_analysis.sh" << "EOF"
#!/usr/bin/env bash
set -e
cd /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
source /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.venv/bin/activate
bash scripts/run_post_training_analysis.sh
EOF
chmod +x "$LOGDIR/run_post_training_analysis.sh"
bsub -q batch -P ma_ai -n 4 -R "rusage[mem=16000]" \
  -o "$LOGDIR/post_training_analysis.%J.out" \
  -e "$LOGDIR/post_training_analysis.%J.err" \
  "$LOGDIR/run_post_training_analysis.sh"
'
```

### If you also need training-set predictions

The master script skips training-set error analysis when the predictions
are missing. To generate them, submit the val-on-train GPU job:

```bash
ssh hpc_muc '
REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
LOGDIR=$REPO/experiments/9cat_diverse_wires_v2_2gpu_val_on_train/bsub_logs
mkdir -p "$LOGDIR"
bsub -q gpu -gpu "num=1:j_exclusive=yes:gmem=42G" -R "osrel=80 && ui==aiml-python" -P ma_ai -n 8 \
  -o "$LOGDIR/val_on_train.%J.out" -e "$LOGDIR/val_on_train.%J.err" \
  "bash -c \"source /ifxhome/\\$USER/.bashrc && source $REPO/.venv/bin/activate && module load <CUDA_MODULE> && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO/sam3/train && python train.py -c configs/full_train_9cat_diverse_wires_v2_2gpu_val_on_train.yaml --num-gpus 1 --num-nodes 1\""
'
```

Use the `gpu_compute` queue with `gmem=80G` instead if you prefer to target
an H200-class host. Replace `<CUDA_MODULE>` with the value resolved by
`configure-repo`.

Once that job finishes, re-run the post-training analysis script above to
produce the training-set bad-case poster and summary.

## What the Colors Mean

In the test bad-case overlay panels:

- **Green** = ground-truth thin-wire-panda mask
- **Red** = model prediction
- **Yellow/orange** = overlap where prediction and ground truth agree

The header text on each panel shows counts, e.g. `GT3 P2 TP1 FP1 FN2`
(3 ground-truth instances, 2 predictions, 1 true positive, 1 false
positive, 2 false negatives).

## Output Format

1. Job ID of the submitted batch job.
2. Paths to the generated posters and JSON summaries.
3. Note if training-set analysis was skipped pending val-on-train predictions.
