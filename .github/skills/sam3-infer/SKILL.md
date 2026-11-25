---
name: sam3-infer
description: 'Run base or fine-tuned SAM3 on images with text prompts using `sam3-infer`, and score or compare models on a COCO split. Use when you need masks or overlays for a folder, a threshold sweep, or a base-versus-checkpoint comparison on the same images. Local, no HPC.'
argument-hint: '[images-or-coco-split]'
user-invocable: true
disable-model-invocation: false
---
# Run SAM3 (`sam3-infer`)

Segments images for one or several text prompts and writes masks, overlays and `summary.json`. Given a COCO split it
also scores the predictions; given a list of checkpoints it runs every model on the same images, so one command gives a
comparison.

## Where things live

- Options, defaults and examples: `sam3_fine_tune/inference/conf/infer.yaml` (Hydra; override any key on the command
  line).
- Code: `sam3_fine_tune/inference/` (`run.py` runs the models, `score.py` scores them).
- Weights: `sam3_checkpoint` and `bpe_path` are required even for a fine-tuned run; local copies are listed in
  CLAUDE.md "Local weights and data".

## Rules

1. The prompt is the category name used in training, exactly as typed; a mismatch reads as a near-zero score.
2. Score on the prepared `test` split (`ROOT/prepared/sam3/test`, `coco_ann=annotation_merged.json`) and only there for
   the final number; thresholds are chosen on `val`.
3. Check that the images of a split exist on disk before trusting a score (CLAUDE.md "Local weights and data").
4. Real-weights check: `tests/inference/test_real_weights.py`.
