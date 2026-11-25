---
name: sam3-finetune
description: 'Fine-tune SAM3 and evaluate the best checkpoint with `sam3-finetune`, locally or as the entry point of an HPC run. Use when training on a dataset root or on the SAM3 view from `sam3-prepare`, re-running only the test evaluation, or choosing where a run belongs (GB10 versus HPC).'
argument-hint: '[data-root]'
user-invocable: true
disable-model-invocation: false
---
# Train SAM3 (`sam3-finetune`)

Prepare the data first (`.github/skills/prepare-data/SKILL.md`), then point `--data-root` at `ROOT/prepared/sam3`. It
trains, then evaluates the best checkpoint on `test`; with a `val` split, in-training validation reads `val`, so `test`
is touched only by the final evaluation.

## Where things live

- Arguments: `python -m sam3_fine_tune.finetune --help` (argparse in `sam3_fine_tune/finetune/cli.py`).
- Training config it builds from: `sam3/train/configs/full_train_config.yaml` (full fine-tuning; never the separate
  `decoder_train_config.yaml`). Edit it, or `category_chunk_size`, only after reading `docs/training-guide.md`.
- Output: `experiments/<run>/` (checkpoints, `config_used.yaml`, `eval/`); the best checkpoint is the one the trainer
  keeps for the highest validation segmentation AP.

## Pick the place

- Short or trial runs: locally on the GB10. Real weights and tokenizer are in CLAUDE.md "Local weights and data".
- Long runs: the HPC. Start at `.github/skills/configure-repo/SKILL.md`, submit with
  `.github/skills/hpc-gpu-submit/SKILL.md`, watch with `.github/skills/training-watchdog/SKILL.md`.
- A NaN or zero loss: `.github/skills/loss-debugging/SKILL.md`.

## After training

Compare the checkpoint with the base model on `test` through `sam3-infer` (`.github/skills/sam3-infer/SKILL.md`). The
category name is the prompt, so use it exactly as trained.
