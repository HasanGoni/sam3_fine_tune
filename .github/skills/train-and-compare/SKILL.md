---
name: train-and-compare
description: 'End to end: take a dataset to a SAM3 versus UNVEIL comparison. Order of the commands (sam3-prepare, sam3-finetune, sam3-train-unveil, sam3-infer, sam3-infer-unveil, sam3-compare), the arguments they share, where val and test are used, and the traps (gate, not_annotated, checkpoint trust, licence, GB10 versus HPC). Start here when asked to train and compare both models.'
argument-hint: '[root]'
user-invocable: true
disable-model-invocation: false
---
# Train and compare SAM3 and UNVEIL

Each step has its own skill and config; this file gives the order and what carries from one step to the next. Do not
copy arguments from here: open the config named in each skill. All commands take the same `ROOT`.

## Order

| # | Step | CLI | Skill | Reads | Writes |
|---|---|---|---|---|---|
| 1 | Prepare both views | `sam3-prepare` | `.github/skills/prepare-data/SKILL.md` | `ROOT` COCO | `ROOT/prepared/{manifest.json,sam3,unveil}` |
| 2 | Fill the recipe | edit `ROOT/prepared/unveil/recipe.yaml`, run step 1 again | prepare-data | | `kind` per category, `families:`, `not_annotated` flags |
| 3a | Train SAM3 | `sam3-finetune --data-root ROOT/prepared/sam3` | `.github/skills/sam3-finetune/SKILL.md` | `train`, `val` | `experiments/<run>/` |
| 3b | Train UNVEIL | `sam3-train-unveil root=ROOT` | `.github/skills/train-unveil/SKILL.md` | `train`, `val` | `out/unveil.safetensors` |
| 4 | Look at predictions (optional) | `sam3-infer`, `sam3-infer-unveil` | `.github/skills/sam3-infer/SKILL.md`, `.github/skills/infer-unveil/SKILL.md` | images | masks, overlays |
| 5 | Score and compare | `sam3-compare root=ROOT` | `.github/skills/compare/SKILL.md` | `test` | table, `compare.json` |

Shared across steps: the same `ROOT`; the same `sam3.pt` and `bpe_simple_vocab_16e6.txt.gz` (CLAUDE.md "Local weights
and data"); the category name as the prompt, exactly as typed; the same frozen split from the manifest.

## val versus test

- `train`: both models fit on it.
- `val`: SAM3 validation during training, and UNVEIL's per-category thresholds. Any choice (epochs, checkpoint,
  threshold) is made here.
- `test`: read only by `sam3-compare` (and `sam3-finetune`'s final evaluation). Never select anything on it.

## Traps

1. **not_annotated.** An image without a label for a category does not mean absent. Fill `families:` in `recipe.yaml`
   and re-run `sam3-prepare` before training and comparing; `sam3-compare` skips flagged images per category.
2. **Gate gated/ungated.** UNVEIL scores differ a lot with the presence gate. Compare with a gated and an ungated
   column of the same checkpoint, and state the setting next to any number.
3. **Checkpoint trust.** SAM3 `checkpoint.pt` needs `trust_checkpoint=true` in `sam3-compare` (unpickling; your own
   checkpoints only). UNVEIL's `.safetensors` does not.
4. **Not trained is not weak.** Give each entry `trained_categories` so absent categories show `not trained`, and a
   `suspect` near-zero on a trained category means check the prompt spelling first.
5. **Licence.** The SAM License applies to both, and several UNVEIL training sets are non-commercial research only. The
   UNVEIL branch is research use until the owner decides otherwise (open in `docs/unveil-recipe.md`).
6. **Images on disk.** Check that split images exist before trusting a score (CLAUDE.md "Local weights and data").

## Where to run

- GB10 (local): prepare, validation, trials, short runs, inference, compare on a subset. Plain `python` is not on PATH;
  use `.venv/bin/python`.
- HPC: long training of either model (`.github/skills/configure-repo/SKILL.md`, `hpc-gpu-submit`,
  `training-watchdog`). Check a change has reached the HPC side of the mount first.
- UNVEIL needs CUDA and the `unveil` package wherever it runs.

## What has never run

The real UNVEIL adapter (`model=unveil`, `sam3-infer-unveil`, UNVEIL columns in `sam3-compare`), the joint module on the
real model and every GB10 timing: the tests use stubs. Expect the first real run to surface bugs; report them, do not
paper over them. The `thin_wire_panda` 0.000 from #18 is unexplained.
