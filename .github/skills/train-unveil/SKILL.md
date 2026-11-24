---
name: train-unveil
description: 'Train the UNVEIL recipe on the UNVEIL view from `sam3-prepare` with `sam3-train-unveil`: validate the recipe, run the short trial that reports step time and peak memory, then train and write a checkpoint with val-fitted thresholds. Use when you edit `recipe.yaml`, choose `epochs`, or need an UNVEIL checkpoint for `sam3-infer-unveil` or `sam3-compare`.'
argument-hint: '[root]'
user-invocable: true
disable-model-invocation: false
---
# Train UNVEIL (`sam3-train-unveil`)

Prepare the data first (`.github/skills/prepare-data/SKILL.md`); this CLI reads `ROOT/prepared/unveil/` and never converts
data. The rules it enforces (loss kind per category, families, no unannotated images, thresholds on `val` only) are in
`docs/unveil-recipe.md`.

## Where things live

- Options and defaults: `sam3_fine_tune/train_unveil/conf/train_unveil.yaml` (Hydra; `python -m sam3_fine_tune.train_unveil`
  or `sam3-train-unveil`). `--help` only prints the Hydra group list; read the YAML.
- Code: `sam3_fine_tune/train_unveil/` (`ops.py` recipe checks, `trainer.py`, `calibrate.py`, `checkpoint.py`).
- Output of `train=true`: `out/unveil.safetensors` (+ `.sha256`), `calibration.json`, `log.jsonl`.

## Order

1. `validate_only=true` (the default) checks the recipe and the view, then stops. Fix what it names: a category without
   `kind`, an unknown category, a negative from the same family, a bad `joint`.
2. `validate_only=false` runs the trial and prints step time and peak memory. Choose `epochs` from that number.
3. `train=true out=DIR validate_only=false` trains. It needs val images (`sam3-prepare val_fraction=...`) and stops
   without them.

## Rules

1. Thresholds (presence and binarization, per category) are fitted on `val`, never `test`; they live in the checkpoint.
2. `model=stub` is a CPU smoke test, not a result. `model=unveil` needs `sam3_checkpoint` and the `unveil` package.
3. Place: the trial and short runs on the GB10; long runs on the HPC (`.github/skills/configure-repo/SKILL.md`, then
   `.github/skills/hpc-gpu-submit/SKILL.md`). One GPU with `accumulate: 8` costs about 8x the time per step.
4. Licence: UNVEIL builds on the SAM License, and several of its training sets are non-commercial research only. Do not
   use an UNVEIL checkpoint in a product or for commercial Acme work until the owner has decided; the decision is
   outside these skills.

## Not tested

The real UNVEIL adapter (`model=unveil`), the joint module on the real model and real GB10 step time and memory have
never run; the tests use `model=stub`. Treat the first real trial as a check of the CLI, not only of the data.
