---
name: prepare-data
description: 'Prepare a dataset for both SAM3 (`sam3-finetune`) and the UNVEIL recipe from one COCO root with `sam3-prepare`: same train/val/test pictures for both models, split frozen across re-runs. Use when starting a new dataset, adding images or categories to one, or checking that two models train and test on identical images. Local, no HPC.'
argument-hint: '[root]'
user-invocable: true
disable-model-invocation: false
---
# Prepare data (`sam3-prepare`)

One COCO root is the source of truth. `sam3-prepare` derives a SAM3 view and an UNVEIL view from it, with one
train/val/test assignment per picture, so both models are compared on identical images. Options and defaults live in
`sam3_fine_tune/prepare/conf/prepare.yaml`; the reasons are in
`docs/superpowers/specs/2026-11-22-sam3-prepare-design.md`.

## Pick the mode

| You have | Run |
|---|---|
| images plus one folder of binary masks per category | `mode=ingest`, then `mode=prepare` |
| a root that already has `{train,test}/{images,annotation.json}` (from `sam3-dataset`, `ingest` or elsewhere) | `mode=prepare` |

```bash
.venv/bin/python -m sam3_fine_tune.prepare mode=ingest images=DIR +masks.chip=DIR +masks.substrate=DIR into=ROOT
.venv/bin/python -m sam3_fine_tune.prepare root=ROOT dry_run=true      # read the report first
.venv/bin/python -m sam3_fine_tune.prepare root=ROOT
```

Masks match images by file stem; an image without a mask for a category is *not annotated* for it. The category name is
the prompt string, used exactly as typed.

`sam3-prepare` is the console-script name; it is installed by `uv pip install -e . --no-deps`, and the `python -m` form above works without it.

## Root layout

`ingest` writes and `prepare` reads `ROOT/{train,test}/{images,annotation.json}`. `prepare` writes only under
`ROOT/prepared/`:

- `manifest.json`: the frozen assignment, keyed by the sha256 of each picture's file bytes.
- `sam3/{train,val,test}/`: symlinked images and `annotation_merged.json`; pass `--data-root ROOT/prepared/sam3` to
  `sam3-finetune` (it validates on `val`).
- `unveil/`: per-category masks, index JSONL per split, `prompts.json`, `recipe.yaml`.

## Rules that hold on every run

1. **A picture is its file bytes.** Two COCO entries with identical bytes are one picture carrying all its categories.
2. **The split is frozen.** Re-running or extending never moves a picture or changes an id. Val is carved only from new
   train pictures. Do not edit `manifest.json`; delete `prepared/` to start over (and accept a new split).
3. **A picture in both train and test stops the run** with a list. `on_conflict=test` keeps it in test.
4. **Pictures with no annotation are dropped**; the report counts them.
5. **Read the report before using the output**: `problems` must be empty; check `missing_images`,
   `similar_category_names` and `recipe_missing_categories`, and the warnings on stderr.

## Start a new dataset

1. Pick a new root folder (one root per image set; never mix image sets in one root).
2. `mode=ingest` for each batch of images and masks, or copy an existing COCO root in.
3. `dry_run=true`, fix what the report names, then run for real.
4. Open `ROOT/prepared/unveil/recipe.yaml` and set `kind: region | balanced` for every category (it ships as `???`).
   Fill `families:` per `docs/unveil-recipe.md`. Run `sam3-prepare root=ROOT` again: families also set the SAM3
   `not_annotated_category_ids`, and the report shows `images_with_not_annotated_flags`.
5. Train: `sam3-finetune` on `ROOT/prepared/sam3`. UNVEIL training: `.github/skills/train-unveil/SKILL.md`; the whole chain: `.github/skills/train-and-compare/SKILL.md`.

## Extend a dataset

Run `mode=ingest` with `into=ROOT` for the new images or categories, then `mode=prepare`. Earlier assignments stay;
`test_fraction` applies to the new pictures only.
