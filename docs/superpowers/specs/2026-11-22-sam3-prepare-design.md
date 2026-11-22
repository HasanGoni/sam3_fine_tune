# sam3-prepare: one data preparation step for SAM3 and UNVEIL

Status: draft for approval, 2026-11-22. Related: #18 (UNVEIL evaluation), #19 (`sam3-train-unveil`),
`docs/unveil-recipe.md` (recipe rules, rule 4 on unannotated images and negatives).

## Intent and why

The owner wants to fine-tune both SAM3 (`sam3-finetune`) and UNVEIL's recipe (#19) and compare which works better.
The data will change: more images of the same categories, extra categories, or a completely different image set with
its own categories. A comparison is only trustworthy if both models train on the same images, are tested on the same
images, and no image is on both sides of the split. Today nothing guarantees this: the two models need different
formats (SAM3 reads COCO; UNVEIL reads a JSONL index with one image path and one mask PNG per image and category), and
the one ingest tool (`sam3-dataset`) breaks on images that carry several categories (see Findings).

## Findings (verified 2026-11-22)

`sam3-dataset` adds ONE category per call and names each copied image `{category}_{name}`. A test with 10 images, all
with a `substrate` mask and 8 with a `chip` mask, added in two calls, gave: every physical image became two unrelated
COCO images (`substrate_im0.png` with only `substrate`, `chip_im0.png` with only `chip`); `im7` landed in test for
`substrate` and in train for `chip` (leakage); and SAM3 sees `chip` as absent in the substrate copy (false negatives).
Forcing the same prefix does not help: the image is skipped as already present, so a second category cannot be added.
COCO itself, and UNVEIL's per-(image, category) mask PNGs, handle overlapping and nested categories (substrate >
chip > lifted chip) without trouble, as long as masks are never derived from a single combined label map.

## Goals

- One command, `sam3-prepare`, that works on any dataset root and produces both models' inputs from one source of truth.
- Multi-category images: one physical image, all its categories, overlaps and nesting preserved.
- Same train/val/test images for both models; split by physical image; frozen once made, so extending the data never moves
  an image across the split.
- Any input: image + mask folders (one mask folder per category) or existing COCO.
- Re-runnable and idempotent: running it again after adding data only adds.
- A new dataset (different images and categories) is a new root with the same command and its own recipe file.

## Non-goals

- Training UNVEIL (#19), evaluation, annotating or fixing labels, instance-level labels.
- Changing the behaviour of `sam3-dataset` (only a warning is added, see below).
- Deleting or rewriting the user's source data.
- Repairing the existing 12-category dataset on disk (it is a possible first test, not a deliverable).

## Anti-goals (failure even if it technically works)

- Any image in both train and test, for either model.
- Moving an existing test image into train, or reshuffling the split when data is extended.
- Overlapping categories fighting over pixels (no combined label map, no exclusivity between categories).
- Unannotated images silently used as negatives (unannotated images are excluded, `docs/unveil-recipe.md` rule 4).
- The two views drifting apart: both are derived from the same manifest, never edited by hand.

## Approaches considered

- A. Extend `sam3-dataset` with a prepare stage. One tool, but it works one category at a time and cannot prepare COCO
  that came from elsewhere.
- **B. New `sam3-prepare` CLI next to `sam3-dataset` (chosen by the owner).** `sam3-dataset` stays as is. The split
  manifest lives in the new tool.
- C. A skill only, chaining existing tools. Fastest, but nothing enforces the shared split, which is the point.

## Decisions (the owner's, unless marked "my call")

- B, plus a skill and a warning in `sam3-dataset`. All multi-category handling lives in `sam3-prepare`.
- COCO is the single source of truth; the SAM3 and UNVEIL views are derived and can be regenerated.
- Images without any annotation are excluded (rule 4); the number dropped is printed.
- Hydra CLI following `sam3_fine_tune/dataset/` (`cli.py`, `__main__.py`, `conf/prepare.yaml` + empty
  `conf/__init__.py`, `[project.scripts]`, package-data, `tests/prepare/`). My call: one entry point with
  `mode: ingest | prepare`.

### `mode=ingest` (my call on the interface)

`sam3-prepare mode=ingest images=DIR masks.chip=DIR masks.substrate=DIR into=ROOT`. Masks are matched to images by file
stem. An image with no mask file for a category is *not annotated* for it (not "absent"). Writes canonical COCO
`ROOT/<split>/annotation.json`, one image entry per physical image carrying all its categories, split by physical image.
Extending an existing root adds images and categories only.

### `mode=prepare` (my call on the layout)

Input: a root with `{train,test}/images` and `annotation.json` (from `ingest`, `sam3-dataset` or elsewhere). Steps:
1. Validate (`validate_dataset` plus the checks below).
2. Identify physical images by content hash (sha256 of file bytes; re-encoded copies are not detected).
3. Merge duplicate entries of one physical image into one image carrying all categories. The same category annotated
   twice for one image with different polygons is an error, not a silent union.
4. Split by physical image into `ROOT/prepared/manifest.json` (hash to train, val or test). Adopts an existing train/test split when
   there is one; stops with a report if one physical image is on both sides. New images are assigned by seeded
   `test_fraction`, then `val_fraction` of the remaining images becomes `val`; existing assignments never change.
5. Drop images without annotations; print the count.
6. Write derived views under `ROOT/prepared/`:
   - `sam3/{train,val,test}/images` (links or copies) and `annotation_merged.json` with `not_annotated_category_ids`,
     so `sam3-finetune --data-root ROOT/prepared/sam3` works once its validation points at `val`.
   - `unveil/masks/<category>/<image id>.png` (one PNG per image and category, rasterized independently),
     `unveil/index/<split>/<category>.jsonl` (image path, mask path, width, height, id, the image's annotated
     categories) for train and val, and `unveil/prompts.json`.
   - `unveil/recipe.yaml` skeleton if absent: categories listed, `kind: ???` for each, `families:` empty,
     `exclude_unannotated: true`. Never overwritten. The training CLI (#19) refuses to run while a `kind` is `???`.
7. Write a report: counts per split and per category, merged duplicates, dropped images, problems.

### `sam3-dataset` warning (my call on the wording)

When an added image has the same content hash as an image already in `into` under another name, print a warning that
names the count and points to `sam3-prepare`. Behaviour is unchanged, and the warning must not make adding slow on
large datasets.

### Skill

`.github/skills/prepare-data/SKILL.md`: when to run which mode, the dataset root layout, the frozen split rule, how to
start a new dataset. It points to the CLI and does not duplicate it.

## Constraints

- Repo rules: Python 3.12, `.venv/bin/python`, ufmt line length 88 on the hand-written packages, mypy flags in
  `pyproject.toml`, `tests/test_repo_standards.py` (Hydra layout), tests on CPU only.
- SAM3 and UNVEIL both stretch input to 1008; `prepare` stores full-resolution masks and does not resize.
- Datasets can be large (thousands of images, some not present locally): hashing and rasterizing must stream, and
  missing image files are reported, not fatal for the whole run.

## Acceptance

- The repro above (two categories added to one image set) is a test: after `ingest`, one image entry per picture with
  both categories, and no picture in both splits.
- Overlap test: nested masks (substrate > chip > lifted chip) keep their full extent in both views.
- Idempotence and extension test: re-running changes nothing; adding images and a category keeps every earlier split
  assignment.
- Duplicate test: a dataset built the old way (two entries per picture) is merged and split by picture.
- The two views list the same train, val and test images (a test compares them).
- Loader patch test: a category listed in `not_annotated_category_ids` produces no query for that image, and one
  short training smoke run on a tiny prepared dataset completes.
- A dry run on the local 12-category dataset reports duplicates, unannotated images (3,017 train / 1,448 test) and the
  split it adopted; nothing in the source dataset is modified.

## Calls made after the builder check (my calls, to be checked by the owner)

- **Split conflicts** (one physical image in both splits, e.g. a dataset built the old way): `on_conflict: error | test`.
  Default `error` with a report listing the images. `test` keeps the image in test and removes it from train, because
  test must never change. The dry run on the 12-category dataset uses `test` and reports how many images moved.
- **Frozen split and ids.** Manifest keys are content hashes, so a re-added duplicate under a new file name inherits
  its old assignment. The manifest also stores each physical image's COCO image id and each category's id, assigned
  once (by first appearance) and never regenerated, so both views stay identical across re-runs.
- **Category names** are used exactly as given (they are the prompt strings): no case or whitespace normalisation.
  Names that differ only in case or whitespace are reported as a probable mistake. The manifest owns category ids and
  unifies train and test category lists.
- **Polygons.** Reuse `mask_to_coco` from `sam3_fine_tune/preprocessing/coco_format.py` (same as `sam3-dataset`). The
  UNVEIL masks are rasterized from the COCO polygons, so both views carry the same labels. The overlap test accepts
  a pixel difference from polygon rounding (mask IoU >= 0.99), not exact equality.
- **Ingest matching.** Image and mask stems must match exactly, any image extension. Unmatched files are reported. A
  size mismatch between an image and its mask is an error for that image, as in `sam3-dataset`.
- **UNVEIL index format** is defined by UNVEIL's loader (`unveil/data/transforms.py`: items with `image`, `gt`, `w`,
  `h`, `id`; masks 0/255, read as `>= 128`; absolute paths), plus our `categories` list. #19 consumes it, so the
  builder reads the loader and does not invent keys.
- **Where files live.** `ingest` writes canonical COCO and images into `ROOT/{train,test}`. `prepare` only reads
  those and writes only under `ROOT/prepared/`.

## Decisions after the builder check (the owner's)

1. **Fair negatives for SAM3: patch the loader.** SAM3's COCO loader (`sam3/train/data/coco_json_loaders.py`) marks
   every query `is_exhaustive: True`, so a category with no annotation in an image is a hard negative. We patch our
   vendored copy so that an image entry may carry `not_annotated_category_ids`; the loader then skips those categories
   for that image instead of querying them as negatives. `prepare` writes the field into the SAM3 view. My call on how
   it is filled: for each image, every category that shares a family (`families:` in `recipe.yaml`) with a category
   the image is annotated for, but is not annotated itself, is listed. This is the same rule UNVEIL uses (same-family
   images are never negatives), so both models see the same negatives. Without `families:` nothing is flagged and SAM3
   behaves as today. The patch needs a check that training still runs (one short GPU or CPU smoke run) and an updated
   reference snapshot under `sam3_fine_tune/modified_script/` if the touched file is mirrored there.
2. **Validation split: yes.** The manifest has three splits, `train`, `val`, `test`, all assigned by physical image.
   `val_fraction` defaults to 0.1 of the non-test images (seeded, frozen). Neither model trains on `val`. SAM3 uses it
   for in-training validation (today its configs validate on `test`, which leaks the test set into model selection),
   UNVEIL fits its presence and binarization thresholds on it, and `test` is touched only for the final comparison.
   Because of this, `prepare` no longer writes a calibration slice of its own: the `val` split replaces it.
   `sam3-finetune`'s generated config must point validation at `val`, which the builder adapts.

## Left to the builder

Hydra config key names, hash implementation (streaming), link versus copy of images, the manifest's exact JSON schema,
how empty masks are treated (my default: no annotation, same as `sam3-dataset` today), progress output, and the test
fixtures.
