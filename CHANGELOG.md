## 2026-07-30

Added a reusable post-training error-analysis step and documented the
available Copilot skills in the README.

- **New skill**: `.github/skills/post-training-error-analysis/SKILL.md`
  describes how to automatically run the standard after-training artifact
  generation: test-set bad-case analysis, test-set overlay poster,
  training-set label overlay poster, and training-set error analysis once
  val-on-train predictions exist.
- **New script**: `scripts/run_post_training_analysis.sh` orchestrates the
  full post-training analysis workflow for the v2 9-category diverse-wires
  model, saving all outputs under the dataset directory.
- **Script updates**:
  - `scripts/run_test_error_analysis.sh` now also generates a test-set
    `thin_wire_panda_overlay_poster.jpg` in the dataset directory.
  - `scripts/run_train_error_analysis.sh` and
    `scripts/run_train_overlay_poster.sh` now save outputs under
    `<dataset>/train/error_analysis`.
  - `scripts/run_test_error_analysis.sh` saves outputs under
    `<dataset>/test/error_analysis`.
- **Skill integration**: `finetune-merged-dataset` skill now ends with a
  pointer to `post-training-error-analysis` after validation.
- **README skills section**: added a table documenting the relevant Copilot
  skills for configure-repo, COCO conversion, merging datasets, diverse-wire
  dataset construction, fine-tuning, post-training error analysis, HPC GPU/CPU
  submission, grid posters, and the training watchdog.

## 2026-07-29

Updated README.md to reflect the current 9-category model, dataset, and
packaged checkpoint location. Copied the latest fine-tuned checkpoint and
metadata to the shared model registry. Started a dedicated error-analysis
branch for thin wire panda, the lowest-performing category. Updated the
thin-wire segmentation inference notebook with a bug fix and a patch-grid
visualization helper.

- **Notebook updates** (`nbs/03_inference.thin_wire_segmentation.ipynb`):
  - Fixed `run_inference` type check from `isinstance(image_path, Path(str, image_path))` to `isinstance(image_path, (str, Path))` so string or `Path` inputs are opened correctly.
  - Added `plot_all_patches(patches, max_cols, figsize_per_patch)` to render all extracted patches in a numbered grid with `(x, y)` coordinates.
- **README corrections**:
  - Expanded the pre-trained checkpoint category table from 4 to 9
    categories (`et_pin`, `eberhard_pin`, `wuxi_pin`, `war_crack`,
    `matrix_chip`, `thick wire HD`, `thin wire HD`, `thick wire panda`,
    `thin wire panda`) with overlay images copied from the validation
    outputs to `readme_images/`.
  - Updated required paths to the current dataset
    (`/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1`)
    and the current fine-tuned checkpoint
    (`/home/hpc_storage/Models/sam3-fintuned/2026-07-29/...`).
  - Added a "Packaged Model Checkpoints" subsection documenting the dated
    release folder, validation metrics (COCO segm AP 0.891), and usage
    example.
  - Updated the merge-dataset example and output `categories` JSON to
    include all 9 categories.
- **Model packaging**: copied
  `experiments/9cat_diverse_wires_v1_gpucompute/checkpoints/checkpoint_10.pt`
  to
  `/home/hpc_storage/Models/sam3-fintuned/2026-07-29/sam3_finetuned_9cat_diverse_wires_v1.pt`
  and wrote `metadata.json` with dataset counts, hyperparameters,
  validation metrics, and usage instructions.
- **Thin wire panda error analysis** (branch
  `feat/thin-wire-panda-error-analysis`):
  - Added `scripts/thin_wire_panda_error_analysis.py` to score every test
    image for category 9, rank the worst cases by TP/FP/FN/mean-IoU, copy
    raw bad-case images to a folder, generate side-by-side overlays, and
    build a single grid poster.
  - First-pass findings: thin wire panda has only 257 GT annotations across
    200 test images (vs. 4,638 for thick wire panda); the dominant failure
    is severe over-segmentation (e.g., GT=1 but 20–100 predictions). False
    negatives are rare; category confusion with thick wire panda is minimal.
  - Generated artifacts (in `experiments/.../dumps/thin_wire_panda_error_analysis/`):
    64-case poster, individual overlays, raw bad-case images, plus
    `error_analysis/thin_wire_panda/summary.txt` and
    `bad_case_summary.json` committed to the branch.
  - Added `scripts/thin_wire_panda_label_quality_analysis.py` to split
    category-9 predictions into `uncorrect_label` and
    `no_ground_truth_available` buckets, and
    `scripts/thin_wire_panda_label_quality_posters.py` to render 4-panel
    (image / GT / prediction / overlay) review posters for each bucket.
  - Added `scripts/thin_wire_panda_clean_annotations.py` to produce a cleaned
    test annotation JSON: 4 new GT annotations were created from the 2
    no-GT images, and 26 `uncorrect_label` images were removed. Result:
    `9cat_diverse_wires_v1/test/annotation_merged_cleaned_thin_wire_panda.json`
    (4,153 images / 13,226 annotations vs. original 4,179 / 13,249).
  - Added `scripts/thin_wire_panda_clean_train_annotations.py` to produce a
    conservatively cleaned training annotation JSON. Of 201 training images
    with thin wire panda predictions but no GT, 162 met the conservative
    criteria (≤5 predictions and best score ≥0.5) and were converted into
    320 new GT annotations; the remaining 39 images were kept unchanged so
    their other-category annotations are preserved. Result:
    `9cat_diverse_wires_v1/train/annotation_merged_cleaned_thin_wire_panda.json`
    (10,228 images / 42,320 annotations vs. original 10,228 / 42,000; thin
    wire panda annotations: 1,302 → 1,622).
  - Added `scripts/thin_wire_panda_converted_train_poster.py` to render a
    grid poster of the 162 converted training images with their new thin
    wire panda annotations overlaid.
  - Fixed `scripts/save_coco_grid_poster.py` to resize masks when their
    shape differs from the source image, supporting mixed-resolution
    datasets.
- **Expanded thin wire panda dataset** (`9cat_diverse_wires_v2`):
  - Extended `sam3_fine_tune/preprocessing/build_diverse_wire_dataset.py`
    with per-category mask filters (`min_total_area`,
    `min_max_component_area`, `min_coverage`) so diverse-category selection
    can exclude images with zero/small/fragmented masks.
  - Added config `sam3_fine_tune/preprocessing/configs/9cat_diverse_wires_v2_config.yaml`
    that raises `thin_wire_panda` from 1,200 to 2,000 total images while
    requiring each selected image to have at least 1,000 px of thin wire
    panda mask and a largest connected component of at least 1,000 px.
  - Built dataset at `sam3_data/9cat_diverse_wires_v2`: train 11,028 images
    / 43,150 annotations, test 4,179 / 13,220; thin wire panda train images
    1,800 (vs. v1 1,000) and annotations 2,412 (vs. v1 1,302). Mean thin
    wire panda train mask area improved from 1,876 px to 2,410 px.
  - Generated overlay posters for v2: all categories, all thin wire panda
    images, and wire-categories comparison.

## 2026-07-28

Added a reusable grid-poster generator for visually inspecting COCO category
labels, plus patch-wise inference helpers for thin-wire segmentation.

- **New script**: `scripts/save_coco_grid_poster.py` renders a single JPEG
  poster of COCO images with mask overlays. Supports single-category mode
  (every image containing the category) and multi-category mode (one labeled
  row per category, capped via `--max-per-category`). Uses the headless `Agg`
  matplotlib backend so it can run on the HPC `batch` CPU queue; configurable
  panel size, grid columns, alpha, DPI, and JPEG quality.
- **New skill**: `.github/skills/coco-grid-poster/SKILL.md` documents how to
  verify category names, submit poster jobs to the HPC `batch` queue, and
  choose between single-category and multi-category modes.
- **HPC poster jobs**: generated
  `9cat_diverse_wires_v1/thin_wire_panda_all_overlay_poster.jpg` (all 1000
  train images) and
  `9cat_diverse_wires_v1/wire_categories_comparison_poster.jpg` (32 images
  each across four wire categories) on the cluster via `bsub -q batch`.
- **Notebook update**: `nbs/03_inference.thin_wire_segmentation.ipynb` extended
  with patch-wise inference helpers (`_extract_patches`,
  `run_patched_inference`), Stratus data paths, and `thick_wire_panda` /
  `thin_wire_HD` prompt examples.

## 2026-07-27

Built a new 4-category diversity-rebalanced COCO wire dataset
(`4cat_diverse_wires_v1`) and generated overlay posters for it. Fixed two
embedding-cache bugs and a GPU-idle job-kill issue discovered along the way.

- **New dataset**: `thick_wire_panda` kept unchanged (2883 train / 748 test),
  `thin_wire_panda`, `thick_wire_HD`, `thin_wire_HD` each rebalanced to 1000
  images (800 train / 200 test) via DINOv3 (ConvNeXt-tiny) embeddings +
  farthest-point-sampling for diversity. Category id order:
  `thick_wire_panda=1, thin_wire_panda=2, thick_wire_HD=3, thin_wire_HD=4`.
- **New code**: `sam3_fine_tune/preprocessing/build_diverse_wire_dataset.py`
  (CLI orchestrator) and `sam3_fine_tune/preprocessing/diversity/`
  (`dinov3_embedder.py`, `select_diverse.py`, `embedding_cache.py`), plus
  `sam3_fine_tune/preprocessing/configs/diverse_wire_dataset_example.yaml`.
- **Embedding cache fixes**: `EmbeddingCache.save()` now saves incrementally
  every `model.cache_save_every_batches` batches (default 20) instead of once
  per category, so partial progress survives interrupted jobs. Fixed two
  bugs found while testing this: (1) `np.savez()` auto-appends `.npz` to its
  target path, which broke the atomic tmp-file rename pattern — now uses
  `<stem>_tmp.npz` as the temp filename; (2) `np.load(..., allow_pickle=False)`
  can't load an object-dtype path array — the cache now stores paths as a
  fixed-width unicode array instead.
- **HPC GPU-idle-kill root cause found & fixed**: two dry-run job attempts
  died with `TERM_OWNER: job killed by owner` after ~15-25 minutes, initially
  suspected to be an SSH-session-tied kill (disproven by testing both `bsub -Is`
  and fully detached `bsub` submissions). The real cause was the cluster's
  documented idle-GPU auto-kill policy (>5 min of <5% GPU utilization):
  `DINOv3Embedder.embed_batch()` loaded every image in a batch serially with
  `PIL.Image.open()` over a network mount (~3.6-6s/batch) before a quick GPU
  forward pass, leaving the GPU mostly idle. Fixed by loading each batch's
  images concurrently with a `ThreadPoolExecutor` (I/O-bound, so the GIL
  isn't a bottleneck) — batch time dropped to ~0.7s (~1.4 it/s), and the
  dry-run then completed successfully.
- **Overlay posters**: `overlay_posters/4cat_diverse_wires_v1_all.jpg` plus
  one per category, generated via `scripts/save_coco_overlay_poster.py`.
- **Skill docs**: `.github/skills/diverse-wire-dataset/SKILL.md` documents
  the pipeline, the incremental-cache fix, and the corrected GPU-idle-kill
  root cause (cross-referencing `.github/skills/hpc-gpu-submit/SKILL.md`'s
  pre-existing entry).

**Correction (same day)**: the 4-category dataset above dropped the other 5
pre-existing categories, which was a mistake — replaced by a 9-category
dataset (`9cat_diverse_wires_v1`) that keeps everything else unchanged and
only rebalances the wire categories:

- **`fixed_category` (singular) → `fixed_categories` (list)** in
  `build_diverse_wire_dataset.py`: now supports copying through several
  categories unchanged from one or more source roots (grouped by root so a
  shared annotation file is parsed once), not just one.
- **`category_display_names` config map**: lets the output COCO category
  `"name"` differ from the internal source-matching name, without touching
  any matching logic. Used to rename the underscore-joined wire category
  names to natural-language (space-separated) prompt strings —
  `thick_wire_panda` → `"thick wire panda"`, `thin_wire_panda` → `"thin wire
  panda"`, `thick_wire_HD` → `"thick wire HD"`, `thin_wire_HD` → `"thin wire
  HD"` — since SAM3 categories double as open-vocabulary text prompts. The
  other 5 categories (`et_pin`, `eberhard_pin`, `wuxi_pin`, `war_crack`,
  `matrix_chip`) keep their original underscore names, unchanged.
- **New dataset** `9cat_diverse_wires_v1`: all 9 categories present.
  `et_pin`/`eberhard_pin`/`wuxi_pin`/`war_crack`/`matrix_chip`/`thick_wire_panda`
  copied through unchanged from the existing 9-category dataset;
  `thin_wire_panda`/`thick_wire_HD`/`thin_wire_HD` each get 1000
  diversity-selected images added from the larger patched pool (in addition
  to any of those categories that already co-occur in the unchanged images).
  Final per-category totals (train+test): et_pin=1500, eberhard_pin=2000,
  wuxi_pin=1500, war_crack=993 (no test split in the source data),
  matrix_chip=1583, thick wire HD=1768, thin wire HD=1825, thick wire
  panda=3631, thin wire panda=1000.
- **`scripts/save_coco_overlay_poster.py`**: added `--one-per-category` +
  `--categories` so a single poster JPEG can show one representative
  overlay row per category (guarantees every requested category appears),
  instead of relying on random sampling to happen to cover every category.
  `overlay_posters/9cat_diverse_wires_v1_all_categories.jpg` shows all 9.

**Training on `9cat_diverse_wires_v1` (same day)**: submitted two fine-tuning
runs — one on `gpu_compute` (1 GPU) and one on `gpu` queue (2 GPUs). The 2-GPU
run crashed at the end of epoch 0 with `AssertionError: Expected 2 batches, got 1`.

- **Root cause**: `gradient_accumulation_steps=2` with `train_batch_size=2`
  requires the train set size to be divisible by `2 × 2 × num_gpus`. The new
  dataset has 10,228 train images, which is not divisible by 8 (2 GPUs) or by
  4 (1 GPU), so the last step of the epoch received only one micro-batch and
  the trainer assertion failed.
- **Fix**: set `gradient_accumulation_steps: 1` in both configs. Effective
  batch size is now 4 for the 2-GPU run and 2 for the 1-GPU run.
- **New configs**: `sam3/train/configs/full_train_9cat_diverse_v1_2gpu.yaml`
  and `sam3/train/configs/full_train_9cat_diverse_v1_gpucompute.yaml`, both
  pointing at `9cat_diverse_wires_v1`.
- **Skill docs**: `.github/skills/finetune-merged-dataset/SKILL.md` updated
  with a hard warning about the divisibility check and an example of how to
  verify it before submitting.
- **H200 `gpu_compute` config (same day)**: added
  `sam3/train/configs/full_train_9cat_diverse_v1_gpucompute_h200.yaml` with
  `category_chunk_size: 9`, `train_batch_size: 6`, `val_batch_size: 6`, and
  `gradient_accumulation_steps: 1`, targeting H200-class hosts via
  `bsub -gpu "num=1:gmem=80G"`. The skill now documents when to use this
  config vs the conservative standard `gpu_compute` config.

## 2026-07-25

Added automatic `*_metadata.json` generation alongside every COCO annotation
file so dataset category counts are visible without parsing the full COCO JSON.

- **New helpers**: `sam3_fine_tune/preprocessing/coco_format.py` now exports
  `build_coco_metadata()` and `save_metadata_file()`. These compute total
  image/annotation/category counts and per-category unique image counts and
  annotation counts.
- **coco_format.py CLI**: writes `<file_name>_metadata.json` next to the COCO
  JSON after conversion.
- **create_patched_dataset.py**: writes `annotation_metadata.json` per split
  alongside `annotation.json`.
- **patch_coco_categories.py**: writes `annotation_merged_metadata.json` per
  split alongside `annotation_merged.json`.
- **Skill docs**: `.github/skills/coco-format-conversion/SKILL.md` and
  `.github/skills/patch-dataset/SKILL.md` document the new metadata files and
  include per-category count examples in their verification sections.
- **Notebook sync**: `nbs/01_coco_format.ipynb` updated with the new exported
  metadata functions.

## 2026-07-24

Started a 2-GPU SAM3 fine-tuning run on the `gpu` queue for the 9-category
unique-panda dataset and captured multi-GPU scheduling/config lessons in the
`finetune-merged-dataset` skill.

- **New config**: `sam3/train/configs/full_train_9cat_2gpu.yaml` for the
  2-GPU run (`category_chunk_size=8`, `train_batch_size=4`,
  `experiment_log_dir=experiments/9cat_unique_panda_2gpu_gpuq`).
- **Training job**: `240845039` on `gpu` queue, `num=2:j_exclusive=yes:gmem=42G`,
  dispatched to `muclcgpu17` and training by 00:38 CEST.
- **Skill update**: `.github/skills/finetune-merged-dataset/SKILL.md` now
  documents multi-GPU queue behavior (`gpu_compute` is single-GPU only; `gpu`
  hosts vary in GPU count), the requirement to create a separate config file
  rather than passing Hydra-style CLI overrides, and a validated `num=2`
  example.

- **Repo cleanup**: reset local and HPC working trees to `origin/main`, added
  `.gitignore` patterns for generated runtime directories (`logs/`, `lsf_logs/`,
  `overlay_check/`, `overlay_posters/`, `deleete/`), and quarantined stray
  artifact files in `deleete/`.

- **New notebook**: added `nbs/03_inference.thin_wire_segmentation.ipynb` for
  running thin-wire segmentation inference with a fine-tuned SAM3 model.

## 2026-07-23

Added a training watchdog script and skill to auto-restart SAM3 fine-tuning jobs
that disappear from the LSF scheduler (e.g. idle-GPU auto-kill), while leaving
successfully completed runs alone.

- **New watchdog script**: `.github/scripts/training_watchdog.sh` monitors a SAM3
  training experiment in LSF and resubmits the batch job if it disappears from the
  scheduler, while exiting cleanly when the latest log reports
  `Successfully completed.`.
- **New skill**: `.github/skills/training-watchdog/SKILL.md` documents how to create
  the bsub command file, start/stop the watchdog, and verify it is running.
- **Watchdog started**: guarding the `9cat_unique_panda` experiment (PID `2417913`).
  An earlier watchdog instance was not stopped cleanly, which caused several
  duplicate training jobs to be submitted; the duplicates were killed and a
  single watchdog was restarted after the merge-conflict cleanup.
## 2026-07-22

Added patch-based dataset creation skills/scripts and corrected HPC queue usage
for preprocessing (CPU-only; GPU only for training/model inference).

- **New script**: `sam3_fine_tune/preprocessing/create_patched_dataset.py` reads a
  YAML config, tiles source images/masks into fixed-size patches (e.g. 256×256)
  with configurable overlap, filters out low-content patches, and writes COCO
  `annotation.json` files per split using the same logic as `coco_format.py`.
- **New script**: `sam3_fine_tune/preprocessing/patch_coco_categories.py` patches
  selected categories in-place inside an existing merged COCO dataset, leaving
  other categories unchanged. Source images containing target categories are
  replaced by overlapping patches; annotations are clipped to each patch.
- **New config example**: `sam3_fine_tune/preprocessing/configs/patch_coco_categories_example.yaml`
  documents the in-place layout (`patch.size`, `patch.overlap`,
  `patch.min_target_area`, `target_categories`, `input.root`, `output.root`).
- **Skill update**: `.github/skills/patch-dataset/SKILL.md` now covers both
  `create_patched_dataset.py` and `patch_coco_categories.py`, explicitly uses the
  HPC `batch` CPU queue (`rusage[mem=8G] -n 1`), and documents the GPU-kill rule
  (low GPU-util jobs are terminated; use GPU queues only for training/inference).
- **Dataset job**: `patch_coco_categories.py` submitted to `batch` CPU queue as
  job `240032281` to produce
  `9cat_separated_wires_v3_patched_wires_256` (256×256 patches, 128 px overlap,
  four wire categories). Earlier GPU-submitted patch jobs (`240022032`,
  `240030716`) were killed after the queue misuse was identified.

## 2026-07-21

Rebuilt the 9-category separated-wire dataset from the original clean labels and
repointed training to the corrected v3 dataset.

- **Corrected dataset**:
  - `9cat_separated_wires_v3`: old 5 categories + `thick_wire_HD`/`thin_wire_HD` (from the
    clean Bent-Wire-ALU-Train/20250512_parted_thin_flat `_lblthick.png`/`_lblthin.png` masks) +
    `thick_wire_panda`/`thin_wire_panda` (from the universal ThinWire segmentation model).
    `gate_pad` was dropped to avoid the recurring thin-wire/gate-pad overlap problem.
    13,988 train / 5,553 test images, 49,560 / 14,159 annotations.
  - Replaced the first v1 build (10,046 / 4,696 images) which had been assembled from an
    incorrect/old 8cat mask source.
- **Visualization**:
  - Generated overlay posters for `9cat_separated_wires_v3` using
    `.github/scripts/coco_visualization_poster.py`.
- **Skills**:
  - `.github/skills/merge-category-dataset/SKILL.md`: added explicit step to count
    categories after merging and check the training config before deciding the new
    `category_chunk_size` value.
  - `.github/skills/finetune-merged-dataset/SKILL.md`: clarified that
    `category_chunk_size` should be set equal to the total category count when it
    fits in GPU memory.
- **Training**:
  - Killed incorrect bsub job `239199510` (trained on v1 source).
  - `sam3/train/configs/full_train_config.yaml` repointed to `9cat_separated_wires_v3`
    (`pin_dir_root` and `experiment_log_dir`).
  - Submitted bsub job `239215766` on `gpu` queue with 3 GPUs, `train_batch_size=4`,
    `category_chunk_size=8`. Killed after deciding to use `category_chunk_size=9` so
    every chunk contains all 9 categories.
  - Submitted bsub job `239233258` on `gpu` queue with 3 GPUs, `train_batch_size=4`,
    `category_chunk_size=9`. OOM'd on 22GB GPUs after ~85 s (training fit on earlier
    29GB cards with `category_chunk_size=8`, but `category_chunk_size=9` exceeded
    22 GB).
  - Reduced `train_batch_size` from `4` to `3` so `category_chunk_size=9` fits on
    22 GB GPUs and submitted bsub job `239295637` on `gpu` queue with 3 GPUs,
    full fine-tune from base `sam3.pt`. Ran ~7.5 h, reached epoch 6, then OOM'd
    on 44 GB GPUs during the resume validation epoch because `val_batch_size=10`
    was too large for `category_chunk_size=9`.
  - Tried reducing `resolution` to `896`; failed immediately with RoPE shape
    assertion in `sam3/model/vitdet.py` (`freqs_cis` precomputed for 1008).
  - Confirmed `gradient_accumulation_steps` is implemented and works for training;
    set `train_batch_size=2`, `gradient_accumulation_steps=2`, `val_batch_size=3`,
    and submitted job `239776660` to resume from epoch 6 and finish training.

## 2026-07-20

Created two final datasets and started two parallel training runs to compare full
8-category bondwire training vs. old categories + thin_wire only.

- **New datasets**:
  - `8cat_bentwire_separate`: old 5 categories + new `thick_wire`/`thin_wire`/`gate_pad`
    from the cleaned Bent-Wire labels (502 source photos merged into 502 multi-category
    images; 6412 train / 3946 test images after merging with the old dataset).
  - `6cat_old5_plus_thin_wire`: old 5 categories + only the new `thin_wire` category
    (excluding new `thick_wire` and `gate_pad`) — 6412 train / 3946 test images.
- **New configs**:
  - `sam3/train/configs/full_train_8cat_bentwire_separate.yaml`
  - `sam3/train/configs/full_train_6cat_old5_plus_thin_wire.yaml`
- **Submitted jobs** (bsub `gpu` queue, `num=1:gmem=40G`, `train_batch_size=4`,
  `category_chunk_size=8`, full fine-tune from base `sam3.pt`):
  - `8cat_bentwire_separate`: job `238802984`
  - `6cat_old5_plus_thin_wire`: job `238802989`

## 2026-07-20

Added a save-to-JPEG poster helper so COCO conversions can be visually verified on the HPC.

- **New script**: `scripts/save_coco_overlay_poster.py` loads a COCO annotation JSON,
  randomly samples images, overlays segmentation masks with per-category colors, and saves
  a JPEG poster.  Defaults point at the merged bondwire dataset; accepts `--coco-json`,
  `--image-dir`, `--output`, `--n-samples`, `--seed`, `--alpha`, `--dpi`, and `--ncols`.
- **Updated `nbs/12_clean_bondwire_dataset.ipynb`**: added an exported `save_overlay_poster()`
  function and example calls for the train/test splits so the same save step can be run
  directly from the notebook.
- Generated and saved on the HPC:
  - `/home/hpc_storage/data/projects/userB/sam3_data/bondwire_clean_merged/train/overlay_poster.jpg`
  - `/home/hpc_storage/data/projects/userB/sam3_data/bondwire_clean_merged/test/overlay_poster.jpg`

## 2026-07-20

Added a notebook and helper module to build COCO datasets from the clean separated
bondwire labels in `Bent-Wire-ALU-Train/20250512_parted_thin_flat`.

- **New notebook**: `nbs/12_clean_bondwire_dataset.ipynb` discovers image/mask triplets
  (`*_original.png`, `*_lblthin.png`, `*_lblthick.png`, `*_gatepads.png`), stages them
  into per-category train/test splits, converts each split to COCO, merges the three
  categories, and runs round-trip mask verification.
- **New module**: `sam3_fine_tune/preprocessing/clean_bondwire_dataset.py` contains the
  same functions so they can be imported without running `nbdev_export`.
- This is the preferred path over the composite-mask bridging workaround: the labels are
  already separated per category, so no synthetic bridge pixels are needed and the model
  learns from true instance masks.

## 2026-07-20

Fixed the bondwire ground-truth annotation pipeline: wires passing behind gate_pads were
split into multiple COCO instances because `gate_pad` has pixel priority in the composite
mask.  Added gap-bridging in the mask splitter, a round-trip verification script, and a
patch script to update an existing merged dataset without touching the other categories.

- **Root cause**: the source composite mask paints `gate_pad` (label 3) on top of
  `thick_wire` (1) and `thin_wire` (2).  `split_multiclass_masks.py` produced binary masks
  with a hole wherever a gate_pad sat on a wire, and `coco_format.py` then used
  `cv2.findContours(RETR_EXTERNAL)` — which emitted two (or more) COCO annotations for a
  single physical wire.  This showed up as thin_wire AP collapsing to 0.376 in the clDice
  validation run (job `238286332`) and was initially mistaken for a model/prediction bug.
- **Added gate-pad gap bridging** (`sam3_fine_tune/preprocessing/split_multiclass_masks.py`):
  new `--bridge_categories` / `--bridge_obstacle_label` / `--max_bridge_px` CLI options.
  For each bridge component (`wire | gate_pad`), the closest disconnected wire pieces are
  joined by a minimal line through the gate_pad region before the binary mask is written.
  This gives `cv2.findContours` one connected component per physical wire while adding the
  smallest possible bridge area.  New `--split` option creates `<category>/<split>/images|masks`
  subdirectories directly, matching the existing `by_category/<cat>/<split>` layout.
- **Added round-trip verification** (`sam3_fine_tune/preprocessing/verify_coco_roundtrip.py`):
  reconstructs per-category masks from a COCO JSON and compares them to the source composite
  masks, reporting mean IoU and source-vs-COCO instance counts.  Supports both composite
  priority mode and `--per_category` mode, plus `--filename_prefix` stripping for prefixed
  merged datasets.
- **Added `patch_bondwire_annotations.py`** to swap fixed bondwire annotations into an
  existing merged dataset without modifying the other categories.  Produced
  `/home/hpc_storage/data/projects/userB/sam3_data/5_plus_bondwire_dataset_v2` from the
  original `5_plus_bondwire_dataset`; thin_wire annotations dropped from 1997 -> 940
  (train) and 277 -> 182 (test) as split pieces were merged, while thick_wire and gate_pad
  counts stayed unchanged.
- **Updated `full_train_config.yaml`**: `paths.pin_dir_root` now points to
  `5_plus_bondwire_dataset_v2` so the next training run uses the fixed annotations.
- **Updated `.github/skills/merge-category-dataset/SKILL.md`**: documented the bondwire
  gap-bridging workflow, round-trip verification, and the patch script.

## 2026-07-20

Added a config-driven full-vs-decoder-only fine-tuning toggle, fixed a false-positive bug
in the anomalous-batch debug dump, and updated the finetune skill to ask about/recommend
the fine-tuning mode.

- **Added `trainer.optim.train_full_model` toggle** (`sam3/train/configs/full_train_config.yaml`
  + `sam3/train/trainer.py`): a new `Trainer._maybe_freeze_non_decoder_params()` method,
  called right after model instantiation, freezes `backbone.vision_backbone.*`,
  `backbone.language_backbone.*`, and `transformer.encoder.*` via `requires_grad=False`
  when `train_full_model: False`, keeping `transformer.decoder.*`, `segmentation_head.*`,
  `geometry_encoder.*`, and `dot_prod_scoring.*` trainable in both modes. Default
  `train_full_model: True` preserves existing full-fine-tune behavior. Frozen params stay
  in their existing optimizer LR param groups unchanged (no param-group surgery needed) —
  `requires_grad=False` means autograd never populates `.grad`, and AdamW already skips
  any param with `.grad is None`.
- **Discovery**: a separate, older `sam3/train/configs/decoder_train_config.yaml` already
  implemented decoder-only fine-tuning (LR=0 param groups +
  `FreezeEncoderBackboneInitializer`), but it predates and lacks every fix made this
  session (annotation-merge fix, bf16/GradScaler fix, per-category evaluator, clDice
  option) — using it as-is would silently reintroduce already-fixed bugs. The new toggle
  lives directly in `full_train_config.yaml` instead, so decoder-only runs inherit every
  current fix.
- **Fixed a false-positive bug in `_maybe_dump_anomalous_batch()`**: during validation, the
  batch dict key is `"pin"`, which isn't a real key in `self.loss` (only `"all"`/`"default"`
  exist), so `_find_loss("pin")` always falls back to `DummyLoss` — which returns exactly
  `0.0` by design every val step (validation only needs predictions for the COCO evaluator,
  not a real loss). The dump guard's `key != "default"` check didn't catch this since `key`
  is the literal string `"pin"`, so every validation step looked like a zero-loss anomaly.
  Confirmed real impact on job `238286332`: all 3 of the shared `MAX_ANOMALOUS_DUMPS=3`
  budget were burned within 8 seconds of the very first validation epoch
  (`val_pin_epoch5_step{0,1,2}`, all `loss=0.0`) — meaning a genuine training collapse later
  in that run would not have been captured, since the budget was already exhausted on
  harmless validation noise. Fixed with a `if phase != Phase.TRAIN: return` guard at the
  top of the function.
- **Updated `finetune-merged-dataset/SKILL.md`**: added a mandatory Inputs question asking
  the user full fine-tune vs. decoder-only (recommending full, per the README's own
  guidance and this repo's validated evidence), wired the answer into
  `trainer.optim.train_full_model`, and flagged not to use the stale
  `decoder_train_config.yaml`.

## 2026-07-19

Follow-up: added a soft clDice loss to address remaining thick_wire/thin_wire mask quality
issues (blobby boundaries and mask fragmentation) noticed after the annotation fix above,
and kicked off an overnight validation run.

- **Motivation**: after the split-photo annotation fix (below) dramatically improved
  detection AP for the 3 new bondwire categories, visual review of the per-category overlay
  posters showed the underlying SEGMENTATION masks for `thick_wire`/`thin_wire` were still
  poor in a specific way: blobby/imprecise boundaries, and — more critically — a single wire
  instance's predicted mask sometimes fragmented into two disconnected pieces. Standard
  Dice/BCE losses barely penalize either failure mode since a thin object's total pixel count
  is tiny relative to the image.
- **Added `soft_cldice_loss()`** (`sam3/train/loss/loss_fns.py`) implementing clDice (Shit
  et al., CVPR 2021) — scores based on skeleton (centerline) recall/precision instead of raw
  pixel overlap, directly penalizing connectivity breaks (a fragmented mask leaves part of
  the GT skeleton "un-recalled") and boundary imprecision. Soft-skeletonization via iterative
  differentiable morphological erosion/dilation (max-pooling based), fully differentiable.
  Validated with synthetic fragmented-vs-connected mask tests (realistic 26px/37px wire
  widths) confirming clDice correctly and proportionally penalizes fragmentation.
- **Wired into the `Masks` loss class** via a new `weight_dict["loss_cldice"]` entry (new
  `Masks.compute_cldice` property) — only active in the non-point-sampled path since it needs
  the full 2D mask. `full_train_config.yaml` sets `loss_cldice: 15.0` by default; **to revert
  to the original default losses (mask + dice only), just remove/comment out that one config
  line — no code change needed.**
- **First attempt OOM'd** (job `238284710`): `TERM_MEMLIMIT`, 64GB host RAM used vs. a 200MB
  reservation, killed after ~80s. Root cause: soft-skeletonization keeps ~10 iterations worth
  of full-resolution (up to 1008x1008) intermediate activations alive for backward, and the
  original CPU-fallback-on-GPU-OOM path moved the same huge tensor to host RAM instead of
  failing fast. **Fixed** by downsampling both masks to 128x128 before skeletonizing (skeleton
  topology doesn't need full pixel resolution — cuts memory/compute ~62x at 1008 res) and
  removing the risky CPU fallback entirely. Re-validated the fix directly against realistic
  wire widths before resubmitting.
- **Validation run started**: job `238286332`, fresh `experiments/8cat_bondwire_cldice`
  experiment dir (starts clean from base `sam3.pt`, not resumed, so the mask-quality change
  can be directly attributed to the new loss). Confirmed running stably past 50 steps with
  the memory fix (GPU mem steady at 44GB, matching the known-tight-config baseline). Overnight
  run — per-category AP + visual overlay poster results pending next check.

Full production fine-tune on the 8-category `8cat_bondwire` dataset (branch
`fix/loss-none-pair-crash`) completed successfully with the GradScaler/bf16 fix from
2026-07-18, and a per-category evaluation breakdown revealed the new bondwire categories
need further work.

- **Training completed**: job `237965304` (batch_size=4, no debug instrumentation) ran all
  10 epochs to completion (`Successfully completed.`, ~8h14m), with **zero** `WEIGHT-NaN`
  hits and zero zero-loss collapses for the entire run — confirms the 2026-07-18 GradScaler
  fix holds for a full production run, not just the debug-settings validation window. Final
  aggregate validation (epoch 9): `segm_AP 0.657`, `AP_50 0.710`, `AR@100 0.889` — a slight
  improvement over epoch 5's `0.651`.
- **New tooling**: added `.github/scripts/per_category_coco_eval.py` — the trainer's
  built-in `val_pin/detection` meter only reports one AGGREGATE AP across all 8 categories
  combined, not a per-category breakdown. This script reuses the predictions already
  dumped by the last validation epoch (`dumps/pins/coco_predictions_segm.json`, no new
  inference needed) and runs `pycocotools.COCOeval` once per category to get real
  per-category AP/AR numbers.
- **Per-category results — the 5 original categories are strong, the 3 new bondwire
  categories are weak**:

  | Category | GT instances (test) | AP | AP_50 |
  |---|---|---|---|
  | et_pin | 1475 | 0.913 | 0.992 |
  | eberhard_pin | 1000 | 0.969 | 1.000 |
  | wuxi_pin | 1538 | 0.955 | 1.000 |
  | matrix_chip | 2853 | 0.961 | 0.990 |
  | war_crack | **0** | — | — |
  | **thick_wire** | 636 | **0.369** | 0.384 |
  | **thin_wire** | 277 | **0.242** | 0.290 |
  | **gate_pad** | 226 | **0.188** | 0.316 |

  Two distinct issues found:
  1. **`war_crack` has a train/test split gap**: 1000 training instances but **0 test
     instances** — every `war_crack` annotation ended up in the train split, none in test,
     so it can't be evaluated at all with the current split. Needs a re-split before it can
     be scored.
  2. **The 3 new bondwire categories (`thick_wire`/`thin_wire`/`gate_pad`) are
     underperforming** — initially hypothesized as thin/elongated-object segmentation
     difficulty (`AP_small: 0.0` for `thick_wire`/`thin_wire`), but investigation below found
     a more fundamental, fully fixable root cause.

- **ROOT CAUSE CONFIRMED via visual spot-check — split-photo contradictory negatives
  (100% of new-category data affected)**: added `.github/scripts/spot_check_category_masks.py`
  to render GT mask overlays for sample `thick_wire`/`thin_wire`/`gate_pad` images.
  Visually, every sample showed OTHER bondwire features (e.g. a `thick_wire`-labeled image
  also clearly showing unlabeled thin wires and gate pads) with no annotation at all — not a
  geometry-difficulty artifact, a genuine missing-label pattern. Traced to the source:
  `Thin_multicat/metadata.json` (`categories: [Thick=1, Thin=2, Gatepads=3]`,
  `overlap_rule: priority`) confirms composite masks genuinely contain all 3 category values
  per photo. `split_multiclass_masks.py` correctly extracts a separate binary mask per
  category per photo (working as designed) — but the downstream COCO conversion + merge
  pipeline kept each category's copy of a SHARED photo as a separate COCO image entry
  (`g_`/`h_`/`i_` filename prefixes), each carrying ONLY its own category's annotations.
  Verified **at scale**: ALL 662 source photos appear as EXACTLY 3 separate image entries
  (0 photos in only 1 or 2). With `include_negatives=True`, every entry implicitly asserts
  the other 2 new categories are absent from a photo where they're actually present —
  systematically contradictory negative supervision across 100% of the new-category data.
  This is now believed to be the PRIMARY driver of the poor per-category AP (more so than
  object geometry, though thin/elongated shapes likely remain a secondary contributing
  factor for `thick_wire`/`thin_wire` specifically).
- **Fix implemented and verified**: added `.github/scripts/merge_duplicate_photo_entries.py`
  — merges each group of duplicate-photo image entries into ONE entry carrying the union of
  all annotations. Verified zero annotation loss: per-category counts identical
  before/after (train: 24666 annotations unchanged across 662 merged groups →
  1324 fewer image entries; test: 8005 annotations unchanged across 113 merged groups →
  226 fewer image entries). Spot-checked the fix directly: the same photo that previously
  showed only `thick_wire` boxes now correctly shows all 3 categories (16 combined
  annotations) under one image entry. `full_train_config.yaml` now points
  train/val/eval at the new `annotation_merged.json` files (original `annotation.json`
  files left untouched — non-destructive fix).
- **Validation run started**: added `nbs/11_debug_per_category_ap_investigation.ipynb`,
  a chronological log of the full per-category-AP investigation (Steps 1-10: aggregate AP
  masking the problem, per-category breakdown tooling, geometry/confidence checks that
  ruled each other out, the visual spot-check smoking gun, tracing to source, ruling out
  the Phase-1 splitter, and confirming + fixing the split-photo bug at scale). Since the
  previous experiment (`8cat_bondwire`)'s checkpoint is already at epoch 10/10 (resuming
  would no-op instead of training), pointed `experiment_log_dir` at a fresh
  `8cat_bondwire_datafix` directory so the run starts clean from base `sam3.pt` on the
  corrected `annotation_merged.json` files — submitted as job `238099599` (epoch 0/9,
  1477 steps/epoch, matching the reduced 5910-image train set post-merge).
- **New**: added `sam3/eval/per_category_eval.py` (`PerCategoryEvalWithOverlays`), wired in
  as a second `pred_file_evaluators` entry alongside the existing aggregate evaluator, so
  every future validation epoch automatically (1) logs per-category AP/AR metrics
  (`coco_eval_<iou_type>_<METRIC>_<category_name>`) and (2) saves one JPEG overlay poster
  per category (GT in green, top predictions in red) to `dumps/category_overlays/` — no
  more manually running the standalone per-category script after training. Categories are
  discovered dynamically from the ground-truth file's own category list, so a newly-added
  category is picked up automatically. Verified end-to-end against already-dumped
  predictions before relying on it for a real run. Takes effect starting with the next
  training run (job `238099599` had already loaded the old single-evaluator config at
  process start).
- **FIX CONFIRMED WORKING — job `238099599` completed** (`Successfully completed.`,
  ~12h30m, fresh from base `sam3.pt` on the corrected `annotation_merged.json` files, zero
  `WEIGHT-NaN` hits): per-category re-evaluation shows the 3 previously-broken categories
  improved dramatically, while the 5 original categories moved only slightly (normal
  run-to-run variance from training fresh rather than resuming):

  | Category | AP before fix | AP after fix | Δ |
  |---|---|---|---|
  | et_pin | 0.913 | 0.857 | -0.056 |
  | eberhard_pin | 0.969 | 0.951 | -0.018 |
  | wuxi_pin | 0.955 | 0.932 | -0.024 |
  | matrix_chip | 0.961 | 0.951 | -0.010 |
  | **thick_wire** | **0.369** | **0.896** | **+0.527** |
  | **thin_wire** | **0.242** | **0.675** | **+0.433** |
  | **gate_pad** | **0.188** | **0.573** | **+0.385** |
  | war_crack | — | — | still unscoreable (0 test instances) |
  | **Overall (aggregate)** | **0.657** | **0.833** | **+0.176** |

  Confirms the split-photo contradictory-negatives bug was indeed the primary driver of
  the poor per-category AP, not object geometry. `war_crack`'s train/test split gap remains
  a separate, still-open issue.

## 2026-07-18

Debugged crashes and a recurring silent zero-loss collapse on the new 8-category
`8cat_bondwire` fine-tune (branch `fix/loss-none-pair-crash`), and significantly upgraded
the anomalous-batch debugging tooling built 2026-07-17.

- **Crash fix**: job `237753606` died after ~8h (epoch 5/10, step 980/1808, healthy
  decreasing loss right up to the crash) with `TypeError: 'NoneType' object is not
  subscriptable` in `sam3_loss.py`'s `_count_matched_pairs()`. Root cause: the function
  guarded a top-level `indices is None`, but not an individual `None` entry inside a list of
  per-sample tuples (which happens for a negative sample with zero matched instances, given
  `include_negatives=True`). Fixed with a one-line guard; resubmitted (job `237892821`) and
  confirmed it resumed cleanly from the epoch-5 checkpoint and passed the exact step that
  crashed before.
- **Anomalous-batch tooling upgrades** (building on the 2026-07-17 dump/inspect mechanism):
  - `trainer.py`'s `_maybe_dump_anomalous_batch()` now also captures `find_targets[].segments`
    (per-object masks) — previously only boxes/num_boxes were saved, even though masks were
    available on the batch at dump time.
  - New `save_post_augmentation_jpegs()` (both in the standalone
    `.github/scripts/inspect_anomalous_batch.py` and duplicated inline in `trainer.py` as
    `_save_anomalous_batch_jpegs()`) denormalizes the post-augmentation image tensor and
    renders box outlines + colored mask overlays to JPEG. `trainer.py` now calls this
    automatically right after every anomalous `.pt` dump (best-effort, wrapped in
    try/except so a visualization bug can never break the training loop), so a JPEG is
    already on disk without a manual follow-up script run.
  - Fixed two real bugs found while building/using this: (1) an operator-precedence bug in
    `_describe_post` that accessed `box[2]` unconditionally regardless of box size; (2) a
    box-shape assumption that broke when a box tensor had an extra leading dim (e.g. shape
    `(1, 4)`) — both fixed by flattening before indexing.
  - **Caught via visual inspection** (the whole reason for building this): the first version
    of the JPEG box/mask attribution was wrong — `num_boxes` has one entry per
    **(image, category)** query (e.g. 32 entries for a 4-image batch with
    `category_chunk_size=8`), not one entry per image. The original code chunked the flat
    boxes/segments list directly by each raw `num_boxes` entry, silently misattributing
    boxes to the wrong image (a box rendered floating in a blank padding region of an image
    that actually had 14 real objects elsewhere in frame). Fixed by summing every
    `len(num_boxes)/batch_size` consecutive entries into the correct per-image total first.
    Verified: the corrected JPEG shows all 14 boxes landing on real chip die pads.
- **Zero-loss collapse, still open**: after the crash fix, job `237892821`'s resumed run hit
  the same *silent exactly-zero-loss* failure mode first seen 2026-07-17 on
  `experiments/5cat_matrix_chip_v2` (loss-debugging skill Step 8) — starting at step 630 of
  epoch 5, ~38% of steps logged zero loss. Cancelled and resubmitted (job `237899477`) to
  pick up the new mask-capture/JPEG tooling; this time the collapse recurred at step 670 of
  the same epoch 5 and did **not** recover — 500+ consecutive zero-loss steps by the time it
  was stopped. Visual inspection of the auto-saved JPEGs (with mask overlays, correct
  attribution) confirmed the underlying data is completely legitimate, ruling out a bad
  sample as the cause again.
- **New diagnostic to actually pinpoint it**: `sam3_loss.py`'s zero-loss diagnostic
  previously only logged the aggregate `total_core_loss` and matched-pair counts. Since
  every individual loss term (`loss_bbox`, `loss_giou`, `loss_ce`, `loss_mask`, `loss_dice`,
  `loss_semantic_seg`, `loss_semantic_dice`) is non-negative and summed, the total can only
  be exactly 0.0 with real matched pairs if **every** individual term (across ~6
  stages x loss_fn x o2o/o2m) is independently 0.0 at once — not plausible as independent
  numerical coincidences, pointing at one shared upstream cause (a weight/scale going to 0)
  rather than genuinely-zero math. Added per-term contribution tracking so the log now
  reports `nonzero_contributions`/`all_zero_terms` explicitly instead of only the total.
  Cancelled job `237899477` and resubmitted (job `237906220`) to catch the (reproducible,
  same-checkpoint) collapse again with this new breakdown — investigation in progress.
- **Root cause #1 confirmed — the zero-loss collapse IS a NaN, just silently absorbed**:
  added a `model_output_debug` capture to `Sam3LossWrapper.compute_loss()` that stashes the
  main stage's `pred_boxes`/`pred_logits`/`pred_masks` at the matched indices whenever the
  zero-loss diagnostic fires. Confirmed the matched predictions are literally `NaN` — the
  `nan_to_num(nan=0.0, ...)` guards added 2026-07-16/17 to stop the earlier
  `FloatingPointError` crash were silently converting every NaN loss term to exactly `0.0`,
  so training kept running with dead (zero-gradient) steps instead of crashing. The first
  version of this capture kept full-size GPU tensors alive and caused a real OOM (job
  `237913353`, step 730); fixed by slicing to matched indices only and moving to CPU
  immediately at the capture site.
- **Root cause #2 confirmed — GradScaler decays into fp32 subnormal territory for
  bfloat16**: added `_check_patch_embed_weight_nan()` to check `patch_embed.proj`'s
  weight/bias for non-finite values right after every optimizer step, and an opt-in
  forward-hook NaN localizer (`_install_nan_debug_hooks()`, gated by
  `SAM3_NAN_DEBUG_HOOKS=1`). Job `237937642` caught the smoking gun:
  `scaler.get_scale()` had decayed to `2.94e-39` (a subnormal fp32 value, ~143 consecutive
  `backoff_factor=0.5x` halvings from `init_scale=65536.0`) at the exact moment
  `patch_embed.proj`'s Conv2d weights went 100% NaN. `torch.amp.GradScaler` is designed for
  `float16`'s narrow exponent range; `bfloat16` has fp32's exponent range and never needed
  loss scaling, but the scaler doesn't know that and kept backing off anyway. Fixed a
  second crash found along the way (`IndexError` on a genuinely-empty, non-`None`
  `indices` entry in `_count_matched_pairs`).
- **Fix implemented and validated**: added `optim.amp.bypass_dynamic_scale_for_bf16` config
  flag (default `True`) that pins `GradScaler`'s scale at (effectively) a constant `1.0` for
  `bfloat16` training (near-identity growth/backoff factors + huge `growth_interval`, since
  `torch.amp.GradScaler`'s constructor hard-asserts `growth_factor > 1.0`/
  `backoff_factor < 1.0`). First validation attempt (job `237943948`) reproduced the
  identical collapse/scale value byte-for-byte, because `_load_resuming_checkpoint()`
  unconditionally restored the checkpoint's saved (old, dynamic) scaler state via
  `load_state_dict()`, silently overwriting the pinned scaler — fixed by skipping that
  restore when the bypass flag is active. Re-validated as job `237950477`: ran clean well
  past step 2530/3617 (past the ~step 1220 point where both prior attempts collapsed), zero
  `WEIGHT-NaN` hits, zero zero-loss steps.
- **Production run started**: reverted `train_batch_size` `2 → 4` (was only dropped for
  `SAM3_NAN_DEBUG_HOOKS` memory headroom during the investigation) and resubmitted without
  debug instrumentation as job `237965304` — running cleanly at the known-tight
  44/44GB baseline.
- Added `nbs/10_debug_zero_loss_collapse.ipynb`, a chronological debugging log covering the
  full investigation (Steps 1-10), and updated `.github/skills/loss-debugging/SKILL.md`
  with the confirmed root causes and both fix attempts, including the checkpoint-resume
  gotcha, for future reference.

## 2026-07-17

Follow-up to the epoch-9 collapse investigation below: found and fixed a reliability bug in
the just-added best-checkpoint mechanism itself, ran a full repro attempt plus a clean-data
audit, and added permanent tooling (anomalous-batch dumps) for the next occurrence.

- Root-caused why two repro jobs (`237539252`, `237570479`) died with `TERM_OWNER: job killed
  by owner.` / exit `143` right at the validation/checkpoint-save step: this HPC auto-kills
  jobs with GPU utilization below ~5% for 5+ minutes, and the just-added
  `save_best_meters` mechanism was saving **all ~9** improving AP/AR sub-metrics as separate
  ~10GB checkpoints on the very first validation (no prior best existed yet) — several minutes
  of near-zero GPU use while writing to the network mount. Documented this policy in
  `hpc-gpu-submit/SKILL.md` and `finetune-merged-dataset/SKILL.md`.
- Fixed it properly: added `checkpoint.save_best_meter_subkeys` (new `CheckpointConf` field in
  `trainer.py`) to restrict which sub-metrics actually trigger a save; set it to
  `["coco_eval_segm_AP"]` in `full_train_config.yaml` so only 1 checkpoint is written per
  improving validation instead of ~9. Resubmitted (job `237598940`) and confirmed it survives
  the validation phase and best-checkpoint save without being auto-killed.
- Ran a full repro from `checkpoint_5.pt` (last known-good) through epochs 6-10 with the same
  config/code: **completed successfully, no collapse** — healthy losses throughout, zero hits
  on the new zero-loss diagnostic. This updates the "reproducible dead state" framing below —
  the collapse did not reproduce from a clean resume.
- Ran `check_dataset_degenerate_samples.py --check-pixels` (via `hpc-cpu-submit`) against the
  full merged dataset (all 5 categories, old and new, both train/test splits): **0 issues
  found** (zero-area boxes, empty segmentations, orphan annotations, missing files, pixel
  decode/NaN-Inf). Rules out a deterministic data-quality bug.
- Reviewed the augmentation pipeline (`RandomizeInputBbox`/`noise_box`, `FilterEmptyTargets`)
  for a plausible "augmentation produces a degenerate box" theory: both already guard against
  it (falls back to the original box / filters `area < 1e-6` GT objects). Combined with the
  clean repro and data audit, this points to a rare/non-deterministic (likely
  hardware/cuDNN-driven) one-off event rather than a reproducible code or data bug.
- Added permanent tooling for the next occurrence: `Trainer._maybe_dump_anomalous_batch()`
  (`trainer.py`) saves the exact batch to `<experiment_log_dir>/dumps/anomalous_batches/`
  whenever the real loss goes non-finite or exactly zero (rate-limited to 3 dumps/run) —
  capturing **both** the post-augmentation batch (`img_batch`, `find_targets`) **and**, via a
  new `Datapoint.raw_boxes` field (mirroring the pre-existing but previously-unused
  `raw_images`), the pre-augmentation source, so a future investigation can directly tell
  augmentation-induced degeneracy apart from source-data degeneracy. Added
  `.github/scripts/inspect_anomalous_batch.py` to print a side-by-side comparison, and
  documented the dump location in `README.md` (new Troubleshooting section) and
  `loss-debugging/SKILL.md` (new Step 8).

Root-caused a silent training collapse on `experiments/5cat_matrix_chip_v2` (val AP dropped
from 0.94 at epoch 5 to exactly 0.0 by epoch 9, with train loss frozen at precisely `0.00000`
from the very first step of epoch 9 onward) and fixed a long-dormant gap that left the
trainer's best-checkpoint mechanism completely disabled, so future regressions like this one
don't silently overwrite the last good checkpoint.

- Confirmed the collapse is a real, reproducible dead state, not a fluke or a notebook
  misconfiguration: job `237436875` showed normal train loss (47–130 range) through epoch 8,
  then exactly `0.00e+00` for every logged step of epoch 9 (val AP also collapsed to `0.0`
  across every metric that epoch), and the subsequent resumed job `237515546` (started from
  that same `checkpoint_10.pt`/`checkpoint.pt`) reproduced the identical all-zero loss at
  epoch 10 — confirming the corruption lives in the saved checkpoint state itself, not a
  one-off epoch-9 fluke.
- Investigated `sam3/train/optim/schedulers.py`'s `InverseSquareRootParamScheduler`
  (`warmup_steps`/`cooldown_steps`/`timescale` all `20`, applied as raw step counts, not
  percentages) as the leading suspect given the collapse's suspicious alignment with
  `Trainer/where` crossing ~0.9. Ruled it out by hand-computing the cooldown factor at the
  exact epoch-9 boundary (`step=2619`, `total_steps=2910`,
  `min(1.0, (2910-2619)/20) = 1.0`) — the cooldown term is mathematically still fully open at
  that point (`cooldown_steps=20` only bites in the last ~20 of 2910 steps), so this specific
  formula does not explain the sudden exact-zero loss. Root cause remains open — likely in the
  data/category-chunk sampling or matcher path rather than the LR schedule; not pursued
  further this session given `checkpoint_5.pt` already provides a clean fallback.
- Killed the futile `237515546` job (continuing from the corrupted `checkpoint_10.pt`, stuck
  at zero loss — confirmed via its own log before killing) rather than let it keep burning
  GPU time on a dead checkpoint.
- Fixed the root enabler for recovering from this class of bug: `trainer.py`'s
  `_log_meters_and_save_best_ckpts`/`save_best_meters` best-checkpoint mechanism already
  existed but was fully inert because `sam3.eval.coco_writer.PredictionDumper` (the `val_pin`
  meter) never implemented the required `is_better` method, so
  `getattr(meter, "is_better", None)` was always `None` and the improvement check was skipped
  unconditionally. Added `PredictionDumper.is_better()` (simple higher-is-better comparison,
  correct for all COCO AP/AR variants it produces) and set
  `trainer.checkpoint.save_best_meters: ["val_pin/detection"]` in
  `sam3/train/configs/full_train_config.yaml`. Going forward, a
  `checkpoint_val_pin_detection_<metric>.pt` is preserved permanently the first time each
  tracked metric improves, so a collapse like this one can no longer silently erase the last
  known-good state — recommend using `checkpoint_5.pt` (last confirmed-good, AP 0.94) instead
  of `checkpoint.pt`/`checkpoint_10.pt` for inference until the root cause is found.
- Did this work on a new branch `fix/best-checkpoint-saving` off `main`, kept independent from
  the same-day `docs/hpc-bsub-escaping-notes` branch (unrelated bsub scripting fix).

Debugged a second NaN-loss crash on `experiments/5cat_matrix_chip_v2` (job `237165894`) after
the 2026-07-16 loss-guard fix, found it only covered two of nine active loss terms, patched
the remaining gap, added permanent instrumentation, and captured the full investigation in a
new `loss-debugging` skill so future NaN crashes don't require re-tracing the whole loss stack.

- Root-caused that the 2026-07-16 fix only added `nan_to_num` guards to `dice_loss()` and `sigmoid_focal_loss()` in `sam3/train/loss/loss_fns.py` — covering `loss_mask`, `loss_dice`, `presence_loss`, `loss_semantic_seg`, `loss_semantic_dice`. Cross-checked the full active `weight_dict` in `sam3/train/configs/full_train_config.yaml` and found `loss_ce` (`IABCEMdetr`, weight 20.0, raw `F.binary_cross_entropy_with_logits`) and `loss_bbox`/`loss_giou` (`Boxes`, weights 5.0/2.0, raw `F.l1_loss` + GIoU reduction) had no NaN/Inf protection at all.
- Added `nan_to_num(nan=0.0, posinf=10.0, neginf=0.0)` guards to `IABCEMdetr.get_loss`'s combined `loss_bce` and to `Boxes.get_loss`'s `loss_bbox`/`loss_giou`, matching the existing `(Sy)`-tagged guard pattern.
- Added a permanent non-finite-loss-term logger (`_log_non_finite_loss_terms`) wired into `Sam3LossWrapper.compute_loss()` (both the `o2o` and `o2m` loss calls) and the semantic-seg loss call in `sam3/train/loss/sam3_loss.py`, so any future NaN/Inf is logged with the exact loss_fn class + key + `is_aux`/`o2m` context before it reaches `trainer.py`'s top-level crash check — kept in place permanently per explicit decision, not removed after this investigation.
- Confirmed the huge wall of `OSError: [Errno 16] Device or resource busy: pymp-*` tracebacks in `.err` logs is benign multiprocessing DataLoader worker temp-dir cleanup noise, unrelated to the NaN — the real failure was the `FloatingPointError: Loss is nan` at the end of the traceback.
- Added `.github/scripts/check_dataset_degenerate_samples.py`: a fast, CPU-only, no-GPU COCO-dataset audit (zero-area boxes, empty segmentations, orphan annotations, missing image files, optional NaN/Inf pixel check) to check for a bad sample before burning multi-hour GPU jobs on a NaN investigation. Smoke-tested against a synthetic COCO json with injected issues.
- Added `.github/skills/loss-debugging/SKILL.md` documenting everything already fixed/checked/still-open in the loss stack, exact file/line locations, the crash point, the instrumentation choke point, and the pymp red herring — so a future NaN investigation starts with full context instead of re-deriving it.
- Did all of the above on a new branch `fix/loss-nan-ce-bbox-guards` rather than the previous working branch.
- Validated the fix: ran `check_dataset_degenerate_samples.py` against the real merged dataset (train: 5248 images/16016 annotations, test: 3331 images/6866 annotations) — 0 issues found in both splits. Resubmitted training as job `237436875` (`gpu` queue, 3 GPUs — `gpu_compute` had PEND=33, above the known threshold), which auto-resumed from `checkpoints/checkpoint.pt` at epoch 8 (the previous crash point) and completed the full remaining schedule through `max_epochs: 10` with LSF status `DONE`/"Successfully completed." Grepped both the `.out` and `.err` logs directly on HPC for `FloatingPointError`, `Loss is nan`, and the new non-finite-loss-term instrumentation message — zero matches in either file, confirming no loss term went non-finite even once. Pushed the branch to `origin` (GitLab MR link provided; no `.gitlab/merge_request_templates/` exists in this repo).

Added a new "bondwire" category set (Thick Wire, Thin Wire, Gate Pad) to the merged SAM3
dataset and started fine-tuning an 8-category model, on branch
`feat/bondwire-8cat-dataset-finetune`.

- Source data: `/home/hpc_storage/data/projects/userB/data/bent-wire/Thin_multicat` (662
  train) and `Thin_multicat_test` (113 test) — flat `<name>.png` + `<name>_mask.png` pairs,
  each mask a **multi-class label map** (pixel value 1 = Thick Wire, 2 = Thin Wire, 3 = Gate
  Pad, 0 = background), confirmed via a new mask-inspection step rather than the user's
  initial (incorrect) filename-substring theory.
- Added `.github/scripts/inspect_mask.sh` + `.github/skills/inspect-mask-format/SKILL.md`:
  a one-shot mask format/pixel-value auto-detector (binary vs multi-class label map vs RGB
  color-coded), wired into `coco-format-conversion` as an optional Step 0, so future
  multi-category conversions need one targeted question instead of a multi-round-trip
  back-and-forth.
- Added `sam3_fine_tune/preprocessing/split_multiclass_masks.py` +
  `.github/scripts/run_split_multiclass_masks.sh`: splits a multi-class label mask into
  per-category binary masks scaled to 0/255 (raw label values like `1`/`2`/`3` would
  otherwise be silently zeroed out by `coco_format.py`'s internal
  `cv2.threshold(mask, 127, 255, THRESH_BINARY)`), staged into per-category `images/`+`masks/`
  folders. All 662/113 samples contained all 3 categories.
- Converted each category/split to COCO format via `coco-format-conversion`: thick_wire
  (662 train / 4810 ann, 113 test / 636 ann), thin_wire (662 / 1997, 113 / 277), gate_pad
  (662 / 1843, 113 / 226).
- Copied the existing 5-category shared dataset
  (`userA/sam3_fine_tune/asset/dataset`) to a new path under userB's own project space
  (`userB/sam3_data/5_plus_bondwire_dataset`) rather than editing the shared original in
  place, then appended the 3 new categories via `merge-category-dataset`'s
  `run_append_dataset.sh` (prefixes `g`/`h`/`i`, following on from the existing `a`-`f`).
  Final merged dataset: 8 categories, train 7234 images/24666 annotations, test 3670
  images/8005 annotations.
- Updated `full_train_config.yaml` via `finetune-merged-dataset`: `pin_dir_root` ->
  the new dataset, `experiment_log_dir` -> `experiments/8cat_bondwire`,
  `category_chunk_size` 5 -> 8 (train + val).
- First submit at `train_batch_size=6` (the prior 5-category known-stable value) OOM'd on a
  44GB card — the `category_chunk_size` 5->8 increase raises per-step memory (more
  category-query pairs processed per image per step), not the batch size or image count
  alone. Reduced `train_batch_size` to `4` and resubmitted (job `237753606`, `gpu` queue,
  `num=1:gmem=40G`); confirmed running stably with decreasing loss and no OOM.

## 2026-07-16

Debugged repeated CUDA OOM crashes while fine-tuning on the 5-category merged dataset and
corrected the `finetune-merged-dataset` skill's queue/OOM guidance based on real test
results — the initial "prefer more/bigger GPUs over shrinking batch size" strategy turned
out to be wrong for this config.

- Reduced `sam3/train/configs/full_train_config.yaml` `train_batch_size` from 20 to 6 — the confirmed-stable value for the 5-category (`category_chunk_size=5`) / `resolution=1008` config. `20` OOM'd on a 22GB card and `12` OOM'd even on a 44GB card, showing this was a real per-batch memory problem, not a queue/hardware-luck problem.
- Corrected `.github/skills/finetune-merged-dataset/SKILL.md`: added a step 0 pre-flight `bqueues` PEND-count check before submitting (calibration data: `PEND=33` on `gpu_compute` was still `PEND` after 4.5 minutes, with `PEND` climbing to 41, not shrinking); replaced the "prefer multi-GPU over shrinking batch size" guidance with a validated OOM fix order (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → reduce `train_batch_size` → reduce `scratch.resolution` as a last resort); explicitly excluded `gradient_accumulation_steps` as an OOM mitigation (confirmed not implemented correctly in this trainer).
- Documented a `resolution` fallback table (ViT `patch_size=16`, so values must stay multiples of 16): `896` (~11% linear / ~21% token-memory cut, try first) → `784` → `672`, in order of smallest quality impact first. Not applied yet — `train_batch_size=6` alone was sufficient.
- Fixed a NaN/Inf gap in `sam3/train/loss/loss_fns.py`'s `sigmoid_focal_loss()`: the default Triton fast-path (`triton=True`, `reduce=True`, `loss_on_multimask=False` — the path `Masks.get_loss()` actually uses for `loss_mask`) returned early before the existing `nan_to_num`/`max(num_boxes, 1)` safety net, so an all-background mask or an unstable logit could still produce a NaN/Inf loss and crash training via `trainer.py`'s `math.isfinite(loss.item())` check (observed as `Loss is nan, attempting to stop training` in job `236913741`'s log). Added the same `nan_to_num` guard and zero-division guard to this fast path so it matches the other (already-protected) branches.
- Added `append_images_to_existing_category()` to `nbs/02_merge_dataset.ipynb` (with markdown explanation + tempfile-based test cell covering the happy path and an unknown-category edge case) to fix a documented gap: `append_dataset()` always minted a brand-new `category_id`, which would silently duplicate a category if re-run for more images of an already-merged class. The new function looks up the category by name and reuses its id, raising if not found. Added the matching `.github/scripts/run_append_images_to_existing_category.sh` one-shot HPC script, and documented the workflow as "Option C" in `.github/skills/merge-category-dataset/SKILL.md` (including the `coco_format.py` image/mask filename-stem-matching gotcha for patch-style datasets with mismatched `_ch0`/`_mask` suffixes).
- Used the new workflow to add 1008 train + 225 test additional `matrix_chip` images (from `curated8bit/train_patches`) into the shared merged dataset at `/home/hpc_storage/data/projects/userA/sam3_fine_tune/asset/dataset`, reusing `category_id=5` (`matrix_chip`) — category count unchanged (still 5), `train` grew to 5248 images/16016 annotations, `test` grew to 3331 images/6866 annotations. Backed up the dataset's `annotation.json` files first (shared asset).
- Sped up `run_append_images_to_existing_category.sh`: it used to run the image-copy/merge loop synchronously inline in the login-node ssh session; extracted that logic into a standalone `.github/scripts/append_images_to_existing_category.py` and now submit it as a `bsub` job on the HPC CPU queue instead (same pattern as `coco-format-conversion`), since appending hundreds/thousands of images is CPU/IO work that shouldn't run on the login node. Verified against a throwaway dummy dataset before use.
- Fixed non-unique COCO annotation IDs in `nbs/01_coco_format.ipynb` `mask_to_coco()` and its exported mirror `sam3_fine_tune/preprocessing/coco_format.py`: the previous `"id": len(annotations) + 1` restarted at `1` for every image, so annotation IDs collided across images and would silently corrupt pycocotools-based evaluation on raw (pre-merge) COCO files. Now uses the running `annotation_id` passed from `form_coco_dataset()` and increments it per contour. Added an explanation markdown cell + a tempfile-based test pair in the notebook covering global uniqueness across two images and the empty-mask edge case.
- Updated `sam3/train/configs/full_train_config.yaml` `experiment_log_dir` to `experiments/5cat_matrix_chip_v2` for the current fine-tuning run (job `237165894` on the `gpu` queue).

## 2026-07-15

Added a `configure-repo` base skill and split the combined append-category-and-finetune
workflow into a data-merge skill and a training skill, then used them end-to-end to add a
5th category (`matrix_chip`) to the shared merged dataset and launch fine-tuning.

- Added `.github/skills/configure-repo/SKILL.md` and `.github/scripts/hpc_repo_context.sh`: resolves repo root, activates the venv, and detects python/torch/cuda versions plus the matching `cuda/X.Y` module in a single ssh round-trip. `hpc-gpu-submit`, `hpc-cpu-submit`, and `coco-format-conversion` now reference it instead of duplicating repo-root/CUDA discovery.
- Split the (unreleased) combined append-category-and-finetune workflow into `.github/skills/merge-category-dataset/SKILL.md` (data-only: validate, convert to COCO, append) and `.github/skills/finetune-merged-dataset/SKILL.md` (config update + non-blocking batch training submission + eval), so each is faster to run standalone.
- Added `.github/scripts/check_new_category_data.sh`: single-round-trip pre-flight validator for a new-category data directory (image/mask counts, existing/missing `annotation.json`, and any leftover COCO json files with mismatched category names).
- Added `.github/scripts/run_append_dataset.sh`: appends a new category into an existing merged COCO dataset by reading only the `#| export`-tagged cells of `nbs/02_merge_dataset.ipynb` directly (stdlib `json`, no `nbdev-export`/`nbdev_export` involved, since this repo's nbdev config is still `settings.ini` and not migrated to `pyproject.toml`).
- Updated `sam3/train/configs/full_train_config.yaml`: `category_chunk_size` 4→5 and `experiment_log_dir` pointed at a new experiment directory, for fine-tuning on the merged dataset with the newly added `matrix_chip` category (5 total categories: `et_pin`, `eberhard_pin`, `wuxi_pin`, `war_crack`, `matrix_chip`).
- Added `experiments/` to `.gitignore` (training run logs/checkpoints/tensorboard events are runtime artifacts, not tracked).
- Reordered this changelog to reverse-chronological (newest date on top) after noticing the 2026-07-15 entry had been appended below 2026-07-14 instead of above it.
- Updated `git-last-change-summary` skill to enforce reverse-chronological `CHANGELOG.md` ordering going forward: new dated entries are inserted before the first existing `## YYYY-MM-DD` heading (never appended at the end), with a `head -5 CHANGELOG.md` validation step before committing.

## 2026-07-14

Updated project documentation and HPC skills.

- Updated `README.md` table formatting, section numbering, and layout notes.
- Added `.github/skills/` (HPC Jupyter Connect, HPC GPU Submit, Git Last Change Summary) and `.github/scripts/discover_repo_root.sh` for dynamic repo discovery on HPC.
- Added `.github/prompts/hpc-submit.prompt.md`.
- Added `ssh-hpc-connect` skill with SSH key/config setup checklist for HPC login nodes (MUC and VIH).
- Added SSH connectivity prerequisites to `hpc-jupyter-connect`, `hpc-gpu-submit`, `hpc-cpu-submit`, and `git-last-change-summary`.
- Added `coco-format-conversion` skill to convert image/mask pairs to COCO format on the HPC CPU queue.
- Replaced `cpu_c` alias in `sam3_fine_tune/preprocessing/coco_format.sh` with the full `bsub` command.
- Validated `coco-format-conversion` skill on the Cegled Econo dataset: produced `annotations.json` with 244 images and 2,199 annotations.
- Added `.github/scripts/run_coco_conversion.sh` one-shot helper: discovers the repo root, activates the venv, submits the CPU conversion job, and validates the output JSON in a single call.
- Fixed the manual-fallback `bsub` command in `coco-format-conversion` to call `coco_format.py` by absolute path instead of `cd <dir> && python coco_format.py`, since the `cd` could silently fail to apply once the job was dispatched to a compute node.
- Added a self-healing repo-root cache (`~/.cache/hpc_repo_root/<project>`) to `discover_repo_root.sh` and `run_coco_conversion.sh`, cutting repeat repo-root lookups from ~70s to ~5ms (validated on HPC; `FORCE_REFRESH=1` bypasses it).

