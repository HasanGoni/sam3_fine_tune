---
name: diverse-wire-dataset
description: 'Build a category-rebalanced COCO wire dataset using DINOv3 (ConvNeXt) embedding diversity selection (farthest-point sampling). One category kept unchanged, others rebalanced to a target total count from a larger candidate pool.'
argument-hint: '[--config <path>]'
user-invocable: true
disable-model-invocation: false
---
# Diverse Wire Dataset

Build a 4-category (or more) COCO dataset where one category is copied through
unchanged and the others are rebalanced to a fixed total image count by
selecting the most *visually diverse* subset from a much larger candidate pool
(instead of a random subset), using DINOv3 image embeddings + farthest-point
(k-center greedy) sampling.

## When To Use

- You have a small/uneven training set for some categories but a much larger
  unlabeled-diversity source pool for the same categories (e.g. a patched
  dataset), and you want a bigger, more visually diverse subset rather than
  just concatenating everything or sampling randomly.
- You want to keep one category's existing curated split completely unchanged
  while growing the others.

## Components

- `sam3_fine_tune/preprocessing/diversity/dinov3_embedder.py` — `DINOv3Embedder`
  class. Loads a DINOv3 **ConvNeXt** backbone (via `timm`) from a local
  checkpoint at `/home/hpc_storage/Models/dinov3_models/` and extracts
  L2-normalized embeddings. ConvNeXt is used (not the DINOv3 ViT variants)
  because its checkpoint keys map cleanly onto `timm`'s built-in
  `convnext.checkpoint_filter_fn` — the ViT variants have several key-name
  mismatches (`storage_tokens`/`reg_token`, `gamma_1`/`ls1.gamma`, RoPE,
  `qkv.bias_mask`) that make them impractical to load via timm as-is.
  `embed_batch()` loads each batch's images **concurrently** via a
  `ThreadPoolExecutor` (I/O-bound work, so the GIL isn't a bottleneck) rather
  than one-at-a-time — see Known Issues for why this matters beyond raw
  throughput.
- `sam3_fine_tune/preprocessing/diversity/select_diverse.py` —
  `farthest_point_sample(embeddings, k, seed)`: greedy k-center diversity
  selection over L2-normalized embeddings (cosine distance = `1 - dot`).
- `sam3_fine_tune/preprocessing/diversity/embedding_cache.py` —
  `EmbeddingCache`: a simple on-disk `{absolute_image_path: embedding}` cache
  (`.npz`). Keyed by path so re-running the build with a different
  `target_count` (e.g. a small dry run, then the real large run) reuses
  already-computed embeddings instead of recomputing the whole candidate pool.
  Default location:
  `/home/hpc_storage/data/projects/userB/sam3_data/_diverse_embedding_cache/<model_name>_cache.npz`
  (override via `model.embedding_cache` in the config). Saved incrementally
  every `model.cache_save_every_batches` batches (default 20) **and** after
  each category, so an interrupted/killed job loses at most a few batches of
  embedding work, not the whole category.
- `sam3_fine_tune/preprocessing/build_diverse_wire_dataset.py` — CLI
  orchestrator. See Config schema + Usage below.

## Split-photo contradictory-negatives guard

This repo previously hit a serious bug (documented in repo memory) where a
source photo containing several target categories got split into multiple
COCO image entries, each asserting the OTHER categories were absent. This
script avoids that: every selected image keeps **all** of its annotations
that belong to **any** of the target categories, not just the category that
triggered its selection.

## Config schema

```yaml
model:
  name: convnext_tiny                # timm model name
  checkpoint: /home/hpc_storage/Models/dinov3_models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
  batch_size: 64
  image_size: 224
  embedding_cache: /path/to/cache.npz   # optional, default under sam3_data/_diverse_embedding_cache/

source:                              # large candidate pool for diverse categories
  root: /home/hpc_storage/data/projects/userB/sam3_data/<patched_dataset>
  splits: [train, test]

fixed_categories:                    # copied through unchanged, original split kept
  - name: et_pin                     # one entry per category; can span several source roots
    source_root: /home/hpc_storage/data/projects/userB/sam3_data/<curated_dataset>
    splits: [train, test]
  - name: thick_wire_panda
    source_root: /home/hpc_storage/data/projects/userB/sam3_data/<curated_dataset>
    splits: [train, test]
  # ... one entry per unchanged category. Entries sharing the same
  # (source_root, splits) have that root's annotation file parsed only once.
  # The older singular `fixed_category: {name, source_root, splits}` (one
  # category only) is still accepted for backward compatibility.

diverse_categories:                  # rebalanced via diversity selection
  - name: thin_wire_panda
    target_count: 1000                # TOTAL selected count (before train/test split)
  - name: thick_wire_HD
    target_count: 1000
  - name: thin_wire_HD
    target_count: 1000

category_order:                      # final category id order (1-based). Values here are
                                      # SOURCE names (used for matching annotations).
  - et_pin
  - thick_wire_panda
  - thin_wire_panda
  - thick_wire_HD
  - thin_wire_HD

category_display_names:              # optional: override the OUTPUT category "name" field
                                      # per source name, without touching any matching logic.
                                      # Useful when category names double as text prompts
                                      # (SAM3 is open-vocabulary) and should read as natural
                                      # language rather than underscore_joined tokens.
  thick_wire_panda: "thick wire panda"
  thin_wire_panda: "thin wire panda"
  thick_wire_HD: "thick wire HD"
  thin_wire_HD: "thin wire HD"

split:
  train_ratio: 0.8
  seed: 42

output:
  root: /home/hpc_storage/data/projects/userB/sam3_data/<output_dataset_name>
```

Notes:
- If a diverse category's candidate pool is smaller than `target_count`, all
  candidates are kept (no embedding/sampling needed for that category).
- Each diverse category's pool excludes any image file already consumed by
  any fixed category or an earlier diverse category (processed in
  `diverse_categories` list order), so the same photo cannot be selected
  twice under different categories. A diverse category's FINAL total image
  count can still end up higher than `target_count` if some fixed-category
  images happen to also contain that diverse category's annotations
  (kept via the contradictory-negatives guard below) — `target_count` caps
  how many *additional* images get picked from the diverse source pool, not
  the category's final total.
- `category_display_names` only affects the output COCO categories list's
  `"name"` field. `category_order`, `fixed_categories[].name`, and
  `diverse_categories[].name` must always use the SOURCE category names
  (as they appear in the input annotation files) for matching to work.

## Procedure

1. **Prerequisite:** run `configure-repo` skill (or just confirm) to resolve
   `REPO_ROOT` and `CUDA_MODULE` on HPC.
2. **IMPORTANT — never use `/tmp` for configs/output/logs when the job will
   run via `bsub`.** `/tmp` is **local to each compute node**, not shared with
   the login node or between nodes. A config/output path under `/tmp` will
   silently not be visible once the job lands on a different node — this
   cost a full debug cycle once (see Known Issues). Always put the config,
   output root, and log file under the shared network mount (e.g. under
   `sam3_data/` or the repo itself).
3. **Do a small dry run first** (e.g. `target_count: 8` per diverse category)
   to validate the full pipeline end-to-end before spending GPU time on the
   full run — checkpoint loading, category ordering, fixed-category
   passthrough counts, and the assembled COCO JSON/metadata.
4. **Submit as a non-blocking (detached) GPU batch job — NOT `bsub -Is`.**
   Embedding is model inference — legitimate GPU-queue work, unlike plain CPU
   preprocessing — but a run over tens of thousands of images can take an
   hour or more, and an interactive `-Is` job unnecessarily ties up a client
   terminal for that whole time. Use a plain `bsub` (no `-Is`) with `-o`/`-e`
   log files so the job is fully detached from the submitting shell:

   ```bash
   ssh hpc_muc -t 'source /ifxhome/$USER/.bashrc && \
     source ${REPO_ROOT}/.venv/bin/activate && \
     module load cuda/12.8 && \
     cd ${REPO_ROOT}/sam3_fine_tune/preprocessing && \
     bsub -q gpu -gpu "num=1:j_exclusive=yes:gmem=10G" \
       -R "select[ui==aiml_batch_training && osrel=80 && type==X64LIN] rusage[mem=32]" -n 2 \
       -o <shared_path>/job.out -e <shared_path>/job.err \
       "python -u build_diverse_wire_dataset.py --config <shared_path>/config.yaml \
        > <shared_path>/run.log 2>&1"'
   ```

   This immediately returns a job ID (e.g. `Job <242194702> is submitted to
   queue <gpu>.`) — do **not** simplify the `-R` select string (see Known
   Issues). Check `bqueues` PEND counts for `gpu` vs `gpu_compute` first and
   prefer whichever has less backlog (see `hpc-gpu-submit` skill).
   Poll status independently with `bjobs -w <job_id>` at any time — this is
   safe and doesn't affect the job.
5. **Poll the log file** (on the shared mount, not `/tmp`) rather than piping
   the live process through `tail` in the same command — `tail`-piping a live
   process makes progress invisible until the process exits or the pipe's
   buffer flushes.
6. **Validate output**: `<output_root>/{train,test}/annotation_merged.json` +
   auto-generated `annotation_merged_metadata.json` (per-category counts),
   and `<output_root>/selection_manifest.json` (pool sizes, selected/train/test
   counts per diverse category, embedder info).
7. **Scale up**: bump `target_count` to the real value and resubmit the same
   way — the embedding cache means only *new* candidate images (not already
   embedded during the dry run) need a fresh forward pass.
8. **Generate an overlay poster.** Use `--one-per-category` (with
   `--categories` listing the desired category names/order, using OUTPUT
   display names if `category_display_names` was used) to produce a single
   JPEG with one guaranteed representative row per category — more reliable
   than random `--n-samples` sampling for confirming every category has
   correct masks, especially once the dataset has many categories:

   ```bash
   python scripts/save_coco_overlay_poster.py \
     --coco-json <output_root>/train/annotation_merged.json \
     --one-per-category \
     --categories 'et_pin,thick wire panda,thin wire panda,thick wire HD,thin wire HD' \
     --output overlay_posters/<dataset_name>_all_categories.jpg
   ```

   `--category <name>` (singular, no `--one-per-category`) still works for a
   single-category poster with random `--n-samples` images, as before.

## Known Issues

- **`/tmp` is per-node local storage, not shared** (confirmed 2026-07-27):
  a config/output/log path under `/tmp` written from the login node is
  invisible from a `bsub`-dispatched compute node (and vice versa) — each
  node has its own local `/tmp`. Symptom: a job appears to run and exit
  successfully (echoes a completion marker after it), but the log file never
  updates and the output directory stays empty, because the actual script
  wrote to a *different node's* `/tmp`. Always use a path under the shared
  network mount.
- **A simplified `bsub -gpu` resource `-R` select string can silently give
  you a CPU-only environment inside an allocated GPU job** — `torch.cuda
  .is_available()` returned `False` with `-R "select[type==X64LIN]
  rusage[mem=32]"` but `True` with the full known-working string
  `select[ui==aiml_batch_training && osrel=80 && type==X64LIN]
  rusage[mem=200]`. Always use the validated full select string from
  `hpc-gpu-submit`'s repo memory notes, not a shortened one.
- **Login-node CPU throttling**: running the heavy JSON-parsing step (COCO
  files can be 1GB+ for large patched datasets) directly on the login node
  (not via `bsub`) can be throttled down to a small fraction of a core after
  a few minutes, making it look stuck. Always submit heavy preprocessing via
  `bsub` (CPU queue for pure JSON/IO work, GPU queue once embedding starts).
- **Serial image loading over a network mount can trigger the cluster's
  idle-GPU auto-kill policy** (root-caused 2026-07-27, see `hpc-gpu-submit`
  skill for the general policy: a GPU job is killed with `TERM_OWNER: job
  killed by owner` / exit 143 if utilization stays below ~5% for more than
  ~5 minutes continuously). Two dry-run jobs died with this exact signature
  after ~15-25 minutes each (`KeyboardInterrupt` inside `PIL.Image`'s
  `convert()`/`load()`/`decode()`). This was initially (incorrectly)
  suspected to be an SSH-session-tied kill from a `bsub -Is` submission, but
  switching to a fully detached `bsub` (no `-Is`, even wrapped in a detached
  `screen` session) still died the same way — proving it wasn't
  session-related. The real cause: `DINOv3Embedder.embed_batch()` originally
  loaded every image in a batch **serially** with `PIL.Image.open()`
  (~3.6-6s/batch on the network mount) before a quick GPU forward pass,
  leaving the GPU idle for most of each cycle — cumulatively past the
  5-minute threshold. **Fix**: load each batch's images concurrently via a
  `ThreadPoolExecutor(max_workers=8)` (I/O-bound, so threads — not
  multiprocessing — are sufficient and simpler). This cut batch time from
  ~3.6-6s down to ~0.7s (~1.4 it/s), and the dry run then completed
  successfully with no further kills. **Lesson**: `TERM_OWNER: job killed by
  owner` does not necessarily mean a session/user-initiated kill — check
  `bjobs -l`'s reported GPU utilization and cross-reference against
  documented idle-kill policies before assuming a connection issue,
  especially when the failure timing matches the policy's stated window
  rather than any observed disconnect.
- **`np.savez(path, ...)` always appends `.npz` if `path` doesn't already end
  with it** — a naive atomic-write temp path like `cache.npz.tmp` actually
  gets written to `cache.npz.tmp.npz`, so the subsequent `tmp_path.replace()`
  raises `FileNotFoundError`. Fix: make the temp path itself end in `.npz`
  (e.g. `<stem>_tmp.npz`), not `<name>.tmp`.
- **`np.load(..., allow_pickle=False)` cannot load an object-dtype array**
  (`ValueError: Object arrays cannot be loaded when allow_pickle=False`) — if
  you were tempted to pass `allow_pickle=True` to work around this, don't:
  store the path list as a fixed-width unicode array (`np.array(paths,
  dtype="U")`) instead, which loads fine under `allow_pickle=False` and
  avoids the pickle security surface entirely.
