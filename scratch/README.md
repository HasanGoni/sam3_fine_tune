# scratch/

This directory holds ad-hoc / temporary Python scripts created during
exploration, debugging, or one-off data audits. They are kept for reference
but are not part of the main package.

If you create a new scratch script, put it here rather than at the repo root.

## UNVEIL zero-shot evaluation (issue #18, 2026-11-22)

- `unveil_zero_shot_eval.py`: per-category pooled IoU of UNVEIL against SAM3's semantic path on a local COCO test split.
- `unveil_overlays.py`: three-panel overlays (ground truth, UNVEIL, SAM3 semantic path) for a few images.
- `unveil_contact_sheets.py`: 12-image sheets per category group (annotations outlined) to review labels by eye.

They run with the repo `.venv` and a clone of `yuhuan-wu/UNVEIL` on `PYTHONPATH` (`PYTHONPATH=<UNVEIL clone>:.`), pass
`--sam3` and `--bpe` explicitly (see "Local weights and data" in `CLAUDE.md`), and need a CUDA GPU. Paths in the sheets
script (dataset root, output folder) are hard-coded: edit them first. Results: issue #18.
