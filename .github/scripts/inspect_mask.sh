#!/usr/bin/env bash
# One-shot mask format / pixel-value auto-detector.
#
# Reads a SINGLE mask image and reports its shape, dtype, and unique
# pixel values, then classifies it automatically as one of:
#   - binary mask (single category, values are {0,255} or {0,1})
#   - multi-class label map (small integer values, e.g. {0,1,2,3})
#   - RGB/multi-channel color-coded mask (distinct colors per category)
#   - unknown (ask the user)
#
# For a multi-class label map, it prints the sorted non-zero label values
# found so the caller only needs to confirm ONE thing with the user (which
# category name each label number maps to) instead of re-discovering the
# whole mask format from scratch each time.
#
# Usage:
#   ssh hpc_muc 'bash -s -- <mask_path> [project_name]' < .github/scripts/inspect_mask.sh
#
# Reuses the same REPO_ROOT cache as hpc_repo_context.sh (run configure-repo
# first if the cache may be stale/missing).

set -e

MASK_PATH="$1"
PROJECT_NAME="${2:-sam3_fine_tune}"
HPC_USER="${USER}"

if [[ -z "${MASK_PATH}" ]]; then
    echo "ERROR: usage: inspect_mask.sh <mask_path> [project_name]" >&2
    exit 1
fi

if [[ ! -f "${MASK_PATH}" ]]; then
    echo "ERROR: mask file not found: ${MASK_PATH}" >&2
    exit 1
fi

CACHE_FILE="${HOME}/.cache/hpc_repo_root/${PROJECT_NAME}"
REPO_ROOT=""
if [[ -f "${CACHE_FILE}" ]]; then
    CACHED=$(cat "${CACHE_FILE}")
    if [[ -n "${CACHED}" && -d "${CACHED}" ]]; then
        REPO_ROOT="${CACHED}"
    fi
fi

if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT=$(find "/home/hpc_storage/data/projects/${HPC_USER}" -maxdepth 4 -type d -name "${PROJECT_NAME}" 2>/dev/null | head -1)
fi

if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "ERROR: could not resolve repo/venv for project '${PROJECT_NAME}'. Run configure-repo first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

python - "${MASK_PATH}" <<'PYEOF'
import sys
import cv2
import numpy as np

path = sys.argv[1]
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if img is None:
    print(f"ERROR: could not read {path}")
    sys.exit(1)

print(f"path: {path}")
print(f"shape: {img.shape}")
print(f"dtype: {img.dtype}")
channels = 1 if img.ndim == 2 else img.shape[2]
print(f"channels: {channels}")

if channels == 1:
    vals, counts = np.unique(img, return_counts=True)
    preview = ", ".join(f"{v}={c}" for v, c in list(zip(vals.tolist(), counts.tolist()))[:20])
    print(f"unique values ({len(vals)}): {preview}")
    nz = vals[vals != 0]
    valset = set(vals.tolist())
    if len(vals) <= 2 and (valset <= {0, 255} or valset <= {0, 1}):
        print("CLASSIFICATION: binary mask (single category)")
    elif len(vals) <= 20 and vals.max() < 50:
        print(f"CLASSIFICATION: multi-class label map. Non-zero label values: {nz.tolist()}")
        print("ACTION: ask the user to confirm the category name for each label value above.")
    else:
        print("CLASSIFICATION: unknown - grayscale but doesn't match binary or small-int label-map pattern. Ask the user.")
else:
    pixels = img.reshape(-1, channels)
    uniq = np.unique(pixels, axis=0)
    print(f"unique colors ({len(uniq)}): {uniq[:10].tolist()}{'...' if len(uniq) > 10 else ''}")
    if len(uniq) <= 20:
        print("CLASSIFICATION: RGB/multi-channel color-coded mask. Ask the user which color maps to which category.")
    else:
        print("CLASSIFICATION: unknown - too many distinct colors for a typical label mask. Ask the user.")
PYEOF
