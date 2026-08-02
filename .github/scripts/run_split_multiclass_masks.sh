#!/usr/bin/env bash
# Split a multi-class label mask dataset (one mask file per image, several
# pixel values each representing a different category) into per-category
# images/+masks/ staging folders, ready for coco-format-conversion.
#
# Runs sam3_fine_tune/preprocessing/split_multiclass_masks.py on the HPC CPU
# queue (bsub), same submission pattern as run_coco_conversion.sh.
#
# Usage (run from your local machine):
#   ssh hpc_muc 'bash -s -- <src_dir> <output_root> <label_map> [project_name]' \
#     < .github/scripts/run_split_multiclass_masks.sh
#
# Example:
#   ssh hpc_muc 'bash -s -- /home/hpc_storage/data/projects/userB/data/bent-wire/Thin_multicat \
#       /home/hpc_storage/data/projects/userB/sam3_data/bondwire_staging/train \
#       "1=Thick Wire,2=Thin Wire,3=Gate Pad"' \
#     < .github/scripts/run_split_multiclass_masks.sh
#
# Expected structure under <src_dir> (flat, no images/masks subfolders):
#   <src_dir>/
#   ├── <name>.png
#   ├── <name>_mask.png        # label values = category ids from <label_map>
#   └── <name>_mask_vis.png    # ignored
#
# label_map format: "<value>=<category name>,<value>=<category name>,..."
#   e.g. "1=Thick Wire,2=Thin Wire,3=Gate Pad"
#
# Output: <output_root>/<category_slug>/{images,masks}/ per category, with
# masks scaled to {0,255} (required by coco_format.py's internal threshold).

set -e

SRC_DIR="${1:-}"
OUTPUT_ROOT="${2:-}"
LABEL_MAP="${3:-}"
PROJECT_NAME="${4:-sam3_fine_tune}"

usage() {
    cat <<EOF
Usage: ssh hpc_muc 'bash -s -- <src_dir> <output_root> <label_map> [project_name]' < run_split_multiclass_masks.sh

  src_dir       flat directory containing <name>.png + <name>_mask.png pairs
  output_root   root dir to stage per-category images/+masks/ folders
  label_map     e.g. "1=Thick Wire,2=Thin Wire,3=Gate Pad"
  project_name  repo directory name (default: sam3_fine_tune)
EOF
}

if [[ -z "${SRC_DIR}" || -z "${OUTPUT_ROOT}" || -z "${LABEL_MAP}" ]]; then
    usage >&2
    exit 1
fi

if [[ ! "${SRC_DIR}" =~ ^/ || ! "${OUTPUT_ROOT}" =~ ^/ ]]; then
    echo "ERROR: src_dir and output_root must be absolute HPC paths." >&2
    exit 1
fi

HPC_USER="${USER}"
SEARCH_ROOT="/home/hpc_storage/data/projects/${HPC_USER}"

if [[ -z "${HPC_USER}" ]]; then
    echo "ERROR: USER is empty." >&2
    exit 1
fi

CACHE_DIR="${HOME}/.cache/hpc_repo_root"
CACHE_FILE="${CACHE_DIR}/${PROJECT_NAME}"
REPO_ROOT=""

if [[ "${FORCE_REFRESH:-0}" != "1" && -f "${CACHE_FILE}" ]]; then
    CACHED=$(cat "${CACHE_FILE}")
    if [[ -n "${CACHED}" && -d "${CACHED}" && -f "${CACHED}/pyproject.toml" ]]; then
        REPO_ROOT="${CACHED}"
    fi
fi

most_recently_modified() {
    local latest="" latest_mtime=0 mtime path
    while IFS= read -r path; do
        [[ -z "${path}" ]] && continue
        mtime=$(stat -c '%Y' "${path}" 2>/dev/null || stat -f '%m' "${path}" 2>/dev/null || echo 0)
        if [[ "${mtime}" -gt "${latest_mtime}" ]]; then
            latest_mtime="${mtime}"
            latest="${path}"
        fi
    done
    echo "${latest}"
}

if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT=$(find "${SEARCH_ROOT}" -maxdepth 4 -type d -name "${PROJECT_NAME}" 2>/dev/null | most_recently_modified)
    if [[ -z "${REPO_ROOT}" ]]; then
        REPO_ROOT=$(find "${SEARCH_ROOT}" -maxdepth 5 -type f -name "pyproject.toml" 2>/dev/null \
            | while read -r f; do dirname "${f}"; done | most_recently_modified)
    fi
    if [[ -n "${REPO_ROOT}" ]]; then
        mkdir -p "${CACHE_DIR}"
        echo "${REPO_ROOT}" > "${CACHE_FILE}"
    fi
fi

if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: Could not find project '${PROJECT_NAME}' under ${SEARCH_ROOT}" >&2
    exit 1
fi

echo "Repo root: ${REPO_ROOT}"

if [[ ! -d "${SRC_DIR}" ]]; then
    echo "ERROR: src_dir not found: ${SRC_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

ACTIVATE="source ${REPO_ROOT}/.venv/bin/activate"
PYTHON_CMD="python ${REPO_ROOT}/sam3_fine_tune/preprocessing/split_multiclass_masks.py --src_dir ${SRC_DIR} --output_root ${OUTPUT_ROOT} --label_map '${LABEL_MAP}'"

echo "Activating project venv and submitting to HPC CPU queue..." >&2
${ACTIVATE} && bsub -Is -R "type=X64LIN && osrel>=70 && ui=pytest rusage[mem=50G]" -n 30 bash -c "${PYTHON_CMD}"

echo ""
echo "=== Split completed ==="
echo "Staged categories under: ${OUTPUT_ROOT}"
ls -la "${OUTPUT_ROOT}"
