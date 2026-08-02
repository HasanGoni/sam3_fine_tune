#!/usr/bin/env bash
# Convert a directory of images/masks to COCO format on the Acme HPC.
#
# Runs directly on the HPC login node using the project virtualenv.
# This is faster than bsub for the typical COCO conversion workload.
#
# Usage (run from your local machine):
#   ssh hpc_muc 'bash -s -- <base_dir> <category_name> <category_id> [file_name] [project_name]' \
#     < .github/scripts/run_coco_conversion.sh
#
# Example:
#   ssh hpc_muc 'bash -s -- /home/hpc_storage/data/projects/userB/data/cegled_econo/47834/curated8bit/train matrix_chip 1' \
#     < .github/scripts/run_coco_conversion.sh
#
# Expected structure under <base_dir>:
#   <base_dir>/
#   ├── images/
#   └── masks/
#
# Output:
#   <base_dir>/<file_name>.json   (default: annotation.json)
#   The images/ directory is kept untouched.

set -e

# Parse arguments after the '--' separator when run via 'bash -s'.
BASE_DIR="${1:-}"
CATEGORY_NAME="${2:-}"
CATEGORY_ID="${3:-}"
FILE_NAME="${4:-annotation}"
PROJECT_NAME="${5:-sam3_fine_tune}"

usage() {
    cat <<EOF
Usage: ssh hpc_muc 'bash -s -- <base_dir> <category_name> <category_id> [file_name] [project_name]' < run_coco_conversion.sh

  base_dir       directory containing images/ and masks/ subdirectories
  category_name  e.g. matrix_chip
  category_id    e.g. 1
  file_name      output JSON base name (default: annotation)
  project_name   repo directory name (default: sam3_fine_tune)
EOF
}

if [[ -z "${BASE_DIR}" || -z "${CATEGORY_NAME}" || -z "${CATEGORY_ID}" ]]; then
    usage >&2
    exit 1
fi

if [[ ! "${BASE_DIR}" =~ ^/ ]]; then
    echo "ERROR: base_dir must be an absolute HPC path. Got: ${BASE_DIR}" >&2
    exit 1
fi

HPC_USER="${USER}"
SEARCH_ROOT="/home/hpc_storage/data/projects/${HPC_USER}"

if [[ -z "${HPC_USER}" ]]; then
    echo "ERROR: USER is empty." >&2
    exit 1
fi

# Discover the repo root on HPC, using a per-user cache to skip the slow
# `find` scan on repeat runs. Cache is validated on every read (path must
# still exist and contain pyproject.toml); set FORCE_REFRESH=1 to bypass it.
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

IMG_DIR="${BASE_DIR}/images"
MASK_DIR="${BASE_DIR}/masks"
OUTPUT_DIR="${BASE_DIR}"

if [[ ! -d "${IMG_DIR}" ]]; then
    echo "ERROR: images/ directory not found: ${IMG_DIR}" >&2
    exit 1
fi

if [[ ! -d "${MASK_DIR}" ]]; then
    echo "ERROR: masks/ directory not found: ${MASK_DIR}" >&2
    exit 1
fi

IMG_COUNT=$(find "${IMG_DIR}" -maxdepth 1 -type f | wc -l)
MASK_COUNT=$(find "${MASK_DIR}" -maxdepth 1 -type f | wc -l)

echo "Found ${IMG_COUNT} images and ${MASK_COUNT} masks"

if [[ "${IMG_COUNT}" -eq 0 || "${MASK_COUNT}" -eq 0 ]]; then
    echo "ERROR: images/ and masks/ must not be empty" >&2
    exit 1
fi

# Activate the project virtualenv on the login node first, then submit to bsub.
# The interactive bsub job inherits this environment, so it does not need to
# re-activate the venv itself.
# Use an absolute path to coco_format.py rather than `cd ... && python ...`:
# the `cd` can silently fail to apply once the job is dispatched to a compute
# node, causing "python: can't open file '.../coco_format.py'" because it
# looks in $REPO_ROOT instead of the preprocessing/ subdirectory.
ACTIVATE="source ${REPO_ROOT}/.venv/bin/activate"
PYTHON_CMD="python ${REPO_ROOT}/sam3_fine_tune/preprocessing/coco_format.py --mask_dir ${MASK_DIR} --img_dir ${IMG_DIR} --output_dir ${OUTPUT_DIR} --file_name ${FILE_NAME} --category_name ${CATEGORY_NAME} --category_id ${CATEGORY_ID}"

echo "Activating project venv and submitting to HPC CPU queue..." >&2
${ACTIVATE} && bsub -Is -R "type=X64LIN && osrel>=70 && ui=pytest rusage[mem=50G]" -n 30 "${PYTHON_CMD}"

JSON_PATH="${OUTPUT_DIR}/${FILE_NAME}.json"

if [[ ! -f "${JSON_PATH}" ]]; then
    echo "ERROR: Expected output file not found: ${JSON_PATH}" >&2
    exit 1
fi

echo ""
echo "=== COCO conversion completed ==="
echo "Output: ${JSON_PATH}"
python3 -c "import json; d=json.load(open('${JSON_PATH}')); print('images:', len(d.get('images', []))); print('annotations:', len(d.get('annotations', []))); print('categories:', d.get('categories', [])); print('===================================')"
