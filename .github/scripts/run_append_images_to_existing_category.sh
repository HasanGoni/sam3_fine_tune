#!/usr/bin/env bash
# Add more images to a category that ALREADY EXISTS in a merged COCO dataset on
# the Acme HPC, using `append_images_to_existing_category()` from
# nbs/02_merge_dataset.ipynb.
#
# This is the counterpart to run_append_dataset.sh: that script's
# `append_dataset()` always creates a brand-new category_id, which is wrong
# when you're just adding more images to a category you already merged in
# before (e.g. more matrix_chip samples) - it would silently create a
# duplicate category. `append_images_to_existing_category()` instead looks up
# the existing category by name and reuses its id, raising if the name isn't
# found (use run_append_dataset.sh for a genuinely new category).
#
# Unlike run_append_dataset.sh (which runs the merge inline in the login-node
# ssh session), this submits the actual copy/merge work as a bsub job to the
# HPC CPU queue (same pattern as run_coco_conversion.sh) - appending hundreds
# or thousands of images is CPU/IO work that shouldn't run synchronously on
# the login node.
#
# This does NOT run `nbdev-export` / `nbdev_export` and does not touch
# sam3_fine_tune/preprocessing/. Instead it submits
# append_images_to_existing_category.py, which reads the notebook's own
# `#| export`-tagged cells directly (stdlib `json` only) and exec's just those
# cells in-memory to get `append_images_to_existing_category`.
#
# Usage (run from your local machine):
#   ssh hpc_muc 'bash -s -- <new_dataset_path> <category_name> <prefix> <existing_output_path> [project_name]' \
#     < .github/scripts/run_append_images_to_existing_category.sh
#
# Example:
#   ssh hpc_muc 'bash -s -- /home/hpc_storage/data/projects/userB/data/cegled_econo/47834/curated8bit/train_patches/coco_staging matrix_chip f /home/hpc_storage/data/projects/userA/sam3_fine_tune/asset/dataset' \
#     < .github/scripts/run_append_images_to_existing_category.sh
#
# Expected structure:
#   <new_dataset_path>/{train,test}/images/ + annotation.json      (more images for an EXISTING category, already in COCO format)
#   <existing_output_path>/{train,test}/images/ + annotation.json  (existing merged dataset, modified in place; must already contain <category_name>)
#
# Output: existing_output_path's annotation.json files updated in place with
# the new images added under the EXISTING category_id (images copied in with
# the given prefix; category count is unchanged).

set -e

NEW_DATASET_PATH="${1:-}"
CATEGORY_NAME="${2:-}"
PREFIX="${3:-}"
EXISTING_OUTPUT="${4:-}"
PROJECT_NAME="${5:-sam3_fine_tune}"

usage() {
    cat <<EOF
Usage: ssh hpc_muc 'bash -s -- <new_dataset_path> <category_name> <prefix> <existing_output_path> [project_name]' < run_append_images_to_existing_category.sh

  new_dataset_path      COCO dataset root for the ADDITIONAL images (must contain train/ and test/ with images/ + annotation.json)
  category_name         must already exist in existing_output_path's categories, e.g. matrix_chip
  prefix                short string prepended to new image filenames to avoid collisions
  existing_output_path  root of the EXISTING merged dataset to append into (modified in place)
  project_name          repo directory name (default: sam3_fine_tune)
EOF
}

if [[ -z "${NEW_DATASET_PATH}" || -z "${CATEGORY_NAME}" || -z "${PREFIX}" || -z "${EXISTING_OUTPUT}" ]]; then
    usage >&2
    exit 1
fi

for p in "${NEW_DATASET_PATH}" "${EXISTING_OUTPUT}"; do
    if [[ ! "${p}" =~ ^/ ]]; then
        echo "ERROR: paths must be absolute HPC paths. Got: ${p}" >&2
        exit 1
    fi
done

HPC_USER="${USER}"
SEARCH_ROOT="/home/hpc_storage/data/projects/${HPC_USER}"

if [[ -z "${HPC_USER}" ]]; then
    echo "ERROR: USER is empty." >&2
    exit 1
fi

# Discover the repo root on HPC, using the same per-user cache convention as
# run_append_dataset.sh / run_coco_conversion.sh. Set FORCE_REFRESH=1 to bypass it.
CACHE_DIR="${HOME}/.cache/hpc_repo_root"
CACHE_FILE="${CACHE_DIR}/${PROJECT_NAME}"
REPO_ROOT=""

if [[ "${FORCE_REFRESH:-0}" != "1" && -f "${CACHE_FILE}" ]]; then
    CACHED=$(cat "${CACHE_FILE}")
    if [[ -n "${CACHED}" && -d "${CACHED}" && -f "${CACHED}/pyproject.toml" ]]; then
        REPO_ROOT="${CACHED}"
    fi
fi

if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT=$(find "${SEARCH_ROOT}" -maxdepth 4 -type d -name "${PROJECT_NAME}" 2>/dev/null | head -n1)
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

NB_PATH="${REPO_ROOT}/nbs/02_merge_dataset.ipynb"
SCRIPT_PATH="${REPO_ROOT}/.github/scripts/append_images_to_existing_category.py"

for req in "${NB_PATH}" "${SCRIPT_PATH}" "${NEW_DATASET_PATH}/train/annotation.json" "${EXISTING_OUTPUT}/train/annotation.json"; do
    if [[ ! -f "${req}" ]]; then
        echo "ERROR: expected file not found: ${req}" >&2
        exit 1
    fi
done

source "${REPO_ROOT}/.venv/bin/activate"

bsub -Is -R "type=X64LIN && osrel>=70 && ui=pytest rusage[mem=50G]" -n 30 \
  "source ${REPO_ROOT}/.venv/bin/activate && python3 ${SCRIPT_PATH} ${NB_PATH} ${NEW_DATASET_PATH} ${CATEGORY_NAME} ${PREFIX} ${EXISTING_OUTPUT}"

