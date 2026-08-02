#!/usr/bin/env bash
# Resolve everything needed before running ANY HPC skill for this repo, in a
# SINGLE ssh round-trip: repo root, activated venv, python/torch/cuda
# versions, and the matching `cuda/X.Y` module name to load for GPU jobs.
#
# This exists so individual skills (hpc-cpu-submit, hpc-gpu-submit,
# coco-format-conversion, merge-category-dataset, finetune-merged-dataset,
# ...) don't each re-derive repo root / venv / cuda module via their own
# separate ad-hoc ssh round-trips. Call this once, reuse the printed values.
#
# Usage:
#   ssh hpc_muc 'bash -s -- [project_name]' < .github/scripts/hpc_repo_context.sh
#
# Prints KEY=VALUE lines to stdout:
#   REPO_ROOT=...
#   PYTHON_BIN=...
#   PYTHON_VERSION=...
#   TORCH_VERSION=...
#   TORCH_CUDA=...
#   CUDA_MODULE=...     best-guess `cuda/X.Y` module name derived from
#                       torch.version.cuda. Verify with `module avail cuda`
#                       before relying on it blindly (see hpc-gpu-submit skill).
#
# Set FORCE_REFRESH=1 to bypass the cached repo root and force rediscovery.

set -e

PROJECT_NAME="${1:-sam3_fine_tune}"
HPC_USER="${USER}"
SEARCH_ROOT="/home/hpc_storage/data/projects/${HPC_USER}"

if [[ -z "${HPC_USER}" ]]; then
    echo "ERROR: USER is empty." >&2
    exit 1
fi

# --- 1. Repo root (same cache convention as discover_repo_root.sh) ---
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
        REPO_ROOT=$(find "${SEARCH_ROOT}" -maxdepth 5 -type f -name "pyproject.toml" 2>/dev/null | while read -r f; do dirname "${f}"; done | most_recently_modified)
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

# --- 2. Activate venv ---
if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "ERROR: venv not found at ${REPO_ROOT}/.venv" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

# --- 3. Python / torch / cuda versions ---
PYTHON_BIN=$(command -v python)
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')

TORCH_VERSION="unknown"
TORCH_CUDA="unknown"
TORCH_INFO=$(python -c "import torch; print(torch.__version__); print(torch.version.cuda)" 2>/dev/null || true)
if [[ -n "${TORCH_INFO}" ]]; then
    TORCH_VERSION=$(echo "${TORCH_INFO}" | sed -n '1p')
    TORCH_CUDA=$(echo "${TORCH_INFO}" | sed -n '2p')
fi

CUDA_MODULE="unknown"
if [[ "${TORCH_CUDA}" != "unknown" && "${TORCH_CUDA}" != "None" ]]; then
    CUDA_MODULE="cuda/${TORCH_CUDA}"
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PYTHON_VERSION=${PYTHON_VERSION}"
echo "TORCH_VERSION=${TORCH_VERSION}"
echo "TORCH_CUDA=${TORCH_CUDA}"
echo "CUDA_MODULE=${CUDA_MODULE}"
