#!/usr/bin/env bash
# Append a new category dataset into an existing merged COCO dataset on the
# Acme HPC, using `append_dataset()` from nbs/02_merge_dataset.ipynb.
#
# This does NOT run `nbdev-export` / `nbdev_export` and does not touch
# sam3_fine_tune/preprocessing/. Instead it reads the notebook's own
# `#| export`-tagged cells directly (stdlib `json` only) and exec's just those
# cells in-memory to define `append_dataset` / `merge_datasets_multi`, then
# calls `append_dataset` with the given arguments. This avoids nbdev's config
# requirements entirely (this repo still uses settings.ini, not pyproject.toml
# nbdev config, so the real `nbdev-export` CLI cannot run here without a
# repo-wide migration).
#
# Usage (run from your local machine):
#   ssh hpc_muc 'bash -s -- <new_dataset_path> <category_name> <prefix> <existing_output_path> [project_name]' \
#     < .github/scripts/run_append_dataset.sh
#
# Example:
#   ssh hpc_muc 'bash -s -- /home/hpc_storage/data/projects/userB/sam3_data/war_crack_coco war_crack d /home/hpc_storage/data/projects/userB/sam3_data/merged_dataset' \
#     < .github/scripts/run_append_dataset.sh
#
# Expected structure:
#   <new_dataset_path>/{train,test}/images/ + annotation.json      (new category, already in COCO format)
#   <existing_output_path>/{train,test}/images/ + annotation.json  (existing merged dataset, modified in place)
#
# Output: existing_output_path's annotation.json files updated in place with
# the new category appended (images copied in with the given prefix).

set -e

NEW_DATASET_PATH="${1:-}"
CATEGORY_NAME="${2:-}"
PREFIX="${3:-}"
EXISTING_OUTPUT="${4:-}"
PROJECT_NAME="${5:-sam3_fine_tune}"

usage() {
    cat <<EOF
Usage: ssh hpc_muc 'bash -s -- <new_dataset_path> <category_name> <prefix> <existing_output_path> [project_name]' < run_append_dataset.sh

  new_dataset_path      COCO dataset root for the NEW category (must contain train/ and test/ with images/ + annotation.json)
  category_name         text prompt for the new category, e.g. war_crack
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
# run_coco_conversion.sh. Set FORCE_REFRESH=1 to bypass it.
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

for req in "${NB_PATH}" "${NEW_DATASET_PATH}/train/annotation.json" "${EXISTING_OUTPUT}/train/annotation.json"; do
    if [[ ! -f "${req}" ]]; then
        echo "ERROR: expected file not found: ${req}" >&2
        exit 1
    fi
done

source "${REPO_ROOT}/.venv/bin/activate"

python3 - "${NB_PATH}" "${NEW_DATASET_PATH}" "${CATEGORY_NAME}" "${PREFIX}" "${EXISTING_OUTPUT}" <<'PYEOF'
import json
import sys

nb_path, new_path, category_name, prefix, existing_output = sys.argv[1:6]

# Load only the notebook's own `#| export`-tagged cells (nbdev's convention)
# and exec them in-memory. This defines append_dataset/merge_datasets_multi
# without running the real nbdev-export CLI or touching the repo.
with open(nb_path) as f:
    nb = json.load(f)

src = []
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    lines = cell.get("source", [])
    text = "".join(lines) if isinstance(lines, list) else lines
    if text.lstrip().startswith("#| export") or text.lstrip().startswith("#|export"):
        src.append(text)

ns = {}
exec(compile("\n".join(src), nb_path, "exec"), ns)
append_dataset = ns["append_dataset"]

append_dataset(
    {"path": new_path, "category_name": category_name, "prefix": prefix},
    existing_output,
)

print("")
print("=== append_dataset completed ===")
for split in ("train", "test"):
    ann_path = f"{existing_output}/{split}/annotation.json"
    try:
        with open(ann_path) as f:
            d = json.load(f)
        print(f"{split}: {len(d['images'])} images, {len(d['annotations'])} annotations, "
              f"categories={[c['name'] for c in d['categories']]}")
    except FileNotFoundError:
        print(f"{split}: annotation.json not found (skipped)")
PYEOF
