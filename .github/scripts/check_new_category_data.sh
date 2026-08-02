#!/usr/bin/env bash
# Pre-flight validation for a new-category dataset, in a SINGLE ssh round-trip.
#
# Before running append-category-and-finetune, use this to discover in one
# shot: image/mask counts per split, whether train/annotation.json and
# test/annotation.json already exist at the exact location the pipeline
# expects, and whether any *other* leftover COCO json files exist under the
# split (e.g. from earlier coco_format.py test runs with a different
# --category_name) that could cause confusion later.
#
# Doing this validation as ad-hoc back-and-forth `ls`/`cat` calls costs one
# slow network round-trip per check. This script batches all of it into one
# ssh call.
#
# Usage:
#   ssh hpc_muc 'bash -s -- <new_data_dir>' < .github/scripts/check_new_category_data.sh
#
# <new_data_dir> is expected to contain train/ and test/ subfolders.

set -e

NEW_DATA_DIR="${1:-}"

if [[ -z "${NEW_DATA_DIR}" || ! "${NEW_DATA_DIR}" =~ ^/ ]]; then
    echo "Usage: ssh hpc_muc 'bash -s -- <absolute_new_data_dir>' < check_new_category_data.sh" >&2
    exit 1
fi

echo "=== Validating ${NEW_DATA_DIR} ==="

for split in train test; do
    echo ""
    echo "--- ${split} ---"
    SPLIT_DIR="${NEW_DATA_DIR}/${split}"

    if [[ ! -d "${SPLIT_DIR}" ]]; then
        echo "MISSING: ${SPLIT_DIR}"
        continue
    fi

    for sub in images masks; do
        D="${SPLIT_DIR}/${sub}"
        if [[ -d "${D}" ]]; then
            CNT=$(find "${D}" -maxdepth 1 -type f | wc -l)
            echo "${sub}/: ${CNT} files"
        else
            echo "${sub}/: MISSING"
        fi
    done

    EXPECTED_ANN="${SPLIT_DIR}/annotation.json"
    if [[ -f "${EXPECTED_ANN}" ]]; then
        python3 -c "
import json
d = json.load(open('${EXPECTED_ANN}'))
print('annotation.json: OK -', len(d.get('images', [])), 'images,',
      len(d.get('annotations', [])), 'annotations, categories=',
      [c['name'] for c in d.get('categories', [])])
"
    else
        echo "annotation.json: NOT FOUND at expected location (${EXPECTED_ANN})"
    fi

    echo "Other .json files found under ${split}/ (possible leftovers from earlier conversions):"
    FOUND_OTHER=0
    while IFS= read -r f; do
        [[ -z "${f}" ]] && continue
        if [[ "${f}" == "${EXPECTED_ANN}" ]]; then
            continue
        fi
        FOUND_OTHER=1
        python3 -c "
import json
f = '${f}'
try:
    d = json.load(open(f))
    cats = [c['name'] for c in d.get('categories', [])]
    print(f'  {f}: images={len(d.get(\"images\", []))} anns={len(d.get(\"annotations\", []))} categories={cats}')
except Exception as e:
    print(f'  {f}: not valid COCO json ({e})')
"
    done < <(find "${SPLIT_DIR}" -maxdepth 3 -type f -name "*.json" 2>/dev/null)
    if [[ "${FOUND_OTHER}" -eq 0 ]]; then
        echo "  (none)"
    fi
done

echo ""
echo "=== End validation ==="
