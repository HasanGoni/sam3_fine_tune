#!/usr/bin/env bash
# Discover the root of this repository on the Acme HPC.
#
# Looks for directories whose basename matches PROJECT_NAME (default: the
# basename of the repo this script ships with). If multiple clones exist,
# returns the most recently modified one so the active workspace is preferred.
# If no matching directory is found, it falls back to the most recently modified
# directory under /home/hpc_storage/data/projects/$USER that contains a
# pyproject.toml file.
#
# Caching: the result is cached at ~/.cache/hpc_repo_root/<PROJECT_NAME> so
# subsequent calls skip the (slow) `find` scan. The cache is validated on
# every read (path must still exist and contain pyproject.toml) and is
# transparently rebuilt if missing or stale. Set FORCE_REFRESH=1 to bypass
# the cache and force rediscovery.
#
# Usage:
#   ssh hpc_muc 'bash -s' < .github/scripts/discover_repo_root.sh [PROJECT_NAME]

set -e

PROJECT_NAME="${1:-sam3_fine_tune}"
HPC_USER="${2:-$USER}"
SEARCH_ROOT="/home/hpc_storage/data/projects/${HPC_USER}"

if [[ -z "${HPC_USER}" ]]; then
    echo "ERROR: HPC_USER is empty. Pass it explicitly or set \$USER." >&2
    exit 1
fi

CACHE_DIR="${HOME}/.cache/hpc_repo_root"
CACHE_FILE="${CACHE_DIR}/${PROJECT_NAME}"

# Fast path: use the cached repo root if it's still valid.
if [[ "${FORCE_REFRESH:-0}" != "1" && -f "${CACHE_FILE}" ]]; then
    CACHED=$(cat "${CACHE_FILE}")
    if [[ -n "${CACHED}" && -d "${CACHED}" && -f "${CACHED}/pyproject.toml" ]]; then
        echo "${CACHED}"
        exit 0
    fi
fi

# Helper: pick the most recently modified path from stdin.
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

# 1. Find directories whose basename matches the project name; prefer the newest.
MATCH=$(find "${SEARCH_ROOT}" -maxdepth 4 -type d -name "${PROJECT_NAME}" 2>/dev/null | most_recently_modified)

if [[ -n "${MATCH}" ]]; then
    mkdir -p "${CACHE_DIR}"
    echo "${MATCH}" > "${CACHE_FILE}"
    echo "${MATCH}"
    exit 0
fi

# 2. Fallback: look for any directory containing pyproject.toml; prefer the newest.
FALLBACK=$(find "${SEARCH_ROOT}" -maxdepth 5 -type f -name "pyproject.toml" 2>/dev/null | while read -r f; do dirname "${f}"; done | most_recently_modified)

if [[ -n "${FALLBACK}" ]]; then
    mkdir -p "${CACHE_DIR}"
    echo "${FALLBACK}" > "${CACHE_FILE}"
    echo "${FALLBACK}"
    exit 0
fi

echo "ERROR: Could not find project '${PROJECT_NAME}' under ${SEARCH_ROOT}" >&2
exit 1

