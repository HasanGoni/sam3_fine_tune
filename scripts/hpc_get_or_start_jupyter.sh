#!/bin/bash
# Reuse an existing HPC Jupyter server if it is still reachable;
# otherwise start a fresh one via scripts/hpc_start_jupyter.exp.

set -euo pipefail

HPC_HOST="hpc_muc"
URL_FILE="/home/ai_dsx/jupyter-homes/userB/.dsx_jl_last.url"
EXPECT_SCRIPT="$(cd "$(dirname "$0")" && pwd)/hpc_start_jupyter.exp"

if [ ! -f "${EXPECT_SCRIPT}" ]; then
    echo "Error: expect script not found: ${EXPECT_SCRIPT}" >&2
    exit 1
fi

# Try to read the last captured URL from HPC.
EXISTING_URL=$(ssh "${HPC_HOST}" "cat ${URL_FILE} 2>/dev/null" || true)

if [ -n "${EXISTING_URL}" ]; then
    # Verify the server is still responding.
    HTTP_CODE=$(ssh "${HPC_HOST}" "curl -s -o /dev/null --max-time 5 -w '%{http_code}' '${EXISTING_URL}'" || echo "000")

    if [ "${HTTP_CODE}" = "200" ]; then
        echo ""
        echo "=== REUSING EXISTING HPC JUPYTER SERVER ==="
        echo "${EXISTING_URL}"
        echo "==========================================="
        echo ""
        exit 0
    else
        echo "Existing server not reachable (HTTP ${HTTP_CODE}); starting a new one..." >&2
    fi
fi

# No usable server found — start a new one.
ssh -t "${HPC_HOST}" 'expect -f -' < "${EXPECT_SCRIPT}"
