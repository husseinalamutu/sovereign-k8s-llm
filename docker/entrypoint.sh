#!/usr/bin/env bash
# entrypoint.sh: Launches llama-server with local GGUF weights or direct Hugging Face Hub stream.

set -euo pipefail

echo "=========================================================="
echo " Starting Sovereign AI Inference Server (llama.cpp)"
echo " Context Window:  ${CTX_SIZE}"
echo " CPU Threads:     ${N_THREADS}"
echo " Serving Port:    ${PORT}"
echo "=========================================================="

# Build arguments array
ARGS=(
    "--host" "0.0.0.0"
    "--port" "${PORT}"
    "-c" "${CTX_SIZE}"
    "-t" "${N_THREADS}"
    "--alias" "tiny-aya-earth-yoruba"
)

if [ -f "${MODEL_PATH}" ]; then
    echo "[*] Found local GGUF model at: ${MODEL_PATH}"
    ARGS+=("-m" "${MODEL_PATH}")
elif [ -n "${HF_REPO:-}" ] && [ -n "${HF_FILE:-}" ]; then
    echo "[*] Local model not mounted. Using Hugging Face Hub stream: ${HF_REPO}/${HF_FILE}"
    ARGS+=("--hf-repo" "${HF_REPO}" "--hf-file" "${HF_FILE}")
else
    echo "[-] Error: No local model found at ${MODEL_PATH} and no HF_REPO specified."
    exit 1
fi

echo "[*] Executing: llama-server ${ARGS[*]}"
exec /usr/local/bin/llama-server "${ARGS[@]}"
