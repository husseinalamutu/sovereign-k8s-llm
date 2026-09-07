#!/usr/bin/env bash
# convert_and_quantize.sh
# Converts a merged Hugging Face checkpoint into GGUF format and quantizes to Q4_K_M using llama.cpp.

set -euo pipefail

MODEL_DIR="${1:-./tiny-aya-earth-yoruba-merged}"
OUTPUT_DIR="${2:-./gguf_output}"
QUANT_TYPE="${3:-Q4_K_M}"
LLAMA_CPP_DIR="${4:-./llama.cpp}"

echo "================================================================="
echo " Sovereign AI: GGUF Conversion & Quantization Pipeline"
echo " Input Model:    ${MODEL_DIR}"
echo " Output Dir:     ${OUTPUT_DIR}"
echo " Quantization:   ${QUANT_TYPE}"
echo "================================================================="

mkdir -p "${OUTPUT_DIR}"

# 1. Check or clone llama.cpp
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    echo "[*] Cloning llama.cpp repository..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"
fi

# 2. Build quantization tools if binaries are missing
if [ ! -f "${LLAMA_CPP_DIR}/build/bin/llama-quantize" ] && [ ! -f "${LLAMA_CPP_DIR}/llama-quantize" ]; then
    echo "[*] Building llama.cpp quantization binaries with CMake..."
    cmake -B "${LLAMA_CPP_DIR}/build" -S "${LLAMA_CPP_DIR}" -DGGML_NATIVE=ON
    cmake --build "${LLAMA_CPP_DIR}/build" --config Release --target llama-quantize
fi

QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
if [ ! -f "${QUANTIZE_BIN}" ]; then
    QUANTIZE_BIN="${LLAMA_CPP_DIR}/llama-quantize"
fi

# 3. Install conversion dependencies
echo "[*] Installing GGUF conversion Python requirements..."
pip install -q -r "${LLAMA_CPP_DIR}/requirements/requirements-convert_hf_to_gguf.txt" || true

# 4. Convert Hugging Face weights to FP16 GGUF
F16_GGUF="${OUTPUT_DIR}/tiny-aya-earth-yoruba-f16.gguf"
echo "[*] Converting Hugging Face model to unquantized GGUF (${F16_GGUF})..."
python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "${MODEL_DIR}" \
    --outfile "${F16_GGUF}" \
    --outtype f16

# 5. Quantize to target format (Q4_K_M)
QUANT_GGUF="${OUTPUT_DIR}/tiny-aya-earth-yoruba-${QUANT_TYPE}.gguf"
echo "[*] Quantizing ${F16_GGUF} -> ${QUANT_GGUF} (${QUANT_TYPE})..."
"${QUANTIZE_BIN}" "${F16_GGUF}" "${QUANT_GGUF}" "${QUANT_TYPE}"

echo "================================================================="
echo "[+] Quantization Complete!"
echo "Raw F16 GGUF size:      $(du -h "${F16_GGUF}" | cut -f1)"
echo "Quantized GGUF size:    $(du -h "${QUANT_GGUF}" | cut -f1)"
echo "Output path:            ${QUANT_GGUF}"
echo "================================================================="
echo "Recommended Kubernetes Memory Sizing:"
echo "  - Idle RAM footprint:     ~2.1 GiB"
echo "  - Peak RAM (2k ctx load): ~3.2 GiB"
echo "  - Suggested K8s limit:    4.5 GiB (avoids OOMKilled)"
echo "================================================================="
