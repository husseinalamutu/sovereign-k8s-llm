# Multi-stage Dockerfile for CPU-optimized llama-server
# Stage 1: Build llama.cpp binaries
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone llama.cpp at stable release
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git .

# Build with cURL support (for direct Hugging Face streaming) and native CPU optimizations
RUN cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU=ON \
    -DLLAMA_CURL=ON \
    && cmake --build build --config Release --target llama-server -j$(nproc)

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8080 \
    CTX_SIZE=2048 \
    N_THREADS=4 \
    MODEL_PATH=/models/tiny-aya-earth-yoruba-Q4_K_M.gguf \
    HF_REPO=HusseinAlamutu/tiny-aya-earth-yoruba-gguf \
    HF_FILE=tiny-aya-earth-yoruba-Q4_K_M.gguf

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libcurl4 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled llama-server from builder
COPY --from=builder /app/build/bin/llama-server /usr/local/bin/llama-server

# Copy startup wrapper
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create models directory
RUN mkdir -p /models

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
