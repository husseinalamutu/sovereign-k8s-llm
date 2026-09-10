#!/usr/bin/env python3
"""
upload_to_huggingface.py
Uploads the quantized GGUF model and auto-generated Model Card to Hugging Face Hub.
This enables direct weight streaming in Kubernetes and provides verifiable links for the KubeCon submission.
"""

import argparse
import os
import sys
from huggingface_hub import HfApi, create_repo

MODEL_CARD_TEMPLATE = """---
language:
- yo
- en
license: apache-2.0
tags:
- sovereign-ai
- yoruba
- quantized
- gguf
- llama.cpp
- kubernetes
- tiny-aya
base_model: CohereLabs/tiny-aya-earth
pipeline_tag: text-generation
---

# TinyAya-Earth Yoruba Q4_K_M (GGUF)

This repository provides quantized **GGUF** weights for **CohereLabs/tiny-aya-earth** (3.35B parameters) fine-tuned on the **Yoruba (`yo`)** instruction split of `masakhane/african-ultrachat`.

Developed as part of the KubeCon + CloudNativeCon Europe 2027 case study:
> **"Serving Sovereign AI: Deploying Quantized, Localized LLMs on Resource-Constrained Kubernetes"**

## Model Details
- **Base Architecture:** Dense decoder-only transformer with Grouped Query Attention (GQA), 36 layers, 8k context window.
- **Base Weights:** `CohereLabs/tiny-aya-earth`
- **Fine-Tuning Dataset:** `masakhane/african-ultrachat` (Yoruba split)
- **Quantization:** `Q4_K_M` via `llama.cpp` (~2.1 GB weight size)
- **Primary Language:** Yoruba (`yo`)

## Usage with llama.cpp CLI
```bash
llama-cli -m tiny-aya-earth-yoruba-Q4_K_M.gguf \\
  -p "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Bawo ni o se le se alaye imo ero komputa ni ede Yoruba?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" \\
  -n 256 -c 2048 --temp 0.7
```

## Running Directly in Kubernetes
This GGUF model can be served using the official `llama.cpp` server container on resource-constrained CPU nodes:
```yaml
containers:
- name: llama-server
  image: ghcr.io/ggerganov/llama.cpp:server
  args:
    - "--hf-repo"
    - "{repo_id}"
    - "--hf-file"
    - "tiny-aya-earth-yoruba-Q4_K_M.gguf"
    - "-c"
    - "2048"
    - "--port"
    - "8080"
  resources:
    requests:
      cpu: "1000m"
      memory: "2.5Gi"
    limits:
      cpu: "2000m"
      memory: "4.5Gi"
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Upload GGUF model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="HusseinAlamutu/tiny-aya-earth-yoruba-gguf", help="Target Hugging Face repository ID")
    parser.add_argument("--file_path", type=str, default="./gguf_output/tiny-aya-earth-yoruba-Q4_K_M.gguf", help="Path to quantized GGUF file")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face access token (or set HF_TOKEN env var)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Sovereign AI: Hugging Face Hub Model Exporter")
    print(f"Target Repo:  {args.repo_id}")
    print(f"GGUF File:    {args.file_path}")
    print("=" * 60)

    if not args.token:
        print("[-] Error: Hugging Face token required. Pass --token or export HF_TOKEN=<your_token>.")
        sys.exit(1)

    if not os.path.exists(args.file_path):
        print(f"[-] Error: File '{args.file_path}' does not exist.")
        sys.exit(1)

    api = HfApi(token=args.token)

    print(f"[*] Creating repository (if not exists): {args.repo_id}...")
    create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, token=args.token)

    # 1. Write README.md (Model Card)
    readme_content = MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id)
    temp_readme = "README_HF.md"
    with open(temp_readme, "w") as f:
        f.write(readme_content)

    print("[*] Uploading Model Card (README.md)...")
    api.upload_file(
        path_or_fileobj=temp_readme,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    if os.path.exists(temp_readme):
        os.remove(temp_readme)

    # 2. Upload GGUF Binary
    file_name = os.path.basename(args.file_path)
    file_size_gb = os.path.getsize(args.file_path) / 1e9
    print(f"[*] Uploading '{file_name}' ({file_size_gb:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=args.file_path,
        path_in_repo=file_name,
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"[+] Model successfully published to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
