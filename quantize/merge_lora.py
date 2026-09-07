#!/usr/bin/env python3
"""
merge_lora.py
Merges a trained PEFT LoRA adapter into the base CohereLabs/tiny-aya-earth weights
to produce a consolidated Hugging Face checkpoint ready for GGUF conversion.
"""

import argparse
import os
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default="CohereLabs/tiny-aya-earth", help="Base Hugging Face model ID")
    parser.add_argument("--adapter_dir", type=str, default="./output_lora", help="Path to trained LoRA adapter directory")
    parser.add_argument("--output_dir", type=str, default="./tiny-aya-earth-yoruba-merged", help="Output directory for merged model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to execute weight merge on")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Sovereign AI: LoRA Weight Consolidation")
    print(f"Base Model:   {args.base_model}")
    print(f"Adapter Dir:  {args.adapter_dir}")
    print(f"Output Dir:   {args.output_dir}")
    print(f"Device:       {args.device}")
    print("=" * 60)

    if not os.path.exists(args.adapter_dir):
        print(f"[-] Error: Adapter directory '{args.adapter_dir}' not found.")
        sys.exit(1)

    print(f"[*] Loading base model '{args.base_model}' in FP16 on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )

    print(f"[*] Loading and applying LoRA adapter from '{args.adapter_dir}'...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("[*] Merging weights and unloading adapter...")
    merged_model = peft_model.merge_and_unload()

    print(f"[*] Saving consolidated model and tokenizer to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[+] Merge complete! Consolidated checkpoint ready at '{args.output_dir}'.")


if __name__ == "__main__":
    main()
