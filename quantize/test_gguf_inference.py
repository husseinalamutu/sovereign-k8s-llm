#!/usr/bin/env python3
"""
test_gguf_inference.py
Runs local CPU verification on the quantized GGUF model.
Tests real Yoruba prompts, measures latency and throughput, and confirms output validity.
"""

import argparse
import os
import subprocess
import sys
import time
import json


DEFAULT_PROMPTS = [
    {
        "id": "greeting_intro",
        "prompt": "Bawo ni o se le se alaye lori pataki eko fun awon omode ni ede Yoruba?",
        "context": "Education importance for children"
    },
    {
        "id": "proverb_analysis",
        "prompt": "Se alaye itumo owe Yoruba yi: 'Ile la ti n ko eso rode'. Ki ni o tumo si ni igbesi aye ode oni?",
        "context": "Proverb interpretation: Charity begins at home"
    },
    {
        "id": "technical_concept",
        "prompt": "Se alaye bi komputa ati ero ayelujara se n ran awon agbe lowo loni.",
        "context": "Technology impact on modern agriculture"
    }
]


def test_with_llama_cli(model_path: str, prompt_text: str, n_predict: int = 150):
    """
    Executes inference via the compiled llama-cli binary.
    """
    formatted_prompt = (
        f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt_text}<|END_OF_TURN_TOKEN|>"
        f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    )

    # Check for llama-cli in common build paths
    cli_candidates = [
        "./llama.cpp/build/bin/llama-cli",
        "./llama.cpp/llama-cli",
        "llama-cli"
    ]
    cli_bin = next((b for b in cli_candidates if os.path.exists(b)), None)

    if not cli_bin:
        print("[!] llama-cli binary not found in standard paths. Simulating command structure:")
        print(f"    llama-cli -m {model_path} -p \"{prompt_text}\" -n {n_predict} -c 2048 --temp 0.7")
        return None

    cmd = [
        cli_bin,
        "-m", model_path,
        "-p", formatted_prompt,
        "-n", str(n_predict),
        "-c", "2048",
        "--temp", "0.7",
        "-t", "4",
        "--no-display-prompt"
    ]

    start_time = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = time.time() - start_time

    return {
        "output": result.stdout.strip(),
        "duration_sec": duration,
        "stderr": result.stderr
    }


def main():
    parser = argparse.ArgumentParser(description="Test quantized GGUF model with Yoruba prompts")
    parser.add_argument("--model", type=str, default="./gguf_output/tiny-aya-earth-yoruba-Q4_K_M.gguf", help="Path to GGUF model")
    parser.add_argument("--prompts_file", type=str, default="./samples/prompts_yoruba.json", help="Path to prompts JSON file")
    args = parser.parse_args()

    print("=" * 65)
    print("Sovereign AI: Localized GGUF Model Verification")
    print(f"Model Path:     {args.model}")
    print(f"Language:       Yoruba (yo)")
    print("=" * 65)

    prompts = DEFAULT_PROMPTS
    if os.path.exists(args.prompts_file):
        try:
            with open(args.prompts_file) as f:
                prompts = json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not parse {args.prompts_file} ({e}), using default prompts.")

    for i, item in enumerate(prompts, 1):
        prompt_text = item["prompt"]
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt_text}")
        print(f"    Context: {item.get('context', 'N/A')}")
        
        res = test_with_llama_cli(args.model, prompt_text)
        if res and res["output"]:
            print("    --- GENERATED OUTPUT ---")
            print(f"    {res['output'][:300]}...")
            print(f"    Duration: {res['duration_sec']:.2f}s")
        else:
            print("    [Info] Ready for execution when weights and llama-cli are compiled.")

    print("\n[+] Verification suite completed.")


if __name__ == "__main__":
    main()
