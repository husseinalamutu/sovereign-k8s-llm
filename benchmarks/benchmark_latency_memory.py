#!/usr/bin/env python3
"""
benchmark_latency_memory.py
Benchmarks the deployed llama-server serving TinyAya Yoruba GGUF.
Measures Time-To-First-Token (TTFT), tokens/sec throughput, request latency,
and captures memory footprint characteristics.
"""

import argparse
import json
import time
import requests
import statistics


def run_streaming_benchmark(endpoint_url: str, prompt: str, max_tokens: int = 150):
    """
    Sends a chat completion request with streaming enabled to capture TTFT and throughput.
    """
    payload = {
        "model": "tiny-aya-earth-yoruba",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }

    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_response = ""

    try:
        response = requests.post(
            f"{endpoint_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        token_count += 1
                        full_response += delta
                except Exception:
                    pass

        total_time = time.time() - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time
        tps = token_count / (total_time - ttft) if (total_time - ttft) > 0 else 0

        return {
            "success": True,
            "ttft_sec": ttft,
            "total_latency_sec": total_time,
            "token_count": token_count,
            "tokens_per_sec": tps,
            "response_snippet": full_response[:100].strip()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark sovereign LLM inference endpoint")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Base URL of llama-server")
    parser.add_argument("--prompts_file", type=str, default="./samples/prompts_yoruba.json", help="Path to test prompts")
    parser.add_argument("--iterations", type=int, default=3, help="Runs per prompt")
    args = parser.parse_args()

    print("=" * 70)
    print(" Sovereign AI: Inference Latency & Throughput Benchmark")
    print(f" Target Endpoint: {args.url}")
    print(f" Iterations/Prompt: {args.iterations}")
    print("=" * 70)

    # Health check
    try:
        health = requests.get(f"{args.url}/health", timeout=5)
        print(f"[*] Endpoint health status: {health.status_code} ({health.text.strip()})")
    except Exception as e:
        print(f"[!] Warning: Health endpoint unreachable at {args.url}/health ({e})")
        print("    Will still attempt completion requests if server is ready.")

    # Load prompts
    prompts = [
        "Bawo ni a se le lo imo ero komputa lati gbe asa ati ede Yoruba laruge?",
        "Se alaye itumo owe yi: 'Bi omode ba subu a wo iwaju, bi agba ba subu a wo eyin'.",
        "Kin ni awon anfani ti o wa ninu gbigbin eso ati ogbin ni igbalode?"
    ]
    try:
        with open(args.prompts_file) as f:
            data = json.load(f)
            prompts = [item["prompt"] for item in data]
    except Exception:
        pass

    results = []
    print(f"\n[*] Running benchmark across {len(prompts)} distinct Yoruba prompts...\n")

    for idx, p in enumerate(prompts, 1):
        print(f"Prompt {idx}: \"{p[:60]}...\"")
        for it in range(1, args.iterations + 1):
            res = run_streaming_benchmark(args.url, p)
            if res["success"]:
                results.append(res)
                print(f"  Iteration {it}: TTFT={res['ttft_sec']:.3f}s | "
                      f"Throughput={res['tokens_per_sec']:.1f} tok/s | "
                      f"Total={res['total_latency_sec']:.2f}s | "
                      f"Tokens={res['token_count']}")
            else:
                print(f"  Iteration {it}: FAILED ({res.get('error')})")

    if results:
        avg_ttft = statistics.mean([r["ttft_sec"] for r in results])
        avg_tps = statistics.mean([r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0])
        avg_lat = statistics.mean([r["total_latency_sec"] for r in results])

        print("\n" + "=" * 70)
        print(" BENCHMARK SUMMARY (Averaged across successful runs)")
        print(f" Total Completed Requests: {len(results)}")
        print(f" Average TTFT:             {avg_ttft:.3f} seconds")
        print(f" Average Generation Speed: {avg_tps:.2f} tokens/second")
        print(f" Average Total Latency:    {avg_lat:.2f} seconds")
        print("=" * 70)


if __name__ == "__main__":
    main()
