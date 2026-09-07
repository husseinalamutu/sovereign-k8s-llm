#!/usr/bin/env bash
# test_curl_chat.sh: Quick curl test against OpenAI-compatible /v1/chat/completions endpoint.

set -euo pipefail

ENDPOINT="${1:-http://localhost:8080}"

echo "================================================================="
echo " Testing Sovereign LLM Endpoint: ${ENDPOINT}"
echo "================================================================="

echo -e "\n[*] Checking Server Health:"
curl -s "${ENDPOINT}/health" || echo "Health check failed"

echo -e "\n\n[*] Sending Yoruba Prompt to /v1/chat/completions:"
curl -s -X POST "${ENDPOINT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tiny-aya-earth-yoruba",
    "messages": [
      {
        "role": "user",
        "content": "Bawo ni o se le se alaye lori bi a se n ko eko nipa imo komputa ni ede Yoruba?"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }' | jq . || cat
