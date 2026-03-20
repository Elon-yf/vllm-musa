#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling with mooncake
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

# meta-llama/Meta-Llama-3.1-8B-Instruct or deepseek-ai/DeepSeek-V2-Lite
MODEL_NAME=${1:-Qwen/Qwen3-8B}

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -i localhost:${port}/v1/models > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# prefilling instance, which is the KV producer
MUSA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
    --port 8100 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' &
PREFILL_PID=$!

# decoding instance, which is the KV consumer
MUSA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}' &
DECODE_PID=$!

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens
#   to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM
#   instance
python3 ./toy_proxy_server.py --prefiller-host 127.0.0.1 --prefiller-port 8100 --decoder-host 127.0.0.1 --decoder-port 8200 &
PROXY_PID=$!
sleep 1

# serve two example requests
output1=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "Santa Clara is a",
"max_tokens": 10,
"temperature": 0
}')


echo "stop serving..."
kill $PREFILL_PID 2>/dev/null
kill $DECODE_PID 2>/dev/null
kill $PROXY_PID 2>/dev/null

echo ""

sleep 1

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""
