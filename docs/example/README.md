# Examples

Supplementary examples for running vLLM on MTGPU. For general vLLM usage, refer to the upstream `vllm/examples` directory.

## Disaggregated Serving

Demonstrates disaggregated prefill/decode serving using the Mooncake KV-transfer connector.

- **`disaggregated_serving.sh`** – Launches two vLLM instances (one prefiller on `MUSA_VISIBLE_DEVICES=0`, one decoder on `MUSA_VISIBLE_DEVICES=1`) with `MooncakeConnector`, starts a proxy server, and runs sample completion requests.
- **`toy_proxy_server.py`** – FastAPI proxy that routes requests to the prefiller for prefill and then to the decoder for token generation.

### Quick Start

```bash
cd example/disaggregated_serving
# Default model: Qwen/Qwen3-8B
bash disaggregated_serving.sh

# Or specify a model:
bash disaggregated_serving.sh meta-llama/Meta-Llama-3.1-8B-Instruct
```
