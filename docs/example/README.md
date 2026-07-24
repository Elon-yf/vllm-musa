# Examples

Supplementary examples for running vLLM on MTGPU. For general vLLM usage, refer to the upstream `vllm/examples` directory.

## Disaggregated Serving

Demonstrates disaggregated prefill/decode serving using the Mooncake KV-transfer
connector.

- **`disaggregated_serving.sh`** – launches one prefiller and one decoder on two
  logical MUSA GPUs, starts the proxy shipped by the pinned upstream vLLM
  checkout, and validates two completion requests. Cleanup targets only the
  processes started by the script.

### Quick Start

```bash
cd example/disaggregated_serving
# Default model: Qwen/Qwen3-8B
bash disaggregated_serving.sh

# Or specify a model:
bash disaggregated_serving.sh meta-llama/Meta-Llama-3.1-8B-Instruct
```

By default, the example leaves the normal compiled serving path enabled. Logs
are written under `/tmp/vllm-musa-mooncake-example-<pid>`; set `LOG_DIR` to
retain them elsewhere. `PREFILL_GPU`, `DECODE_GPU`, service ports,
`MAX_MODEL_LEN`, and `STARTUP_TIMEOUT` can also be overridden through the
environment.
