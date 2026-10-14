# PaddleOCR-VL-1.6 BF16 on vLLM-MUSA

## Overview

This recipe configures the PaddleOCR-VL-1.6 recognition model on one S5000
through the vLLM OpenAI-compatible server. It covers the vLLM process only.
Document layout and page-level post-processing run outside this server.

> [!TIP]
> Use the normal compiled serving path. The page count or outer batch size of
> a document client does not set vLLM's active-sequence or token limits.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision / tensor parallelism | BF16 / TP1 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 512 |
| Maximum batched tokens | 16,384 |
| GPU memory utilization | 0.8 |
| Prefix caching | Off |
| Multimodal processor cache | 0 GB |
| Compilation and CUDAGraph | Normal vLLM defaults; verify the resolved startup configuration |

## Launching the server

Use a vLLM-MUSA environment with the matching PaddleOCR-VL-1.6 checkpoint and
model source installed. Set `MODEL_PATH` to that checkpoint before starting.

```bash
export MODEL_PATH="${MODEL_PATH:?set the PaddleOCR-VL-1.6 checkpoint path}"
export MTHREADS_VISIBLE_DEVICES="${MTHREADS_VISIBLE_DEVICES:?set the assigned S5000 GPU index}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name PaddleOCR-VL-1.6-0.9B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```

The frozen full-run launcher sets the same model, dtype, memory, context,
scheduler, prefix-cache, and processor-cache values. It leaves compilation
and multimodal attention at the installed vLLM-MUSA defaults; the command
above does the same. Its TP1 setting is an effective default, made explicit
here.

## Verifying the vLLM configuration

Check the startup log for `enforce_eager=False`, the resolved
`max_num_seqs=512` and `max_num_batched_tokens=16384`, and successful graph
capture. A prior full-run log reported `FULL_AND_PIECEWISE` graphs and
`FLASH_ATTN` for the visual encoder; confirm those routes in the runtime you
actually start. A `/v1/models` response confirms that the API is listening,
but a real image request is needed to validate model execution.

## Configuration notes

- `max_num_seqs` limits active vLLM sequences. It is independent of an
  upstream document client's page batch.
- `max_num_batched_tokens` is the scheduler's per-iteration token budget;
  chunked multimodal prefill may span iterations.
- `--mm-processor-cache-gb 0` disables the processor cache. It is separate
  from the multimodal encoder cache and encoder CUDAGraph settings.
- Keep `--enforce-eager` off for performance measurements. Record any change
  to graph mode or capture sizes as a separate serving configuration.

Return to the [OCR recipe index](README.md).
