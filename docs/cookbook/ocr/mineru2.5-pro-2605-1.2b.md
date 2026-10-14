# MinerU2.5-Pro-2605-1.2B-BF16

## Overview

Document-parsing vision-language recipe for one S5000 GPU. The official MinerU
two-step pipeline detects page layout, recognizes text, formula, and table
regions, and serializes the result to Markdown.

> [!TIP]
> Use TP1 and the official `vllm-async-engine` client for the FULL pipeline.
> Keep the outer page batch at 100. This page batch is different from
> `max_num_seqs`.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000, MPX30 |
| Driver | 5.2.0-server |
| Precision / tensor parallelism | BF16 / TP1 |
| Architecture | `Qwen2VLForConditionalGeneration` |
| Maximum context | 8,192 tokens |
| Trace maximum sequences / batched tokens | 1 / 4,096 |
| FULL maximum sequences / batched tokens | Runner defaults / 2,048 in the recorded run |
| FULL outer page batch | 100 |
| GPU memory utilization | 0.8 |
| Trace prefix / multimodal processor cache | Off / 0 GB |
| FULL prefix cache / eager mode | Runner defaults (compiled path) |
| FULL client | `MinerUClient(backend="vllm-async-engine")` |
| Dataset | OmniDocBench v1.6 Full, 1,651 original pages |

The **Trace** profile is a single-request diagnostic server: it uses the
explicit sequence and token limits in the table and is paired with a direct
request. The **FULL** profile is the official two-step client below: it creates
its own in-process `AsyncLLM` with the runner defaults. Choose one profile for a
run; do not start the trace server while running the FULL client on the same
GPU.

## Launching the trace server (diagnostic profile)

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

vllm serve /model \
  --served-model-name MinerU2.5-Pro-2605-1.2B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --enforce-eager \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor
```

## Running the official MinerU client

```python
engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
    model="/model",
    hf_overrides={"tie_word_embeddings": True},
    logits_processors=[MinerULogitsProcessor],
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=8192,
))

client = MinerUClient(
    backend="vllm-async-engine",
    vllm_async_llm=engine,
    image_analysis=False,
    use_tqdm=False,
)

blocks = await client.aio_two_step_extract(page_image.convert("RGB"))
markdown = json2md(blocks)
```

Read dataset rows in original order. Submit up to 100 pages per outer P2 batch
and save Markdown and JSON outputs.

## Configuration notes

- Install the pinned MinerU SDK dependencies before starting the client.
- `image_analysis=False` disables optional image and chart analysis. It does
  not disable page layout detection or region recognition.
- The model-local rotary adapter uses text
  `(896, 24, 14, 2, 64)`, vision `(1280, 32, 16, 80)`, and MRoPE
  `(8, 12, 12)`.
- Use the deployment-provided runtime and checkpoint paths.

Return to the [OCR recipe index](README.md).
