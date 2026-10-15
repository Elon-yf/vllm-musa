# MinerU2.5-Pro-2605-1.2B

## Overview

Run this vision-language OCR model with vLLM-MUSA on one S5000 GPU. This page
shows two independent vLLM entry points: an in-process `AsyncLLM` engine and a
`vllm serve` HTTP server.

> [!TIP]
> Recommended: use the in-process `AsyncLLM` + `vllm-async-engine` entry point
> for MinerU2.5-Pro. It produced the fastest completed full-page result with
> the configuration below. Use `vllm serve` when an HTTP endpoint is needed.
> Run only one entry point on a GPU.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |
| Maximum batched tokens | 2,048 |
| GPU memory utilization | 0.80 |
| Prefix cache | On |
| Chunked prefill | On |
| Compilation and CUDA graphs | `VLLM_COMPILE`, `FULL_AND_PIECEWISE` |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- The MinerU2.5-Pro-2605-1.2B checkpoint available at `MODEL_PATH`.
- `mineru-vl-utils` installed in the same environment. It provides the
  `MinerULogitsProcessor` used by the vLLM engine.

## HTTP server

This entry point exposes the vLLM engine through HTTP. Replace `<MODEL_PATH>`
with the checkpoint directory.

```bash
export MODEL_PATH="<MODEL_PATH>"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name MinerU2.5-Pro-2605-1.2B \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.80 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --hf-overrides '{"tie_word_embeddings":true}' \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## In-process `AsyncLLM` engine

This entry point creates the vLLM engine in the caller process. It does not
start `vllm serve`; do not run both entry points on the same GPU.
Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before running the Python code.

Use this vLLM parameter fragment when the application creates the in-process
engine:

```python
from mineru_vl_utils import MinerULogitsProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
    model="<MODEL_PATH>",
    hf_overrides={"tie_word_embeddings": True},
    logits_processors=[MinerULogitsProcessor],
    tensor_parallel_size=1,
    dtype="bfloat16",
    gpu_memory_utilization=0.80,
    max_model_len=8192,
    max_num_seqs=128,
    max_num_batched_tokens=2048,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
))
```

The compiled and graph settings are the vLLM-MUSA runtime defaults validated
for this profile: `VLLM_COMPILE` and `FULL_AND_PIECEWISE`.

## Send an image

The in-process example uses the `engine` created above and MinerU's official
asynchronous client:

```python
import asyncio
from pathlib import Path

from PIL import Image
from mineru_vl_utils import MinerUClient

client = MinerUClient(
    backend="vllm-async-engine",
    vllm_async_llm=engine,
    image_analysis=False,
    use_tqdm=False,
)

async def request() -> None:
    with Image.open(Path("<IMAGE_PATH>")) as image:
        blocks = await client.aio_two_step_extract(image.convert("RGB"))
    print(blocks)
    if hasattr(client, "aclose"):
        await client.aclose()
    engine.shutdown()

asyncio.run(request())
```

For the HTTP server, use the same two-step extraction method with MinerU's
HTTP client:

```python
import asyncio
from pathlib import Path

from PIL import Image
from mineru_vl_utils import MinerUClient

client = MinerUClient(
    backend="http-client",
    server_url="http://127.0.0.1:8000",
    model_name="MinerU2.5-Pro-2605-1.2B",
    image_analysis=False,
    use_tqdm=False,
)

async def request() -> None:
    with Image.open(Path("<IMAGE_PATH>")) as image:
        blocks = await client.aio_two_step_extract(image.convert("RGB"))
    print(blocks)
    if hasattr(client, "aclose"):
        await client.aclose()

asyncio.run(request())
```

## Configuration notes

- The in-process request uses `vllm-async-engine`; the HTTP request uses
  `http-client`. Both call MinerU's two-step extraction method.
- Do not leave scheduler or cache parameters implicit: use the explicit values
  shown above when reproducing this profile.
- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically.
  Do not add the language-model `--attention-config` option to select it.
- The image processor uses the checkpoint's own defaults. Do not add a pixel
  override to this profile.
- Keep BF16, TP1, 0.80 GPU memory utilization, context 8,192, 2,048 batched
  tokens, prefix caching on, chunked prefill on, and
  `VLLM_COMPILE`/`FULL_AND_PIECEWISE` together.

Return to the [OCR recipe index](README.md).
