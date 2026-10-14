# MinerU2.5-Pro-2605-1.2B

## Overview

Serve this vision-language OCR model with one S5000 GPU. The configuration uses
BF16, compiled execution, and CUDA graphs for the language-model path.

> [!TIP]
> Use BF16, TP1, 0.80 GPU memory utilization, an 8,192-token context, 2,048
> batched tokens, prefix caching and chunked prefill on, and
> `VLLM_COMPILE`/`FULL_AND_PIECEWISE`. Keep the vLLM default scheduler sequence
> limit and the checkpoint's image-processor defaults. MUSA selects FLASH_ATTN
> for the ViT path.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Maximum context | 8,192 tokens |
| Maximum sequences | vLLM default |
| Maximum batched tokens | 2,048 |
| GPU memory utilization | 0.80 |
| Prefix cache | On |
| Chunked prefill | On |
| Compilation and CUDA graphs | `VLLM_COMPILE`, `FULL_AND_PIECEWISE` |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- The MinerU2.5-Pro-2605-1.2B checkpoint available at `MODEL_PATH`.
- `mineru-vl-utils` installed in the same environment. It provides the
  `MinerULogitsProcessor` used by the serving command.

## Launching the server

Replace `<MODEL_PATH>` with the checkpoint directory. Run the command where the
intended S5000 GPU is visible to vLLM.

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
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.80 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --hf-overrides '{"tie_word_embeddings":true}' \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Send an image to vLLM

The direct vLLM request uses the model's asynchronous engine client:

```python
import asyncio
from pathlib import Path

from PIL import Image
from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
    model="<MODEL_PATH>",
    hf_overrides={"tie_word_embeddings": True},
    logits_processors=[MinerULogitsProcessor],
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
    max_model_len=8192,
))
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

## Configuration notes

- The model's vLLM integration uses the in-process `AsyncLLM` API; this page
  shows the corresponding `vllm serve` deployment form.
- Leave `max_num_seqs` unset so vLLM uses its default scheduler limit.
- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically.
  Do not add the language-model `--attention-config` option to select it.
- The image processor uses the checkpoint's own defaults. Do not add a
  `--mm-processor-kwargs` pixel override to this profile.
- Keep BF16, TP1, 0.80 GPU memory utilization, context 8,192, 2,048 batched
  tokens, prefix caching on, chunked prefill on, and
  `VLLM_COMPILE`/`FULL_AND_PIECEWISE` together.

Return to the [OCR recipe index](README.md).
