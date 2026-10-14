# PaddleOCR-VL-1.6 BF16

## Overview

Serve the PaddleOCR-VL-1.6 recognition model with vLLM-MUSA on one S5000 GPU.
The server accepts images through the OpenAI-compatible chat completions API.
Document layout analysis and page assembly, when needed, are handled by the
client application.

> [!TIP]
> Use BF16, TP1, 0.80 GPU memory utilization, an 8,192-token context,
> `max_num_seqs=512`, 16,384 batched tokens, prefix caching off, and a zero GB
> multimodal processor cache. MUSA selects FLASH_ATTN for the ViT path.

## At a glance

| Setting | Single node |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 512 |
| Maximum batched tokens | 16,384 |
| GPU memory utilization | 0.80 |
| Prefix caching | Off |
| Multimodal processor cache | 0 GB |
| CUDA graphs | `FULL_AND_PIECEWISE` with vLLM-MUSA defaults |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- One S5000 GPU visible to the vLLM process.
- A Transformers-format PaddleOCR-VL-1.6-0.9B checkpoint. Set `MODEL_PATH` to
  its directory, which contains the model configuration, weights, tokenizer,
  and image processor.

## Single node

```bash
export MODEL_PATH="<MODEL_PATH>"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name PaddleOCR-VL-1.6-0.9B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```

## Send an image

Install the `openai` Python client if needed. Replace `<IMAGE_PATH>` with a
local JPEG or PNG image:

```python
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

image_path = Path("<IMAGE_PATH>")
mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="PaddleOCR-VL-1.6-0.9B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    },
                },
                {"type": "text", "text": "OCR:"},
            ],
        }
    ],
    temperature=0,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## Configuration notes

- `--max-num-seqs` limits active requests in the vLLM scheduler.
- `--max-num-batched-tokens` limits tokens scheduled in one iteration;
  multimodal prefill can span multiple iterations.
- `--mm-processor-cache-gb 0` disables the multimodal **processor** cache;
  it does not disable the vision encoder or vLLM's CUDA graphs.
- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically. Do
  not add the language-model `--attention-config` option to select it.
- Leave compilation and CUDA graph settings at the vLLM-MUSA defaults shown
  above; do not add `--enforce-eager` to this performance configuration.

Return to the [OCR recipe index](README.md).
