# MinerU2.5-Pro-2605-1.2B

## Overview

Serve this vision-language OCR model with one S5000 GPU. The configuration uses
BF16, compiled execution, and CUDA graphs for the language-model path.

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
  --max-num-seqs 128 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.80 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --hf-overrides '{"tie_word_embeddings":true}' \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Send an image request

Replace `<IMAGE_PATH>` with a JPEG or PNG file. This verifies the vLLM image
request path; an application may then send its model-specific prompts.

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
    model="MinerU2.5-Pro-2605-1.2B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"}},
            {"type": "text", "text": "\nLayout Detection:"},
        ]},
    ],
    temperature=0,
    top_p=0.01,
    max_tokens=2048,
    extra_body={"top_k": 1, "skip_special_tokens": False},
)
print(response.choices[0].message.content)
```

`--hf-overrides` keeps the checkpoint's tied embeddings enabled.
`--logits-processors` loads the model-specific processor for sampling.

Return to the [OCR recipe index](README.md).
