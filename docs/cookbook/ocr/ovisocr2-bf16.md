# OvisOCR2 BF16

## Overview

Run OvisOCR2 with vLLM-MUSA on one S5000 GPU. Use either the in-process `LLM`
engine or the `vllm serve` HTTP server.

> [!TIP]
> Recommended: use the in-process `LLM` entry point shown in the
> [OvisOCR2 model card](https://huggingface.co/ATH-MaaS/OvisOCR2#inference).
> Our fastest completed full-page configuration also uses this entry point.
> The HTTP server is available when the application needs a separate endpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Maximum context | 262,144 tokens |
| Maximum sequences | 1,024 |
| Maximum batched tokens | 16,384 |
| GPU memory utilization | 0.80 |
| Prefix caching | Off |
| Chunked prefill | On |
| GDN prefill backend | Triton |
| Compilation / CUDA graphs | `VLLM_COMPILE` / `FULL_AND_PIECEWISE` |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- The OvisOCR2 BF16 checkpoint available at `MODEL_PATH`.

## HTTP server

Replace `<MODEL_PATH>` with the checkpoint directory. Run this command where
the intended S5000 GPU is visible to vLLM:

```bash
export MODEL_PATH="<MODEL_PATH>"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name OvisOCR2 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 262144 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --gdn-prefill-backend triton \
  --mm-processor-kwargs '{"images_kwargs":{"min_pixels":200704,"max_pixels":8294400}}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}'
```


Keep the Ovis image processor bounds at `min_pixels=448 * 448` and
`max_pixels=2880 * 2880`, and use `temperature=0`, `max_tokens=16384`, and
`enable_thinking=False` in the application's generation request.

## Send an image

For the HTTP server, read the image and send the same OCR prompt to the
OpenAI-compatible chat endpoint:

```python
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

OCR_PROMPT = (
    '\nExtract all readable content from the image in natural human reading order '
    'and output the result as a single Markdown document. For charts or images, '
    'represent them using an HTML image tag: <img '
    'src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, '
    'right, bottom are bounding box coordinates scaled to [0, 1000). Format '
    'formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe '
    'all other text as standard Markdown. Preserve the original text without '
    'translation or paraphrasing.'
)
image_path = Path("<IMAGE_PATH>")
mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="OvisOCR2",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {
            "url": f"data:{mime_type};base64,{image_data}"
        }},
        {"type": "text", "text": OCR_PROMPT},
    ]}],
    temperature=0,
    max_tokens=16384,
)
print(response.choices[0].message.content)
```

## Configuration notes

- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically.
  Do not add the language-model `--attention-config` option to select it.
- Keep prefix caching off, BF16, TP1, 0.80 GPU memory utilization, Triton GDN
  prefill, and the 262,144-token context together.

Return to the [OCR recipe index](README.md).
