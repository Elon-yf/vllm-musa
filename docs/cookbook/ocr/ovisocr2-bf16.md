# OvisOCR2 BF16

## Overview

Serve OvisOCR2 on one S5000 GPU with vLLM-MUSA. The server accepts image and
text requests through the OpenAI-compatible chat API.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Maximum context | 262,144 tokens |
| Maximum sequences | vLLM default (left unset in the full contract) |
| Maximum batched tokens | 16,384 |
| GPU memory utilization | 0.80 |
| Prefix caching | Off |
| GDN prefill backend | Triton |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- The OvisOCR2 BF16 checkpoint available at `MODEL_PATH`.

> **Performance configuration.** Keep the scheduler sequence limit unset. The
> historical full-performance contract left `max_num_seqs` at the vLLM default;
> it resolves to 1,024 in the validated vLLM-MUSA environment. Do not replace it
> with an application batch value.

## Launching the server

Replace `<MODEL_PATH>` with the checkpoint directory.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MODEL_PATH="<MODEL_PATH>"

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --served-model-name OvisOCR2 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 262144 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --gdn-prefill-backend triton \
  --mm-processor-kwargs '{"images_kwargs":{"min_pixels":200704,"max_pixels":8294400}}' \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The image processor bounds above are 448² and 2,880² pixels. They apply to all
requests handled by this server.

## Configuration notes

- The historical full path uses vLLM's in-process `LLM` engine. This server
  command carries the same effective vLLM settings; HTTP transport is not a
  performance baseline.
- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically.
  Do not add the language-model `--attention-config` option to select it.
- `--max-num-batched-tokens 16384`, prefix caching off, BF16, TP1, 0.80 GPU
  memory utilization, Triton GDN prefill, and the 262,144-token context are the
  performance settings shown by the full-run logs.

## Send an image request

Replace `<IMAGE_URL>` with an image URL reachable by the server.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "OvisOCR2",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "<IMAGE_URL>"}},
        {"type": "text", "text": "\nExtract all readable content from the image in natural human reading order and output the result as a single Markdown document. For charts or images, represent them using an HTML image tag: <img src=\"images/bbox_{left}_{top}_{right}_{bottom}.jpg\" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."}
      ]
    }],
    "temperature": 0.0,
    "max_tokens": 16384
  }'
```

Return to the [OCR recipe index](README.md).
