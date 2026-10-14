# OvisOCR2 BF16

## Overview

Serve OvisOCR2 on one S5000 GPU with vLLM-MUSA. The server accepts image and
text requests through the OpenAI-compatible chat API.

> [!TIP]
> Use BF16, TP1, 0.80 GPU memory utilization, a 262,144-token context,
> 16,384 batched tokens, prefix caching off, Triton GDN prefill, the Ovis pixel
> bounds below, and the vLLM default scheduler sequence limit. MUSA selects
> FLASH_ATTN for the ViT path.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Maximum context | 262,144 tokens |
| Maximum sequences | vLLM default |
| Maximum batched tokens | 16,384 |
| GPU memory utilization | 0.80 |
| Prefix caching | Off |
| GDN prefill backend | Triton |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- The OvisOCR2 BF16 checkpoint available at `MODEL_PATH`.

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

- The model's vLLM integration uses the synchronous in-process `LLM` API; this
  page shows the corresponding `vllm serve` deployment form.
- Leave `max_num_seqs` unset so vLLM uses its default scheduler limit.
- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically.
  Do not add the language-model `--attention-config` option to select it.
- Keep `--max-num-batched-tokens 16384`, prefix caching off, BF16, TP1, 0.80
  GPU memory utilization, Triton GDN prefill, and the 262,144-token context
  together.

## Send an image to vLLM

The direct vLLM request uses the same fields as the model parser:

```python
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

engine = LLM(
    model="<MODEL_PATH>",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
    gdn_prefill_backend="triton",
    enable_prefix_caching=False,
)
prompt = engine.get_tokenizer().apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "<OCR_PROMPT>"}
    ]}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
with Image.open(Path("<IMAGE_PATH>")) as image:
    result = engine.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {"images_kwargs": {
                "min_pixels": 448 * 448,
                "max_pixels": 2880 * 2880,
            }},
        }],
        SamplingParams(max_tokens=16384, temperature=0.0),
    )
print(result[0].outputs[0].text)
```

Return to the [OCR recipe index](README.md).
