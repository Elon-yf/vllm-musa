# OvisOCR2 BF16

## Overview

Run OvisOCR2 through the in-process vLLM `LLM` API on one S5000 GPU with
vLLM-MUSA.

> [!TIP]
> Use BF16, TP1, 0.80 GPU memory utilization, a 262,144-token context,
> 16,384 batched tokens, `max_num_seqs=1024`, prefix caching off, Triton GDN
> prefill, and the Ovis pixel bounds below. MUSA selects
> FLASH_ATTN for the ViT path.

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
| GDN prefill backend | Triton |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- The OvisOCR2 BF16 checkpoint available at `MODEL_PATH`.

## Recommended vLLM server configuration

Use this server configuration for deployment. The in-process request code below
is an alternative path; do not run both on the same GPU.

```bash
export MODEL_PATH="<MODEL_PATH>"
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name OvisOCR2 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 262144 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --gdn-prefill-backend triton \
  --mm-processor-kwargs '{"images_kwargs":{"min_pixels":200704,"max_pixels":8294400}}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Configuration notes

- The request below starts the model's synchronous in-process `LLM` engine. It
  pins the sweep-selected scheduler values shown above.
- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically.
  Do not add the language-model `--attention-config` option to select it.
- Keep prefix caching off, BF16, TP1, 0.80 GPU memory utilization, Triton GDN
  prefill, and the 262,144-token context together.

## Run an image request in-process

Run this instead of the server command above. The direct vLLM request uses the
same fields as the model parser:

```python
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

engine = LLM(
    model="<MODEL_PATH>",
    tensor_parallel_size=1,
    dtype="bfloat16",
    gpu_memory_utilization=0.80,
    max_model_len=262144,
    max_num_seqs=1024,
    max_num_batched_tokens=16384,
    gdn_prefill_backend="triton",
    enable_prefix_caching=False,
)
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
prompt = engine.get_tokenizer().apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": OCR_PROMPT}
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
