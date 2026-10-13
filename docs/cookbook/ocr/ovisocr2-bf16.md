# OvisOCR2-BF16

## Overview

BF16 serving recipe for OvisOCR2, an OCR model based on the Qwen3.5-0.8B
multimodal architecture. This profile targets one S5000 MPX30.

> [!TIP]
> Use TP1, BF16, prefix caching disabled, Triton GDN prefill, and
> FLASH_ATTN for the vision tower.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 MPX30 |
| Precision / tensor parallelism | BF16 / TP1 |
| Speculative decoding | Off |
| Prefix caching | Off |
| GDN prefill | Triton |
| Vision attention | FLASH_ATTN |
| Maximum context | 32,768 tokens |
| Maximum sequences | 1 |
| Maximum batched tokens | 4,096 |
| Image pixels | 448² to 2880² |
| Maximum output tokens | 16,384 |

## Launching the server

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

vllm serve /model \\
  --trust-remote-code \\
  --tensor-parallel-size 1 \\
  --served-model-name OvisOCR2 \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --max-num-seqs 1 \\
  --max-num-batched-tokens 4096 \\
  --gpu-memory-utilization 0.80 \\
  --no-enable-prefix-caching \\
  --gdn-prefill-backend triton \\
  --attention-config '{"backend":"FLASH_ATTN"}'
```

## Request settings

Use the OvisOCR2 OCR prompt without translation or paraphrasing:

```text
temperature=0
enable_thinking=false
max_tokens=16384
image min_pixels=200704 (448²)
image max_pixels=8294400 (2880²)
```

## Configuration notes

- `--no-enable-prefix-caching` keeps independent OCR pages from sharing
  cached prefixes.
- `--gdn-prefill-backend triton` selects the GDN prefill path.
- `--attention-config '{"backend":"FLASH_ATTN"}'` selects the vision attention
  backend used by this profile.
- `max-num-seqs=1` and `max-num-batched-tokens=4096` provide predictable
  single-request behavior for OCR pages.
- Keep `max_tokens=16384` to avoid truncating long OCR pages.
