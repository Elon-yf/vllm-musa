# PaddleOCR-VL-1.6-BF16-Hybrid

## Overview

This recipe uses one S5000 for PaddleOCR-VL-1.6 recognition and one H20 for
PP-DocLayoutV3 layout detection. PaddleOCRVL connects the H20 client to the
remote recognition server and produces page-level Markdown.

> [!TIP]
> Keep the S5000 recognition server at TP1. On the H20 client, use `gpu:0`
> for layout detection and `use_queues=True`.

## At a glance

| Setting | Value |
|---|---|
| Recognition model | PaddleOCR-VL-1.6-0.9B |
| Layout model | PP-DocLayoutV3 |
| Hardware | 1x S5000 recognition + 1x H20 layout |
| Recognition precision / tensor parallelism | BF16 / TP1 |
| Maximum context / sequences | 8,192 tokens / 512 |
| Maximum batched tokens | 16,384 |
| Prefix / multimodal processor cache | Off / 0 GB |
| H20 client software | PaddleOCR 3.6.0, PaddleX 3.6.1, paddlepaddle-gpu 3.2.1 |
| H20 client batching | 512 input pages; pipeline batch 64; layout batch 8 |

## Launching the server

On the S5000, use the deployment-provided checkpoint at `/model`:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_SERVER_HOST=SERVER_HOST
export VLLM_SERVER_PORT=SERVER_PORT

vllm serve /model \
  --served-model-name PaddleOCR-VL-1.6-0.9B \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --host "$VLLM_SERVER_HOST" \
  --port "$VLLM_SERVER_PORT"
```

## Running the H20 client

Set `PADDLE_VLM_SERVER_URL` to the reachable server endpoint:

```python
import os
from paddleocr import PaddleOCRVL

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    vl_rec_backend="vllm-server",
    vl_rec_server_url=os.environ["PADDLE_VLM_SERVER_URL"],
    vl_rec_api_model_name="PaddleOCR-VL-1.6-0.9B",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_layout_detection=True,
    use_queues=True,
    device="gpu:0",
)
```

Read dataset rows in original order. Warm up with the first eight pages, then
process the full dataset in four outer calls: 512, 512, 512, and 115 pages.
Consume every returned result and save its Markdown output.

## Configuration notes

- The outer batch of 512 is the client page list size. It is separate from
  server `max_num_seqs=512`, pipeline batch 64, and layout batch 8.
- Do not pass `batch_size=512` to the PaddleOCRVL constructor.
- Keep `use_queues=True`, the normal compiled serving path, and do not add
  `--enforce-eager` to the recognition server.
- The input is the dataset page images; source PDFs are not needed.
