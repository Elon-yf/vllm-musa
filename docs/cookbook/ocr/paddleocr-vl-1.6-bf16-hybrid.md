# PaddleOCR-VL-1.6 BF16 hybrid recipe

This recipe runs the official 1,651-page PaddleOCR-VL v1.6 pipeline with PP-DocLayoutV3 on an H20 and VLM recognition on a single S5000 vLLM-MUSA process. The H20 client owns layout detection; the S5000 endpoint owns recognition. This is a layout/recognition pipeline split, not prefill/decode disaggregation. Keep this topology when measuring end-to-end page/s.

> [!TIP]
> The client page batch (512), Paddle pipeline batch (64), layout batch (8), and vLLM `max_num_seqs` (512) are separate controls. `use_queues=True` is part of the official contract.

## Contract

| Item | Value |
|---|---|
| Checkpoint | `PaddleOCR-VL-1.6-0.9B` |
| Precision / TP | BF16 / TP1 |
| S5000 vLLM context | 8,192 tokens |
| S5000 limits | `max_num_seqs=512`, `max_num_batched_tokens=16384` |
| Prefix / MM processor cache | disabled / `0` GB |
| H20 layout | `gpu:0`, PP-DocLayoutV3, batch 8 |
| Paddle pipeline | v1.6, pipeline batch 64, queues enabled |
| Full workload | OmniDocBench v1.6 original images, 1,651 pages |
| Measured outer batches | `512, 512, 512, 115` |
| Warmup | first 8 pages, excluded from measured wall time |

The full-run acceptance check requires three rounds with 1,651 ordered rows and zero errors. Preserve the documented image-only empty row separately from unexpected empty output.

## Environment and S5000 server

Use deployment-assigned paths and IDs. The placeholders below are intentional and keep host, image, model, and lease identities out of the cookbook.

```bash
export VLLM_MUSA_IMAGE="<approved-vllm-musa-image>"
export PADDLE_LEASE_ID="<active-broker-lease-id>"
export PADDLE_GPU="<broker-granted-physical-index>"
export PADDLE_GPU_UUID="<broker-granted-gpu-uuid>"
export PADDLE_PORT="<unused-authorized-port>"
export PADDLE_RUN_ROOT="<fresh-absolute-task-root>"
export PADDLE_MODEL_DIR="<PaddleOCR-VL-1.6-0.9B-checkpoint>"
export OMNIDOC_DATA_DIR="<OmniDocBench-root>"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run on the leased S5000 host, from the task bundle directory.
bash scripts/run_s5000_server.sh
```

`run_s5000_server.sh` stages and validates the source overlay, starts a read-only task runtime, and launches the following exact vLLM arguments through `serve.py`:

```text
vllm serve /model
  --host 0.0.0.0 --port <PADDLE_PORT>
  --served-model-name PaddleOCR-VL-1.6-0.9B
  --trust-remote-code --dtype bfloat16
  --gpu-memory-utilization 0.8 --max-model-len 8192
  --max-num-seqs 512 --max-num-batched-tokens 16384
  --no-enable-prefix-caching --mm-processor-cache-gb 0
```

Do not add `--enforce-eager` to this performance recipe. Confirm a real request and the effective backend/graph state before launching the full client. The server URL supplied to the H20 client must be a credential-free `http(s)://<host>:<port>/v1` endpoint.

## H20 client and full 1,651-page flow

Run on the authorized H20 host. The ticket launcher repeats this flow three times with fresh output directories. It sets `CUDA_VISIBLE_DEVICES=0`, binds the approved H20 UUID, and starts the frozen Paddle CUDA image.

```bash
export H20_GPU_UUID="<authorized-h20-gpu-uuid>"
export H20_PARENT_LEASE="$PADDLE_LEASE_ID"
export H20_S5000_URL="http://<s5000-host>:<port>/v1"
export H20_PERF_ROOT="<fresh-absolute-h20-output-root>"
export H20_RUN_ID="<lowercase-run-id>"
export OMNIDOC_DATA_DIR="<OmniDocBench-root>"

# This performs rounds 1, 2, and 3: each is 8-page warmup plus 512+512+512+115.
bash scripts/stage/h20/run_performance.sh
```

The underlying client code is equivalent to the following complete workload and validation flow:

```python
import json
from pathlib import Path
from paddleocr import PaddleOCRVL

GT = Path("<OmniDocBench-root>/OmniDocBench.json")
IMAGE_ROOT = Path("<OmniDocBench-root>/images")
OUT = Path("<fresh-round-output>")
rows = json.loads(GT.read_text())
assert len(rows) == 1651
paths = [IMAGE_ROOT / row["page_info"]["image_path"] for row in rows]
assert all(path.is_file() for path in paths)

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://<s5000-host>:<port>/v1",
    vl_rec_api_model_name="PaddleOCR-VL-1.6-0.9B",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_layout_detection=True,
    use_queues=True,
    device="gpu:0",
)

# Warmup is deliberately outside the measured interval.
list(pipeline.predict([str(p) for p in paths[:8]]))
OUT.mkdir(parents=True, exist_ok=False)
(OUT / "markdown").mkdir()
records = []
measured_batches = [512, 512, 512, 115]
position = 0
for batch_size in measured_batches:
    batch = paths[position:position + batch_size]
    results = list(pipeline.predict([str(p) for p in batch]))
    assert len(results) == len(batch)
    for offset, (path, result) in enumerate(zip(batch, results)):
        markdown = result.markdown
        text = markdown.get("markdown_texts", "") if isinstance(markdown, dict) else str(markdown)
        text = text if isinstance(text, str) else str(text)
        (OUT / "markdown" / f"{path.stem}.md").write_text(text, encoding="utf-8")
        records.append({
            "row_pos": position + offset,
            "image_path": rows[position + offset]["page_info"]["image_path"],
            "ok": True,
            "markdown_empty": not text.strip(),
        })
    position += batch_size
assert position == 1651
assert len(records) == 1651
assert [r["row_pos"] for r in records] == list(range(1651))
assert all(r["ok"] for r in records)
(OUT / "outputs.jsonl").write_text(
    "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    encoding="utf-8",
)
```

The task runner also records PaddleX operation timings, runtime identity, effective server arguments, and coverage. Count the known empty row separately; any unexpected empty output, duplicate/missing row, or error fails the round.

## Parameter notes

- The S5000 command is a recognition service, not a second layout pipeline. Keep layout on H20 `gpu:0` and recognition on the configured `/v1` endpoint.
- `batch_size=512` is the outer list size in the client runner. Do not pass `batch_size=512` to `PaddleOCRVL`; the effective pipeline batch is 64.
- `use_queues=True` is required for the official overlap behavior. The measured wall time includes official Markdown serialization and persistence, while initialization and the eight warmup pages are excluded.
- Keep the original dataset ordering and all four outer batches. A partial batch or a one-page profiler cell is not a full-run result.
- For comparisons, change vLLM controls only. Keep model, image preprocessing, H20/S5000 topology, BF16, TP1, pipeline batch, layout batch, and output contract fixed.

Return to the [OCR recipe index](README.md).
