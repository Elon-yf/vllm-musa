# MinerU2.5-Pro-2605-1.2B BF16 two-step recipe

This recipe runs MinerU2.5-Pro-2605-1.2B on one S5000 through the official in-process `vllm-async-engine`. Each page goes through MinerU layout extraction and region recognition, then `json2md` post-processing. Keep the two-step client and outer page batch at 100 for the full 1,651-page contract.

> [!TIP]
> The outer batch of 100 is a pipeline/client control. It is separate from engine active sequences and batched tokens. The official runner leaves those engine limits at their defaults.

## Contract

| Item | Value |
|---|---|
| Hardware | 1x S5000 MPX30 |
| Precision / TP | BF16 / TP1 |
| GPU memory utilization | `0.8` |
| Context | 8,192 tokens |
| Engine | `AsyncLLM` through `vllm-async-engine` |
| Model override | `tie_word_embeddings=True` |
| Logits processor | `MinerULogitsProcessor` |
| Image analysis | disabled (`image_analysis=False`) |
| Full workload | OmniDocBench v1.6 original images, 1,651 pages |
| Full P2 batch | 100 |
| Same-process phases | cold 1, warmup 10, P1 30×3 at batch 1, P2 1,651×3 at batch 100 |
| Engine defaults | `max_num_seqs=128`, `max_num_batched_tokens=2048` in vLLM 0.28; verify resolved startup values |
| Prefix / eager mode | runner defaults; do not add a one-page diagnostic override |

The full-run acceptance check requires 1,651 ordered rows, zero inference errors, and preservation of image-only empty Markdown pages. Run three P2 rounds before ranking a current profile.

## Environment and launch

Use deployment-assigned paths and IDs. `run_full_performance.sh` resolves the approved image, validates the source and data, binds read-only model/data/source trees, installs the pinned local wheels, and starts the official runner.

```bash
export VLLM_MUSA_IMAGE="<approved-vllm-musa-image>"
export GPU="<broker-granted-physical-index>"
export GPU_UUID="<broker-granted-gpu-uuid>"
export LEASE_ID="<active-broker-lease-id>"
export HOST_OUT="<fresh-absolute-output-directory>"
export MODEL="<MinerU2.5-Pro-2605-1.2B-checkpoint>"
export DATA="<OmniDocBench-root>"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_PLUGINS=musa,musa_custom_ops
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash scripts/run_full_performance.sh
```

The launcher executes this exact command in the task runtime:

```bash
python3 -u /recipe/run_bench_musa.py \
  --model mineru --batch-size 100 --repeats 3
```

Do not add `--disable-prefix-caching` for MinerU; the wrapper rejects that exception for this model. Do not add `--enforce-eager` to the full recipe.

There is no separate `vllm serve` process. The vLLM startup arguments are the `AsyncEngineArgs` used by the official client:

```python
AsyncEngineArgs(
    model="/model",
    hf_overrides={"tie_word_embeddings": True},
    logits_processors=[MinerULogitsProcessor],
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=8192,
)
```

## Full two-step client, post-processing, and validation

The following is the model-specific client used by the runner. It preserves two-step extraction, RGB conversion, `json2md`, and asynchronous submission within each outer batch.

```python
import asyncio, json, time
from pathlib import Path
from PIL import Image
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
from mineru_vl_utils.post_process import json2md

engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
    model="/model",
    hf_overrides={"tie_word_embeddings": True},
    logits_processors=[MinerULogitsProcessor],
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=8192,
))
client = MinerUClient(
    backend="vllm-async-engine",
    vllm_async_llm=engine,
    image_analysis=False,
    use_tqdm=False,
)

async def extract_one(path: Path):
    with Image.open(path) as image:
        blocks = await client.aio_two_step_extract(image.convert("RGB"))
    return json2md(blocks), blocks

async def full_p2(data_root: str, output_root: str):
    root = Path(data_root)
    rows = json.loads((root / "OmniDocBench.json").read_text())
    assert len(rows) == 1651
    paths = [root / "images" / row["page_info"]["image_path"] for row in rows]
    assert all(path.is_file() for path in paths)
    out = Path(output_root); (out / "markdown").mkdir(parents=True)
    records = []; t0 = time.perf_counter()
    for start in range(0, 1651, 100):
        selected = paths[start:start + 100]
        results = await asyncio.gather(*(extract_one(path) for path in selected))
        assert len(results) == len(selected)
        for offset, (path, (markdown, blocks)) in enumerate(zip(selected, results)):
            pos = start + offset
            (out / "markdown" / f"{path.stem}.md").write_text(markdown, encoding="utf-8")
            records.append({
                "row_pos": pos,
                "image_path": rows[pos]["page_info"]["image_path"],
                "ok": True,
                "markdown_empty": not markdown.strip(),
                "blocks": [dict(block) for block in blocks],
            })
    wall = time.perf_counter() - t0
    assert len(records) == 1651
    assert [r["row_pos"] for r in records] == list(range(1651))
    (out / "outputs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    return {"pages": 1651, "wall_s": wall, "page_per_s": 1651 / wall,
            "empty_markdown": sum(r["markdown_empty"] for r in records)}

try:
    print(asyncio.run(full_p2("<OmniDocBench-root>", "<output>")))
finally:
    if hasattr(client, "aclose"):
        asyncio.run(client.aclose())
    engine.shutdown()
```

The production ticket runner wraps this P2 phase with cold, warmup, and three P1 rounds in the same process, repeats P2 three times, and writes per-batch summaries plus a final arithmetic mean and sample standard deviation. Validate each P2 output for 1,651 unique dataset identities, zero inference errors, and preserved image-only empty pages.

## Parameter notes

- `aio_two_step_extract` must remain the page entry point. A one-stage `generate`, a sequential client, or a layout-only shortcut changes the MinerU contract.
- `image_analysis=False` disables optional chart/image analysis; it does not disable layout extraction or region recognition.
- `json2md` and output persistence are inside the measured P2 wall time. Initialization and cold/warmup phases are reported separately.
- Keep `batch-size=100` for every P2 round and preserve original dataset order. Report the three P2 rates only after all three rounds pass; one-page eager/profiler rows are diagnostic only.
- For parameter sweeps, vary vLLM controls while preserving the two-step client, logits processor, RGB preprocessing, BF16, TP1, output format, and full 1,651-page input.

Return to the [OCR recipe index](README.md).
