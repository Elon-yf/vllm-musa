# OvisOCR2 BF16 in-process recipe

This recipe runs OvisOCR2 inside one vLLM-MUSA process on one S5000. The official workload calls the model directly through `LLM.generate`; it does not start a separate HTTP server. Keep the client batch at 32 for the full 1,651-page contract.

> [!TIP]
> A page batch is a list passed to `LLM.generate`. It is independent of engine scheduling. The performance wrapper records the actual vLLM runtime and disables prefix caching explicitly.

## Contract

| Item | Value |
|---|---|
| Hardware | 1x S5000 MPX30 |
| Precision / TP | BF16 / TP1 |
| GPU memory utilization | `0.8` |
| GDN prefill | `triton` |
| Prefix caching | disabled by `--disable-prefix-caching` |
| Image pixels | `448²` to `2880²` |
| Sampling | temperature `0`, max tokens `16384`, thinking disabled |
| Full workload | OmniDocBench v1.6 original images, 1,651 pages |
| Full P2 batch | 32 |
| Same-process phases | cold 1, warmup 10, P1 30×3 at batch 1, P2 1,651×3 at batch 32 |

The full-run acceptance check requires three complete P2 rounds with zero inference errors. Keep empty and length-finished rows in the output and repeat the contract for current performance claims.

## Environment and launch

The launcher resolves the approved image and binds the model, data, source overlay, and writable caches. Use placeholders for deployment-specific values.

```bash
export VLLM_MUSA_IMAGE="<approved-vllm-musa-image>"
export OVIS_GPU="<broker-granted-physical-index>"
export OVIS_LEASE_ID="<active-broker-lease-id>"
export OVIS_OUT_DIR="<fresh-absolute-output-path>"
export OVIS_MODEL_DIR="<OvisOCR2-checkpoint>"       # optional launcher override
export OMNIDOC_DATA_DIR="<OmniDocBench-root>"       # optional launcher override
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

scripts/launch.sh
```

The launcher invokes this exact task-runtime command:

```bash
python3 -u /ticket/perf/run_bench_musa.py \
  --model ovis --batch-size 32 --repeats 3 --disable-prefix-caching
```

The wrapper also captures `MTHREADS_VISIBLE_DEVICES=$OVIS_GPU`, cache directories, source identity, package versions, and backend environment. Leave `TORCHDYNAMO_DISABLE` unset for the performance lane.

There is no `vllm serve` command in this recipe. The vLLM startup arguments are the `LLM` constructor below; adding a separate server changes the request contract.

## Full parser, postprocess, and validation flow

This is the parser used by the ticket runner, with deployment paths replaced by placeholders. It preserves the official prompt, image bounds, repeat cleanup, and image-tag filtering.

```python
from PIL import Image
from vllm import LLM, SamplingParams

class OvisOCR2Parser:
    def __init__(self, model_path: str):
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            gdn_prefill_backend="triton",
            enable_prefix_caching=False,
        )
        prompt = (
            "Extract all readable content from the image in natural human reading order "
            "and output the result as a single Markdown document. For charts or images, "
            "represent them using an HTML image tag: <img src=\"images/bbox_{left}_"
            "{top}_{right}_{bottom}.jpg\" />, where left, top, right, bottom are bounding "
            "box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables "
            "as HTML: <table>...</table>. Transcribe all other text as standard Markdown. "
            "Preserve the original text without translation or paraphrasing."
        )
        self.prompt = self.model.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}
            ]}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        self.sampling = SamplingParams(max_tokens=16384, temperature=0.0)

    def _clean_truncated_repeats(self, text: str) -> str:
        # The official runner removes a repeated tail only for long outputs.
        if len(text) < 8000:
            return text
        for unit_len in range(1, min(200, len(text) - 1) + 1):
            if text[-1] != text[-1 - unit_len]:
                continue
            match_len, idx = 1, len(text) - 2
            while idx >= unit_len and text[idx] == text[idx - unit_len]:
                match_len += 1; idx -= 1
            total = match_len + unit_len
            if total >= 100 and total // unit_len >= 5:
                tail = total % unit_len
                return text[:-total + unit_len] + (text[-tail:] if tail else "")
        return text

    def parse(self, images: list[Image.Image]) -> list[str]:
        requests = [{
            "prompt": self.prompt,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {"images_kwargs": {
                "min_pixels": 448 * 448, "max_pixels": 2880 * 2880,
            }},
        } for image in images]
        outputs = self.model.generate(requests, self.sampling)
        self.raw_outputs = outputs
        texts = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            text = "\n\n".join(
                block for block in text.split("\n\n")
                if not block.strip().startswith('<img src="images/bbox_')
            )
            texts.append(self._clean_truncated_repeats(text))
        return texts
```

The full runner reads the 1,651 rows in original order and persists every output. The phase loop below shows the exact measured P2 shape and validation; the ticket runner additionally records cold, warmup, P1, package, source, and runtime manifests.

The launch wrapper injects `enable_prefix_caching=False` into the in-process
engine for this full-run profile; keep that injection when reproducing the
baseline.

```python
import asyncio, json, time
from pathlib import Path
from PIL import Image

async def run_full(model_path, data_root, out_root):
    gt = Path(data_root) / "OmniDocBench.json"
    rows = json.loads(gt.read_text())
    assert len(rows) == 1651
    paths = [Path(data_root) / "images" / r["page_info"]["image_path"] for r in rows]
    assert all(p.is_file() for p in paths)
    parser = OvisOCR2Parser(model_path)
    try:
        # cold + warmup are excluded from P2 timing but run in this process.
        with Image.open(paths[0]) as im: parser.parse([im.copy()])
        with Image.open(paths[0]) as im: parser.parse([im.copy() for _ in range(10)])
        rates = []
        for repeat in range(3):
            phase = Path(out_root) / f"p2-{repeat + 1}"
            (phase / "markdown").mkdir(parents=True)
            records = []
            t0 = time.perf_counter()
            for start in range(0, 1651, 32):
                batch_paths = paths[start:start + 32]
                images = [Image.open(p) for p in batch_paths]
                try:
                    texts = parser.parse(images)
                    raw_outputs = parser.raw_outputs
                finally:
                    for image in images: image.close()
                assert len(texts) == len(batch_paths)
                for pos, (path, text) in enumerate(zip(batch_paths, texts), start):
                    (phase / "markdown" / f"{path.stem}.md").write_text(text, encoding="utf-8")
                    generated = raw_outputs[offset].outputs[0]
                    records.append({"row_pos": pos,
                                    "image_path": rows[pos]["page_info"]["image_path"],
                                    "ok": True, "markdown_empty": not text.strip(),
                                    "raw_output": generated.text,
                                    "output_tokens": len(generated.token_ids),
                                    "finish_reason": generated.finish_reason})
            wall = time.perf_counter() - t0
            assert len(records) == 1651
            assert [r["row_pos"] for r in records] == list(range(1651))
            (phase / "outputs.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                encoding="utf-8")
            rates.append(1651 / wall)
        return {"p2_rates": rates, "mean_page_per_s": sum(rates) / 3}
    finally:
        # The production runner closes its in-process engine during cleanup.
        if hasattr(parser.model, "shutdown"):
            parser.model.shutdown()

asyncio.run(run_full("<OvisOCR2-checkpoint>", "<OmniDocBench-root>", "<output>"))
```

## Parameter notes

- `--disable-prefix-caching` is an explicit Ovis compatibility exception in the wrapper; do not infer it from a one-page profiler configuration.
- `gdn_prefill_backend="triton"`, `temperature=0`, `enable_thinking=False`, and `max_tokens=16384` belong to the official parser contract.
- The model is in-process, so `--max-num-seqs` and `--max-num-batched-tokens` are not cookbook launch arguments. If a sweep adds them, record the changed engine controls separately from this baseline.
- Batch 32 means 32 images per `generate` call. The runner still records batch completion latency, not isolated per-page TTFT/TPOT; those streaming metrics are unavailable in this aggregate API.
- Preserve cold, warmup, P1, and all three P2 phases in one process. A single one-page run cannot establish full-run throughput.

Return to the [OCR recipe index](README.md).
