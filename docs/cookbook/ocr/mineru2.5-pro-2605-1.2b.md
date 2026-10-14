# MinerU2.5-Pro-2605-1.2B vLLM-MUSA Configuration

## Overview

MinerU uses an in-process vLLM `AsyncLLM` engine. The vLLM startup point is
`AsyncLLM.from_engine_args`, so this configuration does not start a separate
`vllm serve` process. Run it with one visible S5000 GPU and a checkpoint that
is available to the Python process.

## At a glance

| Setting | Value |
|---|---|
| Engine | In-process `AsyncLLM` |
| Precision | BF16 resolved from checkpoint; runner leaves `dtype` unset |
| Tensor parallelism | TP1 |
| GPU memory utilization | 0.8 |
| Maximum context | 8,192 tokens |
| Checkpoint override | `tie_word_embeddings=True` |
| Sampling integration | `MinerULogitsProcessor` |
| Scheduler, prefix cache, and graph | Runner leaves defaults unset; inspect resolved startup values |

## vLLM Environment

Set the worker start method before importing vLLM. If the runtime uses separate
cache volumes, make the vLLM cache and config paths writable.

```bash
export MTHREADS_VISIBLE_DEVICES="${MTHREADS_VISIBLE_DEVICES:?set the assigned S5000 GPU index}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT="<writable-vllm-cache-directory>"
export VLLM_CONFIG_ROOT="<writable-vllm-config-directory>"
```

The cache paths are optional when the default paths are writable. The GPU
visible to the process must match the intended TP1 device.

## Engine Startup

Use the following arguments where the application creates the vLLM engine.
Replace `/model` with the checkpoint path visible to that process.

```python
import torchada  # Activate MUSA mappings before importing vLLM.
from mineru_vl_utils import MinerULogitsProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

engine = AsyncLLM.from_engine_args(
    AsyncEngineArgs(
        model="/model",
        hf_overrides={"tie_word_embeddings": True},
        logits_processors=[MinerULogitsProcessor],
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=8192,
    )
)
```

`MinerULogitsProcessor` is a model-specific dependency passed through vLLM's
`logits_processors` argument. Keep it enabled for this model.

| vLLM setting | Value passed by the runner |
|---|---|
| Engine API | `AsyncLLM.from_engine_args` |
| Model | `/model` in the tested runtime; use the deployed checkpoint path |
| `hf_overrides` | `{"tie_word_embeddings": True}` |
| `logits_processors` | `[MinerULogitsProcessor]` |
| `tensor_parallel_size` | `1` |
| `gpu_memory_utilization` | `0.8` |
| `max_model_len` | `8192` |

The runner leaves `dtype`, `max_num_seqs`, `max_num_batched_tokens`,
`enable_prefix_caching`, `enforce_eager`, and `compilation_config` unset. The
resolved values depend on the vLLM version, checkpoint, and available GPU
memory. Check the engine startup log for the resolved dtype, scheduler limits,
prefix-caching state, compilation mode, and CUDAGraph mode before interpreting
performance results. Do not infer a fixed scheduler limit or graph mode from
the arguments above.

Return to the [OCR recipe index](README.md).
