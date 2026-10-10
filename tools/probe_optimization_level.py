"""Probe the resolved CAR-RMSNorm pass default on a real MUSA ``VllmConfig``.

Source reading says ``optimization_level`` defaults to O2 (=2), that upstream's
O2 preset routes ``pass_config.fuse_allreduce_rms`` through
``enable_allreduce_rms_fusion``, and that ``_set_config_default`` only writes a
still-``None`` field. That predicts the MUSA platform hook wins on O2 and yields
on O1/O0. This probe checks that prediction against a live config.

The model is synthetic: ``EngineArgs`` only needs ``config.json``, so no weights
and no NFS mount are required.
"""

import glob
import json
import os
import traceback

OUT = {}
SYNTH = "/tmp/probe-qwen35-27b"

# hidden 5120 / tp 2 is a CAR policy cell. The real Qwen3.5-27B config is a
# multimodal one: `hidden_size` lives under `text_config`, and the top level
# only carries the outer architecture. Mirror that shape exactly -- a flat
# `hidden_size` makes `model_arch_config.hidden_size` fall back to 4096, which
# is outside every CAR policy cell and turns the probe into a false negative.
SYNTH_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 4,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
    },
    "vision_config": {
        "hidden_size": 1152,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "image_size": 768,
        "patch_size": 16,
    },
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}


def _discover_model():
    override = os.environ.get("PROBE_MODEL")
    if override:
        return override
    for pattern in (
        "/home/dist/models/*Qwen3.5*27B*",
        "/mnt/nfs/models/*Qwen3.5*27B*",
        "/home/dist/models/*",
    ):
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(os.path.join(path, "config.json")):
                return path
    os.makedirs(SYNTH, exist_ok=True)
    with open(os.path.join(SYNTH, "config.json"), "w") as handle:
        json.dump(SYNTH_CONFIG, handle, indent=2)
    return SYNTH


MODEL = _discover_model()
OUT["_model"] = MODEL
OUT["_synthetic"] = MODEL == SYNTH


def probe(level, label):
    try:
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(
            model=MODEL,
            tensor_parallel_size=2,
            max_model_len=2048,
            dtype="bfloat16",
            optimization_level=level,
        )
        cfg = args.create_engine_config()
        entry = {
            "optimization_level": int(cfg.optimization_level),
            "compilation_mode": str(cfg.compilation_config.mode),
            "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode),
            "fuse_allreduce_rms": cfg.compilation_config.pass_config.fuse_allreduce_rms,
            "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
            "hidden_size": cfg.model_config.get_hidden_size(),
            "dtype": str(cfg.model_config.dtype),
        }
        # What upstream's O-level preset *wanted* to write for this same config.
        # A difference from the resolved value means the platform hook won.
        try:
            from vllm.config.vllm import enable_allreduce_rms_fusion

            entry["upstream_preset_would_write"] = enable_allreduce_rms_fusion(cfg)
        except Exception as exc:  # noqa: BLE001
            entry["upstream_preset_would_write"] = f"error: {exc}"
        try:
            from vllm_musa.optimization_contract.car_rmsnorm import (
                can_enable_fused_allreduce_rmsnorm,
                infer_car_rmsnorm_model_family,
            )

            entry["contract_family"] = infer_car_rmsnorm_model_family(cfg)
            entry["contract_accepts"] = can_enable_fused_allreduce_rmsnorm(
                tp_size=cfg.parallel_config.tensor_parallel_size,
                pp_size=cfg.parallel_config.pipeline_parallel_size,
                dtype=cfg.model_config.dtype,
                hidden_size=cfg.model_config.get_hidden_size(),
                model_family=entry["contract_family"],
            )
        except Exception as exc:  # noqa: BLE001
            entry["contract_family"] = f"error: {exc}"
        OUT[label] = entry
    except Exception as exc:  # noqa: BLE001 - the probe must always report
        OUT[label] = {
            "error": f"{type(exc).__name__}: {exc}",
            "tb": traceback.format_exc()[-800:],
        }


try:
    from vllm.config.vllm import enable_allreduce_rms_fusion
    from vllm.platforms import current_platform

    OUT["_platform"] = {
        "device_name": getattr(current_platform, "device_name", None),
        "is_cuda": current_platform.is_cuda(),
        "is_musa": current_platform.is_musa(),
    }

    class _Stub:
        parallel_config = type("p", (), {"tensor_parallel_size": 2})()

    OUT["_upstream_predicate_on_musa"] = enable_allreduce_rms_fusion(_Stub())
except Exception as exc:  # noqa: BLE001
    OUT["_platform"] = {"error": f"{type(exc).__name__}: {exc}"}

for _level, _label in ((2, "O2_default"), (1, "O1"), (0, "O0"), (None, "unset")):
    probe(_level, _label)

print("PROBE_JSON_START")
print(json.dumps(OUT, indent=2, default=str))
print("PROBE_JSON_END")
