from __future__ import annotations

from dataclasses import replace

from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    MusaOptimizationContract,
    OptimizationFeature,
)

QWEN_V2_SAMPLING_ARCHITECTURES = frozenset(
    {
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
    }
)
QWEN_LEGACY_SAMPLING_ARCHITECTURES = frozenset(
    {
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
QWEN_FA3_ARCHITECTURES = frozenset(
    {*QWEN_LEGACY_SAMPLING_ARCHITECTURES, "CosyVoice3Model"}
)
_QWEN35_36_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
    }
)
_QWEN35_36_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def matches_mineru_qwen2_vl_config(config: object) -> bool:
    """Match MinerU against its raw Qwen2-VL HF config, not hf_text_config."""
    text = getattr(config, "text_config", config)
    vision = getattr(config, "vision_config", None)
    if getattr(config, "model_type", None) != "qwen2_vl" or vision is None:
        return False
    if (
        getattr(text, "hidden_size", None),
        getattr(text, "num_hidden_layers", None),
        getattr(text, "num_attention_heads", None),
        getattr(text, "num_key_value_heads", None),
        getattr(text, "head_dim", None) or 64,
        getattr(vision, "embed_dim", None) or getattr(vision, "hidden_size", None),
        getattr(vision, "depth", None) or getattr(vision, "num_hidden_layers", None),
        getattr(vision, "num_heads", None)
        or getattr(vision, "num_attention_heads", None),
    ) != (896, 24, 14, 2, 64, 1280, 32, 16):
        return False

    section = None
    for owner in (config, text):
        section = getattr(owner, "mrope_section", None)
        if section is not None:
            break
        for name in ("rope_parameters", "rope_scaling"):
            rope = getattr(owner, name, None)
            if isinstance(rope, dict) and rope.get("mrope_section") is not None:
                section = rope["mrope_section"]
                break
        if section is not None:
            break
    try:
        return tuple(int(value) for value in section) == (8, 12, 12)
    except (TypeError, ValueError):
        return False


def install_mineru_qwen2_vl_rotary(owner: object, vllm_config: object) -> None:
    """Apply the Qwen2-VL rotary route only to the selected model's layers."""
    from vllm.platforms import current_platform

    if not current_platform.is_musa():
        return
    from .resolver import resolve_optimization_contract

    if not resolve_optimization_contract(vllm_config).prefers(
        OptimizationFeature.MINERU_QWEN2_VL_ROTARY
    ):
        return
    from .rotary import MusaMRotaryEmbedding, install_vision_rotary

    install_vision_rotary(
        (
            getattr(block, "attn", None)
            for block in getattr(getattr(owner, "visual", None), "blocks", ())
        ),
        expected_blocks=32,
        required_bf16_neox_shape=(80, 40),
    )
    language = getattr(getattr(owner, "language_model", None), "model", None)
    for layer in getattr(language, "layers", ()):
        attention = getattr(layer, "self_attn", None)
        rotary = getattr(attention, "rotary_emb", None)
        if (
            rotary is not None
            and tuple(getattr(rotary, "mrope_section", ()) or ()) == (8, 12, 12)
            and getattr(rotary, "head_size", None) == 64
            and getattr(rotary, "rotary_dim", None) == 64
        ):
            attention.rotary_emb = MusaMRotaryEmbedding(
                rotary, qk_hidden_sizes=(896, 128)
            )


def matches_qwen35_moe_bf16_prefill_layer(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    global_num_experts: int | None,
    *,
    min_tokens: int,
) -> bool:
    """Match the layer-bound Qwen3.5/3.6-35B-A3B prefill signature.

    The caller is the generic fused-MoE custom-op boundary, so tensor and
    layout checks remain here rather than attempting to recover model config
    from global runtime state on every token.
    """

    def dtype_name(value) -> str:
        return str(value).lower().removeprefix("torch.")

    try:
        return (
            global_num_experts == 256
            and hidden_states.ndim == 2
            and dtype_name(hidden_states.dtype) == "bfloat16"
            and dtype_name(w1.dtype) == "bfloat16"
            and dtype_name(w2.dtype) == "bfloat16"
            and hidden_states.shape[0] >= min_tokens
            and hidden_states.shape[1] == 2048
            and tuple(w1.shape) == (256, 256, 2048)
            and tuple(w2.shape) == (256, 2048, 128)
            and topk_weights.ndim == 2
            and topk_ids.ndim == 2
            and topk_weights.shape == topk_ids.shape
            and topk_ids.shape[1] == 8
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def matches_qwen35_moe_bf16_decode_gemv_layer(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    global_num_experts: int | None,
    *,
    max_tokens: int,
) -> bool:
    """Match the TP4-local Qwen3.5/3.6 BF16 decode GEMV shape.

    The local TP4 weights have two supported runtime forms.  With shared
    expert folding they are ``E=257`` and ``top_k=9``; without folding they
    are ``E=256`` and ``top_k=8``.  Both use the same local ``N/K/V`` sizes.
    Keeping this shape contract at the dispatch boundary makes the small-M
    native GEMV opt-in fail closed for other MoE families and for prefill.
    """

    def dtype_name(value) -> str:
        return str(value).lower().removeprefix("torch.")

    try:
        folded = global_num_experts == 257
        unfolded = global_num_experts == 256
        expected_experts = 257 if folded else 256
        expected_top_k = 9 if folded else 8
        return (
            (folded or unfolded)
            and hidden_states.ndim == 2
            and dtype_name(hidden_states.dtype) == "bfloat16"
            and dtype_name(w1.dtype) == "bfloat16"
            and dtype_name(w2.dtype) == "bfloat16"
            and 0 < hidden_states.shape[0] <= max_tokens
            and hidden_states.shape[1] == 2048
            and tuple(w1.shape) == (expected_experts, 256, 2048)
            and tuple(w2.shape) == (expected_experts, 2048, 128)
            and topk_weights.ndim == 2
            and topk_ids.ndim == 2
            and topk_weights.shape == topk_ids.shape
            and topk_ids.shape[0] == hidden_states.shape[0]
            and topk_ids.shape[1] == expected_top_k
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _has_architecture(model: ModelSignature, allowed: frozenset[str]) -> bool:
    architectures = model.outer_architectures or model.architectures
    return any(architecture in allowed for architecture in architectures)


def _single_device(execution: ExecutionSignature, *, include_dcp: bool) -> bool:
    sizes = (
        execution.tensor_parallel_size,
        execution.pipeline_parallel_size,
        execution.data_parallel_size,
    )
    if include_dcp:
        sizes = (*sizes, execution.decode_context_parallel_size)
    return all(size == 1 for size in sizes)


def _cache_supports_fused_qwen_attention(execution: ExecutionSignature) -> bool:
    return execution.cache_dtype in (None, "auto", "bfloat16") and (
        execution.cache_block_size in (None, 64)
    )


def _qwen2_rope_kv_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    if model.family is not ModelFamily.QWEN2:
        return False
    if model.model_type not in ("qwen2", "cosyvoice3"):
        return False
    if model.model_type == "qwen2":
        if model.num_key_value_heads != 2 or model.intermediate_size != 4864:
            return False
    elif model.num_key_value_heads not in (None, 2) or model.intermediate_size not in (
        None,
        4864,
    ):
        return False
    return (
        model.hidden_size == 896
        and model.num_hidden_layers == 24
        and model.num_attention_heads == 14
        and model.dtype == "bfloat16"
        and model.quantization in (None, "none")
        and not execution.has_quant_config
        and not execution.has_speculative_config
        and execution.has_parallel_config
        and _cache_supports_fused_qwen_attention(execution)
        and not model.enforce_eager
        and _single_device(execution, include_dcp=True)
    )


def _qwen3_rope_kv_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    geometry = (
        model.hidden_size,
        model.intermediate_size,
        model.num_hidden_layers,
        model.num_attention_heads,
        model.num_key_value_heads,
        model.head_dim,
    )
    return (
        model.architectures == ("Qwen3ForCausalLM",)
        and model.model_type == "qwen3"
        and geometry
        in {
            (1024, 3072, 28, 16, 8, 128),
            (4096, 12288, 36, 32, 8, 128),
        }
        and model.dtype == "bfloat16"
        and model.quantization in (None, "none")
        and not execution.has_quant_config
        and not execution.has_speculative_config
        and execution.has_parallel_config
        and _cache_supports_fused_qwen_attention(execution)
        and not model.enforce_eager
        and _single_device(execution, include_dcp=True)
    )


def _qwen3_dense_fp8_fusions_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    return (
        model.architectures == ("Qwen3ForCausalLM",)
        and model.model_type == "qwen3"
        and model.has_routed_experts is False
        and (
            model.hidden_size,
            model.intermediate_size,
            model.num_hidden_layers,
        )
        == (4096, 12288, 36)
        and model.quantization == "fp8"
        and model.dtype == "bfloat16"
        and not execution.has_speculative_config
        and _single_device(execution, include_dcp=False)
    )


def _qwen35_gdn_prefill_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    return (
        model.family is ModelFamily.QWEN35_36
        and model.has_routed_experts is False
        and model.dtype == "bfloat16"
        and model.gdn_conv_width == 4
        and model.gdn_conv_dim == 10240
        and execution.has_parallel_config
        and execution.tensor_parallel_size == 1
    )


def _qwen35_moe_prefill_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    return (
        model.family is ModelFamily.QWEN35_36
        and model.has_routed_experts is True
        and model.dtype == "bfloat16"
        and model.hidden_size == 2048
        and model.num_experts == 256
        and model.num_experts_per_tok == 8
        and model.moe_intermediate_size == 512
        and execution.has_parallel_config
        and execution.tensor_parallel_size == 4
        and execution.pipeline_parallel_size == 1
    )


def resolve_qwen_contract(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> MusaOptimizationContract | None:
    architectures = set(model.outer_architectures or model.architectures)
    mineru = model.mineru_qwen2_vl_config_match and (
        "Qwen2VLForConditionalGeneration" in architectures
    )
    if mineru:
        family = ModelFamily.QWEN2
        role = ModelRole.TEXT
    elif "CosyVoice3Model" in architectures or model.model_type == "cosyvoice3":
        family = ModelFamily.QWEN2
        role = ModelRole.COSYVOICE_TALKER
    # Current Qwen3.6 checkpoints deliberately reuse the Qwen3.5 HF schema:
    # Qwen3_5[Moe]ForConditionalGeneration with qwen3_5[_moe][_text].
    # A future Qwen3.6 schema change fails closed until explicitly added.
    elif (
        architectures & _QWEN35_36_ARCHITECTURES
        or model.model_type in _QWEN35_36_MODEL_TYPES
    ):
        family = ModelFamily.QWEN35_36
        role = ModelRole.TEXT
    elif (
        architectures
        & {
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
        }
        or model.model_type == "qwen3"
    ):
        family = ModelFamily.QWEN3
        role = ModelRole.TEXT
    elif (
        architectures
        & {
            "Qwen2ForCausalLM",
            "Qwen2MoeForCausalLM",
        }
        or model.model_type == "qwen2"
    ):
        family = ModelFamily.QWEN2
        role = ModelRole.TEXT
    else:
        return None

    model = replace(model, family=family, role=role)
    preferred: set[OptimizationFeature] = set()
    if mineru:
        preferred.add(OptimizationFeature.MINERU_QWEN2_VL_ROTARY)
    if _has_architecture(model, QWEN_V2_SAMPLING_ARCHITECTURES):
        preferred.add(OptimizationFeature.QWEN_V2_SAMPLING)
    if (
        _has_architecture(model, QWEN_LEGACY_SAMPLING_ARCHITECTURES)
        and not execution.has_speculative_config
        and not execution.is_pooling_model
    ):
        preferred.add(OptimizationFeature.QWEN_LEGACY_SAMPLING)
    if _has_architecture(model, QWEN_FA3_ARCHITECTURES):
        preferred.add(OptimizationFeature.QWEN_FA3_SCHEDULER)
        if (
            execution.has_parallel_config
            and not execution.has_speculative_config
            and _single_device(execution, include_dcp=True)
            and execution.max_num_seqs is not None
            and execution.max_num_seqs > 0
        ):
            preferred.add(OptimizationFeature.QWEN_FA3_SINGLE_REQUEST_METADATA)
    if _qwen2_rope_kv_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN2_ROPE_KV_PRESPLIT)
    if _qwen3_rope_kv_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN3_QK_ROPE_KV_PRESPLIT)
    if _qwen3_dense_fp8_fusions_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN3_DENSE_FP8_POST_GRAD_FUSIONS)
    if _qwen35_gdn_prefill_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN35_GDN_WIDTH4_PREFILL)
    if _qwen35_moe_prefill_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN35_MOE_BF16_PREFILL)
    if model.family is ModelFamily.QWEN35_36:
        # The FP8 fold is safe once the quant method observes a route that has
        # already been extended by the fused routed+shared gate. The quant
        # apply hook is idempotent for that combined top-k shape.
        if model.has_routed_experts is True:
            preferred.add(OptimizationFeature.QWEN35_SHARED_EXPERT_FOLD)
        if model.dtype == "bfloat16":
            preferred.add(OptimizationFeature.QWEN35_INTERLEAVED_MROPE_QK)

    if model.vocab_size in (151936, 152064, 248320):
        if OptimizationFeature.QWEN_V2_SAMPLING in preferred:
            preferred.update(
                {
                    OptimizationFeature.QWEN_V2_GUMBEL,
                    OptimizationFeature.QWEN_UNIFORM_DECODE_VIEWS,
                    OptimizationFeature.QWEN_UNIFORM_SAMPLE_COUNTS,
                    OptimizationFeature.QWEN_SAMPLE_INPUT_VIEWS,
                }
            )
        if OptimizationFeature.QWEN_LEGACY_SAMPLING in preferred:
            preferred.update(
                {
                    OptimizationFeature.QWEN_LEGACY_GUMBEL,
                    OptimizationFeature.QWEN_TP_LOGITS_IPC_GATHER,
                }
            )
        if (
            OptimizationFeature.QWEN_LEGACY_SAMPLING in preferred
            and execution.tensor_parallel_size == 4
            and execution.pipeline_parallel_size == 1
        ):
            preferred.add(OptimizationFeature.QWEN_TP4_SHARDED_GUMBEL)

    profile = f"{family.value}.{'moe' if model.has_routed_experts else role.value}"
    return MusaOptimizationContract(
        model=model,
        execution=execution,
        profile=profile,
        # Keep the two sets independent even while the first Qwen rollout
        # promotes every proven feature automatically. Future providers can
        # expose a supported-but-not-yet-preferred implementation safely.
        supported_features=frozenset(preferred),
        preferred_features=frozenset(preferred),
    )
