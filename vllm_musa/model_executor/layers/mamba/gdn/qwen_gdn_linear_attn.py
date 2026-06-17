# SPDX-License-Identifier: Apache-2.0
"""MUSA runtime override: use MATE GDN prefill/decode for Qwen3.5."""

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_MATE_GDN_PREFILL_AVAILABLE: bool | None = None
_MATE_CHUNK_GATED_DELTA_RULE = None
_MATE_GDN_PREFILL_HAS_IS_LOG_SPACE = False
_MATE_GDN_DECODE_AVAILABLE: bool | None = None
_MATE_GATED_DELTA_RULE_DECODE = None


def _log_once(method_name: str, message: str, *args) -> None:
    log_method = getattr(logger, f"{method_name}_once", None)
    if log_method is None:
        log_method = getattr(logger, method_name)
    log_method(message, *args)


def _get_mate_gdn_prefill_kernel():
    """Return the MATE GDN prefill kernel when the runtime supports MP31."""
    global _MATE_GDN_PREFILL_AVAILABLE, _MATE_CHUNK_GATED_DELTA_RULE
    global _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE

    if _MATE_GDN_PREFILL_AVAILABLE is None:
        try:
            import os

            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            musa = getattr(torch, "musa", None)
            if musa is None or not musa.is_available():
                raise RuntimeError("torch.musa is not available")

            device_capability = tuple(musa.get_device_capability())
            if device_capability != (3, 1):
                raise RuntimeError(
                    f"MATE GDN prefill requires MP31, got {device_capability}"
                )

            import inspect

            from mate.gdn_prefill import chunk_gated_delta_rule

            _MATE_CHUNK_GATED_DELTA_RULE = chunk_gated_delta_rule
            _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE = (
                "is_log_space" in inspect.signature(chunk_gated_delta_rule).parameters
            )
            _MATE_GDN_PREFILL_AVAILABLE = True
            _log_once(
                "info",
                "Enabled MATE GDN prefill kernel for Qwen3.5 " "(is_log_space=%s).",
                _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE,
            )
        except Exception as e:
            _MATE_CHUNK_GATED_DELTA_RULE = None
            _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE = False
            _MATE_GDN_PREFILL_AVAILABLE = False
            _log_once(
                "warning",
                "MATE GDN prefill is unavailable; using recurrent fallback: %s",
                e,
            )

    return _MATE_GDN_PREFILL_AVAILABLE, _MATE_CHUNK_GATED_DELTA_RULE


def _get_mate_gdn_decode_kernel():
    """Return the MATE GDN decode kernel when the runtime supports MP31."""
    global _MATE_GDN_DECODE_AVAILABLE, _MATE_GATED_DELTA_RULE_DECODE

    if _MATE_GDN_DECODE_AVAILABLE is None:
        try:
            import os

            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            musa = getattr(torch, "musa", None)
            if musa is None or not musa.is_available():
                raise RuntimeError("torch.musa is not available")

            device_capability = tuple(musa.get_device_capability())
            if device_capability != (3, 1):
                raise RuntimeError(
                    f"MATE GDN decode requires MP31, got {device_capability}"
                )

            from mate.gdn_decode import gated_delta_rule_decode

            _MATE_GATED_DELTA_RULE_DECODE = gated_delta_rule_decode
            _MATE_GDN_DECODE_AVAILABLE = True
            _log_once("info", "Enabled MATE GDN decode kernel for Qwen3.5.")
        except Exception as e:
            _MATE_GATED_DELTA_RULE_DECODE = None
            _MATE_GDN_DECODE_AVAILABLE = False
            _log_once(
                "warning",
                "MATE GDN decode is unavailable; using upstream decode: %s",
                e,
            )

    return _MATE_GDN_DECODE_AVAILABLE, _MATE_GATED_DELTA_RULE_DECODE


def _try_mate_gdn_prefill(
    self,
    mixed_qkv_non_spec: torch.Tensor,
    a_non_spec: torch.Tensor,
    b_non_spec: torch.Tensor,
    ssm_state: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor | None,
):
    available, mate_chunk_gated_delta_rule = _get_mate_gdn_prefill_kernel()
    if not available or mate_chunk_gated_delta_rule is None:
        return None

    from vllm.model_executor.layers.fla.ops import fused_post_conv_prep

    try:
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=mixed_qkv_non_spec,
            a=a_non_spec,
            b=b_non_spec,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=False,
            output_g_exp=not _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE,
        )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        if not _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE:
            # Clamp exp-space g away from exact zero to avoid MATE NaNs on
            # long real-model prefills with very negative log-space gates.
            g = g.clamp_min(1e-30)

        state_indices = non_spec_state_indices_tensor.to(torch.int64).contiguous()
        initial_state = ssm_state[state_indices].contiguous().to(torch.float32)
        if has_initial_state is not None:
            initial_state[~has_initial_state, ...] = 0
        cu_seqlens = non_spec_query_start_loc.to(torch.int64).contiguous()

        mate_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "scale": None,
            "initial_state": initial_state,
            "output_final_state": True,
            "cu_seqlens": cu_seqlens,
            "use_qk_l2norm_in_kernel": True,
        }
        if _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE:
            mate_kwargs["is_log_space"] = True

        output, final_state = mate_chunk_gated_delta_rule(**mate_kwargs)
        ssm_state.index_copy_(0, state_indices, final_state.to(ssm_state.dtype))
        return output.unsqueeze(0)
    except Exception as e:
        _log_once(
            "warning",
            "MATE GDN prefill failed; using recurrent fallback: %s",
            e,
        )
        return None


def _try_mate_gdn_decode(
    self,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    attn_metadata,
) -> bool:
    available, mate_gated_delta_rule_decode = _get_mate_gdn_decode_kernel()
    if not available or mate_gated_delta_rule_decode is None:
        return False
    if attn_metadata.spec_sequence_masks is not None or attn_metadata.num_decodes <= 0:
        return False

    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
    assert non_spec_state_indices_tensor is not None

    num_actual_tokens = attn_metadata.num_actual_tokens
    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]

    self_kv_cache = self.kv_cache
    conv_state = (
        self_kv_cache[0]
        if is_conv_state_dim_first()
        else self_kv_cache[0].transpose(-1, -2)
    )
    ssm_state = self_kv_cache[1]

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0),
        self.conv1d.weight.size(2),
    )
    mixed_qkv = causal_conv1d_update(
        mixed_qkv,
        conv_state,
        conv_weights,
        self.conv1d.bias,
        self.activation,
        conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
        validate_data=False,
    )

    query, key, value = self.rearrange_mixed_qkv(mixed_qkv)
    state_indices = (
        non_spec_state_indices_tensor[:num_actual_tokens].to(torch.int64).contiguous()
    )
    initial_state = (
        ssm_state.index_select(0, state_indices).contiguous().to(torch.float32)
    )

    try:
        output, final_state = mate_gated_delta_rule_decode(
            q=query.transpose(0, 1).contiguous(),
            k=key.transpose(0, 1).contiguous(),
            v=value.transpose(0, 1).contiguous(),
            state=initial_state,
            state_layout="VK",
            scale=self.head_k_dim**-0.5,
            A_log=self.A_log.detach().float(),
            a=a.view(num_actual_tokens, 1, -1).contiguous(),
            dt_bias=self.dt_bias.detach().float(),
            b=b.view(num_actual_tokens, 1, -1).contiguous(),
            disable_state_update=False,
            use_qk_l2norm=True,
        )
        ssm_state.index_copy_(0, state_indices, final_state.to(ssm_state.dtype))
        core_attn_out[:num_actual_tokens] = output.view(
            num_actual_tokens,
            self.num_v_heads // self.tp_size,
            self.head_v_dim,
        )
    except Exception as e:
        _log_once(
            "warning",
            "MATE GDN decode failed; using recurrent fallback: %s",
            e,
        )
        core_attn_out_non_spec, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    return True


def _forward_core_mate_gdn(self, mixed_qkv, b, a, core_attn_out) -> None:
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata

    if attn_metadata_raw is None:
        self._warmup_prefill_kernels(mixed_qkv, 0)
        return

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata = attn_metadata_raw[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)

    if attn_metadata.num_prefills <= 0:
        if _try_mate_gdn_decode(self, mixed_qkv, b, a, core_attn_out, attn_metadata):
            return
        return self._musa_original_forward_core(mixed_qkv, b, a, core_attn_out)

    has_initial_state = attn_metadata.has_initial_state
    spec_query_start_loc = attn_metadata.spec_query_start_loc
    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    spec_sequence_masks = attn_metadata.spec_sequence_masks
    spec_token_indx = attn_metadata.spec_token_indx
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    self_kv_cache = self.kv_cache
    conv_state = (
        self_kv_cache[0]
        if is_conv_state_dim_first()
        else self_kv_cache[0].transpose(-1, -2)
    )
    ssm_state = self_kv_cache[1]

    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0),
        self.conv1d.weight.size(2),
    )

    if spec_sequence_masks is not None:
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            mixed_qkv_spec = mixed_qkv
            mixed_qkv_non_spec = None
        else:
            mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
            mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
    else:
        mixed_qkv_spec = None
        mixed_qkv_non_spec = mixed_qkv

    if spec_sequence_masks is not None:
        assert spec_state_indices_tensor is not None
        mixed_qkv_spec = causal_conv1d_update(
            mixed_qkv_spec,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=spec_state_indices_tensor[:, 0][
                : attn_metadata.num_spec_decodes
            ],
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=spec_query_start_loc,
            max_query_len=spec_state_indices_tensor.size(-1),
            validate_data=False,
        )

    assert mixed_qkv_non_spec is not None
    mixed_qkv_non_spec = causal_conv1d_fn(
        mixed_qkv_non_spec.transpose(0, 1),
        conv_weights,
        self.conv1d.bias,
        activation=self.activation,
        conv_states=conv_state,
        has_initial_state=has_initial_state,
        cache_indices=non_spec_state_indices_tensor,
        query_start_loc=non_spec_query_start_loc,
        metadata=attn_metadata,
    ).transpose(0, 1)

    query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

    if spec_sequence_masks is not None:
        a_non_spec = a.index_select(0, non_spec_token_indx)
        b_non_spec = b.index_select(0, non_spec_token_indx)
    else:
        a_non_spec = a
        b_non_spec = b

    if spec_sequence_masks is not None:
        core_attn_out_spec, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            q=query_spec,
            k=key_spec,
            v=value_spec,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
            ssm_state_indices=spec_state_indices_tensor,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out_spec = None

    assert non_spec_state_indices_tensor is not None
    core_attn_out_non_spec = _try_mate_gdn_prefill(
        self,
        mixed_qkv_non_spec,
        a_non_spec,
        b_non_spec,
        ssm_state,
        non_spec_state_indices_tensor,
        non_spec_query_start_loc,
        has_initial_state,
    )
    if core_attn_out_non_spec is None:
        if has_initial_state is not None:
            zero_mask = ~has_initial_state
            if bool(torch.any(zero_mask).item()):
                ssm_state[non_spec_state_indices_tensor[zero_mask]] = 0

        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )
        core_attn_out_non_spec, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a_non_spec,
            b=b_non_spec,
            dt_bias=self.dt_bias,
            q=query_non_spec,
            k=key_non_spec,
            v=value_non_spec,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc,
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
        )

    if spec_sequence_masks is not None:
        merged_out = torch.empty(
            (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
            dtype=core_attn_out_non_spec.dtype,
            device=core_attn_out_non_spec.device,
        )
        merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
    else:
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


def apply() -> None:
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    except Exception as e:
        logger.debug("Skipping GDN MATE prefill/decode override: %s", e)
        return

    QwenGatedDeltaNetAttention = getattr(
        qwen_gdn_linear_attn,
        "QwenGatedDeltaNetAttention",
        None,
    )
    if QwenGatedDeltaNetAttention is None:
        return

    original_forward_core = QwenGatedDeltaNetAttention._forward_core
    if getattr(original_forward_core, "_musa_gdn_mate_patch", False) or getattr(
        original_forward_core, "_musa_gdn_prefill_recurrent_patch", False
    ):
        return

    @wraps(original_forward_core)
    def forward_core_with_mate_gdn(self, mixed_qkv, b, a, core_attn_out):
        return _forward_core_mate_gdn(self, mixed_qkv, b, a, core_attn_out)

    QwenGatedDeltaNetAttention._musa_original_forward_core = original_forward_core
    forward_core_with_mate_gdn._musa_gdn_mate_patch = True
    forward_core_with_mate_gdn._musa_gdn_prefill_recurrent_patch = True
    QwenGatedDeltaNetAttention._forward_core = forward_core_with_mate_gdn
    logger.info(
        "Enabled MUSA Qwen3.5 GDN MATE prefill/decode override with recurrent fallback."
    )


apply()

__all__ = ["apply"]
