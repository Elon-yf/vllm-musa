# SPDX-License-Identifier: Apache-2.0
"""MUSA backend for the MATE-backed FlashInfer Sparse MLA wrapper."""

from __future__ import annotations

import torch
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
    FlashInferMLASparseMetadata,
    FlashInferMLASparseTRTLLMBackend,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_musa.utils.flashinfer import (
    has_musa_flashinfer_sparse_mla,
    trtllm_batch_decode_with_kv_cache_mla,
)

logger = init_logger(__name__)


class MUSAFlashInferMLASparseImpl(FlashInferMLASparseImpl):
    """Use the MATE provider while retaining vLLM sparse metadata handling."""

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
        if self.topk_indices_buffer is None:
            raise RuntimeError("MUSA FlashInfer Sparse MLA requires top-k indices")

        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        if self.dcp_world_size > 1:
            from vllm.v1.attention.backends.mla.sparse_utils import (
                triton_filter_and_convert_dcp_index,
            )

            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            from vllm.v1.attention.backends.mla.sparse_utils import (
                triton_convert_req_index_to_global_index,
            )

            topk_indices_physical, seq_lens = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

        query = q.unsqueeze(1)
        block_tables = topk_indices_physical.unsqueeze(1)
        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.topk_tokens,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            return_lse=self.need_to_return_lse_for_decode,
        )

        if self.need_to_return_lse_for_decode:
            if not isinstance(kernel_out, tuple):
                raise RuntimeError("MATE Sparse MLA did not return the requested LSE")
            output, lse = kernel_out
        else:
            if not isinstance(kernel_out, torch.Tensor):
                raise RuntimeError("MATE Sparse MLA returned an unexpected output")
            output, lse = kernel_out, None

        output = output.view(-1, output.shape[-2], output.shape[-1])
        if lse is not None:
            lse = self._normalize_lse(lse, output.shape[0], output.shape[1])
            empty_rows = (topk_indices_physical == -1).all(dim=-1)
            output.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return output, lse


@register_backend(AttentionBackendEnum.FLASHINFER_MLA_SPARSE)
class MUSAFlashInferMLASparseBackend(FlashInferMLASparseTRTLLMBackend):
    """Register the MATE FlashInfer Sparse MLA provider on MUSA."""

    @staticmethod
    def get_impl_cls() -> type[MUSAFlashInferMLASparseImpl]:
        return MUSAFlashInferMLASparseImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 3 and has_musa_flashinfer_sparse_mla()

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not has_musa_flashinfer_sparse_mla():
            return "MATE FlashInfer Sparse MLA wrapper is unavailable"
        return super().supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )


__all__ = ["MUSAFlashInferMLASparseBackend", "MUSAFlashInferMLASparseImpl"]
