import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_musa.jit_kernel import rotary_embedding


@RotaryEmbedding.register_oot
class MusaRotaryEmbedding(RotaryEmbedding):
    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # MUSA-0202: do NOT reassign self.cos_sin_cache inside forward.
        # CUDAGraph-aware Dynamo flags `self.cos_sin_cache = ...` as a
        # buffer mutation and refuses to compile (RuntimeError: Assigning
        # / modifying buffers of nn.Module during forward pass is not
        # allowed when using cudagraph). Use a local variable instead;
        # .to() is a no-op when device/dtype already match.
        cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

        rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query, key
