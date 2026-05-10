# # SPDX-License-Identifier: Apache-2.0
# # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# """
# Patch for vllm.v1.sample.ops.topk_topp_sampler.
# """

_ORIGINAL_APPLY_TOP_K_TOP_P = """def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)
"""

_MUSA_GUARDED_APPLY_TOP_K_TOP_P = """def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if HAS_TRITON and logits.shape[0] >= 8 and not current_platform.is_musa():
        return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)
"""

_MUSA_FAST_APPLY_TOP_K_TOP_P = """def _musa_sampler_fast_path_enabled() -> bool:
    value = __import__("os").environ.get("VLLM_MUSA_SAMPLER_FAST_PATH", "1")
    return value.lower() not in ("0", "false", "no", "off")


def _apply_top_k_top_p_musa_topk_prefilter(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    vocab_size = logits.shape[1]
    k_long = k.to(torch.long)
    if bool((k_long >= vocab_size).any().item()):
        return apply_top_k_top_p_pytorch(logits, k, p)

    max_top_k = int(k_long.max().item())
    if max_top_k <= 0 or max_top_k > 1024:
        return apply_top_k_top_p_pytorch(logits, k, p)

    values, indices = logits.topk(max_top_k, dim=-1, largest=True, sorted=True)
    gather_idx = (k_long.clamp_min(1) - 1).unsqueeze(1)
    threshold = values.gather(1, gather_idx)

    # Preserve the current strict tie semantics: the PyTorch fallback keeps
    # every token equal to the kth logit. If a row has ties, use the fallback.
    num_ge = (logits >= threshold).sum(dim=-1)
    if bool((num_ge != k_long).any().item()):
        return apply_top_k_top_p_pytorch(logits, k, p)

    valid = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
    values = values.masked_fill(~(valid < k_long.unsqueeze(1)), -float("inf"))
    logits_sort = values.flip(dims=(-1,))
    logits_idx = indices.flip(dims=(-1,))

    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    output = logits.new_full(logits.shape, -float("inf"))
    output.scatter_(dim=-1, index=logits_idx, src=logits_sort)
    logits.copy_(output)
    return logits


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if current_platform.is_musa() and _musa_sampler_fast_path_enabled():
        if k is not None and logits.shape[0] >= 16:
            if p is None and logits.shape[1] >= 65536:
                max_top_k = int(k.to(torch.long).max().item())
                if 0 < max_top_k <= 1024:
                    return apply_top_k_only(logits, k)
            elif logits.shape[1] >= 65536:
                return _apply_top_k_top_p_musa_topk_prefilter(logits, k, p)

    if HAS_TRITON and logits.shape[0] >= 8 and not current_platform.is_musa():
        return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)
"""


PATCHES = [
    # Patch to keep unsafe Triton sampling disabled on MUSA and add a
    # correctness-preserving top-k prefilter for large decode batches.
    (_ORIGINAL_APPLY_TOP_K_TOP_P, _MUSA_FAST_APPLY_TOP_K_TOP_P),
    (_MUSA_GUARDED_APPLY_TOP_K_TOP_P, _MUSA_FAST_APPLY_TOP_K_TOP_P),
]
