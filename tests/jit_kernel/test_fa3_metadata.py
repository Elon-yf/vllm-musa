# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from vllm_musa.jit_kernel import fa3_metadata


def test_qwen_fa3_scheduler_lookup_builder_direct_and_fallback(monkeypatch):
    from types import SimpleNamespace

    from vllm_musa.v1.attention.backends import flash_attn

    builder = object.__new__(flash_attn.FlashAttentionMetadataBuilder)
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.reorder_batch_threshold = 1
    builder.use_full_cuda_graph = True
    builder.max_cudagraph_size = 2
    builder.max_num_splits = 32
    builder.aot_schedule = True
    builder.aot_sliding_window = (-1, -1)
    builder._sm_count = 60
    builder._sm_count_query_succeeded = True
    builder._use_qwen_single_request_scheduler_lookup = True
    builder._cu_seqlens_k_buffer = torch.zeros(2, dtype=torch.int32)
    builder.scheduler_metadata = torch.full((17,), -1, dtype=torch.int32)
    builder.num_heads_q = 16
    builder.num_heads_kv = 8
    builder.headdim = 128
    builder.block_size = 64
    builder.kv_cache_dtype = torch.bfloat16
    builder.cache_config = SimpleNamespace(cache_dtype="auto")

    common_metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=128,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([128], dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        causal=True,
    )
    monkeypatch.setattr(
        flash_attn,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: (1, 0, 1, 0),
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")

    direct_enabled = True
    direct_calls = []
    mate_calls = []

    def direct_builder(seq_lens, cu_seqlens_k, scheduler_dst, *args):
        direct_calls.append((seq_lens.clone(), args))
        if not direct_enabled:
            return False
        scheduler_dst.zero_()
        scheduler_dst[0] = 2
        scheduler_dst[8] = 1
        scheduler_dst[12] = 8
        cu_seqlens_k.copy_(torch.tensor([0, 128], dtype=torch.int32))
        return True

    def mate_builder(**_kwargs):
        mate_calls.append(True)
        return torch.arange(16, dtype=torch.int32)

    monkeypatch.setattr(
        fa3_metadata,
        "try_build_qwen_single_request_fa3_metadata",
        direct_builder,
    )
    monkeypatch.setattr(
        flash_attn,
        "get_scheduler_metadata",
        mate_builder,
        raising=False,
    )

    direct_result = builder.build(0, common_metadata)
    assert len(direct_calls) == 1
    assert direct_calls[0][1] == (128, 16, 8, 128, 8)
    assert not mate_calls
    assert direct_result.scheduler_metadata.tolist() == [
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        8,
        0,
        0,
        0,
    ]
    assert direct_result.cu_seqlens_k.tolist() == [0, 128]

    direct_enabled = False
    direct_calls.clear()
    fallback_result = builder.build(0, common_metadata)
    assert len(direct_calls) == 1
    assert len(mate_calls) == 1
    assert fallback_result.scheduler_metadata.tolist() == list(range(16))
    assert fallback_result.cu_seqlens_k.tolist() == [0, 128]
    assert builder.scheduler_metadata[16].item() == 0

    builder._sm_count = 56
    direct_calls.clear()
    mate_calls.clear()
    builder.build(0, common_metadata)
    assert not direct_calls
    assert len(mate_calls) == 1


def test_qwen_single_request_fa3_scheduler_lookup_support_gate():
    seq_lens = torch.zeros(1, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(2, dtype=torch.int32)
    scheduler_dst = torch.zeros(17, dtype=torch.int32)

    assert fa3_metadata._supports_qwen_fa3_scheduler_geometry(4096, 16, 8, 128, 8)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(4097, 16, 8, 128, 8)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(128, 16, 8, 256, 8)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(128, 8, 2, 256, 30)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(128, 24, 25, 128, 3)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(128, 16, 8, 128, 7)

    # Tensor support must fail closed away from a real MUSA device.
    assert not fa3_metadata.supports_qwen_single_request_fa3_scheduler_lookup(
        seq_lens, cu_seqlens_k, scheduler_dst, 128, 16, 8, 128, 8
    )


def test_qwen_single_request_fa3_scheduler_lookup_launches_once(monkeypatch):
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(
        fa3_metadata,
        "_build_qwen_single_request_fa3_metadata_kernel",
        FakeKernel(),
    )
    monkeypatch.setattr(
        fa3_metadata,
        "supports_qwen_single_request_fa3_scheduler_lookup",
        lambda *args: True,
    )
    seq_lens = torch.zeros(1, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(2, dtype=torch.int32)
    scheduler_dst = torch.zeros(513, dtype=torch.int32)

    assert fa3_metadata.try_build_qwen_single_request_fa3_metadata(
        seq_lens, cu_seqlens_k, scheduler_dst, 128, 16, 8, 128, 8
    )
    assert len(launches) == 1
    assert launches[0][0] == (3,)


@pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="requires a MUSA device",
)
@pytest.mark.parametrize(
    ("num_heads_q", "num_heads_kv", "head_dim"),
    [
        (12, 2, 128),
        (14, 2, 64),
        (16, 2, 128),
        (16, 8, 128),
        (28, 4, 128),
        (32, 4, 128),
        (32, 8, 128),
        (40, 8, 128),
        (64, 4, 128),
        (64, 8, 128),
    ],
)
@pytest.mark.parametrize("seq_len", [1, 64, 65, 384, 385, 4096])
def test_qwen_single_request_fa3_scheduler_lookup_values(
    seq_len,
    num_heads_q,
    num_heads_kv,
    head_dim,
):
    from vllm_musa.v1.attention.backends.fa_utils import get_scheduler_metadata

    max_num_splits = (60 + num_heads_kv - 1) // num_heads_kv
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="musa")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="musa")
    cu_seqlens_k = torch.full((2,), -1, dtype=torch.int32, device="musa")
    scheduler_dst = torch.full((513,), -1, dtype=torch.int32, device="musa")
    reference = get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=seq_len,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        headdim=head_dim,
        cache_seqlens=seq_lens,
        qkv_dtype=torch.bfloat16,
        cu_seqlens_q=cu_seqlens_q,
        page_size=64,
        causal=True,
        window_size=(-1, -1),
        num_splits=max_num_splits,
    )

    assert fa3_metadata.try_build_qwen_single_request_fa3_metadata(
        seq_lens,
        cu_seqlens_k,
        scheduler_dst,
        seq_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        max_num_splits,
    )
    torch.musa.synchronize()

    semantic_indices = [0, 4, 8, 12]
    candidate_values = scheduler_dst[:16].cpu().tolist()
    reference_values = reference.cpu().tolist()
    assert [candidate_values[index] for index in semantic_indices] == [
        reference_values[index] for index in semantic_indices
    ]
    assert all(
        value == 0
        for index, value in enumerate(candidate_values)
        if index not in semantic_indices
    )
    assert torch.count_nonzero(scheduler_dst[16:]).item() == 0
    assert cu_seqlens_k.cpu().tolist() == [0, seq_len]
