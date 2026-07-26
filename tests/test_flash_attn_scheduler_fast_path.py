# SPDX-License-Identifier: Apache-2.0
"""Exact gates for the MUSA Qwen direct FA3 decode schedule."""

from types import SimpleNamespace

import pytest

from vllm_musa.v1.attention.backends.flash_attn import (
    _is_musa_qwen_family,
    _use_musa_qwen_direct_decode_schedule,
)


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["Qwen2ForCausalLM"], True),
        (["Qwen3ForCausalLM"], True),
        (["Qwen3_5ForConditionalGeneration"], True),
        (["Qwen3_5MoeForConditionalGeneration"], True),
        (["LlamaForCausalLM"], False),
        (None, False),
    ],
)
def test_musa_qwen_family_gate(architectures, expected: bool) -> None:
    model_config = SimpleNamespace(architectures=architectures)
    assert _is_musa_qwen_family(model_config) is expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),
        ({"aot_schedule": False}, False),
        ({"is_bfloat16": False}, False),
        ({"is_qwen_family": False}, False),
        ({"common_prefix_len": 1}, False),
        ({"dcp_world_size": 2}, False),
        ({"num_reqs": 63, "num_decodes": 63}, False),
        ({"num_decodes": 63}, False),
        ({"num_actual_tokens": 63, "num_decode_tokens": 63}, False),
        ({"num_decode_tokens": 63}, False),
        ({"max_query_len": 2}, False),
        ({"max_num_splits": 2}, False),
    ],
)
def test_qwen_direct_decode_schedule_gate(kwargs, expected: bool) -> None:
    values = {
        "aot_schedule": True,
        "is_bfloat16": True,
        "is_qwen_family": True,
        "common_prefix_len": 0,
        "dcp_world_size": 1,
        "num_reqs": 64,
        "num_decodes": 64,
        "num_actual_tokens": 64,
        "num_decode_tokens": 64,
        "max_query_len": 1,
        "max_num_splits": 1,
    }
    values.update(kwargs)
    assert _use_musa_qwen_direct_decode_schedule(**values) is expected
