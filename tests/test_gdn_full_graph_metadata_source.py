# SPDX-License-Identifier: Apache-2.0
"""Source contract for sync-free FULL-graph GDN decode padding."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0089-perf-skip-empty-full-graph-GDN-metadata-fill.patch"
)


def test_full_occupancy_skips_empty_gdn_query_padding_fill():
    source = PATCH.read_text()

    assert source.count("diff --git") == 1
    assert (
        "diff --git a/vllm/v1/attention/backends/gdn_attn.py "
        "b/vllm/v1/attention/backends/gdn_attn.py"
    ) in source
    assert "+            if num_decodes < batch_size:" in source
    assert (
        "-            non_spec_query_start_loc[num_decodes + 1 :].fill_("
        "non_spec_num_query_tokens)"
    ) in source


def test_partial_bucket_uses_existing_cpu_terminal_token_count():
    source = PATCH.read_text()

    assert "+                assert non_spec_query_start_loc_cpu is not None" in source
    assert "+                    non_spec_query_start_loc_cpu[-1].item()" in source
    assert source.count("non_spec_query_start_loc_cpu[-1].item()") == 1
