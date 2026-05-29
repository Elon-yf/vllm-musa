# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0400/0402/0403: dflash.py compatibility + draft-loop FULL CUDAGraph capture.

dflash overrides DFlashProposer.dummy_run. This patch rewrites it so that, in
addition to the #34880 4-tuple unpack, it does the FULL-mode BOOT capture the
base proposer's dummy_run (Hunk 11) does for Eagle:

- accept common_attn_metadata (the MUSA gpu_model_runner passes it to every
  drafter; the upstream dflash override lacked the kwarg -> boot TypeError);
- dispatch at uniform_decode_query_len = 1 + num_speculative_tokens (dflash is a
  PARALLEL block proposer: each request contributes that many query tokens, not
  Eagle's 1) so the boot-captured batch_descriptor matches the inference one;
- build dflash's (non-causal) per-layer attn metadata for the captured forward;
- pass batch_descriptor=batch_desc to set_forward_context so the FULL graph is
  registered under the key the inference propose() dispatches to (otherwise the
  CUDAGraphWrapper finds no graph at request time and trips
  validate_cudagraph_capturing_enabled -> "capturing at an inappropriate time").

Requires the MUSA llm_base_proposer patch (for_draft_model dispatcher,
_determine_batch_execution_and_padding 4-tuple + uniform_decode_query_len, the
qdl in initialize_cudagraph_keys) and supports_sd_full_graph for dflash in the
gpu_model_runner patch (gated VLLM_MUSA_DFLASH_FULL_WRAP). CommonAttentionMetadata
is imported in dflash.py.
"""

# Hunk 1: dummy_run signature gains common_attn_metadata (absorb the runner call).
_OLD_SIG = """    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:"""

_NEW_SIG = """    def dummy_run(
        self,
        num_tokens: int,
        common_attn_metadata: CommonAttentionMetadata | None = None,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:"""

# Hunk 2: dummy_run body — 4-tuple + qdl + dflash attn metadata + batch_descriptor.
_OLD_BODY = """        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):"""

_NEW_BODY = """        import dataclasses as _dc
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        # MUSA-0403: dflash parallel-drafting query_len (= 1 + num_speculative
        # tokens per request); dispatch FULL keys at this width, not Eagle's 1.
        uniform_decode_query_len = 1 + self.num_speculative_tokens
        uniform_decode = common_attn_metadata is not None
        (
            cudagraph_runtime_mode,
            batch_desc,
            num_input_tokens,
            num_tokens_across_dp,
        ) = self._determine_batch_execution_and_padding(
            num_query_tokens,
            uniform_decode,
            use_cudagraphs=use_cudagraphs,
            uniform_decode_query_len=uniform_decode_query_len,
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # MUSA-0403: build dflash's non-causal per-layer attn metadata for the
        # captured forward so the FA kernel captures at the right shape and the
        # FULL graph registers under the inference batch_descriptor.
        if common_attn_metadata is not None:
            _, dummy_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                _dc.replace(common_attn_metadata, causal=False)
            )
        else:
            dummy_attn_metadata = None

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        context_states = self.hidden_states[:num_tokens]

        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            dummy_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_desc,
            slot_mapping=slot_mapping_dict,
        ):"""

PATCHES = [(_OLD_SIG, _NEW_SIG), (_OLD_BODY, _NEW_BODY)]
