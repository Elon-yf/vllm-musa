# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0203: backport of vllm-project/vllm#34880 to vllm v0.20.0-dev.

Adds a `for_draft_model` flag to `CudagraphDispatcher` so a draft-model
proposer can own its own dispatcher and register FULL-mode capture
keys with `uniform_decode_query_len=1` (i.e. one token per request per
step, matching each Eagle decode step). Also threads
`uniform_decode_query_len` through `_create_padded_batch_descriptor`
and `dispatch` so the second-and-later draft steps can dispatch with
the per-step descriptor while the first step still shares the target
model's batch descriptor.

When upstream #34880 merges, this patch file becomes a no-op and can be
deleted.
"""

# ---- Hunk 1: __init__ signature + self.for_draft_model ----
_OLD_INIT = """    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config"""

_NEW_INIT = """    def __init__(self, vllm_config: VllmConfig, for_draft_model: bool = False):
        self.vllm_config = vllm_config
        self.for_draft_model = for_draft_model
        self.compilation_config = vllm_config.compilation_config"""

# ---- Hunk 2: _create_padded_batch_descriptor accepts override ----
_OLD_CREATE = """    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]"""

_NEW_CREATE = """    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
        uniform_decode_query_len: int | None = None,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        if uniform_decode_query_len is None:
            uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]"""

# ---- Hunk 3: initialize_cudagraph_keys adds draft-FULL keys block ----
# Insert before the trailing `self.keys_initialized = True` line that closes
# `initialize_cudagraph_keys`. Anchor on a unique multi-line region that
# includes the closing `)` of the prior add_cudagraph_key call.
_OLD_INIT_KEYS_TAIL = """                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras
                    ),
                )

        self.keys_initialized = True

    def dispatch("""

_NEW_INIT_KEYS_TAIL = """                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras
                    ),
                )

        if self.for_draft_model and cudagraph_mode.has_full_cudagraphs():
            max_num_tokens = self.vllm_config.scheduler_config.max_num_seqs
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when full mode is enabled."
            )
            capture_sizes_for_draft_model = []
            for size in self.compilation_config.cudagraph_capture_sizes:
                capture_sizes_for_draft_model.append(size)
                if size >= max_num_tokens:
                    break
            for bs, num_active_loras in product(
                capture_sizes_for_draft_model, lora_cases
            ):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras, 1
                    ),
                )

        self.keys_initialized = True

    def dispatch("""

# ---- Hunk 4: dispatch() signature ----
_OLD_DISPATCH_SIG = """        num_active_loras: int = 0,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:"""

_NEW_DISPATCH_SIG = """        num_active_loras: int = 0,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
        uniform_decode_query_len: int | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:"""

# ---- Hunk 5: dispatch() body — normalized_uniform + pass qdl through ----
_OLD_DISPATCH_BODY = """        normalized_uniform = uniform_decode and self.cudagraph_mode.separate_routine()
        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )"""

_NEW_DISPATCH_BODY = """        normalized_uniform = uniform_decode and (
            self.cudagraph_mode.separate_routine() or self.for_draft_model
        )
        batch_desc = self._create_padded_batch_descriptor(
            num_tokens,
            normalized_uniform,
            has_lora,
            effective_num_active_loras,
            uniform_decode_query_len,
        )"""

PATCHES = [
    (_OLD_INIT, _NEW_INIT),
    (_OLD_CREATE, _NEW_CREATE),
    (_OLD_INIT_KEYS_TAIL, _NEW_INIT_KEYS_TAIL),
    (_OLD_DISPATCH_SIG, _NEW_DISPATCH_SIG),
    (_OLD_DISPATCH_BODY, _NEW_DISPATCH_BODY),
]
