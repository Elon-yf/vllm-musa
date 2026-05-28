# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0203: backport of vllm-project/vllm#34880 — gpu_input_batch.py hunks.

Adds `num_reqs_padded` property to ``InputBatch`` so the spec-decode
draft path can request a padded batch_size for FULL cudagraph capture
keys without changing the actual scheduler request count.
"""

# ---- Hunk 1: __init__ adds _num_reqs_padded attribute ----
_OLD_INIT = """        self.sampled_token_ids_cpu: torch.Tensor | None = None
        self.async_copy_ready_event: torch.Event | None = None

    @property
    def req_ids(self) -> list[str]:"""

_NEW_INIT = """        self.sampled_token_ids_cpu: torch.Tensor | None = None
        self.async_copy_ready_event: torch.Event | None = None

        self._num_reqs_padded: int | None = None

    @property
    def req_ids(self) -> list[str]:"""

# ---- Hunk 2: condense() — reset on empty-batch path ----
_OLD_CONDENSE_EMPTY = """            self._req_ids.clear()
            self.req_output_token_ids.clear()
            self.spec_token_ids.clear()
            return"""

_NEW_CONDENSE_EMPTY = """            self._req_ids.clear()
            self.req_output_token_ids.clear()
            self.spec_token_ids.clear()
            self._num_reqs_padded = None
            return"""

# ---- Hunk 3: refresh_metadata() — reset cached padded count ----
_OLD_REFRESH = """    def refresh_metadata(self):
        \"\"\"Apply any batch updates to sampling metadata.\"\"\"

        if self.is_pooling_model:"""

_NEW_REFRESH = """    def refresh_metadata(self):
        \"\"\"Apply any batch updates to sampling metadata.\"\"\"

        self._num_reqs_padded = None

        if self.is_pooling_model:"""

# ---- Hunk 4: num_reqs property — add num_reqs_padded ----
_OLD_NUM_REQS = """    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:"""

_NEW_NUM_REQS = """    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def num_reqs_padded(self) -> int:
        if self._num_reqs_padded is None:
            return self.num_reqs
        return self._num_reqs_padded

    @num_reqs_padded.setter
    def num_reqs_padded(self, num_reqs_padded: int) -> None:
        self._num_reqs_padded = num_reqs_padded

    @property
    def all_greedy(self) -> bool:"""

PATCHES = [
    (_OLD_INIT, _NEW_INIT),
    (_OLD_CONDENSE_EMPTY, _NEW_CONDENSE_EMPTY),
    (_OLD_REFRESH, _NEW_REFRESH),
    (_OLD_NUM_REQS, _NEW_NUM_REQS),
]
