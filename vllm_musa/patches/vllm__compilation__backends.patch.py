# SPDX-License-Identifier: Apache-2.0
"""MUSA cat-6 object patch: accept (and ignore) torch.compile backend keyword
options on this vLLM snapshot (was inline
``_patch_vllm_backend_call_options``)."""

from functools import wraps

from vllm.logger import init_logger

logger = init_logger(__name__)

PATCHES: list = []


def apply() -> None:
    try:
        from vllm.compilation.backends import VllmBackend
    except Exception as e:
        logger.debug("Skipping VllmBackend options patch: %s", e)
        return

    original_call = VllmBackend.__call__
    if getattr(original_call, "_musa_accepts_backend_options", False):
        return

    @wraps(original_call)
    def call_with_ignored_options(self, graph, example_inputs, **kwargs):
        return original_call(self, graph, example_inputs)

    call_with_ignored_options._musa_accepts_backend_options = True
    VllmBackend.__call__ = call_with_ignored_options
