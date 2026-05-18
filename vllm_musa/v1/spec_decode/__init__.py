# Eagle3 full-loop spec-decode runner for MUSA.
#
# Per MUSA-0090, this package implements an SGLang-style EAGLEDraftCudaGraphRunner
# adapted to vLLM's types. The runner captures the full N-step Eagle3 draft loop
# in one cudagraph (vs vLLM's iterative per-step PIECEWISE dispatch), reducing
# host-side overhead from ~30 ms per spec round to ~3 ms.
#
# Design doc: generated/musa0090_impl/step1-design-doc.md
# Q1 smoke: generated/musa0090_impl/step1_5-q1-cudagraph-smoke-result.md
#
# Public surface — used by the patch at:
#     vllm-musa/vllm_musa/patches/vllm__v1__spec_decode__eagle.patch.py
#
# Imports are lazy on first use to avoid pulling vllm.v1.spec_decode types
# at vllm-musa-init time (the patches/ system needs to monkey-patch them).

# Existing MUSA-Triton kernel for prepare_next_token_padded — pre-MUSA-0090.
from . import utils  # noqa: F401
