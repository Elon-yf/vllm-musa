# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.sample.rejection_sampler.

MUSA-0064: vLLM's spec-decode rejection-sampler Triton kernels do not
compile on MUSA's Triton (3.1.0), crashing engine init whenever
speculative decoding is enabled (observed with the MiniMax-M2.5 Eagle3
draft). The triton.compiler.errors.CompilationError caret points at
`if not is_greedy:` in rejection_greedy_sample_kernel:

    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
       ^
    ValueError('Cannot bitcast data-type of size 8 to data-type of size 1')

Root cause: `is_greedy_ptr` points to a `torch.bool` tensor; MUSA Triton
3.1.0 mishandles `not` / truthiness applied to a bool-typed `tl.load`'d
value (it tries to bitcast the size-8 pointer/value to the size-1 bool
type). CUDA Triton handles it; MUSA Triton 3.1.0 does not.

Fix (three coordinated replacements):
  1. Call site `rejection_sample()` — pass `is_greedy` as int32 instead
     of torch.bool so the kernels load a plain integer.
  2. rejection_greedy_sample_kernel — `True if ... else tl.load` becomes
     `1 if ... else tl.load`, and `if not is_greedy:` becomes the
     explicit `if is_greedy == 0:`.
  3. rejection_random_sample_kernel — `if is_greedy:` becomes the
     explicit `if is_greedy != 0:`.
Plus a defensive 1-element dummy tensor for the `synthetic_conditional_
rates` None pointer (kernels gate every read of it behind
`SYNTHETIC_MODE: tl.constexpr`, so the dummy is never read).

All replacements are behaviourally identical on CUDA; they only avoid
the MUSA-Triton-3.1.0 bool-bitcast path. `device` / `torch` are in
scope at the `rejection_sample()` insertion point.
"""

PATCHES = [
    # 1 + dummy: call site -- is_greedy as int32 + synthetic dummy tensor.
    (
        """    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
""",
        """    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        # MUSA-0064: int32, not torch.bool -- MUSA Triton 3.1.0 chokes on
        # `not` / truthiness of a bool-typed tl.load'd value.
        is_greedy = (
            sampling_metadata.temperature == GREEDY_TEMPERATURE
        ).to(torch.int32)

    # MUSA-0064: MUSA Triton rejects the None-pointer handling for the
    # `synthetic_conditional_rates_ptr` kernel arg. Pass a 1-element
    # dummy tensor when synthetic mode is off; both rejection-sampler
    # kernels gate every read of it behind `SYNTHETIC_MODE: tl.constexpr`
    # so the dummy is never read.
    if synthetic_conditional_rates is None:
        synthetic_conditional_rates = torch.empty(
            1, dtype=torch.float32, device=device
        )
""",
    ),
    # 2: rejection_greedy_sample_kernel -- avoid `not` on a loaded value.
    (
        """    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return""",
        """    # MUSA-0064: `1`/`== 0` instead of `True`/`not` -- MUSA Triton
    # 3.1.0 cannot bitcast the bool-typed loaded value.
    is_greedy = 1 if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if is_greedy == 0:
        # Early exit for non-greedy sampling requests.
        return""",
    ),
    # 3: rejection_random_sample_kernel -- avoid truthiness on a loaded value.
    (
        """    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return""",
        """    req_idx = tl.program_id(0)
    # MUSA-0064: explicit `!= 0` instead of truthiness on the loaded
    # value -- MUSA Triton 3.1.0 cannot bitcast the bool-typed value.
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy != 0:
        # Early exit for greedy sampling requests.
        return""",
    ),
]
