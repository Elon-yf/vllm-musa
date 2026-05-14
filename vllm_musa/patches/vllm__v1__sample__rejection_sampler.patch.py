# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.sample.rejection_sampler.

MUSA-0064: vLLM's spec-decode rejection-sampler Triton kernels
(rejection_greedy_sample_kernel / rejection_random_sample_kernel) take a
`synthetic_conditional_rates_ptr` argument that is `None` in the
non-synthetic mode (the default). MUSA's Triton (3.1.0) rejects the
None-pointer handling for that kernel arg at compile time with
`ValueError('Cannot bitcast data-type of size 8 to data-type of size 1')`,
crashing engine init whenever speculative decoding is enabled (observed
with the MiniMax-M2.5 Eagle3 draft, num_speculative_tokens 3 and 5).

Both kernels gate every read of that pointer behind
`SYNTHETIC_MODE: tl.constexpr`, so substituting a 1-element dummy
float32 tensor for `None` gives Triton a well-typed pointer to compile
against while the dummy is never actually read (SYNTHETIC_MODE is False
whenever synthetic_conditional_rates was None). `device` and `torch` are
both already in scope at the insertion point in `rejection_sample()`.
"""

PATCHES = [
    (
        """    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
""",
        """    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

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
]
