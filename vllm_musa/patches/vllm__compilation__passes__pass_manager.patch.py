# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.passes.pass_manager.

MUSA-0047: upstream pass_manager.py gates several fusion-pass imports
on `if current_platform.is_cuda():`, which excludes MUSA. The
`MiniMaxQKNormPass` class is in turn referenced unconditionally when
`pass_config.fuse_minimax_qk_norm=True` (line ~142), producing:

    NameError: name 'MiniMaxQKNormPass' is not defined

during Inductor backend init on MiniMax-M2.7 at TP=8 with the
upstream-recommended compile config
`{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}`.

The MUSA build includes the kernel
`minimax_reduce_rms_kernel.cu` (in VLLM_CSRC_SOURCES per
`vllm-musa/setup.py`) which the pass replaces inlined QK
allreduce+RMS with. Extend the import gate to also fire on MUSA.

The two other CUDA-gated imports — AllReduceFusionPass and AsyncTPPass
— also touch MUSA-affecting paths but are out of scope for MUSA-0047.
Bring them along; they currently aren't activated unless their
pass_config flags are flipped on.
"""

PATCHES = [
    (
        """if current_platform.is_cuda():
    from .fusion.allreduce_rms_fusion import AllReduceFusionPass
    from .fusion.collective_fusion import AsyncTPPass
    from .fusion.minimax_qk_norm_fusion import MiniMaxQKNormPass""",
        """if current_platform.is_cuda() or current_platform.is_musa():
    from .fusion.allreduce_rms_fusion import AllReduceFusionPass
    from .fusion.collective_fusion import AsyncTPPass
    from .fusion.minimax_qk_norm_fusion import MiniMaxQKNormPass""",
    ),
]
