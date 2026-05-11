# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.utils.deep_gemm.
"""

PATCHES = [
    # Patch is_deep_gemm_supported to support musa device type
    (
        "is_supported_arch = current_platform.is_cuda()",
        "is_supported_arch = current_platform.is_musa()",
    ),
    # Patch is_device_capability to support musa device type
    (
        "current_platform.is_device_capability(90)",
        "current_platform.is_device_capability(31)",
    ),
    # Patch get_mk_alignment_for_contiguous_layout to support musa deepep
    (
        "mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()",
        "mk_align_size = 128",
    ),
    # MUSA DeepGEMM does not support grouped FP8 UE8M0 casts yet. Keep the
    # default DeepGEMM FP8 path enabled, but force the oracle to use FLOAT32
    # scales on MUSA so MoE warmup and runtime do not select UE8M0 layouts.
    (
        """        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return
""",
        """        if use_e8m0 and current_platform.is_musa():
            logger.info_once(
                "DeepGEMM E8M0 disabled on MUSA: grouped FP8 UE8M0 cast is "
                "not supported by the MUSA DeepGEMM backend."
            )
            use_e8m0 = False

        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return
""",
    ),
    (
        """    if envs.VLLM_USE_DEEP_GEMM_E8M0:
        logger.info_once("DeepGEMM E8M0 enabled on current platform.")
        return True
""",
        """    if envs.VLLM_USE_DEEP_GEMM_E8M0:
        if current_platform.is_musa():
            logger.info_once(
                "DeepGEMM E8M0 disabled on MUSA: grouped FP8 UE8M0 cast is "
                "not supported by the MUSA DeepGEMM backend."
            )
            return False
        logger.info_once("DeepGEMM E8M0 enabled on current platform.")
        return True
""",
    ),
]
