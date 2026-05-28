# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# MUSA spec-decode helpers. The only surviving module is `utils`, which
# installs a MUSA-Triton-adapted `eagle_prepare_next_token_padded_kernel`
# replacing the upstream version that fails MUSA Triton compile with
# `mismatched type for valid_count`.
#
# The MUSA-0090/0109 EagleFullLoopRunner was deleted in MUSA-0203 (the
# giant-chain CUDAGraph capture corrupted draft MCCL barrier state on
# yeahdongcn70 / mcc 5.1.0). Its replacement follows upstream PR #34880's
# pattern: wrap the draft model with the standard `CUDAGraphWrapper`
# + per-step dispatch instead of a one-shot N-step capture.

from . import utils  # noqa: F401
