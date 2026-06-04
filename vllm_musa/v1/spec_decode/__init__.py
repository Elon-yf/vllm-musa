# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# MUSA spec-decode helpers. The only surviving module is `utils`, which
# installs a MUSA-Triton-adapted `eagle_prepare_next_token_padded_kernel`
# replacing the upstream version that fails MUSA Triton compile with
# `mismatched type for valid_count`.

from . import utils  # noqa: F401
