# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared empirically measured kernel-selection thresholds."""

# Minimum rows where the MUSA JIT fused-add RMSNorm provider was measured to
# outperform the fallback for contiguous BF16 H5120 workloads on S5000.
FUSED_ADD_RMSNORM_MIN_ROWS = 64
