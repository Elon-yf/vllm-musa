// SPDX-License-Identifier: Apache-2.0
// MUSA-0123: Python-facing torch op wrapper for fused_ar_add_rmsnorm.
//
// Bridges torch tensors → the C-ABI launcher in fused_ar_add_rmsnorm.mu
// by looking up CustomAllreduce state (RankData / RankSignals / Signal)
// from the fptr_t handle.
//
// This file is a .cu so it can #include "custom_all_reduce.cuh"
// (which the .mu file can't, see comments there).

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "custom_all_reduce.cuh"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

// Forward decl of the launcher (defined in fused_ar_add_rmsnorm.mu).
extern "C" void fused_ar_add_rmsnorm_bf16_tp8_launch(
    void* dp,
    void* sg_signals,
    void* self_sg,
    const __mt_bfloat16* residual,
    const __mt_bfloat16* gamma,
    __mt_bfloat16* output,
    int num_rows,
    int hidden_size,
    float epsilon,
    int rank,
    musaStream_t stream);

using fptr_t = int64_t;

// musa_fused_ar_rmsnorm:
//   - _fa: opaque CustomAllreduce* handle (same fptr_t used by
//     existing all_reduce op)
//   - input: 2-D bf16 tensor [rows, hidden]. Must be a
//     CustomAllreduce-registered buffer (caller is responsible for
//     ensuring this — typically achieved by passing the same buffer
//     used for the AR input).
//   - residual: 2-D bf16 tensor, same shape as input.
//   - weight: 1-D bf16 tensor of length hidden (RMS-norm gamma).
//   - output: 2-D bf16 tensor, same shape as input, written by the op.
//   - epsilon: RMS-norm epsilon.
//
// Restrictions of this first cut:
//   - bf16 only
//   - TP=8 only
//   - hidden % 8 == 0
//   - vec_hidden (hidden/8) must be <= 2048
//
// API: matches the existing `_C_custom_ar.all_reduce` pattern of taking
// (fa, input, output, reg_buffer, reg_buffer_sz). `input` is the local
// rank's data tensor; the wrapper copies it into the IPC-registered
// `reg_buffer` (pre-AR), then runs the fused kernel reading from peer
// reg_buffers.
void musa_fused_ar_rmsnorm(fptr_t _fa,
                           torch::Tensor& input,
                           torch::Tensor& residual,
                           torch::Tensor& weight,
                           torch::Tensor& output,
                           fptr_t _reg_buffer,
                           int64_t reg_buffer_sz_bytes,
                           double epsilon) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
              "MUSA-0123 op requires BF16 (got ", input.scalar_type(), ")");
  TORCH_CHECK(residual.scalar_type() == at::ScalarType::BFloat16,
              "residual must be BF16");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16,
              "weight must be BF16");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "output must be BF16");
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
                  weight.is_contiguous() && output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && residual.dim() == 2 && output.dim() == 2,
              "input/residual/output must be 2-D [rows, hidden]");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D [hidden]");
  TORCH_CHECK(input.sizes() == residual.sizes(),
              "input/residual shape mismatch");
  TORCH_CHECK(input.sizes() == output.sizes(), "input/output shape mismatch");
  TORCH_CHECK(input.size(1) == weight.size(0), "weight size mismatch");

  const int num_rows = static_cast<int>(input.size(0));
  const int hidden_size = static_cast<int>(input.size(1));
  TORCH_CHECK(hidden_size % 8 == 0, "hidden_size must be multiple of 8");
  TORCH_CHECK(hidden_size <= 16384, "hidden_size must be <= 16384");

  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(fa->world_size_ == 8,
              "MUSA-0123 op currently supports TP=8 only (got world_size=",
              fa->world_size_, ")");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();

  // Copy input → reg_buffer (the IPC-registered buffer). After this,
  // each rank's reg_buffer contains its own pre-AR data; the fused
  // kernel reads ALL 8 peers' reg_buffers via the RankData* pointers.
  void* reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  const int64_t input_bytes =
      static_cast<int64_t>(input.numel() * input.element_size());
  TORCH_CHECK(reg_buffer != nullptr, "reg_buffer must be non-null");
  TORCH_CHECK(input_bytes <= reg_buffer_sz_bytes,
              "input (", input_bytes, "B) exceeds reg_buffer (",
              reg_buffer_sz_bytes, "B)");
  AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, input.data_ptr(), input_bytes,
                                cudaMemcpyDeviceToDevice, stream));

  // Look up RankData* for reg_buffer. CustomAllreduce already
  // registered this buffer at init time (its own buffer_ptrs[rank]).
  vllm::RankData* ptrs;
  cudaStreamCaptureStatus status;
  CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));
  if (status == cudaStreamCaptureStatusActive) {
    ptrs = fa->d_rank_data_base_ + fa->graph_unreg_buffers_.size();
    fa->graph_unreg_buffers_.push_back(reg_buffer);
  } else {
    auto it = fa->buffers_.find(reg_buffer);
    TORCH_CHECK(it != fa->buffers_.end(),
                "MUSA-0123: reg_buffer not registered in CustomAllreduce");
    ptrs = it->second;
  }

  fused_ar_add_rmsnorm_bf16_tp8_launch(
      ptrs,
      fa->sg_.signals,        // address of host-side Signal*[8] array
      fa->self_sg_,
      static_cast<const __mt_bfloat16*>(residual.data_ptr()),
      static_cast<const __mt_bfloat16*>(weight.data_ptr()),
      static_cast<__mt_bfloat16*>(output.data_ptr()),
      num_rows,
      hidden_size,
      static_cast<float>(epsilon),
      fa->rank_,
      stream);
}
