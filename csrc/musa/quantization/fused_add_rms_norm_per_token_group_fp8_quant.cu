#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>
#include <torch/all.h>

#include "torch_musa/csrc/core/MUSAGuard.h"

namespace {

constexpr int kHidden = 4096;
constexpr int kHalfHidden = kHidden / 2;
constexpr int kGroupSize = 128;
constexpr int kThreads = 512;
constexpr int kValuesPerHalf = 4;
constexpr float kQuantEpsilon = 1.0e-10f;
constexpr float kFp8Max = 448.0f;
constexpr float kFp8Min = -448.0f;

template <int BLOCK_SIZE>
__device__ __forceinline__ float BlockReduceSum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  if constexpr (BLOCK_SIZE <= 32) {
    return value;
  }

  __shared__ float warp_sums[BLOCK_SIZE / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < BLOCK_SIZE / 32 ? warp_sums[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

__device__ __forceinline__ float WarpReduceMax(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    fused_add_rms_norm_per_token_group_fp8_quant_kernel(
        const T* __restrict__ input, const T* __restrict__ residual,
        const T* __restrict__ weight, T* __restrict__ residual_out,
        __nv_fp8_e4m3* __restrict__ output_q,
        float* __restrict__ output_scale, float epsilon) {
  static_assert(sizeof(T) == 2, "the fused kernel expects a 2-byte input");
  static_assert(kThreads * kValuesPerHalf == kHalfHidden,
                "one block must cover each half-row exactly once");

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int first_offset = row * kHidden + tid * kValuesPerHalf;
  const int second_offset = first_offset + kHalfHidden;

  alignas(8) T input_first[kValuesPerHalf];
  alignas(8) T input_second[kValuesPerHalf];
  alignas(8) T residual_first[kValuesPerHalf];
  alignas(8) T residual_second[kValuesPerHalf];
  *reinterpret_cast<uint2*>(input_first) =
      *reinterpret_cast<const uint2*>(input + first_offset);
  *reinterpret_cast<uint2*>(input_second) =
      *reinterpret_cast<const uint2*>(input + second_offset);
  *reinterpret_cast<uint2*>(residual_first) =
      *reinterpret_cast<const uint2*>(residual + first_offset);
  *reinterpret_cast<uint2*>(residual_second) =
      *reinterpret_cast<const uint2*>(residual + second_offset);

  float sums_first[kValuesPerHalf];
  float sums_second[kValuesPerHalf];
  float square_pairs[kValuesPerHalf];
  alignas(8) T rounded_first[kValuesPerHalf];
  alignas(8) T rounded_second[kValuesPerHalf];
#pragma unroll
  for (int index = 0; index < kValuesPerHalf; ++index) {
    const float first = static_cast<float>(input_first[index]) +
                        static_cast<float>(residual_first[index]);
    const float second = static_cast<float>(input_second[index]) +
                         static_cast<float>(residual_second[index]);
    sums_first[index] = first;
    sums_second[index] = second;
    rounded_first[index] = static_cast<T>(first);
    rounded_second[index] = static_cast<T>(second);
    // Match the current Inductor R0_BLOCK=2048 loop: corresponding squares
    // from the two half-rows are accumulated before the 2048-element reduce.
    square_pairs[index] = first * first + second * second;
  }
  *reinterpret_cast<uint2*>(residual_out + first_offset) =
      *reinterpret_cast<uint2*>(rounded_first);
  *reinterpret_cast<uint2*>(residual_out + second_offset) =
      *reinterpret_cast<uint2*>(rounded_second);

  // Match the generated sizePerThread=4 reduction order.
  float local_square_sum = square_pairs[0] + square_pairs[1];
  local_square_sum = square_pairs[2] + local_square_sum;
  local_square_sum = square_pairs[3] + local_square_sum;

  const float total_square_sum = BlockReduceSum<kThreads>(local_square_sum);
  const float inv_rms = rsqrtf(total_square_sum * (1.0f / 4096.0f) + epsilon);

  alignas(8) T weight_first[kValuesPerHalf];
  alignas(8) T weight_second[kValuesPerHalf];
  *reinterpret_cast<uint2*>(weight_first) =
      *reinterpret_cast<const uint2*>(weight + tid * kValuesPerHalf);
  *reinterpret_cast<uint2*>(weight_second) =
      *reinterpret_cast<const uint2*>(weight + kHalfHidden +
                                     tid * kValuesPerHalf);

  alignas(8) T weighted_first[kValuesPerHalf];
  alignas(8) T weighted_second[kValuesPerHalf];
  float first_absmax = kQuantEpsilon;
  float second_absmax = kQuantEpsilon;
#pragma unroll
  for (int index = 0; index < kValuesPerHalf; ++index) {
    // The pinned MUSA Inductor lowering folds the nominal intermediate cast
    // through the gain multiply. Keep normalized values in FP32 and round
    // only the weighted result, matching the compiled serving graph.
    const float first_normalized = sums_first[index] * inv_rms;
    const float second_normalized = sums_second[index] * inv_rms;
    weighted_first[index] = static_cast<T>(
        first_normalized * static_cast<float>(weight_first[index]));
    weighted_second[index] = static_cast<T>(
        second_normalized * static_cast<float>(weight_second[index]));
    first_absmax = fmaxf(
        first_absmax, fabsf(static_cast<float>(weighted_first[index])));
    second_absmax = fmaxf(
        second_absmax, fabsf(static_cast<float>(weighted_second[index])));
  }

  const float first_group_absmax = WarpReduceMax(first_absmax);
  const float second_group_absmax = WarpReduceMax(second_absmax);
  const float first_scale = first_group_absmax / kFp8Max;
  const float second_scale = second_group_absmax / kFp8Max;
  const int lane = tid & 31;
  const int first_group = tid >> 5;
  const int second_group = first_group + kHalfHidden / kGroupSize;
  if (lane == 0) {
    output_scale[row * (kHidden / kGroupSize) + first_group] = first_scale;
    output_scale[row * (kHidden / kGroupSize) + second_group] = second_scale;
  }

  union {
    uint8_t bytes[kValuesPerHalf];
    uint32_t packed;
  } quantized_first, quantized_second;
#pragma unroll
  for (int index = 0; index < kValuesPerHalf; ++index) {
    const float first_value = static_cast<float>(weighted_first[index]);
    const float second_value = static_cast<float>(weighted_second[index]);
    const float first_clamped =
        fminf(fmaxf(first_value / first_scale, kFp8Min), kFp8Max);
    const float second_clamped =
        fminf(fmaxf(second_value / second_scale, kFp8Min), kFp8Max);
    const __nv_fp8_e4m3 first_fp8 = __nv_fp8_e4m3(first_clamped);
    const __nv_fp8_e4m3 second_fp8 = __nv_fp8_e4m3(second_clamped);
    quantized_first.bytes[index] =
        *reinterpret_cast<const uint8_t*>(&first_fp8);
    quantized_second.bytes[index] =
        *reinterpret_cast<const uint8_t*>(&second_fp8);
  }
  *reinterpret_cast<uint32_t*>(output_q + first_offset) =
      quantized_first.packed;
  *reinterpret_cast<uint32_t*>(output_q + second_offset) =
      quantized_second.packed;
}

inline bool IsAligned16(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

struct TensorRange {
  uintptr_t begin;
  uintptr_t end;
};

TensorRange GetTensorRange(const torch::Tensor& tensor) {
  const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  return {begin, begin + tensor.nbytes()};
}

bool RangesOverlap(const TensorRange& left, const TensorRange& right) {
  return left.begin < right.end && right.begin < left.end;
}

}  // namespace

void fused_add_rms_norm_per_token_group_fp8_quant(
    const torch::Tensor& input, const torch::Tensor& residual,
    const torch::Tensor& weight, torch::Tensor& residual_out,
    torch::Tensor& output_q, torch::Tensor& output_scale, double epsilon) {
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
                  weight.is_contiguous() && residual_out.is_contiguous() &&
                  output_q.is_contiguous() && output_scale.is_contiguous(),
              "fused RMSNorm quant requires contiguous tensors");
  TORCH_CHECK(input.device() == residual.device() &&
                  input.device() == weight.device() &&
                  input.device() == residual_out.device() &&
                  input.device() == output_q.device() &&
                  input.device() == output_scale.device(),
              "all fused RMSNorm quant tensors must share one device");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHidden,
              "fused RMSNorm quant requires input shape [M, 4096]");
  TORCH_CHECK(residual.sizes() == input.sizes(),
              "residual shape must equal input shape");
  TORCH_CHECK(residual_out.sizes() == input.sizes(),
              "residual_out shape must equal input shape");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == kHidden,
              "weight shape must be [4096]");
  TORCH_CHECK(output_q.sizes() == input.sizes(),
              "output_q shape must equal input shape");
  TORCH_CHECK(output_scale.dim() == 2 &&
                  output_scale.size(0) == input.size(0) &&
                  output_scale.size(1) == kHidden / kGroupSize,
              "output_scale shape must be [M, 32]");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
              "fused RMSNorm quant initially supports BF16 only");
  TORCH_CHECK(residual.scalar_type() == input.scalar_type() &&
                  weight.scalar_type() == input.scalar_type() &&
                  residual_out.scalar_type() == input.scalar_type(),
              "input, residual, weight, and residual_out dtypes must match");
  TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "output_q must be float8_e4m3fn");
  TORCH_CHECK(output_scale.scalar_type() == at::ScalarType::Float,
              "output_scale must be float32");
  TORCH_CHECK(IsAligned16(input.data_ptr()) &&
                  IsAligned16(residual.data_ptr()) &&
                  IsAligned16(weight.data_ptr()) &&
                  IsAligned16(residual_out.data_ptr()) &&
                  IsAligned16(output_q.data_ptr()) &&
                  IsAligned16(output_scale.data_ptr()),
              "fused RMSNorm quant requires 16-byte aligned tensor bases");

  const torch::Tensor tensors[] = {input, residual, weight, residual_out,
                                   output_q, output_scale};
  for (int left = 0; left < 6; ++left) {
    for (int right = left + 1; right < 6; ++right) {
      TORCH_CHECK(!RangesOverlap(GetTensorRange(tensors[left]),
                                 GetTensorRange(tensors[right])),
                  "fused RMSNorm quant does not support overlapping tensors");
    }
  }

  if (input.size(0) == 0) {
    return;
  }
  const c10::musa::OptionalMUSAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_add_rms_norm_per_token_group_fp8_quant_kernel<c10::BFloat16>
      <<<input.size(0), kThreads, 0, stream>>>(
          static_cast<const c10::BFloat16*>(input.data_ptr()),
          static_cast<const c10::BFloat16*>(residual.data_ptr()),
          static_cast<const c10::BFloat16*>(weight.data_ptr()),
          static_cast<c10::BFloat16*>(residual_out.data_ptr()),
          static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
          static_cast<float*>(output_scale.data_ptr()),
          static_cast<float>(epsilon));
}
