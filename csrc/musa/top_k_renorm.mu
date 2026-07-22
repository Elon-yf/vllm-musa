/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stateless top-k=50 probability renormalization using three-pass radix
 * selection over probabilities. Values outside [0, 1], including NaN/Inf,
 * are treated as zero so malformed rows cannot index the radix histogram out
 * of bounds.
 */

#include <cstdint>

#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include "musa_ops.h"

namespace {

constexpr int kTopK = 50;
constexpr int kCandidateCapacity = 512;
constexpr unsigned int kOneBits = 0x3f800000u;

// IEEE-754 bit order is monotonic for finite non-negative values. Therefore a
// single unsigned comparison accepts exactly +0 through +1 and rejects
// negative values, NaN, Inf, and values greater than one.
__device__ __forceinline__ float sanitize_probability(float value) {
  return __float_as_uint(value) <= kOneBits ? value : 0.0f;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int delta = 16; delta; delta >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, delta);
  }
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < (blockDim.x >> 5) ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int delta = 16; delta; delta >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, delta);
    }
  }
  if (threadIdx.x == 0) {
    scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// Select the rank-th value from two 1024-bin radix shards. Values are
// non-negative finite probabilities, so their IEEE-754 bit order is monotonic.
__device__ __forceinline__ void select_radix2(const unsigned int* histogram,
                                               unsigned int prefix,
                                               unsigned int* selected,
                                               int* rank) {
  const int lane = threadIdx.x & 31;
  if (threadIdx.x < 32) {
    const int begin = lane << 5;
    int count = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      count += static_cast<int>(histogram[begin + i] +
                                histogram[1024 + begin + i]);
    }
    int suffix = count;
    #pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
      const int next = __shfl_down_sync(0xffffffffu, suffix, delta);
      if (lane + delta < 32) {
        suffix += next;
      }
    }
    const int above = suffix - count;
    const int wanted = *rank;
    const unsigned int mask =
        __ballot_sync(0xffffffffu, wanted > above && wanted <= suffix);
    const int winner = __ffs(static_cast<int>(mask)) - 1;
    if (lane == winner) {
      int local_rank = wanted - above;
      for (int bin = begin + 31; bin >= begin; --bin) {
        const int bin_count = static_cast<int>(
            histogram[bin] + histogram[1024 + bin]);
        if (local_rank > bin_count) {
          local_rank -= bin_count;
        } else {
          *selected = (prefix << 10) | static_cast<unsigned int>(bin);
          *rank = local_rank;
          break;
        }
      }
    }
  }
  __syncthreads();
}

// Two-read radix-select path. The kernel is stateless across launches and uses
// no device-global counters or inter-block rendezvous.
template <int Capacity, int BlockSize>
__global__ __launch_bounds__(BlockSize, 1) void top_k_renorm_kernel(
    const float4* __restrict__ input, float4* __restrict__ output, int rows,
    int vector_count) {
  const int thread = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int shard = thread / (BlockSize / 2);
  const int local_thread = thread & (BlockSize / 2 - 1);
  __shared__ unsigned int histogram[2][1024];
  __shared__ unsigned int prefix;
  __shared__ int rank;
  __shared__ int candidate_count;
  __shared__ int invalid_probability;
  __shared__ float candidates[Capacity];
  __shared__ int candidate_indices[Capacity];
  __shared__ float reduction[BlockSize / 32];

  if (thread == 0) {
    prefix = 0;
    rank = kTopK;
    candidate_count = 0;
    invalid_probability = 0;
  }
  __syncthreads();

  for (int i = local_thread; i < 1024; i += BlockSize / 2) {
    histogram[shard][i] = 0;
  }
  __syncthreads();

  const int64_t row_base = static_cast<int64_t>(row) * vector_count;
  for (int vector = thread; vector < vector_count; vector += BlockSize) {
    const float4 value = input[row_base + vector];
    output[row_base + vector] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    unsigned int bits[4] = {
        __float_as_uint(value.x), __float_as_uint(value.y),
        __float_as_uint(value.z), __float_as_uint(value.w)};
    bool invalid = false;
    #pragma unroll
    for (int component = 0; component < 4; ++component) {
      if (bits[component] > kOneBits) {
        bits[component] = 0;
        invalid = true;
      }
      atomicAdd(&histogram[shard][bits[component] >> 20], 1);
    }
    if (invalid) {
      atomicExch(&invalid_probability, 1);
    }
  }
  __syncthreads();
  select_radix2(&histogram[0][0], 0, &prefix, &rank);

  const unsigned int exponent_prefix = prefix;
  float greater_sum = 0.0f;
  for (int i = local_thread; i < 1024; i += BlockSize / 2) {
    histogram[shard][i] = 0;
  }
  __syncthreads();

  for (int vector = thread; vector < vector_count; vector += BlockSize) {
    const float4 value = input[row_base + vector];
    float values[4] = {
        invalid_probability ? sanitize_probability(value.x) : value.x,
        invalid_probability ? sanitize_probability(value.y) : value.y,
        invalid_probability ? sanitize_probability(value.z) : value.z,
        invalid_probability ? sanitize_probability(value.w) : value.w};
    #pragma unroll
    for (int component = 0; component < 4; ++component) {
      const unsigned int bits = __float_as_uint(values[component]);
      const unsigned int exponent = bits >> 20;
      if (exponent >= exponent_prefix) {
        if (exponent > exponent_prefix) {
          greater_sum += values[component];
        } else {
          atomicAdd(&histogram[shard][(bits >> 10) & 1023], 1);
        }
        const int index = atomicAdd(&candidate_count, 1);
        if (index < Capacity) {
          candidates[index] = values[component];
          candidate_indices[index] = vector * 4 + component;
        }
      }
    }
  }
  __syncthreads();
  select_radix2(&histogram[0][0], exponent_prefix, &prefix, &rank);

  for (int i = local_thread; i < 1024; i += BlockSize / 2) {
    histogram[shard][i] = 0;
  }
  __syncthreads();
  const unsigned int mantissa_prefix = prefix;
  if (candidate_count <= Capacity) {
    for (int i = thread; i < candidate_count; i += BlockSize) {
      const unsigned int bits = __float_as_uint(candidates[i]);
      if ((bits >> 10) == mantissa_prefix) {
        atomicAdd(&histogram[shard][bits & 1023], 1);
      }
    }
  } else {
    for (int vector = thread; vector < vector_count; vector += BlockSize) {
      const float4 value = input[row_base + vector];
      const float values[4] = {
          invalid_probability ? sanitize_probability(value.x) : value.x,
          invalid_probability ? sanitize_probability(value.y) : value.y,
          invalid_probability ? sanitize_probability(value.z) : value.z,
          invalid_probability ? sanitize_probability(value.w) : value.w};
      #pragma unroll
      for (int component = 0; component < 4; ++component) {
        const unsigned int bits = __float_as_uint(values[component]);
        if ((bits >> 10) == mantissa_prefix) {
          atomicAdd(&histogram[shard][bits & 1023], 1);
        }
      }
    }
  }
  __syncthreads();
  select_radix2(&histogram[0][0], mantissa_prefix, &prefix, &rank);
  const unsigned int threshold = prefix;

  float selected_sum = 0.0f;
  if (candidate_count <= Capacity) {
    for (int i = thread; i < candidate_count; i += BlockSize) {
      const float value = candidates[i];
      const unsigned int bits = __float_as_uint(value);
      if ((bits >> 20) == exponent_prefix && bits >= threshold) {
        selected_sum += value;
      }
    }
  } else {
    for (int vector = thread; vector < vector_count; vector += BlockSize) {
      const float4 value = input[row_base + vector];
      const float values[4] = {
          invalid_probability ? sanitize_probability(value.x) : value.x,
          invalid_probability ? sanitize_probability(value.y) : value.y,
          invalid_probability ? sanitize_probability(value.z) : value.z,
          invalid_probability ? sanitize_probability(value.w) : value.w};
      #pragma unroll
      for (int component = 0; component < 4; ++component) {
        const unsigned int bits = __float_as_uint(values[component]);
        if ((bits >> 20) == exponent_prefix && bits >= threshold) {
          selected_sum += values[component];
        }
      }
    }
  }

  const float selected_total =
      block_sum(greater_sum + selected_sum, reduction);
  const float inverse = selected_total > 0.0f ? 1.0f / selected_total : 0.0f;
  if (candidate_count <= Capacity) {
    float* output_row = reinterpret_cast<float*>(output + row_base);
    for (int i = thread; i < candidate_count; i += BlockSize) {
      const float value = candidates[i];
      output_row[candidate_indices[i]] =
          __float_as_uint(value) >= threshold ? value * inverse : 0.0f;
    }
  } else {
    for (int vector = thread; vector < vector_count; vector += BlockSize) {
      const float4 value = input[row_base + vector];
      const float values[4] = {
          invalid_probability ? sanitize_probability(value.x) : value.x,
          invalid_probability ? sanitize_probability(value.y) : value.y,
          invalid_probability ? sanitize_probability(value.z) : value.z,
          invalid_probability ? sanitize_probability(value.w) : value.w};
      float4 normalized;
      normalized.x = __float_as_uint(values[0]) >= threshold
                         ? values[0] * inverse
                         : 0.0f;
      normalized.y = __float_as_uint(values[1]) >= threshold
                         ? values[1] * inverse
                         : 0.0f;
      normalized.z = __float_as_uint(values[2]) >= threshold
                         ? values[2] * inverse
                         : 0.0f;
      normalized.w = __float_as_uint(values[3]) >= threshold
                         ? values[3] * inverse
                         : 0.0f;
      output[row_base + vector] = normalized;
    }
  }
}

template <int BlockSize>
void launch_top_k_renorm(const float* input, float* output, int rows,
                         int vocab, musaStream_t stream) {
  const int vector_count = vocab / 4;
  auto input_vec = reinterpret_cast<const float4*>(input);
  auto output_vec = reinterpret_cast<float4*>(output);
  top_k_renorm_kernel<kCandidateCapacity, BlockSize>
      <<<rows, BlockSize, 0, stream>>>(input_vec, output_vec, rows,
                                        vector_count);
}

}  // namespace

void musa_rubymine_top_k_renorm_probs(at::Tensor probs,
                                      at::Tensor renorm_probs,
                                      int64_t top_k_val) {
  TORCH_CHECK(renorm_probs.device() == probs.device(),
              "top-k renorm output must be on the input device");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32,
              "top-k renorm fast path requires float32 probabilities");
  TORCH_CHECK(renorm_probs.scalar_type() == torch::kFloat32,
              "top-k renorm output must be float32");
  TORCH_CHECK(probs.dim() == 2 && renorm_probs.sizes() == probs.sizes(),
              "top-k renorm expects matching 2-D tensors");
  TORCH_CHECK(probs.is_contiguous() && renorm_probs.is_contiguous(),
              "top-k renorm expects contiguous tensors");
  TORCH_CHECK(top_k_val == kTopK,
              "top-k fast path is specialized for k=50");

  const int rows = static_cast<int>(probs.size(0));
  const int vocab = static_cast<int>(probs.size(1));
  TORCH_CHECK(rows >= 0 && rows <= 256,
              "top-k renorm fast path supports 0 <= batch <= 256");
  TORCH_CHECK(vocab >= kTopK && (vocab % 4) == 0,
              "top-k renorm fast path requires vocab >= 50 and divisible by 4");
  if (rows == 0) {
    return;
  }

  const c10::musa::OptionalMUSAGuard device_guard(probs.device());
  auto stream = at::musa::getCurrentMUSAStream();
  // The 1024-thread launch is useful only for the smallest decode batches;
  // keep the larger serving batches at 512 threads for register pressure.
  if (rows <= 8) {
    launch_top_k_renorm<1024>(probs.data_ptr<float>(),
                              renorm_probs.data_ptr<float>(), rows, vocab,
                              stream);
  } else {
    launch_top_k_renorm<512>(probs.data_ptr<float>(),
                             renorm_probs.data_ptr<float>(), rows, vocab,
                             stream);
  }
  const musaError_t error = musaGetLastError();
  TORCH_CHECK(error == musaSuccess, "top-k renorm kernel failed: ",
              musaGetErrorString(error));
}
