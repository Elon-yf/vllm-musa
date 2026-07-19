/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shape-specialized min-p sampling for Qwen's 151936-token vocabulary.
 * The chunked reduction and multi-CTA dataflow are adapted from RubyMine
 * problem 900015, submission 20732.  The production wrapper below restores
 * FlashInfer's Philox, per-row threshold, and row-index contracts.
 */

#include <ATen/Utils.h>
#include <ATen/core/Generator.h>
#include <flashinfer/sampling.cuh>
#include <torch/all.h>

#include <cfloat>
#include <mutex>
#include <optional>

#include "musa.h"
#include "pytorch_extension_utils.h"
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kRowsPerBlock = kThreads / kWarp;
constexpr int kChunk = 2048;
constexpr int kQwenVocab = 151936;
constexpr int kMaxBatch = 64;

struct alignas(16) Float4 {
  float x, y, z, w;
};

__device__ __forceinline__ Float4 load4_cached(const float* ptr) {
  return *reinterpret_cast<const Float4*>(ptr);
}

__device__ __forceinline__ Float4 load4_bypass(const float* ptr) {
  // The standalone RubyMine donor uses ld.global.cs.v4.f32.  In the full
  // vLLM extension, combining that four-output inline assembly with Philox
  // exhausts the mp31 output-register allocator.  Keep the chunked multi-CTA
  // dataflow first; a later isolated ticket can reintroduce cache bypass with
  // a lower-pressure load helper if end-to-end evidence warrants it.
  return load4_cached(ptr);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_max(float value) {
  __shared__ float warp_values[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) {
    warp_values[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < 8 ? warp_values[lane] : -FLT_MAX;
  if (warp == 0) {
    value = warp_max(value);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float block_sum(float value) {
  __shared__ float warp_values[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) {
    warp_values[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < 8 ? warp_values[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void partial_max_kernel(const float* __restrict__ probs,
                                   const int* __restrict__ indices,
                                   float* __restrict__ partial, int vocab,
                                   int chunks) {
  const int job = blockIdx.x;
  const int row = job / chunks;
  const int chunk = job - row * chunks;
  const int row_idx = indices == nullptr ? row : indices[row];
  const int begin = chunk * kChunk;
  const int end = min(vocab, begin + kChunk);
  const float* row_probs = probs + static_cast<size_t>(row_idx) * vocab;
  float local = -FLT_MAX;
  for (int index = begin + threadIdx.x * 4; index < end;
       index += blockDim.x * 4) {
    if (index + 3 < end) {
      const Float4 value = load4_bypass(row_probs + index);
      local = fmaxf(local, value.x);
      local = fmaxf(local, value.y);
      local = fmaxf(local, value.z);
      local = fmaxf(local, value.w);
    } else {
      for (int tail = index; tail < end; ++tail) {
        local = fmaxf(local, row_probs[tail]);
      }
    }
  }
  const float maximum = block_max(local);
  if (threadIdx.x == 0) {
    partial[job] = maximum;
  }
}

__global__ void threshold_rng_kernel(
    const float* __restrict__ partial, float* __restrict__ threshold,
    float* __restrict__ uniform, const float* __restrict__ min_p_arr,
    float min_p_val, int* __restrict__ output, int batch, int vocab,
    int chunks, uint64_t philox_seed, uint64_t philox_offset) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x & (kWarp - 1);
  const int row = blockIdx.x * kRowsPerBlock + warp;
  if (row >= batch) {
    return;
  }

  float maximum = -FLT_MAX;
  for (int chunk = lane; chunk < chunks; chunk += kWarp) {
    maximum = fmaxf(maximum, partial[row * chunks + chunk]);
  }
  maximum = warp_max(maximum);
  if (lane == 0) {
    const float min_p = min_p_arr == nullptr ? min_p_val : min_p_arr[row];
    murandStatePhilox4_32_10_t state;
    murand_init(philox_seed, row, philox_offset, &state);
    threshold[row] = maximum * min_p;
    uniform[row] = murand_uniform(&state);
    output[row] = vocab - 1;
  }
}

__global__ void partial_sum_kernel(const float* __restrict__ probs,
                                   const int* __restrict__ indices,
                                   const float* __restrict__ threshold,
                                   float* __restrict__ partial, int vocab,
                                   int chunks) {
  const int job = blockIdx.x;
  const int row = job / chunks;
  const int chunk = job - row * chunks;
  const int row_idx = indices == nullptr ? row : indices[row];
  const int begin = chunk * kChunk;
  const int end = min(vocab, begin + kChunk);
  const float* row_probs = probs + static_cast<size_t>(row_idx) * vocab;
  const float cutoff = threshold[row];
  float local = 0.0f;
  for (int index = begin + threadIdx.x * 4; index < end;
       index += blockDim.x * 4) {
    if (index + 3 < end) {
      const Float4 value = load4_bypass(row_probs + index);
      local += value.x >= cutoff ? value.x : 0.0f;
      local += value.y >= cutoff ? value.y : 0.0f;
      local += value.z >= cutoff ? value.z : 0.0f;
      local += value.w >= cutoff ? value.w : 0.0f;
    } else {
      for (int tail = index; tail < end; ++tail) {
        const float value = row_probs[tail];
        local += value >= cutoff ? value : 0.0f;
      }
    }
  }
  const float total = block_sum(local);
  if (threadIdx.x == 0) {
    partial[job] = total;
  }
}

__device__ __forceinline__ float warp_prefix(float value) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float incoming = __shfl_up_sync(0xffffffffu, value, offset);
    if (lane >= offset) {
      value += incoming;
    }
  }
  return value;
}

__global__ void select_kernel(const float* __restrict__ probs,
                              const int* __restrict__ indices,
                              const float* __restrict__ threshold,
                              const float* __restrict__ uniform,
                              const float* __restrict__ partial,
                              int* __restrict__ output, int batch, int vocab,
                              int chunks) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x & (kWarp - 1);
  const int row = blockIdx.x * kRowsPerBlock + warp;
  if (row >= batch) {
    return;
  }

  __shared__ int selected_chunks[kRowsPerBlock];
  __shared__ float selected_targets[kRowsPerBlock];
  __shared__ int selected_tokens[kRowsPerBlock];
  __shared__ int last_valid_tokens[kRowsPerBlock];

  if (lane == 0) {
    float total = 0.0f;
    int last_nonzero_chunk = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      const float chunk_sum = partial[row * chunks + chunk];
      total += chunk_sum;
      if (chunk_sum > 0.0f) {
        last_nonzero_chunk = chunk;
      }
    }
    const float target = uniform[row] * total;
    float prefix = 0.0f;
    int selected = last_nonzero_chunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      const float next = prefix + partial[row * chunks + chunk];
      if (next > target) {
        selected = chunk;
        break;
      }
      prefix = next;
    }
    selected_chunks[warp] = selected;
    selected_targets[warp] = target - prefix;
    selected_tokens[warp] = vocab;
    last_valid_tokens[warp] = -1;
  }
  __syncwarp();

  const int chunk = selected_chunks[warp];
  const int begin = chunk * kChunk;
  const int end = min(vocab, begin + kChunk);
  const int row_idx = indices == nullptr ? row : indices[row];
  const float* row_probs = probs + static_cast<size_t>(row_idx) * vocab;
  const float cutoff = threshold[row];
  const int interval = kChunk / kWarp;
  const int lane_begin = begin + lane * interval;
  const int lane_end = min(end, lane_begin + interval);

  float lane_sum = 0.0f;
  int lane_last_valid = -1;
  for (int index = lane_begin; index < lane_end; ++index) {
    const float value = row_probs[index];
    if (value >= cutoff) {
      lane_sum += value;
      lane_last_valid = index;
    }
  }
  const float inclusive = warp_prefix(lane_sum);
  const float lane_prefix = inclusive - lane_sum;
  if (lane_last_valid >= 0) {
    atomicMax(&last_valid_tokens[warp], lane_last_valid);
  }

  const float target = selected_targets[warp];
  if (lane_prefix <= target && inclusive > target) {
    float cumulative = lane_prefix;
    for (int index = lane_begin; index < lane_end; ++index) {
      const float value = row_probs[index];
      if (value >= cutoff) {
        cumulative += value;
        if (cumulative > target) {
          atomicMin(&selected_tokens[warp], index);
          break;
        }
      }
    }
  }
  __syncwarp();
  if (lane == 0) {
    const int selected = selected_tokens[warp];
    const int fallback = last_valid_tokens[warp] >= 0
                             ? last_valid_tokens[warp]
                             : vocab - 1;
    output[row] = selected < vocab ? selected : fallback;
  }
}

}  // namespace

void musa_chunked_min_p_sampling_from_probs(
    at::Tensor probs, at::Tensor output,
    std::optional<at::Tensor> maybe_indices,
    std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
    bool deterministic, std::optional<at::Generator> gen_) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DIM(2, probs);
  CHECK_DIM(1, output);
  CHECK_EQ(probs.dtype(), torch::kFloat);
  CHECK_EQ(output.dtype(), torch::kInt);
  CHECK_EQ(probs.device(), output.device());

  const int batch = output.size(0);
  const int vocab = probs.size(1);
  TORCH_CHECK(batch > 0 && batch <= kMaxBatch,
              "chunked min-p sampler requires 1 <= batch <= ", kMaxBatch);
  TORCH_CHECK(vocab == kQwenVocab,
              "chunked min-p sampler requires vocab=", kQwenVocab);
  TORCH_CHECK(probs.size(0) >= batch,
              "chunked min-p sampler received fewer probability rows than outputs");

  int* indices = nullptr;
  if (maybe_indices.has_value()) {
    at::Tensor indices_tensor = maybe_indices.value();
    CHECK_INPUT(indices_tensor);
    CHECK_EQ(indices_tensor.dtype(), torch::kInt);
    CHECK_EQ(indices_tensor.device(), probs.device());
    TORCH_CHECK(indices_tensor.numel() >= batch,
                "chunked min-p indices must cover every output row");
    indices = indices_tensor.data_ptr<int>();
  }

  float* min_p_arr = nullptr;
  if (maybe_min_p_arr.has_value()) {
    at::Tensor min_p_tensor = maybe_min_p_arr.value();
    CHECK_INPUT(min_p_tensor);
    CHECK_EQ(min_p_tensor.dtype(), torch::kFloat);
    CHECK_EQ(min_p_tensor.device(), probs.device());
    TORCH_CHECK(min_p_tensor.numel() >= batch,
                "chunked min-p thresholds must cover every output row");
    min_p_arr = min_p_tensor.data_ptr<float>();
  }

  auto gen = at::get_generator_or_default<at::MUSAGeneratorImpl>(
      gen_, at::musa::detail::getDefaultMUSAGenerator());
  uint64_t philox_seed;
  uint64_t philox_offset;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    // Match FlashInfer's production min-p contract exactly: one Philox
    // counter reservation per sampled row.
    at::PhiloxMusaState state = gen->philox_musa_state(batch);
    philox_seed = state.seed_.val;
    philox_offset = state.offset_.val;
  }

  const int chunks = (vocab + kChunk - 1) / kChunk;
  auto scratch = at::empty({static_cast<int64_t>(batch) * (chunks + 2)},
                           probs.options());
  float* partial = scratch.data_ptr<float>();
  float* threshold = partial + static_cast<size_t>(batch) * chunks;
  float* uniform = threshold + batch;

  const c10::musa::OptionalMUSAGuard device_guard(probs.device());
  auto stream = at::musa::getCurrentMUSAStream();
  partial_max_kernel<<<batch * chunks, kThreads, 0, stream>>>(
      probs.data_ptr<float>(), indices, partial, vocab, chunks);
  threshold_rng_kernel<<<(batch + kRowsPerBlock - 1) / kRowsPerBlock,
                         kThreads, 0, stream>>>(
      partial, threshold, uniform, min_p_arr, static_cast<float>(min_p_val),
      output.data_ptr<int>(), batch, vocab, chunks, philox_seed, philox_offset);
  partial_sum_kernel<<<batch * chunks, kThreads, 0, stream>>>(
      probs.data_ptr<float>(), indices, threshold, partial, vocab, chunks);
  select_kernel<<<(batch + kRowsPerBlock - 1) / kRowsPerBlock, kThreads, 0,
                  stream>>>(probs.data_ptr<float>(), indices, threshold, uniform,
                            partial, output.data_ptr<int>(), batch, vocab,
                            chunks);
  const musaError_t error = musaGetLastError();
  TORCH_CHECK(error == musaSuccess, "chunked min-p sampler failed: ",
              musaGetErrorString(error));
  (void)deterministic;
}
