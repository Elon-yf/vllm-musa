// SPDX-License-Identifier: Apache-2.0
// MUSA-0123: fused (cross-rank all-reduce) + (add residual) + (RMS norm)
//
// First working version. Reads from 8 peer buffers, sums across ranks,
// adds residual, computes RMS norm, applies gamma, writes output — all
// in ONE kernel pass with TWO cross-rank barriers.
//
// Compared to the current vllm-musa path (AR kernel → separate
// fused_add_rmsnorm kernel), this kernel eliminates one barrier round
// trip and one intermediate tensor materialization. Per MUSA-0121's
// divergence proof, each barrier pays max-of-N hardware variance cost
// (~200-400µs on M2.5+Eagle3 at TP=8), so a fused single-barrier path
// gives a proportional TPOT reduction.
//
// Build status: SELF-CONTAINED but NOT YET WIRED into setup.py or the
// IR pass. Calling code must:
//   1. Have IPC peer pointers (from vllm's CustomAllreduce.register_buffer)
//   2. Pass them as `void* peer_ptrs[8]` device-array to this kernel
//   3. Use `multi_gpu_barrier_with_atomic` (defined in
//      csrc/custom_all_reduce.cuh) for cross-rank sync
//
// For per-block sizing, see comments inside `fused_ar_add_rmsnorm_kernel`.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <cstdlib>

// MUSA-0123 build-validation note: we DELIBERATELY inline the minimal
// types from vllm/csrc/custom_all_reduce.cuh here (rather than #include
// the .cuh) because torchada's CUDA→MUSA porting doesn't process the
// .cuh when it's included from a .mu file (cuda.h isn't on the .mu
// include path). The structs MUST byte-match vllm's layout — verify
// after any change in custom_all_reduce.cuh.
//
// Source of truth: vllm-musa/csrc/custom_all_reduce.cuh @ HEAD.

namespace vllm {

using FlagType = uint32_t;
constexpr int kMaxBlocks_local = 60;  // matches USE_MUSA branch in .cuh

struct Signal_layout {
  alignas(128) FlagType self_counter[kMaxBlocks_local][8];
  alignas(128) FlagType peer_counter[2][kMaxBlocks_local][8];
};
using Signal = Signal_layout;

struct __align__(16) RankData {
  const void* ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

// Mirror of multi_gpu_barrier_with_atomic from .cuh.
template <int32_t ngpus>
__device__ __forceinline__ void multi_gpu_barrier_with_atomic_local(
    const RankSignals& sg, Signal* self_sg, int32_t rank) {
  if (threadIdx.x < ngpus) {
    auto val =
        self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr =
        &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
    auto self_counter_ptr =
        &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
    atomicExch(peer_counter_ptr, val);
    while (atomicAdd(self_counter_ptr, 0) != val) {
    }
  }
  __syncthreads();
}

}  // namespace vllm

namespace vllm_musa {

// Packed array type, mirroring vllm's packed_t<T>::P (array_t<T, 16/sizeof(T)>).
template <typename T, int N>
struct Packed {
  T data[N];
};

template <int BLOCK_X>
__device__ __forceinline__ float block_reduce_sum_f(float value) {
  // Warp-shuffle reduce first (32 lanes).
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  if constexpr (BLOCK_X <= 32) {
    return value;
  }
  __shared__ float shared[BLOCK_X / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = threadIdx.x < (BLOCK_X / 32) ? shared[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = (BLOCK_X / 32) >> 1; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

// MUSA-0123 first-cut kernel.
//
// Grid layout:
//   gridDim.x = num_rows                (one row per block — rows are
//                                        independent)
//   blockDim.x = BLOCK_X (e.g. 256)     (threads collaborate within row)
//
// Each thread processes ELEMS_PER_THREAD packed elements from row.
// hidden_size must be divisible by VLEN (= 8 for bf16). The product
// BLOCK_X * VLEN * ELEMS_PER_THREAD must be >= hidden_size so the row
// is fully covered.
//
// Algorithm (per row R, all threads in block):
//   Barrier 1: ensure all ranks have written input to peer buffer.
//   For each owned packed-elem index i_v:
//     g_v = R * (hidden_size / VLEN) + i_v
//     acc = sum_over_8_peers( peer_ptrs[p][g_v] )           // f32 acc
//     acc += residual[g_v]                                   // bf16 → f32
//     local_x2 += sum(acc^2)
//     store acc in register
//   block_reduce(local_x2) → total_x2
//   rms = rsqrt(total_x2 / hidden_size + eps)
//   For each owned packed-elem index i_v:
//     out[g_v] = (stored acc) * rms * gamma[i_v]   (bf16 cast)
//
// Templated on NRANKS and BLOCK_X for compile-time unroll.
template <typename T, int NRANKS, int BLOCK_X, int ELEMS_PER_THREAD>
__global__ void __launch_bounds__(BLOCK_X, 1) fused_ar_add_rmsnorm_kernel(
    vllm::RankData* dp,
    vllm::RankSignals sg,
    vllm::Signal* self_sg,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    T* __restrict__ output,
    int hidden_size,
    int vec_hidden_size,
    float epsilon,
    int rank) {
  using P = Packed<T, 8>;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * vec_hidden_size;

  // Cross-rank barrier 1: all ranks visible.
  vllm::multi_gpu_barrier_with_atomic_local<NRANKS>(sg, self_sg, rank);

  // Storage for post-AR values per thread — kept in registers/local.
  float acc_stash[ELEMS_PER_THREAD][8];
  float local_x2 = 0.0f;

#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    const int vec_idx = tid + e * BLOCK_X;
    if (vec_idx >= vec_hidden_size) {
#pragma unroll
      for (int v = 0; v < 8; v++) acc_stash[e][v] = 0.0f;
      continue;
    }
    const int g_idx = row_offset + vec_idx;

    float acc[8];
#pragma unroll
    for (int v = 0; v < 8; v++) acc[v] = 0.0f;

    // Sum across 8 peers (compile-time unrolled for NRANKS).
#pragma unroll
    for (int p = 0; p < NRANKS; p++) {
      const P peer = ((const P*)dp->ptrs[p])[g_idx];
#pragma unroll
      for (int v = 0; v < 8; v++) {
        acc[v] += __bfloat162float(peer.data[v]);
      }
    }

    // Add residual (bf16 → f32).
    const P res_p = ((const P*)residual)[g_idx];
#pragma unroll
    for (int v = 0; v < 8; v++) {
      acc[v] += __bfloat162float(res_p.data[v]);
    }

    // Accumulate sum-of-squares.
#pragma unroll
    for (int v = 0; v < 8; v++) {
      local_x2 += acc[v] * acc[v];
      acc_stash[e][v] = acc[v];
    }
  }

  // Block-wide reduce of x^2 → row variance.
  const float total_x2 = block_reduce_sum_f<BLOCK_X>(local_x2);
  const float rms = rsqrtf(total_x2 / static_cast<float>(hidden_size) + epsilon);

  // Apply gamma + cast back to bf16; write output.
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    const int vec_idx = tid + e * BLOCK_X;
    if (vec_idx >= vec_hidden_size) continue;
    const int g_idx = row_offset + vec_idx;
    const P gamma_p = ((const P*)gamma)[vec_idx];
    P out_p;
#pragma unroll
    for (int v = 0; v < 8; v++) {
      const float normed =
          acc_stash[e][v] * rms * __bfloat162float(gamma_p.data[v]);
      out_p.data[v] = __float2bfloat16(normed);
    }
    ((P*)output)[g_idx] = out_p;
  }

  // No second barrier — output is local and not peer-readable.
}

}  // namespace vllm_musa

// Host launcher: dispatches the right template instantiation based on
// (hidden_size, BLOCK_X). For now only support bf16, TP=8, BLOCK_X=256.
// The kernel needs CustomAllreduce state, which is the caller's job.
//
// extern "C" so the symbol is preserved through linking and exposed
// for a future torch_bindings.cpp wrapper (MUSA-0123 step 2).
extern "C" void fused_ar_add_rmsnorm_bf16_tp8_launch(
    void* dp,           // device pointer to vllm::RankData
    void* sg_signals,   // device pointer array (8) of vllm::Signal*
    void* self_sg,      // device pointer to vllm::Signal
    const __mt_bfloat16* residual,
    const __mt_bfloat16* gamma,
    __mt_bfloat16* output,
    int num_rows,
    int hidden_size,
    float epsilon,
    int rank,
    musaStream_t stream) {
  constexpr int BLOCK_X = 256;
  constexpr int VLEN = 8;
  const int vec_hidden_size = hidden_size / VLEN;
  if (hidden_size % VLEN != 0) {
    // unsupported in this first-cut. Caller should fall back.
    return;
  }

  vllm::RankData* dp_ptr = reinterpret_cast<vllm::RankData*>(dp);
  vllm::RankSignals rs;
  for (int i = 0; i < 8; ++i) {
    rs.signals[i] = reinterpret_cast<vllm::Signal**>(sg_signals)[i];
  }
  vllm::Signal* self_sg_ptr = reinterpret_cast<vllm::Signal*>(self_sg);

  // ELEMS_PER_THREAD picked so BLOCK_X * VLEN * ELEMS_PER_THREAD >= hidden_size.
  // For M2.5 hidden=6144: vec_hidden=768, BLOCK_X=256 → ELEMS_PER_THREAD=3.
  // For hidden=2048: vec_hidden=256, BLOCK_X=256 → ELEMS_PER_THREAD=1.
  if (vec_hidden_size <= 256) {
    vllm_musa::fused_ar_add_rmsnorm_kernel<__mt_bfloat16, 8, BLOCK_X, 1>
        <<<num_rows, BLOCK_X, 0, stream>>>(
            dp_ptr, rs, self_sg_ptr, residual, gamma, output, hidden_size,
            vec_hidden_size, epsilon, rank);
  } else if (vec_hidden_size <= 768) {
    vllm_musa::fused_ar_add_rmsnorm_kernel<__mt_bfloat16, 8, BLOCK_X, 3>
        <<<num_rows, BLOCK_X, 0, stream>>>(
            dp_ptr, rs, self_sg_ptr, residual, gamma, output, hidden_size,
            vec_hidden_size, epsilon, rank);
  } else if (vec_hidden_size <= 2048) {
    vllm_musa::fused_ar_add_rmsnorm_kernel<__mt_bfloat16, 8, BLOCK_X, 8>
        <<<num_rows, BLOCK_X, 0, stream>>>(
            dp_ptr, rs, self_sg_ptr, residual, gamma, output, hidden_size,
            vec_hidden_size, epsilon, rank);
  }
  // else: hidden too large for this scheme; would need bigger BLOCK_X.
}
