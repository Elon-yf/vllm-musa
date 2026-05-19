// SPDX-License-Identifier: Apache-2.0
// MUSA-0123 v2: fused (cross-rank all-reduce) + (add residual) + (RMS norm)
//
// v2 redesign (2026-05-19): switched internal structure from broadcast
// peer-reads (v1, perf-negative at 73.82µs vs 26.80µs baseline) to a
// 2shot-style reduce-scatter + allreduce-variance + allgather pattern.
//
// Algorithm (each block handles ONE row; BLOCK_X threads collaborate):
//   STAGE 1 (reduce-scatter + add residual + partial variance):
//     - Each rank owns slice [rank * part, (rank+1) * part) of the row
//     - For each owned vec-idx: read packed-elem from ALL 8 peer
//       reg_buffers, sum (cross-rank reduce), add residual, write to
//       OWN tmp_buf (peer-IPC visible scratch after Signal struct)
//     - Accumulate sum-of-squares over OUR slice → local partial variance
//
//   CROSS-RANK BARRIER + VARIANCE SUM:
//     - Block-reduce local_x2 to single float per rank
//     - Store partial variance in our tmp scratch tail
//     - Barrier
//     - Read peer partial variances, sum → total_x2 (full row variance)
//     - Compute rms = rsqrt(total_x2 / hidden_size + epsilon)
//
//   STAGE 2 (allgather + normalize):
//     - Each thread reads peer tmp slices in skewed order
//     - For each (rank+i)%NRANKS target: load tmp slice values,
//       apply rms * gamma, write to local output
//
// Bandwidth per rank (BS=1, H=6144, TP=8, vec_H=768):
//   v1 (broadcast): each thread reads H/BLOCK_X × 8 peers = 8H/BLOCK_X per thread
//                   per rank total = H × 8 = 8H reads
//   v2 (2shot):     stage 1 reads H/8 × 8 = H reads + stage 2 reads H reads
//                   per rank total = 2H reads (4x less than v1)
//
// This matches the cross_device_reduce_2stage kernel's bandwidth profile
// while fusing add + RMS into a single pass with TWO barriers (instead of
// the AR's two-stage barrier + fused_add_rmsnorm's one barrier = 3 effective
// barriers in the standard path).

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
// .cuh when it's included from a .mu file. The structs MUST byte-match
// vllm's layout — verify after any change in custom_all_reduce.cuh.
namespace vllm {

using FlagType = uint32_t;
constexpr int kMaxBlocks_local = 60;

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

// Returns pointer to the scratch area immediately after Signal struct.
// Mirrors get_tmp_buf<P>(Signal*) from .cuh.
template <typename P>
__device__ __forceinline__ P* get_tmp_buf_local(Signal* sg) {
  return reinterpret_cast<P*>(reinterpret_cast<Signal*>(sg) + 1);
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

// MUSA-0123 v2 kernel: 2shot-style internal structure.
//
// Grid: gridDim.x = num_rows (one block per row).
// Block: BLOCK_X threads.
//
// vec_hidden_size = hidden_size / 8 (packed by 8 bf16 elements).
// Requires: vec_hidden_size % NRANKS == 0 (each rank owns equal slice).
//
// Storage layout in each rank's tmp_buf (peer-IPC region):
//   [0 .. num_rows * vec_hidden_size) : packed bf16 (reduced+residual data)
//   [num_rows * vec_hidden_size * sizeof(P) .. +num_rows*sizeof(float)) : float variance scratch
template <typename T, int NRANKS, int BLOCK_X>
__global__ void __launch_bounds__(BLOCK_X, 1) fused_ar_add_rmsnorm_kernel_v2(
    vllm::RankData* dp,
    vllm::RankSignals sg,
    vllm::Signal* self_sg,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    T* __restrict__ output,
    int num_rows,
    int hidden_size,
    int vec_hidden_size,
    float epsilon,
    int rank) {
  using P = Packed<T, 8>;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * vec_hidden_size;

  // Slice ownership: each rank handles vec_hidden_size/NRANKS packed-elements.
  const int part = vec_hidden_size / NRANKS;
  const int my_start = rank * part;
  const int my_end = (rank == NRANKS - 1) ? vec_hidden_size : my_start + part;

  // Own tmp buffer (peer-readable scratch in self_sg).
  P* my_tmp = vllm::get_tmp_buf_local<P>(self_sg);

  // ENTRY BARRIER: all peers have written their pre-AR inputs.
  vllm::multi_gpu_barrier_with_atomic_local<NRANKS>(sg, self_sg, rank);

  // ===== STAGE 1: REDUCE-SCATTER + ADD RESIDUAL + PARTIAL VARIANCE =====
  float local_x2 = 0.0f;
  for (int vec_idx = my_start + tid; vec_idx < my_end; vec_idx += BLOCK_X) {
    const int g_idx = row_offset + vec_idx;
    float acc[8];
#pragma unroll
    for (int v = 0; v < 8; v++) acc[v] = 0.0f;

    // Sum across NRANKS peers, skewed access to spread bandwidth.
#pragma unroll
    for (int p = 0; p < NRANKS; p++) {
      int target = (rank + p) % NRANKS;
      const P peer = ((const P*)dp->ptrs[target])[g_idx];
#pragma unroll
      for (int v = 0; v < 8; v++) {
        acc[v] += __bfloat162float(peer.data[v]);
      }
    }

    // Add residual slice.
    const P res_p = ((const P*)residual)[g_idx];
#pragma unroll
    for (int v = 0; v < 8; v++) {
      acc[v] += __bfloat162float(res_p.data[v]);
    }

    // Sum-of-squares for variance.
#pragma unroll
    for (int v = 0; v < 8; v++) {
      local_x2 += acc[v] * acc[v];
    }

    // Write reduced+residual to OWN tmp (peer-readable).
    P out_p;
#pragma unroll
    for (int v = 0; v < 8; v++) {
      out_p.data[v] = __float2bfloat16(acc[v]);
    }
    my_tmp[g_idx] = out_p;
  }

  // Block-wide variance reduce (within rank).
  local_x2 = block_reduce_sum_f<BLOCK_X>(local_x2);

  // Store our partial variance in our tmp scratch tail.
  // Layout: variance area immediately follows the packed data region.
  float* my_var = reinterpret_cast<float*>(my_tmp + num_rows * vec_hidden_size);
  if (tid == 0) {
    my_var[row] = local_x2;
  }

  // BARRIER 2: stage 1 results visible to peers.
  vllm::multi_gpu_barrier_with_atomic_local<NRANKS>(sg, self_sg, rank);

  // ===== CROSS-RANK VARIANCE SUM =====
  // Read peer partial variances, sum to get total_x2 for full row.
  float total_x2;
  if (tid < NRANKS) {
    P* peer_tmp = vllm::get_tmp_buf_local<P>(sg.signals[tid]);
    float* peer_var =
        reinterpret_cast<float*>(peer_tmp + num_rows * vec_hidden_size);
    total_x2 = peer_var[row];
  } else {
    total_x2 = 0.0f;
  }
  // Warp-reduce across first NRANKS lanes.
#pragma unroll
  for (int offset = NRANKS >> 1; offset > 0; offset >>= 1) {
    total_x2 += __shfl_xor_sync(0xffffffff, total_x2, offset);
  }
  // Broadcast to whole block via shared memory.
  __shared__ float s_total_x2;
  if (tid == 0) s_total_x2 = total_x2;
  __syncthreads();
  total_x2 = s_total_x2;

  const float rms =
      rsqrtf(total_x2 / static_cast<float>(hidden_size) + epsilon);

  // ===== STAGE 2: ALLGATHER + NORMALIZE + WRITE =====
  // Read peer tmp slices in skewed order, apply rms*gamma, write to output.
#pragma unroll
  for (int p = 0; p < NRANKS; p++) {
    int target = (rank + p) % NRANKS;
    const int target_start = target * part;
    const int target_part_size =
        (target == NRANKS - 1) ? (vec_hidden_size - target_start) : part;
    const P* peer_tmp = vllm::get_tmp_buf_local<P>(sg.signals[target]);

    for (int local_idx = tid; local_idx < target_part_size;
         local_idx += BLOCK_X) {
      const int vec_idx = target_start + local_idx;
      const int g_idx = row_offset + vec_idx;
      const P tmp_p = peer_tmp[g_idx];
      const P gamma_p = ((const P*)gamma)[vec_idx];
      P out_p;
#pragma unroll
      for (int v = 0; v < 8; v++) {
        const float normed = __bfloat162float(tmp_p.data[v]) * rms *
                             __bfloat162float(gamma_p.data[v]);
        out_p.data[v] = __float2bfloat16(normed);
      }
      ((P*)output)[g_idx] = out_p;
    }
  }
}

}  // namespace vllm_musa

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
    musaStream_t stream) {
  constexpr int BLOCK_X = 256;
  constexpr int VLEN = 8;
  const int vec_hidden_size = hidden_size / VLEN;
  if (hidden_size % VLEN != 0) return;
  // v2 requires vec_hidden_size divisible by NRANKS=8.
  if (vec_hidden_size % 8 != 0) return;

  vllm::RankData* dp_ptr = reinterpret_cast<vllm::RankData*>(dp);
  vllm::RankSignals rs;
  for (int i = 0; i < 8; ++i) {
    rs.signals[i] = reinterpret_cast<vllm::Signal**>(sg_signals)[i];
  }
  vllm::Signal* self_sg_ptr = reinterpret_cast<vllm::Signal*>(self_sg);

  vllm_musa::fused_ar_add_rmsnorm_kernel_v2<__mt_bfloat16, 8, BLOCK_X>
      <<<num_rows, BLOCK_X, 0, stream>>>(
          dp_ptr, rs, self_sg_ptr, residual, gamma, output, num_rows,
          hidden_size, vec_hidden_size, epsilon, rank);
}
