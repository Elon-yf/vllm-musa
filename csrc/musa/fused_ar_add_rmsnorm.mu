// SPDX-License-Identifier: Apache-2.0
// MUSA-0123: fused (cross-rank all-reduce) + (add residual) + (RMS norm)
//
// Goal: eliminate the multi-barrier divergence-amplification of running
// these as separate ops. Pattern mirrors MT-Transformer's
// sqmma_moe_ffn2_decode_*_comm_nodes_8_all_reduce_True_*_epi_rms_quant_True
// algorithm, implemented natively with vllm's RankData + Signal*
// infrastructure (no proprietary libmtt link).
//
// Hot-path workflow (per layer in M2.5+Eagle3):
//   BEFORE: AR(x)  → tmp1 ; fused_add_rmsnorm(tmp1, res, gamma)
//   AFTER:  fused_ar_add_rmsnorm(peer_x, res, gamma)  → out  in ONE kernel
//
// Two barriers in current path (one each AR + rmsnorm sync) become ONE
// barrier here. Each barrier pays the max-of-N hardware-variance cost
// (~200-400µs per call per MUSA-0121 profile), so reducing barriers
// proportionally cuts TPOT.
//
// Status: SKELETON. The reduction algorithm + sync are TODO. See
// `tickets/todo/MUSA-0123-fused-ar-rms-kernel.md` for the full spec.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <cstdlib>

#include "vec_utils.muh"

namespace vllm_musa {

// Per-rank state struct, opaque to caller. Caller must obtain via the
// custom_all_reduce API (cuda_communicator's CustomAllreduce instance).
// On the device side, this is passed as a device-pointer to a struct
// holding the 8 peer pointers + sync metadata.
struct FusedArRmsState {
  void* peer_ptrs[8];  // device pointers, one per TP rank
  // Sync signal pointers (analog to vllm's Signal*). TODO: wire up.
};

template <int BLOCK_X>
__device__ __forceinline__ float block_reduce_sum(float value) {
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

// MUSA-0123: kernel sketch.
//
// Layout:
//   gridDim.x = num_rows (one row per block)
//   blockDim.x = BLOCK_X (e.g. 256 threads)
//   Each thread covers (hidden / vlen / BLOCK_X) packed elements per row.
//
// Algorithm:
//   1. Cross-rank barrier: ensure all ranks have written input to their
//      peer-IPC buffer.
//   2. For each packed element this thread owns:
//      a. Load that index from all 8 peer buffers; sum to f32 acc.
//      b. Add the residual at that index (bf16 → f32).
//      c. Accumulate x^2 in thread-local register.
//   3. Block-reduce sum-of-squares → row variance.
//   4. Compute rms = rsqrt(variance / hidden + eps).
//   5. For each packed element this thread owns:
//      a. Multiply by rms and gamma.
//      b. Store to output (bf16).
//   6. Cross-rank barrier (only needed if downstream readers consume
//      the output via peer access; otherwise can be omitted).
//
// TODOs in this skeleton:
//   - Wire up FusedArRmsState + RankSignals (use vllm's existing types
//     from custom_all_reduce.cuh).
//   - Add the cross-rank barrier (use multi_gpu_barrier_with_atomic).
//   - Validate alignment: input must be a vec-16 multiple.
//   - Add fp8 quant epilogue (parallels MT-Transformer's variants).
//   - Hook into vllm's IR pass via a TORCH_LIBRARY op so that the
//     fuse_allreduce_rms IR pass can rewrite to this kernel.
template <typename T, int BLOCK_X, int CHUNK, int NRANKS>
__global__ void __launch_bounds__(BLOCK_X, 1) fused_ar_add_rmsnorm_kernel_stub(
    /*const*/ FusedArRmsState* state,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int hidden_size,
    int vec_hidden_size,
    float epsilon) {
  // TODO(MUSA-0123): implement per the algorithm sketch above.
  //
  // For now, mark this as a build-validation-only stub. Calling it does
  // NOT produce correct output. Once the algorithm is wired, register
  // as torch.ops._C_musa_ops.musa_fused_ar_add_rmsnorm and integrate
  // with vllm's fuse_allreduce_rms IR pass.
  (void)state;
  (void)residual;
  (void)weight;
  (void)output;
  (void)hidden_size;
  (void)vec_hidden_size;
  (void)epsilon;
}

}  // namespace vllm_musa
