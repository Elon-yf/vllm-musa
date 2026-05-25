#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <cstdlib>

#include "vec_utils.muh"

// =============================================================================
// PERF NOTE  (MUSA-0150, verified 2026-05-24)
// =============================================================================
//
// This kernel is the V14 design from the sphere-kb silicon-validated
// optimization arc for fused_add_rmsnorm on MTT-S5000 (PH1, mp_31). See:
//
//   sphere-kb/probes/sgl_kernel_ports_20260428/01-fused_add_rmsnorm/
//     RESULTS.md            -- iteration history V1..V14, benchmark CSV
//     sphere_rmsnorm.mu     -- 813-line variant catalog (V1..V11 diagnostic)
//     v14_release/          -- packaged V14 reference (functionally identical
//                              to this kernel; same single-pass register-
//                              resident design, same __musa_memcpy_g2s async
//                              weight prefetch, same vload16_byp_slc LSU-
//                              bypass loads, same adaptive block_x heuristic
//                              frozen 2026-04-29)
//
// Verified achievement (silicon, worker33017, 1000 iters/trial, 5-trial median,
// WARM-cache torch.musa.Event timer — superseded as primary perf reference):
//   - geomean +31 % over sgl-kernel `LayerNormGlobalKernelVlen<T,float,1024,1,8>`
//   - max +93 % at M=256, N=4096
//   - effective bandwidth ~1470 GB/s at large shapes (warm-L2 artifact)
//
// COLD-CACHE re-bench 2026-05-25 (yeahdongcn60, mate.bench_kineto +
// flush_l2=True, num_tests=30, MUSA_VISIBLE_DEVICES=0). Discipline per
// .claude/rules/musa-kernel-bench.md. Numbers below are the real
// production-relevant ceilings, not the warm-cache probe values:
//
//   shape (M, N)         GB/s      % of 1600 GB/s peak
//   (4096, 12288)        1222      76.4 %
//   (4096,  8192)        1148      71.8 %
//   (4096,  3072)        1085      67.8 %   <- M2.5 prefill per-rank
//   (4096, 12288) probe  1222      76.4 %
//   ( 512, 12288)         942      58.9 %
//   ( 256,  4096)         641      40.1 %
//   ( 128,  4096)         548      34.2 %
//   (  64,  3072) M2.5    270      16.9 %   <- dispatcher fixed (was 200)
//   (  16,  3072) M2.5     85       5.3 %   <- dispatcher fixed (was 70)
//
// At cold L2, the kernel reaches ~76 % of GDDR6 peak at the largest shapes,
// not the warm-cache 1470 GB/s "roofline" the probe published. Real
// headroom remains in the deferred candidates below for big-M shapes.
//
// Small-M regime (e.g. cookbook 4k/1k BS=1 decode where M = 1 verify token
// or M = 6 verify chain): this kernel is launch-overhead + SM-occupancy
// bound, NOT bandwidth-bound. One block of 1024 threads (≈2.8 % of 36-MP
// utilization) cannot saturate GDDR6. The lever there is fusion with
// neighboring ops -- tracked in MUSA-0156 (layernorm_quant_kernels.cu),
// MUSA-0157 (fused_qknorm_rope_kernel.cu), MUSA-0158
// (fused_silu_mul_block_quant.cu). The multi-row-per-block variant (V6) was
// tested in the probe and consistently 5-30 % WORSE due to extra
// per-row reduce overhead.
//
// Deferred next-step optimization candidates (from the probe's "Open
// questions" section, not yet measured):
//   - TME tile-stride load to overlap x/r streams (gated behind
//     MUTE_ARCH_TME_MP31_ACTIVATED, est. +5..10 %)
//   - Pipeline pass-1 chunks with async copy on x/r as well as w (only
//     profitable at CHUNK >= 2, e.g. N=12288)
//
// Reference cold-cache benchmark:
//   benchmarks/op_perf/bench_fused_add_rmsnorm.py
//     -- uses mate.testing.utils.bench_kineto with flush_l2=True per the
//     .claude/rules/musa-kernel-bench.md discipline. The previous bench
//     (time.perf_counter_ns + no L2 flush) produced warm-cache artifacts
//     and is replaced by this script.
//
// =============================================================================

extern "C" {
extern __device__ void __musa_memcpy_g2s(
    __attribute__((address_space(3))) void* dst,
    __attribute__((address_space(1))) const void* src, int size, int prefetch);
extern __device__ void __musa_memcpy_g2s_wait();
}

namespace vllm_musa {

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
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < (BLOCK_X / 32) ? shared[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = (BLOCK_X / 32) >> 1; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

template <typename T, int BLOCK_X, int CHUNK>
__global__ void __launch_bounds__(BLOCK_X, 1) fused_add_rmsnorm_kernel(
    T* __restrict__ input, T* __restrict__ residual,
    const T* __restrict__ weight, int hidden_size, int vec_hidden_size,
    float epsilon) {
  constexpr int VEC_SIZE = 8;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  T* input_row = input + static_cast<size_t>(row) * hidden_size;
  T* residual_row = residual + static_cast<size_t>(row) * hidden_size;

  __shared__ T shared_weight[CHUNK * BLOCK_X * VEC_SIZE];

#pragma unroll
  for (int chunk = 0; chunk < CHUNK; ++chunk) {
    const int vec_idx = chunk * BLOCK_X + tid;
    if (vec_idx < vec_hidden_size) {
      __attribute__((address_space(3))) void* smem_ptr =
          (__attribute__((address_space(3))) void*)(
              shared_weight + vec_idx * VEC_SIZE);
      __attribute__((address_space(1))) const void* gmem_ptr =
          (__attribute__((address_space(1))) const void*)(
              weight + vec_idx * VEC_SIZE);
      __musa_memcpy_g2s(smem_ptr, gmem_ptr, 16, 128);
    }
  }

  Vec16<T> fused_local[CHUNK];
  float variance = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < CHUNK; ++chunk) {
    const int vec_idx = chunk * BLOCK_X + tid;
    if (vec_idx < vec_hidden_size) {
      Vec16<T> input_vec = vload16_byp_slc(input_row + vec_idx * VEC_SIZE);
      Vec16<T> residual_vec =
          vload16_byp_slc(residual_row + vec_idx * VEC_SIZE);
      Vec16<T> sum_vec;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        const float value =
            static_cast<float>(input_vec.x[i]) + static_cast<float>(residual_vec.x[i]);
        sum_vec.x[i] = static_cast<T>(value);
        variance += value * value;
      }
      fused_local[chunk] = sum_vec;
      vstore16(residual_row + vec_idx * VEC_SIZE, sum_vec);
    }
  }

  const float total = block_reduce_sum<BLOCK_X>(variance);
  const float inv_rms = rsqrtf(total / static_cast<float>(hidden_size) + epsilon);

  __musa_memcpy_g2s_wait();
  __syncthreads();

#pragma unroll
  for (int chunk = 0; chunk < CHUNK; ++chunk) {
    const int vec_idx = chunk * BLOCK_X + tid;
    if (vec_idx < vec_hidden_size) {
      Vec16<T> weight_vec =
          *reinterpret_cast<Vec16<T>*>(shared_weight + vec_idx * VEC_SIZE);
      Vec16<T> out_vec;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        out_vec.x[i] = static_cast<T>(static_cast<float>(fused_local[chunk].x[i]) *
                                      inv_rms *
                                      static_cast<float>(weight_vec.x[i]));
      }
      vstore16(input_row + vec_idx * VEC_SIZE, out_vec);
    }
  }
}

template <typename T>
void dispatch_fused_add_rmsnorm(T* input, T* residual, const T* weight,
                                int rows, int hidden_size, float epsilon,
                                musaStream_t stream) {
  const int vec_hidden_size = hidden_size / 8;

  static const int forced_block = []() {
    const char* env = std::getenv("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X");
    return env == nullptr ? 0 : std::atoi(env);
  }();

  int block_x;
  if (forced_block > 0) {
    TORCH_CHECK(forced_block == 128 || forced_block == 256 ||
                    forced_block == 512 || forced_block == 1024,
                "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X must be one of "
                "128, 256, 512, or 1024");
    block_x = forced_block;
  } else if (rows >= 512 && vec_hidden_size <= 640) {
    block_x = 128;
  } else if (vec_hidden_size <= 480 && rows >= 256) {
    // MUSA-0150 (re-bench 2026-05-25, mate.bench_kineto cold-cache): for
    // narrow hidden (vec_hidden_size <= 480, i.e. hidden <= 3840 — covers
    // M2.5 per-rank N=3072) with rows >= 256, block_x=128 wins cold-cache:
    //   rows=256 hidden=3072: 600.3 -> 630.6 GB/s (+5 %)
    //   rows=4096 hidden=3072: 1030.9 -> 1085.9 GB/s (+5 %)
    // Rule is tightened from vec<=512 to vec<=480 to avoid the N=4096
    // (vec=512) regression: at vec=512 M=128, BLOCK=128 needs CHUNK=4 and
    // loses to BLOCK=256 CHUNK=2 by ~17 %. The probe's wider-hidden
    // benches over-credited 256-thread blocks because their warm-cache
    // numbers couldn't expose the cross-warp reduce overhead.
    block_x = 128;
  } else if (rows >= 256) {
    block_x = 256;
  } else if (rows >= 128 && vec_hidden_size >= 896) {
    block_x = 512;
  } else if (rows >= 128) {
    block_x = 256;
  } else if (vec_hidden_size <= 480) {
    // MUSA-0150 (re-bench 2026-05-25): for narrow hidden and small M
    // (rows < 128, vec_hidden <= 480), block_x=512 wins decisively over the
    // previous default 1024. Cold-cache deltas at hidden=3072:
    //   rows=6:  27.4 -> 33.0 GB/s  (+20 %)
    //   rows=16: 69.3 -> 84.2 GB/s  (+21 %)
    //   rows=64: 200.5 -> 275.6 GB/s (+37 %)
    // The probe's else-branch=1024 assumed small M needs to saturate all
    // 1024 lanes; cold-cache the cross-warp shared-mem reduce dominates,
    // and a 2-warp block (512) is the right amortization point.
    block_x = 512;
  } else if (rows >= 8 && vec_hidden_size >= 1280) {
    block_x = 512;
  } else {
    block_x = 1024;
  }

#define LAUNCH(BLOCK, CHUNK)                                                \
  fused_add_rmsnorm_kernel<T, BLOCK, CHUNK>                                 \
      <<<rows, BLOCK, 0, stream>>>(input, residual, weight, hidden_size,     \
                                   vec_hidden_size, epsilon)

  if (block_x == 128) {
    if (vec_hidden_size <= 128) LAUNCH(128, 1);
    else if (vec_hidden_size <= 256) LAUNCH(128, 2);
    else if (vec_hidden_size <= 384) LAUNCH(128, 3);
    else if (vec_hidden_size <= 512) LAUNCH(128, 4);
    else if (vec_hidden_size <= 640) LAUNCH(128, 5);
    else if (vec_hidden_size <= 768) LAUNCH(128, 6);
    else if (vec_hidden_size <= 896) LAUNCH(128, 7);
    else if (vec_hidden_size <= 1024) LAUNCH(128, 8);
    else if (vec_hidden_size <= 1536) LAUNCH(128, 12);
    else TORCH_CHECK(false, "BLOCK=128 hidden_size too large");
  } else if (block_x == 256) {
    if (vec_hidden_size <= 256) LAUNCH(256, 1);
    else if (vec_hidden_size <= 512) LAUNCH(256, 2);
    else if (vec_hidden_size <= 768) LAUNCH(256, 3);
    else if (vec_hidden_size <= 1024) LAUNCH(256, 4);
    else if (vec_hidden_size <= 1536) LAUNCH(256, 6);
    else if (vec_hidden_size <= 2048) LAUNCH(256, 8);
    else TORCH_CHECK(false, "BLOCK=256 hidden_size too large");
  } else if (block_x == 512) {
    if (vec_hidden_size <= 512) LAUNCH(512, 1);
    else if (vec_hidden_size <= 1024) LAUNCH(512, 2);
    else if (vec_hidden_size <= 1536) LAUNCH(512, 3);
    else if (vec_hidden_size <= 2048) LAUNCH(512, 4);
    else TORCH_CHECK(false, "BLOCK=512 hidden_size too large");
  } else {
    if (vec_hidden_size <= 1024) LAUNCH(1024, 1);
    else if (vec_hidden_size <= 2048) LAUNCH(1024, 2);
    else TORCH_CHECK(false, "BLOCK=1024 hidden_size too large");
  }

#undef LAUNCH
}

}  // namespace vllm_musa

void musa_fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                             torch::Tensor& weight, double epsilon) {
  TORCH_CHECK(input.device().is_privateuseone(), "input must be a MUSA tensor");
  TORCH_CHECK(residual.device().is_privateuseone(),
              "residual must be a MUSA tensor");
  TORCH_CHECK(weight.device().is_privateuseone(), "weight must be a MUSA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D");
  TORCH_CHECK(residual.dim() == 2, "residual must be 2-D");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  TORCH_CHECK(input.sizes() == residual.sizes(), "input/residual shape mismatch");
  TORCH_CHECK(input.size(1) == weight.size(0), "weight size mismatch");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
              "input/residual dtype mismatch");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input/weight dtype mismatch");
  TORCH_CHECK(input.size(1) % 8 == 0,
              "hidden_size must be a multiple of 8");
  TORCH_CHECK(input.size(1) <= 16384,
              "hidden_size must be <= 16384");

  const c10::musa::OptionalMUSAGuard guard(device_of(input));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  const int rows = static_cast<int>(input.size(0));
  const int hidden_size = static_cast<int>(input.size(1));

  if (input.scalar_type() == at::ScalarType::Half) {
    vllm_musa::dispatch_fused_add_rmsnorm<__half>(
        static_cast<__half*>(input.data_ptr()),
        static_cast<__half*>(residual.data_ptr()),
        static_cast<const __half*>(weight.data_ptr()), rows, hidden_size,
        static_cast<float>(epsilon), stream);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    vllm_musa::dispatch_fused_add_rmsnorm<__mt_bfloat16>(
        static_cast<__mt_bfloat16*>(input.data_ptr()),
        static_cast<__mt_bfloat16*>(residual.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()), rows, hidden_size,
        static_cast<float>(epsilon), stream);
  } else {
    TORCH_CHECK(false, "only fp16 and bf16 are supported");
  }
}
