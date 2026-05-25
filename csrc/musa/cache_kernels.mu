#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <cstdlib>

#include "vec_utils.muh"

// =============================================================================
// PERF NOTE  (MUSA-0151, characterized 2026-05-24)
// =============================================================================
//
// Kernel: reshape_and_cache_flash_nhd_kernel<T, BLOCK_X=512, TOKENS_PER_BLOCK>
// Workload: pure bandwidth-bound gather (key, value) -> scatter (kv-cache),
// 4 × num_tokens × num_heads × head_size × sizeof(T) bytes per call,
// no compute beyond index arithmetic.
//
// Current design layers (V2, 2026-05-25):
//   1. LSU cache-bypass loads (`vload16_byp_slc`) — the V3 floor from the
//      sphere-kb arc (without LSU bypass plain loads are 2-5× slower).
//   2. Vec16<T> (128-bit) vectorized loads + stores.
//   3. Shape-adaptive dispatcher: TOKENS_PER_BLOCK ∈ {1,2,4,8} chosen by
//      vecs_per_token band; env override
//      VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK.
//   4. Adaptive BLOCK_X (V2, 2026-05-25): BLOCK_X = next pow2 >=
//      TOKENS_PER_BLOCK * vecs_per_token, clamped to [32, 512]. Recovers
//      the 75 % idle-thread overhead at narrow-KV shapes (vec_token=16
//      drops BLOCK_X 512 -> 128 and quadruples concurrent CTAs/MP).
//
// sphere-kb status: no probe exists yet for this kernel (closest is
// `probes/sgl_kernel_ports_20260428/03-apply_rope_pos_ids/` which fuses
// RoPE + KV-write but is a separate op). Verify-and-document baseline
// recorded by the regression bench
// `benchmarks/op_perf/bench_reshape_and_cache_flash.py`.
//
// Cold-cache bench (mate.bench_kineto + flush_l2=True, yeahdongcn60,
// 2026-05-25). Roofline anchor: sphere-kb claim
// s5000.practical_gddr6_bandwidth_updated = 1200 GB/s read+write.
//
//   shape (T, H, D)     V1 GB/s   V2 GB/s   % of 1200 R+W  Δ
//   ----------------------------------------------------------
//   M2.5 narrow-KV (num_heads_kv=1, head_size=128):
//     T=1     H=1 D=128     0.28      0.33      0.0 %   +18 %
//     T=6     H=1 D=128     1.48      1.60      0.1 %   +8 %
//     T=16    H=1 D=128     3.78      3.91      0.3 %   +3 %
//     T=64    H=1 D=128    13.49     13.64      1.1 %   +1 %
//     T=256   H=1 D=128    48.54     47.82      4.0 %   -1.5 %
//     T=4096  H=1 D=128   348.28    413.11     34.4 %   +19 %  ←
//   Wider configs (already BLOCK_X=512 in both V1 and V2):
//     T=4096  H=4 D=128   623.62    612.69     51.1 %   noise
//     T=4096  H=8 D=128   599.33    599.56     50.0 %   0 %
//
// At narrow KV (M2.5 prefill T=4096) V2 lifts +19 % by trimming
// BLOCK_X 512 -> 128 (4 x more concurrent CTAs/MP, zero idle threads).
// At wider H V1==V2 (identical template params); the small T=256 H=4
// regression is GPU-to-GPU variance, not algorithmic.
//
// Deferred optimization candidates:
//   - Non-temporal store hint for kv-cache writes (kv-cache is rarely
//     re-read until attention; NT store may bypass an L2 fill that's
//     immediately discarded). Needs a probe-style mu-asm 'slc=nt' load
//     equivalent for store; speculative until measured.
//   - TME tile-stride load to overlap key + value streams. Gated behind
//     MUTE_ARCH_TME_MP31_ACTIVATED. Speculative.
//   - Batch decode-step writes across multiple iterations to amortize
//     launch overhead at small T (currently 4 µs/launch + 0.1 µs/MB of
//     traffic; at small-T the launch overhead dominates).
//
// =============================================================================

namespace vllm_musa {

template <typename T, int BLOCK_X, int TOKENS_PER_BLOCK>
__global__ void __launch_bounds__(BLOCK_X, 1)
    reshape_and_cache_flash_nhd_kernel(
        const T* __restrict__ key, const T* __restrict__ value,
        T* __restrict__ key_cache, T* __restrict__ value_cache,
        const int64_t* __restrict__ slot_mapping, int num_tokens,
        int vecs_per_token, int block_size, int64_t key_stride,
        int64_t value_stride, int64_t block_stride, int64_t page_stride) {
  constexpr int VEC_SIZE = 8;
  const int base_token = blockIdx.x * TOKENS_PER_BLOCK;
  const int total_vecs = TOKENS_PER_BLOCK * vecs_per_token;

  for (int linear = threadIdx.x; linear < total_vecs; linear += BLOCK_X) {
    const int local_token = linear / vecs_per_token;
    const int token_idx = base_token + local_token;
    if (token_idx >= num_tokens) {
      continue;
    }

    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }

    const int vec_idx = linear - local_token * vecs_per_token;
    const int elem_offset = vec_idx * VEC_SIZE;
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx - block_idx * block_size;

    const int64_t src_k = static_cast<int64_t>(token_idx) * key_stride +
                          elem_offset;
    const int64_t src_v = static_cast<int64_t>(token_idx) * value_stride +
                          elem_offset;
    const int64_t dst = block_idx * block_stride + block_offset * page_stride +
                        elem_offset;

    Vec16<T> key_vec = vload16_byp_slc(key + src_k);
    Vec16<T> value_vec = vload16_byp_slc(value + src_v);
    vstore16(key_cache + dst, key_vec);
    vstore16(value_cache + dst, value_vec);
  }
}

template <typename T>
void dispatch_reshape_and_cache_flash_nhd(
    const T* key, const T* value, T* key_cache, T* value_cache,
    const int64_t* slot_mapping, int num_tokens, int num_heads, int head_size,
    int block_size, int64_t key_stride, int64_t value_stride,
    int64_t block_stride, int64_t page_stride, musaStream_t stream) {
  const int vecs_per_token = (num_heads * head_size) / 8;

  static const int forced_tokens_per_block = []() {
    const char* env = std::getenv("VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK");
    return env == nullptr ? 0 : std::atoi(env);
  }();

  // MUSA-0151 (cold-cache re-bench 2026-05-25, mate.bench_kineto +
  // flush_l2=True): V1 fixed BLOCK_X=512 leaves up to 75 % of threads idle
  // at narrow-KV shapes. M2.5 per-rank has num_heads_kv=1 head_size=128 ->
  // vecs_per_token=16 -> TPB*vecs = 128, leaving 384/512 threads idle in
  // each block (the for-loop's `linear < total_vecs` condition exits
  // immediately for them). V2 adapts BLOCK_X to the actual work
  // (TPB * vecs_per_token rounded up to next power of 2, floored at 32 to
  // avoid sub-warp blocks, capped at 512 to keep cross-warp shared-mem
  // reduce dormant). Smaller BLOCK_X recovers 4 x more concurrent CTAs
  // per MP at narrow shapes; wider shapes (vecs >= 128) are unaffected
  // because they already saturate BLOCK_X=512.

#define LAUNCH(BX, TPB)                                                    \
  reshape_and_cache_flash_nhd_kernel<T, BX, TPB>                           \
      <<<(num_tokens + (TPB) - 1) / (TPB), BX, 0, stream>>>(               \
          key, value, key_cache, value_cache, slot_mapping, num_tokens,    \
          vecs_per_token, block_size, key_stride, value_stride,            \
          block_stride, page_stride)

  // Pick TOKENS_PER_BLOCK first (same heuristic as V1).
  int tpb;
  if (forced_tokens_per_block == 1 || forced_tokens_per_block == 2 ||
      forced_tokens_per_block == 4 || forced_tokens_per_block == 8) {
    tpb = forced_tokens_per_block;
  } else if (vecs_per_token <= 64) {
    tpb = 8;
  } else if (vecs_per_token <= 128) {
    tpb = 4;
  } else if (vecs_per_token <= 256) {
    tpb = 2;
  } else {
    tpb = 1;
  }

  // Then pick the smallest BLOCK_X >= TPB * vecs_per_token (pow2, in
  // [32, 512]). At narrow shapes this trims to 32/64/128 and quadruples
  // concurrent CTAs per MP. At wider shapes BLOCK_X stays 512.
  const int total = tpb * vecs_per_token;
  if (total <= 32) {
    if (tpb == 1) { LAUNCH(32, 1); }
    else if (tpb == 2) { LAUNCH(32, 2); }
    else if (tpb == 4) { LAUNCH(32, 4); }
    else { LAUNCH(32, 8); }
  } else if (total <= 64) {
    if (tpb == 1) { LAUNCH(64, 1); }
    else if (tpb == 2) { LAUNCH(64, 2); }
    else if (tpb == 4) { LAUNCH(64, 4); }
    else { LAUNCH(64, 8); }
  } else if (total <= 128) {
    if (tpb == 1) { LAUNCH(128, 1); }
    else if (tpb == 2) { LAUNCH(128, 2); }
    else if (tpb == 4) { LAUNCH(128, 4); }
    else { LAUNCH(128, 8); }
  } else if (total <= 256) {
    if (tpb == 1) { LAUNCH(256, 1); }
    else if (tpb == 2) { LAUNCH(256, 2); }
    else if (tpb == 4) { LAUNCH(256, 4); }
    else { LAUNCH(256, 8); }
  } else {
    if (tpb == 1) { LAUNCH(512, 1); }
    else if (tpb == 2) { LAUNCH(512, 2); }
    else if (tpb == 4) { LAUNCH(512, 4); }
    else { LAUNCH(512, 8); }
  }

#undef LAUNCH
}

}  // namespace vllm_musa

void musa_reshape_and_cache_flash_nhd(torch::Tensor& key, torch::Tensor& value,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache,
                                      torch::Tensor& slot_mapping) {
  TORCH_CHECK(key.device().is_privateuseone(), "key must be a MUSA tensor");
  TORCH_CHECK(value.device().is_privateuseone(), "value must be a MUSA tensor");
  TORCH_CHECK(key_cache.device().is_privateuseone(),
              "key_cache must be a MUSA tensor");
  TORCH_CHECK(value_cache.device().is_privateuseone(),
              "value_cache must be a MUSA tensor");
  TORCH_CHECK(slot_mapping.device().is_privateuseone(),
              "slot_mapping must be a MUSA tensor");
  TORCH_CHECK(key.dim() == 3, "key must be [tokens, heads, head_size]");
  TORCH_CHECK(value.dim() == 3, "value must be [tokens, heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be [blocks, block_size, heads, head_size]");
  TORCH_CHECK(value_cache.dim() == 4,
              "value_cache must be [blocks, block_size, heads, head_size]");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1-D");
  TORCH_CHECK(key.scalar_type() == value.scalar_type(),
              "key/value dtype mismatch");
  TORCH_CHECK(key.scalar_type() == key_cache.scalar_type(),
              "key/key_cache dtype mismatch");
  TORCH_CHECK(value.scalar_type() == value_cache.scalar_type(),
              "value/value_cache dtype mismatch");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int64");
  TORCH_CHECK(key.stride(2) == 1 && key.stride(1) == key.size(2),
              "key head/head_size dimensions must be contiguous");
  TORCH_CHECK(value.stride(2) == 1 && value.stride(1) == value.size(2),
              "value head/head_size dimensions must be contiguous");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0),
              "key/value cache block count mismatch");
  TORCH_CHECK(key_cache.size(1) == value_cache.size(1),
              "key/value cache block size mismatch");
  TORCH_CHECK(key_cache.size(2) == value_cache.size(2),
              "key/value cache head count mismatch");
  TORCH_CHECK(key_cache.size(3) == value_cache.size(3),
              "key/value cache head size mismatch");
  TORCH_CHECK(key.size(1) == key_cache.size(2), "head count mismatch");
  TORCH_CHECK(value.size(1) == value_cache.size(2), "value head count mismatch");
  TORCH_CHECK(key.size(2) == key_cache.size(3), "head size mismatch");
  TORCH_CHECK(value.size(2) == value_cache.size(3), "value head size mismatch");
  TORCH_CHECK(slot_mapping.size(0) <= key.size(0),
              "slot_mapping longer than key rows");
  TORCH_CHECK(slot_mapping.size(0) <= value.size(0),
              "slot_mapping longer than value rows");
  TORCH_CHECK(key.size(2) % 8 == 0, "head_size must be a multiple of 8");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "cache head dimension must be contiguous");
  TORCH_CHECK(key_cache.stride(2) == key_cache.size(3),
              "key_cache must use NHD layout");
  TORCH_CHECK(value_cache.stride(2) == value_cache.size(3),
              "value_cache must use NHD layout");
  TORCH_CHECK(key_cache.stride(1) == key_cache.size(2) * key_cache.size(3),
              "key_cache page stride must be contiguous NHD");
  TORCH_CHECK(value_cache.stride(1) ==
                  value_cache.size(2) * value_cache.size(3),
              "value_cache page stride must be contiguous NHD");

  const c10::musa::OptionalMUSAGuard guard(device_of(key));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  const int num_tokens = static_cast<int>(slot_mapping.size(0));
  const int num_heads = static_cast<int>(key.size(1));
  const int head_size = static_cast<int>(key.size(2));
  const int block_size = static_cast<int>(key_cache.size(1));
  const int64_t key_stride = key.stride(0);
  const int64_t value_stride = value.stride(0);
  const int64_t block_stride = key_cache.stride(0);
  const int64_t page_stride = key_cache.stride(1);

  if (num_tokens == 0) {
    return;
  }

  if (key.scalar_type() == at::ScalarType::Half) {
    vllm_musa::dispatch_reshape_and_cache_flash_nhd<__half>(
        static_cast<const __half*>(key.data_ptr()),
        static_cast<const __half*>(value.data_ptr()),
        static_cast<__half*>(key_cache.data_ptr()),
        static_cast<__half*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), num_tokens, num_heads, head_size,
        block_size, key_stride, value_stride, block_stride, page_stride,
        stream);
  } else if (key.scalar_type() == at::ScalarType::BFloat16) {
    vllm_musa::dispatch_reshape_and_cache_flash_nhd<__mt_bfloat16>(
        static_cast<const __mt_bfloat16*>(key.data_ptr()),
        static_cast<const __mt_bfloat16*>(value.data_ptr()),
        static_cast<__mt_bfloat16*>(key_cache.data_ptr()),
        static_cast<__mt_bfloat16*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), num_tokens, num_heads, head_size,
        block_size, key_stride, value_stride, block_stride, page_stride,
        stream);
  } else {
    TORCH_CHECK(false, "only fp16 and bf16 are supported");
  }
}
