// SPDX-License-Identifier: Apache-2.0
// first-pass port of SGLang's quick_all_reduce_base.h. NOT YET COMPILED.
//
#pragma once

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cstdint>

#define __quickreduce_device_inline__ __device__ __forceinline__
#define __quickreduce_launch_bounds_two_shot__ __launch_bounds__(128, 4)  // kBlockSize=128
#define __quickreduce_launch_bounds_one_shot__ __launch_bounds__(256, 4)  // half of HIP

namespace quickreduce {

typedef __mt_bfloat16 nv_bfloat16;  // __hip_bfloat16 -> __mt_bfloat16
typedef __mt_bfloat162 nv_bfloat162;  // __hip_bfloat162 -> __mt_bfloat162

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

// Setup acquire-release semantics for vector memory reads (mubuf instruction)
// as per architecture.
#if defined(__gfx942__)
// CDNA3: Scope bits sc0, sc1
#define MUBUF_ACQUIRE 16
#define MUBUF_RELEASE 16
#elif (defined(__gfx908__) || defined(__gfx90a__))
// CDNA1 and CDNA2 - glc bit
#define MUBUF_ACQUIRE 1
#define MUBUF_RELEASE 0
#endif

static constexpr int kNegOne = 0xBC00BC00;  // {-1, -1}, fp16x2_t

// Number of atoms (4xf16x2_t) processed by a single thread
static constexpr int kAtoms = 8;

// warp/block-size adaptation for MTT S5000.
// HIP/AMD CDNA wavefront = 64; MUSA on S5000 = 32. To preserve the
// "4 warps per block" ratio that the codec macros assume, drop block
// size from 256 (= 4*64) to 128 (= 4*32). This is the conservative
// choice; if mcc compiles cleanly, an A/B at block=256 (= 8 warps on
// MUSA) is worth probing - more warps may yield better occupancy on
// S5000's 96-CU silicon.
static constexpr int kBlockSize = 128;
static constexpr int kAtomStride = kBlockSize;

// Size and atom stride of source/destination data that the block will
// process.
// Workgroup scope = Tile = (kBlockSize threads x 8 atoms x 16B)
static constexpr int kTileSize = kBlockSize * kAtoms * sizeof(int32x4_t);

// Max number of blocks. MTT S5000 has 96 CUs (vs MI300's 304); keep the
// 4x factor for occupancy headroom.
static constexpr int kMaxNumBlocks = 96 * 4;

// MUSA wavefront size on MTT S5000.
static constexpr int kWavefront = 32;

// kBlockSize threads, 4 wavefronts (kBlockSize/kWavefront = 4).
static dim3 constexpr kBlockTwoShot = {kWavefront, kBlockSize / kWavefront, 1};

// Number of threads in a group for quantization
// It corresponds to 32 F16 elements in quantization block
static constexpr int kThreadGroupSize = 8;

// Methods
__quickreduce_device_inline__ __host__ unsigned long divceil(unsigned long x, unsigned long y) {
  return ((x + y - 1) / y);
}

// BufferResource was originally a UNION holding an int32x4_t
// descriptor that AMD GCN's vector buffer-fetch unit consumes (the
// `buffer_load_dwordx4` / `buffer_store_dwordx4` AMD intrinsics). MUSA's
// mp_31 backend does not have an equivalent buffer-fetch unit and cannot
// lower `llvm.amdgcn.raw.buffer.{load,store}.v4i32` — observed as
// `error in backend: lsu.st.2d requires constant blx, bly, stride` and
// `v4i32,ch = llvm.amdgcn.raw.buffer.load` in iter-3
// (`generated/musa0088/iter3-final-state-and-mcc-segfault.md`).
//
// Replacement: plain pointer wrapper + portable `int32x4_t*` vector
// load/store. 16-byte alignment is preserved because the callers compute
// `voffset` as multiples of `sizeof(int32x4_t)` (= 16 bytes).
struct BufferResource {
  __quickreduce_device_inline__ constexpr BufferResource() : address(nullptr), range(0) {}

  __quickreduce_device_inline__ constexpr BufferResource(void* buffer_address, uint32_t buffer_size)
      : address(buffer_address), range(buffer_size) {}

  void* address;
  uint32_t range;
};

// AMD `llvm.amdgcn.raw.buffer.load.v4i32` -> portable int4 load.
// The original AMD signature took `(int32x4_t srsrc, int32_t voffset,
// int32_t soffset, int32_t aux)` where srsrc was the buffer descriptor and
// aux was AMD-specific. We take a `BufferResource const&` to keep the
// portable address-bearing variant; soffset and aux are kept as ignored
// parameters so the calling-side diff stays minimal (existing kernels pass
// soffset=0, aux=0 — pure AMD-isms).
__quickreduce_device_inline__ static int32x4_t buffer_load_dwordx4(
    BufferResource const& buf, int32_t voffset, int32_t soffset = 0, int32_t aux = 0) {
  (void)soffset;
  (void)aux;
  const char* base = reinterpret_cast<const char*>(buf.address);
  return *reinterpret_cast<const int32x4_t*>(base + voffset);
}

__quickreduce_device_inline__ static void buffer_store_dwordx4(
    int32x4_t data, BufferResource const& buf, int32_t voffset, int32_t soffset = 0, int32_t aux = 0) {
  (void)soffset;
  (void)aux;
  char* base = reinterpret_cast<char*>(buf.address);
  *reinterpret_cast<int32x4_t*>(base + voffset) = data;
}

__quickreduce_device_inline__ static void set_fp16_ovfl(bool const value) {
#if defined(__gfx942__)
  if (value) {
    asm volatile("s_setreg_imm32_b32 0xdc1, 1;" ::);
  } else {
    asm volatile("s_setreg_imm32_b32 0xdc1, 0;" ::);
  }
#endif
}
union bf162_int_union {
  int i;
  nv_bfloat162 bf2;
};

template <typename T>
__quickreduce_device_inline__ void packed_assign_add(int32x4_t* A, int32x4_t* B);

template <>
__quickreduce_device_inline__ void packed_assign_add<half>(int32x4_t* A, int32x4_t* B) {
  int32x4_t& tR_fragment = A[0];
  int32x4_t& tA_fragment = B[0];

  // AMD GCN v_pk_add_f16 -> portable __hadd2. Use cast-and-offset
  // because tR_fragment is a clang vector (__vector_size__ attribute) — taking
  // address of [i] is rejected.
  int32_t* rp = reinterpret_cast<int32_t*>(&tR_fragment);
  int32_t* ap = reinterpret_cast<int32_t*>(&tA_fragment);
  for (int i = 0; i < 4; ++i) {
    __half2 r = *reinterpret_cast<__half2*>(&rp[i]);
    __half2 a = *reinterpret_cast<__half2*>(&ap[i]);
    *reinterpret_cast<__half2*>(&rp[i]) = __hadd2(r, a);
  }
}

template <>
__quickreduce_device_inline__ void packed_assign_add<nv_bfloat16>(int32x4_t* A, int32x4_t* B) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(A);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(B);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    tA[i] = __hadd2(tA[i], tB[i]);
  }
}

template <typename T>
__quickreduce_device_inline__ int packed_max(int a, int b);

template <>
__quickreduce_device_inline__ int packed_max<half>(int a, int b) {
  // AMD GCN v_pk_max_f16 -> portable __hmax2.
  __half2 a2 = *reinterpret_cast<__half2*>(&a);
  __half2 b2 = *reinterpret_cast<__half2*>(&b);
  __half2 r2 = __hmax2(a2, b2);
  int result;
  *reinterpret_cast<__half2*>(&result) = r2;
  return result;
}

template <>
__quickreduce_device_inline__ int packed_max<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hmax2(A.bf2, B.bf2);
  return R.i;
}

template <typename T>
__quickreduce_device_inline__ int packed_min(int a, int b);

template <>
__quickreduce_device_inline__ int packed_min<half>(int a, int b) {
  // AMD GCN v_pk_min_f16 -> portable __hmin2.
  __half2 a2 = *reinterpret_cast<__half2*>(&a);
  __half2 b2 = *reinterpret_cast<__half2*>(&b);
  __half2 r2 = __hmin2(a2, b2);
  int result;
  *reinterpret_cast<__half2*>(&result) = r2;
  return result;
}

template <>
__quickreduce_device_inline__ int packed_min<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hmin2(A.bf2, B.bf2);
  return R.i;
}

template <typename T>
__quickreduce_device_inline__ int packed_abs_max(int a, int b);

template <>
__quickreduce_device_inline__ int packed_abs_max<half>(int a, int b) {
  half2 wmaxh2 = __builtin_bit_cast(half2, a);
  half2 wminh2 = __builtin_bit_cast(half2, b);
  half2 wblockmaxh2;

  wblockmaxh2.x = __hgt(__habs(wmaxh2.x), __habs(wminh2.x)) ? wmaxh2.x : wminh2.x;
  wblockmaxh2.y = __hgt(__habs(wmaxh2.y), __habs(wminh2.y)) ? wmaxh2.y : wminh2.y;
  return __builtin_bit_cast(int, wblockmaxh2);
}

template <>
__quickreduce_device_inline__ int packed_abs_max<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2.x = __hgt(__habs(A.bf2.x), __habs(B.bf2.x)) ? A.bf2.x : B.bf2.x;
  R.bf2.y = __hgt(__habs(A.bf2.y), __habs(B.bf2.y)) ? A.bf2.y : B.bf2.y;
  return R.i;
}

template <typename T>
__quickreduce_device_inline__ int packed_add(int a, int b);

template <>
__quickreduce_device_inline__ int packed_add<half>(int a, int b) {
  // AMD GCN v_pk_add_f16 -> portable __hadd2.
  __half2 a2 = *reinterpret_cast<__half2*>(&a);
  __half2 b2 = *reinterpret_cast<__half2*>(&b);
  __half2 r2 = __hadd2(a2, b2);
  int result;
  *reinterpret_cast<__half2*>(&result) = r2;
  return result;
}

template <>
__quickreduce_device_inline__ int packed_add<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hadd2(A.bf2, B.bf2);
  return R.i;
}

template <>
__quickreduce_device_inline__ int packed_add<int16_t>(int a, int b) {
  // AMD GCN v_pk_add_i16 -> scalar int16x2 add via bit-fiddle.
  int16_t a_lo = static_cast<int16_t>(a & 0xFFFF);
  int16_t a_hi = static_cast<int16_t>((a >> 16) & 0xFFFF);
  int16_t b_lo = static_cast<int16_t>(b & 0xFFFF);
  int16_t b_hi = static_cast<int16_t>((b >> 16) & 0xFFFF);
  int16_t r_lo = static_cast<int16_t>(a_lo + b_lo);
  int16_t r_hi = static_cast<int16_t>(a_hi + b_hi);
  int result = (static_cast<int>(r_hi) << 16) | (static_cast<int>(static_cast<uint16_t>(r_lo)));
  return result;
}

template <typename T>
__quickreduce_device_inline__ int packed_sub(int a, int b);

template <>
__quickreduce_device_inline__ int packed_sub<half>(int a, int b) {
  int result;

  // AMD GCN v_pk_fma_f16(-1, b, a) = a - b -> portable __hsub2.
  __half2 a2 = *reinterpret_cast<__half2*>(&a);
  __half2 b2 = *reinterpret_cast<__half2*>(&b);
  __half2 r2 = __hsub2(a2, b2);
  *reinterpret_cast<__half2*>(&result) = r2;
  return result;
}

template <>
__quickreduce_device_inline__ int packed_sub<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hsub2(A.bf2, B.bf2);
  return R.i;
}

template <typename T>
__quickreduce_device_inline__ int packed_mul(int a, int b);

template <>
__quickreduce_device_inline__ int packed_mul<half>(int a, int b) {
  // AMD GCN v_pk_mul_f16 -> portable __hmul2.
  __half2 a2 = *reinterpret_cast<__half2*>(&a);
  __half2 b2 = *reinterpret_cast<__half2*>(&b);
  __half2 r2 = __hmul2(a2, b2);
  int result;
  *reinterpret_cast<__half2*>(&result) = r2;
  return result;
}

template <>
__quickreduce_device_inline__ int packed_mul<nv_bfloat16>(int a, int b) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(&b);
  nv_bfloat162 tR = __hmul2(*tA, *tB);
  return *(reinterpret_cast<int*>(&tR));
}

template <typename T>
__quickreduce_device_inline__ int packed_rcp(int a);

template <>
__quickreduce_device_inline__ int packed_rcp<half>(int a) {
  return __builtin_bit_cast(int, h2rcp(__builtin_bit_cast(half2, a)));
}

template <>
__quickreduce_device_inline__ int packed_rcp<nv_bfloat16>(int a) {
  bf162_int_union A, R;
  A.i = a;
  R.bf2 = h2rcp(A.bf2);
  return R.i;
}

// changes dtype
__quickreduce_device_inline__ float T2float_cast(half a) {
  return __half2float(a);
}

__quickreduce_device_inline__ float T2float_cast(nv_bfloat16 a) {
  return __bfloat162float(a);
}

template <typename T>
__quickreduce_device_inline__ int group_abs_max(int32x4_t atom) {
  const int group_leader = (threadIdx.x / kThreadGroupSize) * kThreadGroupSize;

  int wmax, wmin, wblockmax;
  int a, b;
  a = packed_max<T>(atom[0], atom[1]);
  b = packed_max<T>(atom[2], atom[3]);

  wmax = packed_max<T>(a, b);

  a = packed_min<T>(atom[0], atom[1]);
  b = packed_min<T>(atom[2], atom[3]);

  wmin = packed_min<T>(a, b);

  // Reduce the max among a group of threads
  // Note: This is basically 2 blocks of values setup as the
  // upper/lower halves of the f16x2_t
  for (int i = 1; i < kThreadGroupSize; i <<= 1) {
    int x = __shfl_down_sync(0xFFFFFFFF, wmax, i);
    wmax = packed_max<T>(wmax, x);

    int y = __shfl_down_sync(0xFFFFFFFF, wmin, i);
    wmin = packed_min<T>(wmin, y);
  }
  wblockmax = packed_abs_max<T>(wmax, wmin);
  // Share with the cohort
  wblockmax = __shfl_sync(0xFFFFFFFF, wblockmax, group_leader);
  return wblockmax;
}

__quickreduce_device_inline__ void set_sync_flag(uint32_t* flag_ptr, uint32_t flag) {
  __atomic_store_n(flag_ptr, flag, __ATOMIC_RELEASE);
}

__quickreduce_device_inline__ void wait_sync_flag(uint32_t* flag_ptr, uint32_t flag) {
  while (__atomic_load_n(flag_ptr, __ATOMIC_RELAXED) != flag) {
  }
}

}  // namespace quickreduce
