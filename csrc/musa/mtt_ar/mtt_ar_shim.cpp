// Thin C-ABI shim around MT-Transformer's custom AR launcher.
// Wraps mt::mtt::kernels::CallCommCustomArSqSumPH1 from libmtt.so so we can
// call it from vllm-musa without dragging in the mttransformer/Base.h
// header surface.

// Minimal mttransformer enum redeclaration. ABI must match libmtt's
// `mt::mtt::Type` enum.
// From core/include/mttransformer/Base.h:
//   enum class Type : int32_t { ... BFLOAT16 = 2, FLOAT16 = 1, ... };
namespace mt {
namespace mtt {
enum class Type : int {
    FLOAT32 = 0,
    FLOAT16 = 1,
    BFLOAT16 = 2,
    FLOAT8_E4M3 = 3,
    INT32 = 4,
    INT64 = 5,
    INT8 = 6,
};
}  // namespace mtt
}  // namespace mt

// musaStream_t is a typedef for `struct MUstream_st*`. We don't need musa
// runtime headers for the shim itself since we just forward the opaque
// pointer.
struct MUstream_st;
typedef MUstream_st* musaStream_t;

namespace mt {
namespace mtt {
namespace kernels {
// libmtt.so exports this with the C++-mangled symbol
// _ZN2mt3mtt7kernels24CallCommCustomArSqSumPH1EP11MUstream_stPvS4_iiiiNS0_4TypeEj
void CallCommCustomArSqSumPH1(musaStream_t s, void* sq_sum_out,
                              void* d_ptrs_array, int bs, int hidden_size,
                              int rank, int nranks, mt::mtt::Type dtype,
                              unsigned counter);
}  // namespace kernels
}  // namespace mtt
}  // namespace mt

extern "C" {

// Public C ABI. Forwards to CallCommCustomArSqSumPH1 with TP=8 BF16.
// d_ptrs is a device-side void** of length 8 (peer pointers, one per rank,
// each pointing to that rank's [bs, hidden] BF16 buffer).
void mtt_ar_bf16_tp8(musaStream_t stream, void* d_ptrs, int bs, int hidden,
                     int rank, unsigned counter) {
    mt::mtt::kernels::CallCommCustomArSqSumPH1(
        stream, /*sq_sum_out=*/nullptr, d_ptrs, bs, hidden, rank, /*nranks=*/8,
        mt::mtt::Type::BFLOAT16, counter);
}

// Variant with fused square_sum epilogue.
void mtt_ar_bf16_tp8_sqs(musaStream_t stream, void* sq_sum_out, void* d_ptrs,
                          int bs, int hidden, int rank, unsigned counter) {
    mt::mtt::kernels::CallCommCustomArSqSumPH1(
        stream, sq_sum_out, d_ptrs, bs, hidden, rank, 8,
        mt::mtt::Type::BFLOAT16, counter);
}

}  // extern "C"
