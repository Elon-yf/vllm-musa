#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"
#include <torch/all.h>
#include <torch/library.h>
#include <torch/version.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops
  //




  // Attention ops
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  // paged_attention_v1 impl stripped (kernel not built on MUSA)

  // PagedAttention V2.
  // paged_attention_v2 impl stripped (kernel not built on MUSA)

  // Merge attn states
  // Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
  // can be used to combine partial attention results (in the split-KV case)

  // Activation ops (quantized only — basic ops moved to _C_stable_libtorch)

  // Fused SiLU+Mul + per-block quantization

  // Horizontally-fused DeepseekV4-MLA: per-head RMSNorm + GPT-J RoPE for Q, and
  // GPT-J RoPE + UE8M0 FP8 quant + paged cache insert for KV, all in one
  // kernel launch.
  // MUSA: fused DeepSeek-V4 qnorm/rope/cache impl stripped;
  // vllm_musa redirects this path to native/JIT MUSA implementations.

  // Apply repetition penalties to logits in-place

  // Optimized top-k per row operation



  // Quantization ops
#ifndef USE_MUSA

  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't interfere
  // with using ScalarType arguments in the ops. If they are marked as mutable,
  // pytorch throws an assert in
  // 'torch._higher_order_ops._register_effectful_op' that prevents these
  // kernels from being torch.compile'd.
  // See the following document for more info on custom types and ops that use
  // custom types:
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  // conditionally compiled so impl registration is in source file

  // Marlin Optimized Quantized GEMM (supports GPTQ, AWQ, FP8, NVFP4, MXFP4).
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
  // conditionally compiled so impl registrations are in source file

  // awq_marlin repack from AWQ.
  // conditionally compiled so impl registrations are in source file

  // preprocess W-int4A-fp8 weight for marlin kernel
  // conditionally compiled so impl registrations are in source file

#endif

#ifndef USE_MUSA
  // Expert-specialization mxfp8 blockscaled grouped quantization (SM100+).
  ops.def(
      "mxfp8_experts_quant("
      " Tensor input, Tensor problem_sizes, Tensor expert_offsets,"
      " Tensor blockscale_offsets, Tensor! quant_output, Tensor! scale_factor)"
      " -> ()");
  // conditionally compiled so impl registration is in source file

  // Expert-specialization mxfp8 blockscaled grouped GEMM (SM100+).
  ops.def(
      "cutlass_mxfp8_grouped_mm("
      " Tensor a, Tensor b, Tensor sfa, Tensor sfb, Tensor! out,"
      " Tensor problem_sizes, Tensor expert_offsets, Tensor blockscale_offsets)"
      " -> ()");
  // conditionally compiled so impl registration is in source file

#endif

  // Mamba selective scan kernel

#ifndef USE_MUSA

  //  conditionally compiled so impl in source file
#endif
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.

  // Gets the maximum shared memory per block device attribute.
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels




#ifdef USE_ROCM
  // Quick Reduce all-reduce kernels
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  custom_ar.def("init_custom_qr", &init_custom_qr);
  custom_ar.def("qr_destroy", &qr_destroy);

  custom_ar.def("qr_get_handle", &qr_get_handle);

  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  // Max input size in bytes
  custom_ar.def("qr_max_size", &qr_max_size);
#endif
}

torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);
// MUSA: get_cuda_view_from_cpu_tensor's kernel lives in cuda_view.cu (a
// regular _C source) and the stable bindings exclude its impl on MUSA, so
// register it here in the regular _C extension.
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, mcv) {
  mcv.impl("get_cuda_view_from_cpu_tensor", torch::kCPU,
           &get_cuda_view_from_cpu_tensor);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
