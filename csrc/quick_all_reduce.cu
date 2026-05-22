// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MUSA-0088 first-pass mechanical port of SGLang's quick_all_reduce.cu.
// Source: sglang/sgl-kernel/csrc/allreduce/quick_all_reduce.cu
// Translation: mechanical sed-pass per MUSA-0080 iter-2 breakdown,
// plus USE_ROCM -> USE_MUSA and HIPStreamMasquerading -> CUDAStream.
// NOT YET COMPILED OR TESTED. Follow-up:
//   - Wire into vllm-musa/setup.py MUSA_KERNELS list
//   - Add Python wrappers + dispatcher integration per the iter-2 breakdown
//   - Compile via mcc + iterate on errors
//   - Correctness gate (greedy parity) + perf A/B per MUSA-0088 ACs
//
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#ifdef USE_MUSA

#include "quick_all_reduce.h"
#include "core/registration.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

// MUSA-0116 DEBUG: definition of the phase-stop symbol declared extern in
// quick_all_reduce.cuh. 99 = full run (inert unless env QR_DBG_STOP set).
namespace quickreduce { __device__ int qr_dbg_stop = 99; }

quickreduce::fptr_t init_custom_qr(int64_t rank, int64_t world_size, std::optional<int64_t> qr_max_size) {
  if (world_size > 8) throw std::invalid_argument("world size > 8 is not supported");
  if (world_size == 6) throw std::invalid_argument("world size == 6 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank passed in");
  quickreduce::DeviceComms* fptr = new quickreduce::DeviceComms();
  fptr->init(world_size, rank, qr_max_size);
  return (quickreduce::fptr_t)fptr;
}

void qr_destroy(quickreduce::fptr_t _fa) {
  if (_fa) {
    auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
    fa->destroy();
    delete fa;
  }
}

torch::Tensor qr_get_handle(quickreduce::fptr_t _fa) {
  auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  musaIpcMemHandle_t handle = fa->get_handle();
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto data_handle = torch::empty({static_cast<int64_t>(sizeof(musaIpcMemHandle_t))}, options);
  std::memcpy(data_handle.data_ptr(), &handle, sizeof(musaIpcMemHandle_t));
  return data_handle;
}

void qr_open_handles(quickreduce::fptr_t _fa, const std::vector<torch::Tensor>& handles) {
  auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  std::vector<musaIpcMemHandle_t> ipc_handles;
  ipc_handles.reserve(handles.size());
  for (auto& handle : handles) {
    // Ensure the tensor is on the same device as the current device.
    musaIpcMemHandle_t ipc_handle;
    std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(musaIpcMemHandle_t));
    ipc_handles.push_back(ipc_handle);
  }
  fa->open_ipc_handles(ipc_handles);
}

void qr_all_reduce(
    quickreduce::fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, int64_t quant_level, bool cast_bf2half) {
  auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  // kMaxProblemSize is a byte budget (default int32_max+1 = 2 GiB),
  // not an element count. Compare against numel * element_size to
  // avoid letting fp16/bf16 buffers > 2 GiB bytes slip past the
  // check and overflow the comm allocation. See PR #40 review
  // comment.
  TORCH_CHECK_LE(out.numel() * out.element_size(), fa->kMaxProblemSize);
  if (out.scalar_type() == at::ScalarType::Half) {
    fa->allreduce<half, false>(
        reinterpret_cast<half*>(inp.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        out.numel(),
        quant_level,
        stream);
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    // MUSA-0116: bf16 always routes through the fp16 cast path. mcc 4.3.4
    // cannot compile the native nv_bfloat16 kernel (musa_bf16.h "h"
    // register-constraint asm). cast_bf2half is forced true here.
    (void)cast_bf2half;
    fa->allreduce<half, true>(
        reinterpret_cast<half*>(inp.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        out.numel(),
        quant_level,
        stream);
  } else {
    throw std::runtime_error("quick allreduce only supports float16 and bfloat16");
  }
}

int64_t qr_max_size() {
  // The default is 2GB (2,147,483,648 bytes)
  return static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
}

#define INSTANTIATE_FOR_WORLDSIZE(T, Codec, cast_bf2half)                      \
  template struct quickreduce::AllReduceTwoshot<T, Codec<T, 2>, cast_bf2half>; \
  template struct quickreduce::AllReduceTwoshot<T, Codec<T, 4>, cast_bf2half>; \
  template struct quickreduce::AllReduceTwoshot<T, Codec<T, 8>, cast_bf2half>;

// MUSA-0088 iter-5: strip to M2.5 production shape only (bf16 CodecFP, no cast).
// The full template surface (36 instantiations) triggers an mcc internal
// segfault in CallGraph Pass Manager. Restricting to this single shape
// keeps the kernel binary small and lets mcc complete. Other shapes
// (CodecQ4/Q6/Q8 quantized, half dtype, cast_bf2half=true) can be
// re-enabled when mate/mcc fixes the segfault.
// MUSA-0116: instantiate only the fp16 CodecFP path (cast=false for fp16
// inputs, cast=true for bf16 routed through fp16). The native nv_bfloat16
// instantiation is dropped - mcc cannot compile its musa_bf16.h asm.
INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecFP, true)
// INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ4, false)
// INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ6, false)
// INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ8, false)
// INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecFP, true)
// INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ4, true)
// INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ6, true)
// INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ8, true)

// MUSA-0088 iter-5: half-dtype variants stripped (only bf16 kept above).
// INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecFP, false)
// INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ4, false)
// INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ6, false)
// INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ8, false)

// MUSA-0116: register the quick_all_reduce ops under torch.ops._C_quick_ar.*
// (the namespace the Python wrapper quick_all_reduce.py probes). Compiled
// into the vllm_musa._C extension; registers when that .so is imported.
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _quick_ar), quick_ar) {
  quick_ar.def("init(int rank, int world_size, int? qr_max_size) -> int",
               &init_custom_qr);
  quick_ar.def("destroy(int fa) -> ()", &qr_destroy);
  quick_ar.def("get_handle(int fa) -> Tensor", &qr_get_handle);
  quick_ar.def("open_handles(int fa, Tensor[] handles) -> ()", &qr_open_handles);
  quick_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int quant_level, "
      "bool cast_bf2half) -> ()");
  quick_ar.impl("all_reduce", torch::kMUSA, &qr_all_reduce);
  quick_ar.def("max_size() -> int", &qr_max_size);
}

#endif  // USE_MUSA
