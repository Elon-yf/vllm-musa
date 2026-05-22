// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MUSA-0088 first-pass port of SGLang's quick_all_reduce.h.
// Translation: mechanical sed-pass per MUSA-0080 iter-2 breakdown
// + hipLaunchKernelGGL -> direct <<<grid, block, sharedMem, stream>>>.
//
// NOT YET COMPILED. Follow-up:
//   - Verify mcc accepts the <<<>>> launch syntax (vs hipLaunchKernelGGL)
//   - Adapt warp-size assumptions (HIP wave 64 -> MUSA wave 32)
//   - Wire csrc/quick_all_reduce.cu into setup.py kernel list
//   - DONE: musaIpcMemLazyEnablePeerAccess exists in driver_types.h.
//   - musaMallocWithFlags / musaDeviceMallocUncached not in MUSA SDK;
//     replaced with plain musaMalloc (microopt only, not correctness).
//   - Build + correctness + perf A/B per MUSA-0088 ACs
//
#pragma once

#include <musa_runtime.h>

#include <vector>

#include "quick_all_reduce.cuh"

#define MUSA_CHECK(err)                                                                               \
  do {                                                                                               \
    musaError_t err_ = (err);                                                                         \
    if (err_ != musaSuccess) {                                                                        \
      std::printf("HIP error %d at %s:%d. %s\n", err_, __FILE__, __LINE__, musaGetErrorString(err_)); \
      throw std::runtime_error("HIP error");                                                         \
    }                                                                                                \
  } while (0)

namespace quickreduce {
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void allreduce_prototype_twoshot(
    T const* A,
    T* B,
    uint32_t N,
    uint32_t num_blocks,
    int rank,
    uint8_t** dbuffer_list,
    uint32_t data_offset,
    uint32_t flag_color,
    int64_t data_size_per_phase) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  while (block < num_blocks) {
    AllReduceKernel::run(A, B, N, block, rank, dbuffer_list, data_offset, flag_color, data_size_per_phase);
    block += grid;
    flag_color++;
  }
}

#define TWOSHOT_DISPATCH(__codec)                                         \
  if (world_size == 2) {                                                  \
    using LineCodec = __codec<T, 2>;                                      \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>; \
    allreduce_prototype_twoshot<AllReduceKernel, T><<<dim3(grid), dim3(kBlockTwoShot), 0, stream>>>(A, B, N, num_blocks, rank, dbuffer_list, data_offset, flag_color, data_size_per_phase);                                           \
  } else if (world_size == 4) {                                           \
    using LineCodec = __codec<T, 4>;                                      \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>; \
    allreduce_prototype_twoshot<AllReduceKernel, T><<<dim3(grid), dim3(kBlockTwoShot), 0, stream>>>(A, B, N, num_blocks, rank, dbuffer_list, data_offset, flag_color, data_size_per_phase);                                           \
  } else if (world_size == 8) {                                           \
    using LineCodec = __codec<T, 8>;                                      \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>; \
    allreduce_prototype_twoshot<AllReduceKernel, T><<<dim3(grid), dim3(kBlockTwoShot), 0, stream>>>(A, B, N, num_blocks, rank, dbuffer_list, data_offset, flag_color, data_size_per_phase);                                           \
  }

enum QuickReduceQuantLevel {
  F16 = 0,
  INT8 = 1,
  INT6 = 2,
  INT4 = 3,
};

struct DeviceComms {
  // Max problem size is 2GB (in bytes) or half of uint32_t max value.
  int64_t kMaxProblemSize = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

  // Max TP-8
  static int constexpr kMaxWorldSize = 8;

  bool initialized = false;
  uint32_t flag_color = 1;
  int world_size;
  int rank;

  uint8_t* dbuffer;
  uint8_t** dbuffer_list;
  musaIpcMemHandle_t buffer_ipc_handle;
  std::vector<musaIpcMemHandle_t> all_buffer_ipc_handles;
  std::vector<uint8_t*> buffer_list;
  uint32_t data_offset;

  DeviceComms() : initialized(false), world_size(1), rank(0) {}
  ~DeviceComms() {
    destroy();
  }

  void init(int world_size, int rank, std::optional<int64_t> max_problem_size = std::nullopt) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;
    if (max_problem_size.has_value() && max_problem_size.value() > 0) {
      this->kMaxProblemSize = max_problem_size.value();
    }
    // Allocate buffer size for worst case: F16 2-stage buffer.
    // NOTE: data_buffer_size must NOT be `static` — function-local
    // statics are initialized exactly once and would reuse the first
    // call's kMaxProblemSize for every subsequent init, allocating an
    // incorrect total_buffer_size when the caller passes a different
    // max_problem_size. See PR #40 review comment.
    uint32_t flags_buffer_size = 2 * world_size * kMaxNumBlocks * sizeof(uint32_t);
    int64_t data_buffer_size = 2 * this->kMaxProblemSize;
    int64_t total_buffer_size = flags_buffer_size + data_buffer_size;
    data_offset = flags_buffer_size;
    // MUSA-0088: hipExtMallocWithFlags(..., hipDeviceMallocUncached) has no
    // direct MUSA equivalent. Use plain musaMalloc; the uncached flag on
    // HIP was a microoptimisation for AMD's MI300 cache hierarchy and isn't
    // strictly necessary for correctness on MTT S5000. If profiling shows
    // this matters, replace with musaMallocManaged + musaMemAdvise to
    // approximate uncached behaviour.
    MUSA_CHECK(musaMalloc((void**)&dbuffer, total_buffer_size));

    // Clear the flags buffer.
    MUSA_CHECK(musaMemset(dbuffer, 0, flags_buffer_size));

    // Device-side list of IPC buffers.
    buffer_list.resize(world_size);
    MUSA_CHECK(musaMalloc(&dbuffer_list, world_size * sizeof(uint8_t*)));

    // Create IPC handles for rank's communication buffer.
    all_buffer_ipc_handles.resize(world_size);
    MUSA_CHECK(musaIpcGetMemHandle(&buffer_ipc_handle, dbuffer));

    initialized = true;
  }
  int get_world_size() {
    return world_size;
  }
  int get_rank() {
    return rank;
  }
  bool status() {
    return initialized;
  }
  musaIpcMemHandle_t const get_handle() {
    return buffer_ipc_handle;
  }

  void destroy() {
    if (initialized) {
      // Close IPC handles against the host-side vector of opened peer
      // pointers (buffer_list), not the device-side mirror
      // (dbuffer_list). dbuffer_list is a device allocation used by
      // the kernel via H2D-copied pointer values; dereferencing it
      // from host code passes garbage to musaIpcCloseMemHandle. See
      // PR #40 review comment.
      for (int i = 0; i < world_size; i++) {
        if (i != rank) {
          MUSA_CHECK(musaIpcCloseMemHandle(buffer_list[i]));
        }
      }

      MUSA_CHECK(musaFree(dbuffer));
      MUSA_CHECK(musaFree(dbuffer_list));

      initialized = false;
    }
  }

  void open_ipc_handles(std::vector<musaIpcMemHandle_t> const& ipc_handles) {
    assert(ipc_handles.size() == all_buffer_ipc_handles.size());
    for (int i = 0; i < world_size; i++) {
      all_buffer_ipc_handles[i] = ipc_handles[i];
    }

    // Open device memory access to the IPC communication buffers.
    // Note: For our own rank, we do not need to open a handle.
    for (int i = 0; i < world_size; i++) {
      if (i != rank) {
        MUSA_CHECK(
            musaIpcOpenMemHandle((void**)&buffer_list[i], all_buffer_ipc_handles[i], musaIpcMemLazyEnablePeerAccess));
      } else {
        buffer_list[i] = dbuffer;
      }
    }

    MUSA_CHECK(musaMemcpy(dbuffer_list, buffer_list.data(), world_size * sizeof(uint8_t*), musaMemcpyHostToDevice));
  }

  template <typename T, bool cast_bf2half>
  void allreduce(T const* A, T* B, uint32_t N, int quant_level, musaStream_t stream) {
    if (world_size != 2 && world_size != 4 && world_size != 8) {
      throw std::runtime_error("All Reduce not supported for world_size = " + std::to_string(world_size));
    }

    // Configuration.
    uint32_t msg_size = N * sizeof(T);
    uint32_t num_blocks = divceil(msg_size, kTileSize);
    uint32_t grid = min(kMaxNumBlocks, num_blocks);
    // MUSA-0116: place Phase-2 data immediately after Phase-1, with
    // stride = the actual aligned message size (num_blocks*kTileSize),
    // NOT +kMaxProblemSize (2 GiB). A cross-rank IPC store at a ~2 GiB
    // offset faults on S5000 (IMA localized to Phase-2A). The 2 GiB
    // stride was only a worst-case buffer-layout constant, never the
    // per-call data size; the comm buffer stays large enough either way.
    int64_t data_size_per_phase = static_cast<int64_t>(num_blocks) * kTileSize;
    // MUSA-0116: only the full-precision CodecFP path is enabled on MUSA.
    // The Int4/6/8 quant codecs pull in group_abs_max warp shuffles + extra
    // surface not needed for the M2.5 all-reduce; keep the build minimal.
    // MUSA-0091: dispatch the quantized codec on quant_level. INT4 moves
    // less cross-rank data; CodecFP is the lossless fallback.
    if (quant_level == INT4) {
      TWOSHOT_DISPATCH(CodecQ4)
    } else {
      TWOSHOT_DISPATCH(CodecFP)
    }
    MUSA_CHECK(cudaGetLastError());
    // Rotate the flag color.
    flag_color += divceil(N, grid);
  }
};

}  // namespace quickreduce
