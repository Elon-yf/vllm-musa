#include <optional>
#include <torch/library.h>

#include "core/scalar_type.hpp"

#include <vector>

void musa_fused_gemv_moe(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    torch::Tensor &topk_weights,
    torch::Tensor &topk_ids,
    bool mul_routed_weight,
    int64_t topk,
    bool use_int4_w4a16,
    bool use_swigelu);

void musa_fused_gemv(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    bool use_int4_w4a16,
    bool use_swigelu,
    bool use_rms_norm,
    const c10::optional<torch::Tensor> &gamma,
    double eps);

void musa_fused_add_rms_norm(
    torch::Tensor &input,
    torch::Tensor &residual,
    torch::Tensor &weight,
    double eps);

// MUSA-0123: fused (all-reduce + add-residual + RMS-norm) op.
// API matches existing _C_custom_ar.all_reduce: takes fa handle and an
// IPC-registered reg_buffer; the wrapper copies input → reg_buffer
// then runs the fused kernel reading from peer reg_buffers.
void musa_fused_ar_rmsnorm(
    int64_t fa,
    torch::Tensor &input,
    torch::Tensor &residual,
    torch::Tensor &weight,
    torch::Tensor &output,
    int64_t reg_buffer,
    int64_t reg_buffer_sz_bytes,
    double epsilon);

void musa_reshape_and_cache_flash_nhd(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &slot_mapping);

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max, bool scale_ue8m0,
    bool dummy_is_scale_transposed = false,
    bool dummy_is_tma_aligned = false);

void silu_and_mul_per_token_group_fp8_quant(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max);
