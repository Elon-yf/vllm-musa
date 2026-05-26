#include <torch/all.h>
#include <torch/library.h>

void musa_fused_gemv_moe(
    torch::Tensor &A, torch::Tensor &B, torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    torch::Tensor &topk_weights, torch::Tensor &topk_ids,
    bool mul_routed_weight, int64_t topk, bool use_int4_w4a16, bool use_swigelu);

TORCH_LIBRARY(gemv_dev, m) {
    m.def(
        "musa_fused_gemv_moe(Tensor! A, Tensor! B, Tensor! C, Tensor? A_scale, "
        "Tensor? B_scale, Tensor! topk_weights, Tensor! topk_ids, "
        "bool mul_routed_weight, int topk, bool use_int4_w4a16, "
        "bool use_swigelu) -> ()");
}
TORCH_LIBRARY_IMPL(gemv_dev, PrivateUse1, m) {
    m.impl("musa_fused_gemv_moe", &musa_fused_gemv_moe);
}
