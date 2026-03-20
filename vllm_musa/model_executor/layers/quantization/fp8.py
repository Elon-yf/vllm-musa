import torch
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_experts


def apply(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True,
        activation=layer.activation,
        global_num_experts=layer.global_num_experts,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        expert_map=layer.expert_map,
        quant_config=self.moe_quant_config,
    )


import vllm.model_executor.layers.quantization.fp8

vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.apply = apply
