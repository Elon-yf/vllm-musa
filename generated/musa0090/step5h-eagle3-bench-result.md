# MUSA-0090 step 5h Eagle3 bench evidence (2026-05-16)

## Server-side
- vllm-musa SHA dfbdb20 (step 5h: FlashAttentionMetadataBuilder.build(fast_build=True))
- Container yeahdongcn60, sglang-0.5.6 venv
- Revision-4 config minus compile_sizes=[1] (incompatible with spec-decode cudagraph padding per ValueError)
- Eagle3 draft: /home/dist/diffusion/yeahdongcn/thoughtworks/MiniMax-M2.5-Eagle3, N=7

## Startup outcome
**PASS** — "Application startup complete." at 03:50:53. No metadata-API AttributeError
crash. The step 5h fix structurally resolved bugs 1-8 + bug 9 (block_table) by
delegating per-step metadata construction to FlashAttentionMetadataBuilder.build()
instead of patching CommonAttentionMetadata with setattr.

## BS=1 4k/1k warm bench (5 runs)

| Run | output_throughput | TTFT | TPOT |
|---|---|---|---|
| 1 (cold) | 13.966 tok/s | high (cudagraph capture) | — |
| 2 | 15.720 tok/s | — | — |
| 3 | 18.956 tok/s | — | — |
| 4 | 18.960 tok/s | — | — |
| 5 | 37.549 tok/s | — | — |
| warm median (2-5) | **18.958** tok/s | — | — |
| warm mean (2-5) | 22.796 tok/s | — | — |
| warm peak | 37.549 tok/s | — | — |

vs **no-spec baseline 53.657 tok/s** (cookbook §4.2.1):
- **−64.67% on warm median**
- **−30.0% on warm peak**

## Spec acceptance (positions 3-6)
All 0.00 — the draft tokens at positions 3+ are **never accepted** by the verify model.
This is the diagnostic signal that the per-step metadata views inside the captured
graph are NOT being mutated correctly between draft steps. The graph captures with
step-0 state and replays with that frozen state instead of updating per step.

## Interpretation
The step 5h fix unblocks the AttributeError crash that prevented Eagle3 from even
starting, but does NOT achieve SGLang's reported +30% MUSA gain. SGLang's gain
requires the **full per-step backend-array architecture**:

  Per SGLang `MusaFlashAttentionMultiStepBackend.__init__` (sglang/srt/hardware_backend/
  musa/attention/flashattention_backend.py:916):
      self.attn_backends = []
      for i in range(speculative_num_steps - 1):
          self.attn_backends.append(MusaFlashAttentionBackend(
              speculative_step_id=i, topk=topk, speculative_num_steps=N))

  Each step's backend instance has `speculative_step_id=i` baked in.
  `init_forward_metadata_capture_cuda_graph` and `init_forward_metadata_replay_cuda_graph`
  on the FlashAttentionMultiStepBackend iterate all step backends.
  In eagle_worker.py:904: `forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]`
  — switches the BACKEND reference per step, not just the metadata.

The vllm-musa step 5h calls builder.build(fast_build=True) once per step at capture
time, but the resulting FlashAttentionMetadata objects share many internal tensor
references that the captured graph cannot independently mutate per step. The
SGLang pattern works because each step's MusaFlashAttentionBackend instance owns
its own metadata state.

## Next steps (NOT done in this commit)
1. **Full SGLang per-step-backend port** (~3-5 days dedicated): refactor
   EagleFullLoopRunner to maintain N-1 independent FlashAttentionBackend instances,
   one per draft step. Each step's instance gets `speculative_step_id=i` baked in
   and owns its per-step metadata. The captured loop indexes by
   `attn_backends[step_idx]` (vs `metadata_array[step_idx]`).
2. Validate spec acceptance > 0 at positions 3-6 before any perf claim.
3. Re-bench against 53.657 no-spec baseline; SGLang reference is +30%.

## Files
- vllm-musa `xd/minimax_m2_5_sota` @ dfbdb20 (step 5h)
- Container: `/tmp/vllm_omni_musa_outputs/m25_bs1_warm_eagle3/run{1..5}.json`
- Container: `/tmp/vllm_omni_musa_logs/m25_eagle3.log`
