# MUSA-0090 step 2: Pre-allocated per-step attention metadata array.
#
# The TRICK that lets vLLM's iterative Eagle3 draft loop be captured as one
# cudagraph: pre-build N copies of CommonAttentionMetadata at capture time
# (one per spec step), wired so each step's tensors are VIEWS into the
# pre-allocated `EagleDraftBuffers`. The captured graph mutates the buffers
# via in-place kernels, and the metadata views see those mutations without
# any Python rebuild.
#
# Reference SGLang pattern: `draft_attn_backend.attn_backends[i]` —
# sglang/python/sglang/srt/speculative/eagle_worker.py:904
#
# Design ref: generated/musa0090_impl/step1-design-doc.md §3.2.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from .spec_info import EagleDraftBuffers


def build_per_step_attn_metadata_array(
    base_metadata: Any,  # CommonAttentionMetadata; typed Any to avoid hard dep at module-load
    buffers: EagleDraftBuffers,
    batch_size: int,
    proposer: Any = None,  # vllm.v1.spec_decode.eagle.EagleProposer (optional)
) -> list[Any]:
    """Pre-allocate N attn-metadata objects whose tensors point into `buffers`.

    The arithmetic that vLLM does in Python at each loop iteration in
    `EagleProposer.propose()` (max_seq_len += 1, slot_mapping recompute,
    seq_lens += 1) is unrolled here:

      - max_seq_len: bumped statically by step index at metadata-build time
      - slot_mapping: VIEW into buffers.slot_mapping_per_step[step_idx, :batch_size]
                      (mutated by the eagle_step_update_slot_mapping_and_metadata
                      kernel inside the captured graph)
      - seq_lens:     VIEW into buffers.seq_lens_per_step[step_idx, :batch_size]
                      (mutated by the same kernel)

    Returns a list of `buffers.num_steps` CommonAttentionMetadata-shaped objects.
    The runner indexes this list by step number inside its capture loop:
      forward_batch.attn_metadata = metadata_array[step_idx]

    Args:
        base_metadata: The first-step attention metadata that vLLM normally
            passes to EagleProposer.propose(). Used as the template; per-step
            metadata objects copy its non-mutable fields (block_table_tensor,
            cu_seq_lens base, query_start_loc, etc.) and override the
            mutable ones (slot_mapping, seq_lens, max_seq_len).
        buffers: The pre-allocated draft buffers. Per-step views are taken
            into `slot_mapping_per_step` and `seq_lens_per_step`.
        batch_size: The actual batch size for this capture context (≤ buffers.max_bs).

    Returns:
        A list of length buffers.num_steps. Each element is a metadata object
        whose tensor fields point into `buffers`. Order matches the spec
        step order: metadata_array[0] is for the first draft fwd, [N-1] is
        for the last.
    """
    if buffers.num_steps <= 0:
        return []

    # MUSA-0090 step 5e: when the proposer is provided, use its
    # build_per_group_and_layer_attn_metadata() to produce per-layer
    # backend-specific metadata (e.g., FlashAttentionMetadata with the
    # use_cascade, common_prefix_len, etc. fields the compiled draft
    # model's forward expects). The compiled forward reads
    # get_forward_context().attn_metadata and accesses .use_cascade
    # unconditionally; passing the raw CommonAttentionMetadata fails
    # with AttributeError.
    # MUSA-0090 step 5g: disable the per-layer build path. Although it
    # provides use_cascade etc., the resulting metadata's seq_lens has
    # a shape that flash_attn rejects ('seqused_k must have shape
    # (batch_size,)'). Instead, use the simpler dataclasses.replace path
    # below and setattr() the flag-fields the compiled forward expects.
    use_per_layer = False  # was: proposer is not None and hasattr(...)
    if use_per_layer:
        metadata_array: list[Any] = []
        for step_idx in range(buffers.num_steps):
            # Mutate base_metadata in place to reflect the step's state
            # (slot_mapping, seq_lens, max_seq_len), then build per-layer
            # metadata. The per-layer call returns a dict-shaped object
            # that the model reads via get_forward_context().
            base_metadata.slot_mapping = buffers.slot_mapping_per_step[step_idx, :batch_size]
            base_metadata.seq_lens = buffers.seq_lens_per_step[step_idx, :batch_size]
            base_metadata.max_seq_len = int(getattr(base_metadata, "max_seq_len", 0)) + (1 if step_idx > 0 else 0)
            try:
                _, per_layer = proposer.build_per_group_and_layer_attn_metadata(
                    base_metadata, draft_index=step_idx
                )
                metadata_array.append(per_layer)
            except Exception as exc:
                raise RuntimeError(
                    f"step {step_idx}: proposer.build_per_group_and_layer_attn_metadata "
                    f"failed: {type(exc).__name__}: {exc}"
                ) from exc
        return metadata_array

    # Fallback: dataclasses.replace on CommonAttentionMetadata. Works for
    # tests + lightweight scenarios but the compiled draft model may
    # need per-layer metadata (see comment block above).
    metadata_array: list[Any] = []
    base_max_seq_len = int(getattr(base_metadata, "max_seq_len", 0))

    for step_idx in range(buffers.num_steps):
        # Per-step views into the pre-allocated buffers. .narrow() returns a
        # contiguous view; the captured graph writes to these slices via the
        # eagle_step_update_slot_mapping_and_metadata kernel and the view
        # reflects the writes without rebuild.
        slot_mapping_view = buffers.slot_mapping_per_step[step_idx, :batch_size]
        seq_lens_view = buffers.seq_lens_per_step[step_idx, :batch_size]
        positions_view = buffers.positions_per_step[step_idx, :batch_size]

        # max_seq_len is a static int (not a tensor); bump by step_idx so the
        # attention backend sizes its scratch correctly for each step. The
        # +1 between steps is the spec-decode invariant: each draft token
        # adds one position to the sequence.
        step_max_seq_len = min(
            base_max_seq_len + step_idx,
            int(getattr(base_metadata, "max_model_len", base_max_seq_len + buffers.num_steps)),
        )

        # Build the per-step metadata. We use dataclasses.replace if base is a
        # dataclass; else fall back to a shallow copy + setattr. CommonAttentionMetadata
        # is a frozen dataclass in upstream vLLM, so `replace()` is the supported path.
        try:
            step_metadata = replace(
                base_metadata,
                slot_mapping=slot_mapping_view,
                seq_lens=seq_lens_view,
                max_seq_len=step_max_seq_len,
            )
            # MUSA-0090 step 5g: the compiled draft model's forward accesses
            # attn_metadata.use_cascade unconditionally (and likely other
            # backend-specific flags). Set defaults via setattr so the
            # compiled forward doesn't AttributeError on missing fields.
            # These are read-only flag fields; we don't need real implementations.
            for attr_name, default in [
                ("use_cascade", False),
                ("common_prefix_len", 0),
                ("query_start_loc", None),
                ("seqused_k", seq_lens_view),  # flash_attn expects this name
                ("max_query_len", 1),  # spec_decode draft is always 1-token-per-step
                ("max_seq_len_k", step_max_seq_len),
            ]:
                if not hasattr(step_metadata, attr_name) or getattr(step_metadata, attr_name, None) is None:
                    try:
                        object.__setattr__(step_metadata, attr_name, default)
                    except (AttributeError, TypeError):
                        pass  # frozen dataclass; can't setattr — accept and continue
        except TypeError:
            # If replace() fails (e.g., the type isn't a dataclass), construct
            # via direct copy. This is the fallback for vllm-musa builds where
            # CommonAttentionMetadata has been monkey-patched to a different
            # shape. Identified at runtime; raise an informative error.
            raise NotImplementedError(
                "build_per_step_attn_metadata_array requires CommonAttentionMetadata "
                "to be a dataclass with `slot_mapping`, `seq_lens`, `max_seq_len` "
                f"fields. Got type {type(base_metadata).__name__}; check that "
                "the vllm version is compatible (this code was written against "
                "vllm v0.20.1.dev0+g88d34c640)."
            )

        metadata_array.append(step_metadata)

        # Sanity check (cheap): each step's views should point at distinct
        # memory regions. data_ptr() differs because of the step offset.
        if step_idx > 0:
            prev = metadata_array[step_idx - 1]
            if getattr(prev, "slot_mapping").data_ptr() == slot_mapping_view.data_ptr():
                raise RuntimeError(
                    f"step {step_idx} slot_mapping view aliases step {step_idx-1}; "
                    "buffers.slot_mapping_per_step is not laid out as expected"
                )

    return metadata_array


@dataclass
class StepMetadataIndexing:
    """Records how each step's metadata reads from the buffers.

    Used by the capture path to verify that the indexing math is correct
    before committing the graph. Construct from a built metadata_array via
    the inspect() helper.
    """

    num_steps: int
    base_max_seq_len: int
    per_step_max_seq_lens: list[int]
    slot_mapping_data_ptrs: list[int]
    seq_lens_data_ptrs: list[int]

    @classmethod
    def inspect(cls, metadata_array: list[Any]) -> StepMetadataIndexing:
        """Read back the structure of a built metadata array for verification.

        Used in tests / debug logging — not on the hot path.
        """
        if not metadata_array:
            return cls(
                num_steps=0,
                base_max_seq_len=0,
                per_step_max_seq_lens=[],
                slot_mapping_data_ptrs=[],
                seq_lens_data_ptrs=[],
            )
        return cls(
            num_steps=len(metadata_array),
            base_max_seq_len=int(getattr(metadata_array[0], "max_seq_len", 0)),
            per_step_max_seq_lens=[int(getattr(m, "max_seq_len", 0)) for m in metadata_array],
            slot_mapping_data_ptrs=[int(m.slot_mapping.data_ptr()) for m in metadata_array],
            seq_lens_data_ptrs=[int(m.seq_lens.data_ptr()) for m in metadata_array],
        )

    def verify_strict(self) -> None:
        """Raise if the metadata array isn't shaped correctly for the captured
        loop. Sanity-check before passing to graph capture."""
        if self.num_steps <= 0:
            raise RuntimeError("metadata array is empty")
        if len(set(self.slot_mapping_data_ptrs)) != self.num_steps:
            raise RuntimeError(
                f"slot_mapping views are not all distinct: "
                f"{self.slot_mapping_data_ptrs}"
            )
        if len(set(self.seq_lens_data_ptrs)) != self.num_steps:
            raise RuntimeError(
                f"seq_lens views are not all distinct: "
                f"{self.seq_lens_data_ptrs}"
            )
        # max_seq_lens should be monotonically non-decreasing across steps
        for i in range(1, self.num_steps):
            if self.per_step_max_seq_lens[i] < self.per_step_max_seq_lens[i - 1]:
                raise RuntimeError(
                    f"max_seq_len decreased at step {i}: "
                    f"{self.per_step_max_seq_lens}"
                )
