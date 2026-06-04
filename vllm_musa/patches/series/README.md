# `vllm_musa/patches/series/` — build-time patch series

**THE** vLLM-MUSA source-patch mechanism (no runtime fallback). A `git format-patch`
series of MUSA's source modifications against the pinned `vllm@<tag>`
(`setup.py` `_VLLM_REPO.git_tag`), applied at build to the cloned `third_party/vllm`
*before* install so the installed vLLM is pre-patched.

- **Applied at build** by `setup.py::_apply_musa_patch_series` → `build_apply.py`
  (`git apply`, idempotent `--reverse --check`).
- **Generated/regenerated** by `make -f Makefile.sync format-patches`
  (`git format-patch --no-signature --zero-commit`, keeping `index` blob lines so
  `git am -3` 3-way works across version bumps).

Bootstrapped 2026-06-03 from the legacy `*.patch.py` source-transforms: **36 patches**
(the live v0.22.0 source edits). Excluded by design: the 2 object/registration patches
(`eagle` prime, `parallel_state` — they patch live objects, stay runtime),
the ~11 dead DeepSeek-V4 old-namespace patches (targets absent on v0.22), and 2
anchor-absent patches (`all2all`, `triton_unified_attention` — to review: obsolete on
v0.22 or stale anchor). Verified: `git am -3` replays all 36 cleanly onto a fresh
`vllm@v0.22.0`.
