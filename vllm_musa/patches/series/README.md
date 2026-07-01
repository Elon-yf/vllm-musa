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

Currently **81 patches** — the MUSA source edits against `vllm@v0.24.0`, applied at
build. Runtime object/registration patches (which patch live objects at import) are
kept separately in `vllm_musa/patches/`, not in this build-time series. Verified:
`git am -3` replays all 81 cleanly onto a fresh `vllm@v0.24.0`.
