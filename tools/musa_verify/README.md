# `tools/musa_verify/` — parallel device-pinned verification harness

A small update→verify harness for vllm-musa changes. It runs the **unit tests**
plus **one functional server smoke per model**, all **concurrently**, each smoke
pinned to its own MUSA device via `MUSA_VISIBLE_DEVICES` (e.g. DeepSeek-V2-Lite on
device 0, Qwen3-8B on device 1).

## Configuration

All host/container/path values are env vars with placeholder defaults — set them
for your MUSA environment, and **never commit real values**:

| Env var | Meaning | Placeholder default |
|---|---|---|
| `REMOTE_PASS` | SSH password — read from the env, never hardcoded | (required) |
| `MUSA_HOST` | `user@host` of the MUSA box | `user@musa-host` |
| `MUSA_CONTAINER` | docker container name | `musa-container` |
| `FORK_REMOTE` | git remote to push the branch under test to | `origin` |
| `CONTAINER_WS` | repo checkout path inside the container | `/workspace` |
| `MUSA_VENV` | python venv to activate in the container | `/path/to/musa-venv` |
| `UNIT_TEST_DEVICE` | spare device id for the unit-test import | `7` |

Pin one consistent test environment for reproducibility — don't switch envs
between runs (it invalidates cross-run comparison).

## Files

| File | Runs where | Purpose |
|---|---|---|
| `models.conf` | — | the verify matrix: `family \| model_path \| device \| port \| tp \| expect` |
| `smoke_one_model.sh` | in container | one `vllm serve` smoke pinned to one device; health-wait → semantic request → PASS/FAIL/SKIP |
| `unit_tests.sh` | in container | `pytest tests/test_patches.py` (the patch-mechanism guard), pinned to a spare device |
| `verify.sh` | local | push → sync container → run unit + all smokes in parallel → summary (pure bash) |
| `verify_workflow.js` | local (Workflow tool) | same parallel verify phase, agent-orchestrated |

## Shared-box safety

Built to be safe alongside other jobs on the same box:

- `smoke_one_model.sh` checks `mthreads-gmi` for its target device and **SKIPs**
  (does not fail, does not kill anything) if free memory is below `MIN_FREE_MIB`
  (default 40 GiB). It never steals a GPU from another job.
- On exit it kills **only its own process group** (`setsid` + `kill -- -PGID`),
  never a blanket `pkill -f vllm`.

## Secrets

Never hardcode the SSH password. `verify.sh` reads `$REMOTE_PASS`; the Workflow
takes it via `args.pass` at invoke time.

```bash
export REMOTE_PASS=...        # from your shell / secrets manager
```

## The update → verify loop

1. Make a change on the branch, commit.
2. Sync + (re)build the container to the new SHA:
   - Python-only diff → `git fetch && git reset --hard <sha>` (editable install picks it up).
   - csrc / `setup.py` / `.cu` diff, **or first switch onto this branch** → also
     `pip install -e . --no-build-isolation -v` (set `REINSTALL=1` for `verify.sh`).
3. Verify (unit + smokes, parallel, device-pinned):

   ```bash
   REMOTE_PASS=… MUSA_HOST=… MUSA_CONTAINER=… REINSTALL=1 tools/musa_verify/verify.sh  # fresh container
   REMOTE_PASS=… MUSA_HOST=… MUSA_CONTAINER=…             tools/musa_verify/verify.sh  # Python-only iterations
   ```

   Or via the Workflow tool (after the container is synced/built) — see the header
   of `verify_workflow.js` for the `args` shape.
4. Read the summary. A run is GREEN only if no line is `FAIL` (`SKIP` is allowed —
   it means a device was busy). Re-run skipped models once their device frees up.

## Add a model

Append a row to `models.conf` on a free device/port. Keep one device per
concurrent smoke.
