# pr43343 — review harness for vllm-project/vllm#43343

Scratch dir inside this fork. Not for upstreaming. Holds the scripts used to
test PR #43343 (Qwen3.5 quantized LM-head + MTP CT ignore-list + VLM prefix)
on B200/GB300 per Shang's ask in `swdl-vllm-dev`.

## Layout

- `sbatch_run.sbatch` — single sbatch: editable install + bench sweep in one
  srun. Install goes into the container's system site-packages (no venv) so
  the pre-installed `flashinfer-jit-cache` / `flashinfer-cubin` /
  `nvidia-cutlass-dsl` stay coherent. Source-only edits in `$VLLM_ROOT`
  propagate via the `.pth` written by editable install — re-run sbatch to
  pick them up; ~30 s of pip work added per job. Canonical command per
  Juhi Mittal (swdl-vllm-dev p1777302459439329): `VLLM_USE_PRECOMPILED=1 pip
  install --no-build-isolation -e .`
- `bench_serve_pr43343.sh` — inner script the sbatch invokes. Spins up
  `vllm serve` per config, runs `vllm bench serve` against ShareGPT at a rate
  sweep, and greps the server log for the failure modes #43343 fixes.
- `lm_head_logger.patch` — one-line `logger.info` in
  `vllm/model_executor/layers/vocab_parallel_embedding.py` that prints the
  `lm_head` quant-method dispatch result every time a `ParallelLMHead` is
  constructed. The bench probe greps for this line to PASS/FAIL hunk 1.
  Apply on **whichever branch you build** (it's not branch-specific):
  `cd .. && git apply pr43343/lm_head_logger.patch`. Already applied on
  this branch.
- `sbatch_build_image.sbatch` — optional, mostly historical. Bakes
  `cuda-nvrtc-dev-13-0` into a derived `.sqsh` from `vllm-openai:nightly`.
  Not needed with the current install pattern (FlashInfer AOT cache covers
  the kernels — no JIT, no `nvrtc.h` needed). Keep only if you want a stable
  custom image for the team.

## Usage

```bash
sbatch sbatch_run.sbatch
```

`outputs/%j-stdout.txt` collects slurm stdout; per-run server logs land in
`$USER_DIR/outputs/$SLURM_JOB_ID/`.

## A/B for the actual review

```bash
# A: stock upstream/main, no #43343
(cd .. && git checkout upstream/main)
A_JOB=$(sbatch --parsable sbatch_run.sbatch)

# B: upstream/main + #43343
(cd .. && git fetch upstream pull/43343/head:pr-43343 && git checkout pr-43343)
B_JOB=$(sbatch --parsable sbatch_run.sbatch)

# After both finish:
diff <(grep -E "PASS|FAIL|SKIP|hunk" $USER_DIR/outputs/$A_JOB/*.server.log) \
     <(grep -E "PASS|FAIL|SKIP|hunk" $USER_DIR/outputs/$B_JOB/*.server.log)
```

No separate install step — `sbatch_run.sbatch` does it every time.

## Known holes (scope mismatch w/ #43343)

- `nvidia/Qwen3.5-397B-A17B-NVFP4` is **W4A4**, not W4A16. `#43343`'s parent
  PR title says W4A16. Confirmed with askliar that this is the only public
  Qwen3.5 NVFP4 ckpt.
- That ckpt has `lm_head` AND `mtp.layers.0*` in `ignore`. Hunks 1 (lm_head
  `quant_config`) and 3 (CT MTP ignore-list) are no-ops on it. Only hunk 2
  (VLM prefix swap) is exercised.
- No public compressed-tensors Qwen3.5 MoE ckpt exists. Hunk 3 fundamentally
  can't be tested from public HF.
- Pending askliar's reply: drop `lm_head` from `ignore` for hunk 1 testing,
  and/or pointer to an internal W4A16 / CT export.
