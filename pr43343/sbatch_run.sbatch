#!/bin/bash
#SBATCH -A coreai_mlperf_inference
#SBATCH --output=outputs/%j-stdout.txt
#SBATCH --error=outputs/%j-stderr.txt
#SBATCH --partition=batch
#SBATCH --gres=gpu:8
#SBATCH --qos=interactive
#SBATCH --time=4:00:00
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=coreai_mlperf_inference-vllm-pr43343.review
#
# Sbatch outer for PR vllm-project/vllm#43343 review on B200/GB300.
# Per-run output dir is outputs/${SLURM_JOB_ID} so sacct/Panoptes can correlate.
#
# Install + bench in a single srun. Editable install lands in the container's
# system site-packages (no venv) so FlashInfer's pre-installed AOT cache + nvidia-
# cutlass-dsl stay coherent. Source-only edits in $VLLM_ROOT propagate via the
# .pth file written by `pip install -e .`.
#
# Canonical command per Juhi Mittal in swdl-vllm-dev (p1777302459439329):
#   VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -e .

set -uo pipefail

USER_DIR="${USER_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_inference/users/yibaim}"
VLLM_ROOT="${VLLM_ROOT:-$USER_DIR/github/yibaimeng/vllm}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$USER_DIR/images/vllm_vllm-openai:nightly-1fe3303983e1829fae25edfb0b93e8cbcfad96e6.sqsh}"
OUTPUT_HOST_DIR="${OUTPUT_HOST_DIR:-$USER_DIR/outputs/${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_HOST_DIR"

mounts="\
$USER_DIR/vllm-cache:/root/.cache,\
$USER_DIR/hf_cache:/workspace/hf_cache,\
$USER_DIR/hf_data:/workspace/hf_data,\
$VLLM_ROOT:/workspace/vllm,\
$HOME/.config/enroot:/root/.config/enroot,\
$OUTPUT_HOST_DIR:/output"

export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1}
export VLLM_USE_FLASHINFER_MOE_FP4=${VLLM_USE_FLASHINFER_MOE_FP4:-1}
export VLLM_FLASHINFER_MOE_BACKEND=${VLLM_FLASHINFER_MOE_BACKEND:-latency}

# Forward common per-run overrides into the container.
PASS_ENV=()
for v in SHAREGPT_PATH RATES NUM_PROMPTS NVFP4_MODEL CT_MODEL HF_HOME \
         VLLM_USE_FLASHINFER_SAMPLER VLLM_USE_FLASHINFER_MOE_FP4 VLLM_FLASHINFER_MOE_BACKEND; do
    [[ -n "${!v:-}" ]] && PASS_ENV+=("$v=${!v}")
done

echo "==== job $SLURM_JOB_ID starting at $(date -Iseconds) ===="
echo "VLLM_ROOT=$VLLM_ROOT"
echo "OUTPUT_HOST_DIR=$OUTPUT_HOST_DIR"
echo "CONTAINER_IMAGE=$CONTAINER_IMAGE"

exec srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$mounts" \
    --no-container-mount-home \
    --container-writable \
    --mpi=pmix \
    env "${PASS_ENV[@]}" \
    bash -c '
        set -eux
        # vllm/vllm-openai slim image has no git; setuptools-scm needs it for
        # the version. Also silences the precompiled-wheel base-commit warning.
        apt-get update -qq && apt-get install -y -qq --no-install-recommends git
        git config --global --add safe.directory "*"

        cd /workspace/vllm

        # Install build deps that pyproject.toml [build-system].requires would
        # have given us if we weren'\''t using --no-build-isolation. The
        # vllm-openai container ships most of these but newer vllm checkouts
        # add deps the image doesn'\''t have yet (e.g. setuptools-rust, added
        # by vllm#43377 after the nightly cut).
        pip install --no-deps -r requirements/build/cuda.txt

        # Editable install on top of the container'\''s system vllm. Targets
        # /usr/local/lib/python3.12/dist-packages by default (running as root,
        # no venv). VLLM_USE_PRECOMPILED=1 -> reuse the .so from the matching
        # precompiled wheel, no C++ rebuild. --no-build-isolation -> use the
        # container'\''s torch/cmake/ninja, not a fresh isolated build env
        # (which would pull a mismatched torch wheel).
        VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -e .

        # Sanity check: confirm vllm imports + FlashInfer AOT cache is reachable.
        # Container has python3 via update-alternatives but no `python` symlink.
        python3 -c "import vllm; print(\"vllm \" + vllm.__version__)"
        python3 -c "import flashinfer, os; from flashinfer.jit.env import FLASHINFER_AOT_DIR; print(\"AOT dir:\", FLASHINFER_AOT_DIR, \"exists:\", os.path.isdir(FLASHINFER_AOT_DIR))"

        bash /workspace/vllm/pr43343/bench_serve_pr43343.sh /output
    '
