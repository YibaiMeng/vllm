#!/bin/bash
#SBATCH -A coreai_mlperf_inference
#SBATCH --output=outputs/%j-build-image-stdout.txt
#SBATCH --error=outputs/%j-build-image-stderr.txt
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=coreai_mlperf_inference-vllm-pr43343.build-image
#
# Derive a new container image from vllm/vllm-openai:nightly-<sha> with the
# CUDA dev headers (nvrtc.h, cublasLt.h) installed so FlashInfer's JIT compile
# works at runtime. Result is saved as a .sqsh that you can point sbatch_run /
# sbatch_install at instead of the stock nightly.
#
# Why: upstream vllm publishes only the slim runtime image; PR #33929 to add
# dev headers was closed unmerged. Issue #35501 tracks the same problem.
# See README for full context.
#
# Pattern: `srun --container-image=<base> --container-save=<out.sqsh>
#   --container-writable bash -c '<modifications>'` is the canonical
# enroot/pyxis way to derive an image without `docker build`.
#
# Output: $OUT_SQSH (default: $USER_DIR/images/vllm-openai-with-nvrtc-<sha>.sqsh)
#
# Run once per base image. Then in sbatch_install.sbatch / sbatch_run.sbatch,
# point CONTAINER_IMAGE at $OUT_SQSH.

set -uo pipefail

USER_DIR="${USER_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_inference/users/${USER}}"
BASE_SHA="${BASE_SHA:-1fe3303983e1829fae25edfb0b93e8cbcfad96e6}"
BASE_IMAGE="${BASE_IMAGE:-docker://vllm/vllm-openai:nightly-${BASE_SHA}}"
OUT_SQSH="${OUT_SQSH:-$USER_DIR/images/vllm-openai-with-nvrtc-${BASE_SHA}.sqsh}"

# CUDA major-minor for the package suffix. Defaults to 13-0 since the upstream
# nightly is built on CUDA 13.0.
CUDA_DASH="${CUDA_DASH:-13-0}"

mkdir -p "$(dirname "$OUT_SQSH")"

echo "==== build-image job $SLURM_JOB_ID at $(date -Iseconds) ===="
echo "BASE_IMAGE   = $BASE_IMAGE"
echo "OUT_SQSH     = $OUT_SQSH"
echo "CUDA_DASH    = $CUDA_DASH"

if [[ -f "$OUT_SQSH" ]]; then
    echo "WARNING: $OUT_SQSH already exists. Delete or set OUT_SQSH= to override." >&2
    exit 1
fi

# Modifications run inside the base container. `--container-save` snapshots
# the writable overlay back into OUT_SQSH on exit.
build_cmd='
    set -eux
    nvcc --version || true
    apt-get update
    apt-get install -y --no-install-recommends \
        cuda-nvrtc-dev-'"${CUDA_DASH}"' \
        cuda-cudart-dev-'"${CUDA_DASH}"' \
        libcublas-dev-'"${CUDA_DASH}"'

    # Make sure /usr/local/cuda symlink points at the installed version so
    # nvcc finds the headers without -I flags.
    if [[ ! -e /usr/local/cuda/include/nvrtc.h ]]; then
        # locate the actually-installed path and symlink it
        for d in /usr/local/cuda-* /usr/local/cuda; do
            if [[ -f "$d/include/nvrtc.h" ]]; then
                ln -sfn "$d" /usr/local/cuda
                break
            fi
        done
    fi

    # Verify
    ls -la /usr/local/cuda/include/nvrtc.h
    test -f /usr/local/cuda/include/cublasLt.h && echo "cublasLt.h OK"
    rm -rf /var/lib/apt/lists/*
'

srun \
    --container-image="$BASE_IMAGE" \
    --container-save="$OUT_SQSH" \
    --container-writable \
    --no-container-mount-home \
    bash -c "$build_cmd"

echo "==== build-image done at $(date -Iseconds) ===="
echo "Use this image by setting:"
echo "  CONTAINER_IMAGE=$OUT_SQSH"
echo "in sbatch_install.sbatch and sbatch_run.sbatch (and remove the apt-get"
echo "step from sbatch_run.sbatch since the headers are now baked in)."
