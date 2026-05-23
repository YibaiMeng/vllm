#!/usr/bin/env bash
# PR vllm-project/vllm#43343 review harness on B200/GB300 x8.
# Reuses ShareGPT sweep at fixed request rates. Adds MTP-enabled configs
# (essential — 2 of 3 hunks in #43343 are MTP-only) and explicit pass/fail
# probes against the failure modes the PR is meant to fix.
#
# This is the INNER script — runs inside the vllm-nightly container.
# Invoke via srun_overlap.sh (interactive) or sbatch_run.sbatch (batch). See README.md.
#
# A/B pattern for the review:
#   1. checkout upstream/main, install vllm, run -> logs/A_main
#   2. checkout pr-43343,      install vllm, run -> logs/B_pr43343
#   3. diff <(grep -E "PASS|FAIL|SKIP|hunk" logs/A_main/*.server.log) \
#           <(grep -E "PASS|FAIL|SKIP|hunk" logs/B_pr43343/*.server.log)
#
# What each config exercises:
#   ep         NVFP4 baseline.  Hits hunk 1 (Qwen3_5ForCausalLMBase lm_head
#              quant_config) + hunk 2 (VLM prefix swap, only on a VLM ckpt).
#   ep_mtp1    NVFP4 + MTP-1.   Adds hunk 1's MTP half (Qwen3_5MTP lm_head
#              quant_config).
#   ep_mtp3    NVFP4 + MTP-3.   Same as ep_mtp1, higher k.
#   ct_*       Only if CT_MODEL is set. Compressed-tensors nvfp4-pack-quantized
#              path. Hunk 3 (Qwen3_5MTP CT ignore-list) ONLY trips on a CT
#              ckpt + MTP. nvidia/Qwen3.5-397B-A17B-NVFP4 is ModelOpt format;
#              ask askliar for the CT W4A16 export path.

set -uo pipefail

LOG_DIR="${1:?usage: $0 <log_dir> [extra vllm serve flags ...]}"
shift

EXTRA_SERVE_ARGS=()
for a in "$@"; do
    [[ "$a" == -* ]] && EXTRA_SERVE_ARGS+=("$a") || EXTRA_SERVE_ARGS+=("--$a")
done

SHAREGPT_PATH="${SHAREGPT_PATH:-/workspace/hf_data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json}"
RATES_STR="${RATES:-16 32}"
read -r -a RATES <<< "$RATES_STR"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-1500}"
PORT="${PORT:-8000}"; HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"

NVFP4_MODEL="${NVFP4_MODEL:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
CT_MODEL="${CT_MODEL:-}"

mkdir -p "$LOG_DIR" || { echo "FATAL: cannot mkdir $LOG_DIR" >&2; exit 1; }
probe="$LOG_DIR/.writetest.$$"
( : > "$probe" ) 2>/dev/null || { echo "FATAL: $LOG_DIR not writable" >&2; exit 1; }
rm -f "$probe"
[[ -f "$SHAREGPT_PATH" ]] || { echo "FATAL: ShareGPT not at $SHAREGPT_PATH" >&2; exit 1; }

# Record vllm build identity so A/B comparisons are unambiguous.
{
    echo "ts=$(date -Iseconds)"
    python3 -c "import vllm; print('vllm_version=' + vllm.__version__)" 2>/dev/null || true
    if [[ -n "${VLLM_REPO:-}" && -d "$VLLM_REPO/.git" ]]; then
        (cd "$VLLM_REPO" \
            && echo "vllm_sha=$(git rev-parse HEAD)" \
            && echo "vllm_branch=$(git rev-parse --abbrev-ref HEAD)" \
            && echo "vllm_dirty=$(git status --porcelain | wc -l)")
    fi
} > "$LOG_DIR/BUILD_INFO.$(date +%Y%m%d-%H%M%S).txt"

pip show flashinfer-python flashinfer-jit-cache flashinfer-cubin 2>&1 | grep -E "^(Name|Version|Location)"
python3 -c "
import flashinfer, os
print('flashinfer:', flashinfer.__version__, flashinfer.__file__)
from flashinfer.jit.env import FLASHINFER_AOT_DIR
print('AOT_DIR:', FLASHINFER_AOT_DIR, 'exists:', os.path.isdir(FLASHINFER_AOT_DIR))
import glob
print('AOT .so files:', glob.glob(str(FLASHINFER_AOT_DIR) + '/*/*.so')[:5])
print('trtllm sm100 present:', any('fused_moe_trtllm_sm100' in p for p in glob.glob(str(FLASHINFER_AOT_DIR) + '/*/*.so')))
"
# See https://recipes.vllm.ai/Qwen/Qwen3.5-397B-A17B?variant=nvfp4&features=text_only%2Cspec_decoding&hardware=b200&strategy=single_node_tep&advanced=ep_weight_filter 
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_DEEP_GEMM=0
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_USE_FLASHINFER_MOE_FP4=1
SERVE_COMMON=(
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --enable-expert-parallel \
    --tensor-parallel-size 8 \
    --language-model-only \
    --enable-ep-weight-filter \
    --host "$HOST" --port "$PORT"
)
BENCH_COMMON=(
    --backend vllm --base-url "$BASE_URL"
    --dataset-name sharegpt --dataset-path "$SHAREGPT_PATH"
    --num-prompts "$NUM_PROMPTS"
)

TS=$(date +%Y%m%d-%H%M%S)
SERVER_PID=""

cleanup_server() {
    [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null || { SERVER_PID=""; return; }
    kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

wait_for_server() {
    local timeout="$1" start=$SECONDS
    while (( SECONDS - start < timeout )); do
        curl -fsS "${BASE_URL}/health" >/dev/null 2>&1 && return 0
        [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null && return 2
        sleep 5
    done
    return 1
}

analyze_server_log() {
    local server_log="$1"
    echo "  --- PR43343 signals in $server_log ---"
    # hunk 3: CT MTP ignore-list. Independent of the logger patch.
    if grep -qE "KeyError.*experts\.(w13_weight|w2_weight)" "$server_log"; then
        echo "    FAIL hunk3 : KeyError on MTP fused expert weight (CT ignore-list incomplete)"
    fi
    # hunk 1: lm_head dispatch. Requires lm_head_logger.patch applied.
    if grep -qE "lm_head\[.*\] dispatch -> ModelOpt.*LinearMethod" "$server_log"; then
        echo "    PASS hunk1 : $(grep -m1 -oE 'lm_head\[.*\] dispatch -> .*' "$server_log")"
    elif grep -qE "lm_head\[.*\] dispatch -> Unquantized" "$server_log"; then
        echo "    FAIL hunk1 : $(grep -m1 -oE 'lm_head\[.*\] dispatch -> .*' "$server_log")"
    else
        echo "    SKIP hunk1 : no lm_head dispatch line — apply pr43343/lm_head_logger.patch and rebuild"
    fi
    grep -qE "language_model\.model\.|model\.language_model\." "$server_log" \
        && echo "    INFO hunk2 : VLM wrapper-prefix path was visited"
}

run_config() {
    local model="$1"; shift
    local tag="$1"; shift
    local serve_extra=("$@")
    local safe_model="${model//\//_}"
    local config_name="${TS}_pr43343_${safe_model}${tag:+_$tag}"
    local server_log="$LOG_DIR/${config_name}.server.log"

    local serve_cmd=(env PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME:-/workspace/hf_cache}" \
        vllm serve "$model" "${SERVE_COMMON[@]}" "${serve_extra[@]}" "${EXTRA_SERVE_ARGS[@]}")

    echo
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] starting server: $config_name"
    printf '    cmd : '; printf '%q ' "${serve_cmd[@]}"; echo
    { printf '# '; printf '%q ' "${serve_cmd[@]}"; echo; } > "$server_log"

    "${serve_cmd[@]}" >> "$server_log" 2>&1 &
    SERVER_PID=$!
    echo "    pid : $SERVER_PID — waiting up to ${SERVE_READY_TIMEOUT}s for /health"

    local rc; wait_for_server "$SERVE_READY_TIMEOUT"; rc=$?
    if (( rc != 0 )); then
        echo "  ERROR: server not ready (rc=$rc) — see $server_log"
        analyze_server_log "$server_log"
        cleanup_server
        return 1
    fi
    echo "    ready at $(date +%H:%M:%S)"
    analyze_server_log "$server_log"

    local run_name="${config_name}" base="$LOG_DIR/${config_name}"
    local bench_cmd=(env PYTHONUNBUFFERED=1 \
        vllm bench serve --model "$model" "${BENCH_COMMON[@]}" \
        --request-rate 32 --save-result \
        --result-dir "$LOG_DIR" --result-filename "${run_name}.json")
    echo "--- [$(date +%H:%M:%S)] bench: $run_name (rate=32)"
    "${bench_cmd[@]}" > >(tee -a "${base}.out") 2> >(tee "${base}.err" >&2)
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  ERROR: server died mid-sweep"; break; }

    cleanup_server
    sleep 10
}

run_config "$NVFP4_MODEL" no_spec      
run_config "$NVFP4_MODEL" mtp1 --enable-expert-parallel --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
run_config "$NVFP4_MODEL" mtp3 --enable-expert-parallel --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

if [[ -n "$CT_MODEL" ]]; then
    run_config "$CT_MODEL" ct_mtp3 --quantization=compressed-tensors --enable-expert-parallel \
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
fi
