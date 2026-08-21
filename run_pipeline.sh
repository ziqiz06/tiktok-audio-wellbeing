#!/usr/bin/env bash
# Runs the full pipeline end-to-end, unattended (e.g. overnight).
#
# Design principles for an unattended run:
#   - Pre-flight gate: tests downloading before committing hours to it. This
#     is what caught, in an earlier sandboxed test, that downloads fail
#     entirely from a blocked IP -- better to find that out in 60 seconds
#     than at 6am with nothing downloaded.
#   - Every step logs to its own timestamped file under logs/, and appends
#     one line to PIPELINE_STATUS.txt -- check that file first when you wake
#     up, it's the whole story in a few lines.
#   - Every Python script in this chain (download_videos.py,
#     audio_feature_extraction.py, embeddingvector.py) is independently
#     checkpointed/resumable, so if this script (or your machine) dies
#     partway through, re-running it picks up where it left off rather than
#     starting over.
#   - permutation_test.py's cost at the real N~736 scale is unmeasured (only
#     tested at the old N=150 pilot) -- this run uses a conservative
#     N_PERMUTATIONS so it finishes and tells you the real per-permutation
#     time, rather than gambling the whole night on an unknown multiplier.
#     Scale up in a follow-up run once you know that number.
#
# Usage: run this with caffeinate so your Mac doesn't sleep partway through
# (a sleeping Mac pauses everything, silently, for however long the lid
# stays shut):
#
#   nohup caffeinate -i bash run_pipeline.sh > logs/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
#
# Then close the lid if you want -- caffeinate -i keeps the machine (not
# just the display) awake for as long as this script's process tree is
# alive. Check progress any time with: cat PIPELINE_STATUS.txt

set -uo pipefail  # NOT set -e: we want to control exactly which failures stop the chain

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
STATUS_FILE="PIPELINE_STATUS.txt"
echo "=== Pipeline run started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$STATUS_FILE"

run_step() {
    local name="$1"; shift
    local logfile="logs/$(date +%Y%m%d_%H%M%S)_${name}.log"
    local start=$(date +%s)
    echo "--- [$(date -u +%H:%M:%S)] Starting: $name ---"
    if "$@" > "$logfile" 2>&1; then
        local dur=$(( $(date +%s) - start ))
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] OK    $name (${dur}s) -- log: $logfile" >> "$STATUS_FILE"
        echo "--- [$(date -u +%H:%M:%S)] Done: $name (${dur}s) ---"
        return 0
    else
        local dur=$(( $(date +%s) - start ))
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] FAILED $name (${dur}s) -- see $logfile" >> "$STATUS_FILE"
        echo "--- [$(date -u +%H:%M:%S)] FAILED: $name -- see $logfile ---"
        return 1
    fi
}

# ── Pre-flight: does downloading even work from this network? ────────────────
echo "--- Pre-flight: testing download on 10 videos before committing to the full batch ---"
preflight_log="logs/$(date +%Y%m%d_%H%M%S)_preflight_download.log"
python3 download_videos.py --limit 10 > "$preflight_log" 2>&1
preflight_ok=$(grep -oE '^Done\. [0-9]+ succeeded' "$preflight_log" | grep -oE '[0-9]+' || echo 0)
echo "Pre-flight: ${preflight_ok}/10 succeeded (log: $preflight_log)"
if [ "$preflight_ok" -lt 3 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ABORTED after pre-flight: only ${preflight_ok}/10 downloads succeeded -- likely IP-blocked or network issue. See $preflight_log" >> "$STATUS_FILE"
    echo "ABORTING: pre-flight download success rate too low (${preflight_ok}/10, need >=3)."
    echo "This usually means the network this is running on is IP-blocked by TikTok -- check $preflight_log for 'Your IP address is blocked' errors."
    exit 1
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] OK    preflight_download (${preflight_ok}/10 succeeded)" >> "$STATUS_FILE"

# ── Full pipeline ──────────────────────────────────────────────────────────
run_step "download_videos"          python3 download_videos.py || exit 1
run_step "audio_feature_extraction" python3 section5.1/audio_feature_extraction.py || exit 1
run_step "embeddingvector"          python3 embeddingvector.py || exit 1
run_step "build_features"           python3 build_features.py || exit 1
run_step "build_final_model_df"     python3 build_final_model_df.py || exit 1
run_step "ridge_models"             python3 run_ridge_models.py || exit 1

# Permutation test: conservative N_PERMUTATIONS for this first run at the new
# scale -- see header comment. Only the primary outcome (happy) for now;
# ridge_models already covers all four outcomes' R²/RMSE.
run_step "permutation_test" python3 permutation_test.py --target happy --comparison both --n-permutations 100

echo "=== Pipeline run finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$STATUS_FILE"
echo ""
echo "=== ALL DONE. Summary: ==="
cat "$STATUS_FILE"
