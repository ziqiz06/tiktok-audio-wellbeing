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
#     extract_audio_features.py, extract_ast_embeddings.py) is independently
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

# Use the project venv's interpreter explicitly, not whatever "python3" the
# invoking shell happens to resolve to. This matters a lot for a
# nohup/background launch: the shell that runs `nohup ... &` might not have
# venvast activated even if you normally work with it active, and a bare
# `python3` call would then silently fall back to system Python -- which is
# missing pandas/yt-dlp/torch/etc and fails immediately, in a way that looks
# nothing like what actually broke (this happened on the first real run of
# this script: every download "failed" because the script never even got
# past `import pandas`, not because of any network/IP issue).
VENV_PYTHON="$(dirname "${BASH_SOURCE[0]}")/venvast/bin/python3"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "VENV_PYTHON not found at $VENV_PYTHON -- falling back to 'python3' on PATH."
    echo "This is risky for a background/nohup run (see comment above). Fix the path if this isn't right."
    VENV_PYTHON="python3"
fi
echo "Using interpreter: $VENV_PYTHON ($($VENV_PYTHON --version 2>&1))"

# Fail fast and clearly if required packages aren't importable with this
# interpreter, instead of discovering it three layers deep in a misleading
# "download failed" log at 3am.
echo "--- Checking dependencies ---"
if ! "$VENV_PYTHON" -c "import pandas, numpy, sklearn, torch, torchaudio, librosa, soundfile, yt_dlp, curl_cffi" 2> logs/dependency_check.log; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ABORTED: missing dependencies under $VENV_PYTHON. See logs/dependency_check.log" >> "$STATUS_FILE"
    echo "ABORTING: $VENV_PYTHON is missing required packages -- see logs/dependency_check.log"
    cat logs/dependency_check.log
    exit 1
fi
echo "Dependencies OK."

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
# Sample size and threshold calibrated against a real measurement: a full
# overnight run on this network genuinely succeeded ~77.5% of the time
# (2297/2964), with most failures being real content attrition (~1-year-old
# videos, since deleted/removed) rather than actual blocking. A 10-video
# sample is too small against that noise floor -- it can occasionally read
# as a near-total failure by chance alone even when nothing is wrong (this
# happened: 0-1/10 succeeded on one run, then an already-succeeded video
# downloaded fine again seconds later, proving the pipeline was never
# actually broken). 25 videos, threshold >=10, keeps real margin below the
# known ~77.5% baseline while still catching a genuine full block/crash.
PREFLIGHT_N=25
PREFLIGHT_MIN_OK=10
echo "--- Pre-flight: testing download on $PREFLIGHT_N videos before committing to the full batch ---"
preflight_log="logs/$(date +%Y%m%d_%H%M%S)_preflight_download.log"
"$VENV_PYTHON" pipeline/download_videos.py --limit "$PREFLIGHT_N" > "$preflight_log" 2>&1
preflight_ok=$(grep -oE '^Done\. [0-9]+ succeeded' "$preflight_log" | grep -oE '[0-9]+' || echo 0)

if grep -q "^Nothing to do\.$" "$preflight_log"; then
    # Nothing left to test isn't a failure -- it means download_videos.py has
    # already exhausted the candidate pool (everything is either downloaded
    # or already logged as a permanent failure from a prior run). Testing
    # "does downloading work" is meaningless when there's nothing left to
    # attempt; skip straight to the real run rather than reporting a false 0%.
    echo "Pre-flight: nothing left to download (pool already exhausted) -- skipping gate, proceeding."
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] OK    preflight_download (skipped -- download pool already exhausted)" >> "$STATUS_FILE"
elif [ "$preflight_ok" -ge "$PREFLIGHT_MIN_OK" ]; then
    echo "Pre-flight: ${preflight_ok}/${PREFLIGHT_N} succeeded (log: $preflight_log)"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] OK    preflight_download (${preflight_ok}/${PREFLIGHT_N} succeeded)" >> "$STATUS_FILE"
else
    # Diagnose from the actual log instead of assuming a cause.
    if grep -qi "ModuleNotFoundError\|ImportError\|Traceback" "$preflight_log"; then
        reason="script crashed on startup (missing dependency/import error) -- this is NOT a network/IP issue, check $preflight_log for the traceback"
    elif grep -qi "impersonat" "$preflight_log"; then
        reason="yt-dlp is missing browser-impersonation support (curl_cffi) -- TikTok's bot detection may be rejecting requests because of this, not the network itself. Try: pip install -U \"yt-dlp[default]\" curl_cffi"
    elif grep -qi "IP address is blocked" "$preflight_log"; then
        # NOTE: this message is known to be misleading -- TikTok returns it for
        # several different real causes (deleted/removed video, geo-restriction,
        # genuine rate-limiting), not only actual IP blocks. At this sample size
        # (25) hitting the threshold is a real signal either way, but don't
        # assume it specifically means "blocked" -- check a few IDs in a browser
        # if this keeps happening across repeated runs.
        reason="most failures were 'IP address is blocked' -- this can mean a real block OR a batch of removed/unavailable videos (TikTok uses the same message for both). Check a few video IDs from $preflight_log in a browser before assuming it's a network problem."
    else
        reason="unclear -- read $preflight_log directly, don't assume network/IP blocking"
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ABORTED after pre-flight: only ${preflight_ok}/${PREFLIGHT_N} downloads succeeded. Diagnosis: $reason" >> "$STATUS_FILE"
    echo "ABORTING: pre-flight download success rate too low (${preflight_ok}/${PREFLIGHT_N}, need >=${PREFLIGHT_MIN_OK})."
    echo "Diagnosis: $reason"
    exit 1
fi

# ── Full pipeline ──────────────────────────────────────────────────────────
run_step "download_videos"        "$VENV_PYTHON" pipeline/download_videos.py || exit 1
run_step "extract_audio_features" "$VENV_PYTHON" pipeline/extract_audio_features.py || exit 1
run_step "extract_ast_embeddings" "$VENV_PYTHON" pipeline/extract_ast_embeddings.py || exit 1
run_step "build_features"         "$VENV_PYTHON" pipeline/build_features.py || exit 1
run_step "build_model_dataset"    "$VENV_PYTHON" pipeline/build_model_dataset.py || exit 1
run_step "ridge_models"           "$VENV_PYTHON" analysis/run_ridge_models.py || exit 1

# Permutation test: conservative N_PERMUTATIONS for this first run at the new
# scale -- see header comment. Only the primary outcome (happy) for now;
# ridge_models already covers all four outcomes' R²/RMSE.
run_step "permutation_test" "$VENV_PYTHON" analysis/permutation_test.py --target happy --comparison both --n-permutations 100

echo "=== Pipeline run finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$STATUS_FILE"
echo ""
echo "=== ALL DONE. Summary: ==="
cat "$STATUS_FILE"
