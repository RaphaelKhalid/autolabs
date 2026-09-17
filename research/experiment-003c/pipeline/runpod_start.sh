#!/usr/bin/env bash
# Pod entry for Experiment 3C. Self-bootstraps: clones the pinned repo commit, installs deps, runs the smoke test.
# Expects /workspace/3c/.env (AUTOLABS_3C_WORKER_URL, AUTOLABS_3C_TOKEN, AUTOLABS_3C_RUN_ID). Never prints it.
set -euo pipefail
WORKDIR=/workspace/3c; REPO=https://github.com/RaphaelKhalid/autolabs.git; REF="${AUTOLABS_3C_GIT_REF:-main}"
mkdir -p "$WORKDIR"; export HF_HOME="$WORKDIR/hf"; mkdir -p "$HF_HOME"
if [ -f "$WORKDIR/.env" ]; then set -a; source "$WORKDIR/.env"; set +a; else echo "warning: $WORKDIR/.env missing" >&2; fi
if [ ! -d "$WORKDIR/repo/.git" ]; then git clone --depth 1 --filter=blob:none --sparse "$REPO" "$WORKDIR/repo"; fi
cd "$WORKDIR/repo"; git sparse-checkout set research/experiment-003c/pipeline; git fetch --depth 1 origin "$REF"; git checkout -q FETCH_HEAD
echo "pipeline commit: $(git rev-parse HEAD)"
cd research/experiment-003c/pipeline
python -m pip install -q --upgrade pip; python -m pip install -q -r requirements.txt
CONFIG="${1:-configs/smoke.json}"
shift $(( $# > 0 ? 1 : 0 ))
exec python run_smoke.py --config "$CONFIG" "$@" 2>&1 | tee -a "$WORKDIR/smoke.log"
