#!/usr/bin/env bash
# Run the full 50-member MPI-ESM1-2-LR Grand Ensemble SMILE
# decomposition in the background. Resume-friendly — each member's
# row is appended to the CSV as it completes, so an interruption only
# loses the in-flight member.
#
# Usage:
#   bash scripts/run_smile_full.sh           # start fresh / resume
#   tail -f logs/smile_esgf_full.log         # watch progress
#   ls /tmp/smile_esgf/                      # see scratch dir
#
# Stop:
#   pkill -f compute_smile_esgf.py
# Restart picks up where it left off.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

LOG="logs/smile_esgf_full.log"
CSV="data/results/fovs_decomposition_smile_esgf.csv"
SCRATCH="/tmp/smile_esgf"

echo "Starting full SMILE run; output CSV: $CSV"
echo "Log:                                  $LOG"
echo "Scratch dir (auto-cleaned per member): $SCRATCH"

nohup python -u scripts/compute_smile_esgf.py \
    --output "$CSV" \
    --scratch "$SCRATCH" >> "$LOG" 2>&1 &

PID=$!
echo "Started PID $PID. Tail the log to watch progress:"
echo "  tail -f $LOG"
