#!/usr/bin/env bash
# Per-model progress bars for the CMIP6 expansion.
#
# For each of the 9 new models, show a 4-segment progress bar where each
# segment is one of (historical_vo, historical_so, ssp585_vo, ssp585_so).
# A segment is filled to:
#   100% if the section file exists in data/cmip6_sections/
#    X%  for the active variable (parsed from the latest "[GET ... NN%" log
#         line for that model, weighted across multi-chunk archives)
#     0% otherwise.
#
# Usage:
#   bash scripts/progress_expansion.sh           # one shot
#   bash scripts/progress_expansion.sh --watch   # refresh every 10s

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

NEW_MODELS=(ACCESS-ESM1-5 BCC-CSM2-MR CESM2-WACCM CIESM CMCC-ESM2
            EC-Earth3 FGOALS-f3-L MRI-ESM2-0 TaiESM1)

# -- helpers ------------------------------------------------------------
# Render a fraction (0.0 .. 1.0) as a 16-char unicode progress bar.
bar() {
    local frac="$1"
    local width=16
    local n_full
    n_full=$(awk -v f="$frac" -v w="$width" 'BEGIN{
        v=int(f*w); if (v<0) v=0; if (v>w) v=w; print v
    }')
    local n_empty=$((width - n_full))
    printf '['
    for ((i=0; i<n_full; i++)); do printf '#'; done
    for ((i=0; i<n_empty; i++)); do printf '.'; done
    printf ']'
}

# Inspect /tmp/cmip6_expansion/<model> to find how many chunks of the
# active variable are downloaded vs total expected, plus % of the
# currently-streaming chunk.  Returns "n_done/n_total LASTPCT" as text.
chunks_status() {
    local model="$1"
    local var_exp="$2"   # e.g. "ssp585_vo"
    local d="/tmp/cmip6_expansion/${model}"
    local var=${var_exp##*_}; local exp=${var_exp%_*}
    if [[ ! -d "$d" ]]; then echo "0 0"; return; fi
    local n
    n=$(find "$d" -name "${var}_Omon_${model}_${exp}_*_gn_*.nc" \
                  -size +100k 2>/dev/null | wc -l)
    # Most recent chunk progress percent from the log that mentions THIS
    # model-name in its line (so we don't mix Job B's % into EC-Earth3).
    local pct=""
    for log in logs/expansion_jobB_mixed.log \
               logs/expansion_jobC_esgf.log \
               logs/expansion_jobA_pangeo.log \
               logs/expansion_jobD_access.log; do
        [[ -f "$log" ]] || continue
        # find the most recent file-progress line whose nearest-prior
        # [GET ...] line includes this model name. Approximation: just
        # require this log file to mention the model in the last 30 lines.
        if tail -n 30 "$log" 2>/dev/null | grep -q "${model}_"; then
            local p
            p=$(tail -n 30 "$log" \
                 | grep -oE '\([0-9]+%, [0-9.]+ MB/s\)' \
                 | tail -n 1 | grep -oE '^\([0-9]+%' | tr -d '(%')
            if [[ -n "$p" ]]; then pct="$p"; fi
        fi
    done
    echo "${n} ${pct:-0}"
}

report_once() {
    local now
    now=$(date "+%H:%M:%S")
    printf '\n========================================================================\n'
    printf 'CMIP6 expansion progress  (%s)\n' "$now"
    printf '========================================================================\n'
    printf '%-18s  %-18s  %s\n' 'model' 'progress (4 files)' 'breakdown'
    printf '%s\n' '------------------------------------------------------------------------'

    for m in "${NEW_MODELS[@]}"; do
        # Count completed final-section files and accumulate fractional
        # progress for the variable currently being downloaded.
        local frac=0
        local stat_str=""
        local n_done=0
        for c in historical_vo historical_so ssp585_vo ssp585_so; do
            local f="data/cmip6_sections/${m}_${c}.nc"
            if [[ -f "$f" && $(stat -c %s "$f") -gt 1000 ]]; then
                n_done=$((n_done+1))
                stat_str="$stat_str ${c}"$'\xe2\x9c\x93'
            else
                # Partial / not started
                local cs
                cs=$(chunks_status "$m" "$c")
                local nchunks=${cs%% *}
                local actpct=${cs##* }
                if (( nchunks > 0 )); then
                    stat_str="$stat_str ${c}~${nchunks}ch+${actpct}%"
                else
                    stat_str="$stat_str ${c}__"
                fi
            fi
        done
        # Fractional progress: each completed file = 0.25, plus partial work.
        # For simplicity report the integer 4-of-4 bar only.
        frac=$(awk -v n="$n_done" 'BEGIN{print n/4}')
        local b
        b=$(bar "$frac")
        printf '%-18s  %s %d/4   %s\n' "$m" "$b" "$n_done" "$stat_str"
    done

    # Aggregate
    local total=0
    for m in "${NEW_MODELS[@]}"; do
        for c in historical_vo historical_so ssp585_vo ssp585_so; do
            local f="data/cmip6_sections/${m}_${c}.nc"
            if [[ -f "$f" && $(stat -c %s "$f") -gt 1000 ]]; then
                total=$((total+1))
            fi
        done
    done
    local total_frac
    total_frac=$(awk -v t="$total" 'BEGIN{print t/36}')
    local tb
    tb=$(bar "$total_frac")
    printf '\n%-18s  %s %d/36 (%d%%)\n' 'TOTAL' "$tb" "$total" \
            "$(awk -v t="$total" 'BEGIN{printf int(t*100/36)}')"

    # Active transfer rates
    printf '\nactive transfers:\n'
    for log in logs/expansion_jobB_mixed.log \
               logs/expansion_jobC_esgf.log logs/expansion_jobD_access.log; do
        [[ -f "$log" ]] || continue
        local last
        last=$(tail -n 5 "$log" | grep -E 'MB/s|done' | tail -n 1)
        [[ -n "$last" ]] && \
            printf '  %s: %s\n' "$(basename "$log" .log)" \
                   "$(echo "$last" | sed 's/^.*\] //')"
    done
    printf '\nlive: %s\n' "$(date "+%H:%M:%S")"
}

if [[ "${1:-}" == "--watch" ]]; then
    while :; do
        clear
        report_once
        sleep 10
    done
else
    report_once
fi
