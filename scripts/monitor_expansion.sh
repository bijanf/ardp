#!/usr/bin/env bash
# Monitor the parallel CMIP6 expansion downloads.
# Reports per-job: PID alive? heartbeat (last log line + age), files done.
# Usage:  bash scripts/monitor_expansion.sh
#         bash scripts/monitor_expansion.sh --watch   # rerun every 30s

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

watch_mode=0
[[ "${1:-}" == "--watch" ]] && watch_mode=1

NEW_MODELS=(ACCESS-ESM1-5 BCC-CSM2-MR CESM2-WACCM CIESM CMCC-ESM2
            EC-Earth3 FGOALS-f3-L MRI-ESM2-0 TaiESM1)

now() { date "+%H:%M:%S"; }

stuck_threshold_sec=900  # 15 min without log activity = stuck

report_once() {
    echo "============================================================"
    echo "Expansion monitor  ($(now))"
    echo "============================================================"

    # 1) process liveness
    local n_alive=$(pgrep -f compute_cmip6_expansion_hybrid | wc -l)
    echo "alive python jobs: $n_alive"

    # 2) per-log heartbeat
    for log in logs/expansion_jobA_pangeo.log \
               logs/expansion_jobB_mixed.log \
               logs/expansion_jobC_esgf.log \
               logs/expansion_jobD_access.log; do
        [[ -f "$log" ]] || continue
        local mtime=$(stat -c %Y "$log")
        local age=$(( $(date +%s) - mtime ))
        local last=$(tail -n 1 "$log" | tr -d '\n')
        local marker="ok"
        (( age > stuck_threshold_sec )) && marker="STALE"
        printf "  [%s] %s  age=%ds  last: %.120s\n" \
                "$marker" "$(basename "$log")" "$age" "$last"
    done

    # 3) per-model file completion + corruption check
    echo ""
    echo "section files (4/4 = complete for that model):"
    for m in "${NEW_MODELS[@]}"; do
        local n=0 mb=0 bad=""
        for c in historical_vo historical_so ssp585_vo ssp585_so; do
            local f="data/cmip6_sections/${m}_${c}.nc"
            if [[ -f "$f" ]]; then
                local sz=$(stat -c %s "$f")
                if (( sz < 1000 )); then
                    bad="$bad ${c}<1KB"
                else
                    n=$((n+1))
                    mb=$(( mb + sz / 1000000 ))
                fi
            fi
        done
        if [[ -n "$bad" ]]; then
            printf "  %-20s %d/4   %4d MB  CRASHED:%s\n" \
                   "$m" "$n" "$mb" "$bad"
        else
            printf "  %-20s %d/4   %4d MB\n" "$m" "$n" "$mb"
        fi
    done

    # 3b) NetCDF integrity check (open every produced section)
    echo ""
    echo "NetCDF integrity check:"
    python3 - <<'PY' 2>&1 | head -30
import os
from pathlib import Path
import xarray as xr
sd = Path("data/cmip6_sections")
new_models = ["ACCESS-ESM1-5","BCC-CSM2-MR","CESM2-WACCM","CIESM",
              "CMCC-ESM2","EC-Earth3","FGOALS-f3-L","MRI-ESM2-0","TaiESM1"]
problems = 0
for m in new_models:
    for c in ("historical_vo","historical_so","ssp585_vo","ssp585_so"):
        f = sd / f"{m}_{c}.nc"
        if not f.exists():
            continue
        try:
            ds = xr.open_dataset(f, decode_times=False)
            v = c.split("_")[1]
            if v not in ds.data_vars:
                print(f"  BAD {f.name}: missing var {v}")
                problems += 1
            elif "section_latitude" not in ds[v].attrs:
                print(f"  WARN {f.name}: no section_latitude attr")
            elif "time" not in ds.dims:
                print(f"  BAD {f.name}: no time dim")
                problems += 1
            ds.close()
        except Exception as e:
            print(f"  BAD {f.name}: open failed -> {e}")
            problems += 1
if problems == 0:
    print("  all section files OK")
else:
    print(f"  {problems} BAD file(s) - delete and re-run pipeline")
PY

    # 3c) leftover scratch from crashed/killed downloads
    echo ""
    echo "scratch leftovers (/tmp/cmip6_expansion):"
    if [[ -d /tmp/cmip6_expansion ]]; then
        local sc_mb=$(du -sm /tmp/cmip6_expansion 2>/dev/null | awk '{print $1}')
        local sc_n=$(find /tmp/cmip6_expansion -name '*.nc' | wc -l)
        echo "  $sc_n in-flight files, $sc_mb MB total"
        # Stale leftover: scratch file > 1h old means a crashed run
        local stale=$(find /tmp/cmip6_expansion -name '*.nc' \
                       -mmin +60 2>/dev/null | wc -l)
        if (( stale > 0 )); then
            echo "  WARN: $stale file(s) >1h old (probable crash leftover)"
        fi
    else
        echo "  no scratch dir (clean)"
    fi

    # 4) live download bandwidth (current ESGF transfers)
    echo ""
    echo "active ESGF transfers (last 30s):"
    local since=$(date -d '30 seconds ago' +%s 2>/dev/null || \
                  date -v-30S +%s 2>/dev/null)
    for log in logs/expansion_jobB_mixed.log \
               logs/expansion_jobC_esgf.log; do
        [[ -f "$log" ]] || continue
        # extract the most recent "MB/s" measurement
        local recent=$(tail -n 5 "$log" | \
                       grep -E 'MB/s|done' | tail -n 1)
        [[ -n "$recent" ]] && \
            printf "  %s: %s\n" "$(basename "$log" .log)" \
                   "$(echo "$recent" | sed 's/^.*\] //')"
    done
}

if (( watch_mode )); then
    while :; do
        clear
        report_once
        sleep 30
    done
else
    report_once
fi
