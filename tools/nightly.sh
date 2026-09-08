#!/usr/bin/env bash
# nightly.sh — the FULL gate on the quiesced reference machine (PRINCIPLES.md P9/P10).
#
# Runs, in order: build · guards · correctness (ctest incl. the QuantLib oracle + consistency binaries, with
# SWAPS_TIMING_ASSERTS=1) · perf gate (self-baseline + targets, min-of-5 reps, refuses under load) and records
# the informational QuantLib reference table. Writes a dated markdown summary to
#   ~/Library/Logs/swapengine-nightly/YYYY-MM-DD.md   and   $ROOT/baselines/NIGHTLY.md  (latest; committed by hand)
# Installed as a launchd agent by tools/install_nightly.sh (03:00 local). Run by hand any time: tools/nightly.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOGDIR="${HOME}/Library/Logs/swapengine-nightly"; mkdir -p "${LOGDIR}"
DAY="$(date +%F)"; LOG="${LOGDIR}/${DAY}.log"; SUM="${LOGDIR}/${DAY}.md"
export SWAPS_TIMING_ASSERTS=1
export SWAPS_PERF_ARGS="--reps 5 --min-time 0.5 --max-load 3.0"   # this desktop idles at ~2-3 with the Claude app open
{
  echo "# SwapEngine nightly — ${DAY} ($(git rev-parse --short HEAD), branch $(git rev-parse --abbrev-ref HEAD))"
  echo; echo "- started: $(date -u +%FT%TZ)  load: $(uptime | sed 's/.*load averages*: //')"
  echo "- fingerprint: $(bash tools/fingerprint.sh | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["key"], d["isa"], d["arch_flag"], "| ql:", d.get("quantlib_toolchain","?"))')"
  echo; echo '```'
  bash tools/verify.sh 2>&1 | tee "${LOG}" | tail -40
  RC=${PIPESTATUS[0]}
  echo '```'
  echo; echo "- verify.sh exit: ${RC}  ($( [ "${RC}" = 0 ] && echo GREEN || echo RED ))"
  if [ -f build/perf/last_run.json ]; then
    echo; echo "## Perf (min of reps; reference is informational)"; echo; echo '```'
    python3 - <<'EOF'
import json
d=json.load(open("build/perf/last_run.json"))
print(f"quiesced={d['quiesced']} load={d['load_avg']:.2f} reps={d['reps']} utc={d['utc']}")
print(f"{'metric':<24}{'ours_ns':>14}{'ref_ns':>14}{'speedup':>9}  verdict")
for k,m in d["metrics"].items():
    v=d["verdicts"].get(k,{})
    ref=m["reference_ns"]; sp=m["speedup"]
    print(f"{k:<24}{m['ours_ns']:>14,}{(f'{ref:,}' if ref else '—'):>14}{(f'{sp:.1f}x' if sp else '—'):>9}  {'PASS' if v.get('ok') else 'FAIL'} {','.join(v.get('flags',[]))}")
EOF
    echo '```'
  fi
  echo; echo "- finished: $(date -u +%FT%TZ)"
} > "${SUM}" 2>&1
cp "${SUM}" "${ROOT}/baselines/NIGHTLY.md"
echo "nightly: $(grep -o 'verify.sh exit: [0-9]* *(.*)' "${SUM}")  -> ${SUM}"
