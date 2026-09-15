#!/bin/bash
# Counts-only runs of the Part 1 / Part 2 probes (AADJIT_NO_TIMING): parity, dispatch counts, allocations, IR coverage -- no timing
# loops. Binary names contain "guardtrace" so every running probe of ours (chain ./coarse, branchprobe) HOLDS its timing while these run.
# Builds one at a time, niced, never during a benchmark/test step (a main-line ninja build only slows down, it measures nothing).
cd "$(dirname "$0")"
E="/Users/kevinbroughton/Desktop/Claude Projects/SwapEngine"
SDK=$(xcrun --show-sdk-path)
F=(-std=c++20 -O3 -DNDEBUG -fno-math-errno -march=x86-64-v3 -isysroot "$SDK" "-DSWAPS_CONVENTIONS_JSON=\"$E/conventions/conventions.json\"" -I. "-I$E/include" "-I$E/build/generated" "-I$E/bench/fixtures" "-I$E/third_party/eigen")
HOLD='shape_ladder_bench|check_perf\.py|verify\.sh|ctest|mutate\.py|probe_dm_stall'
wait_steps() {
  local quiet=0
  while [ $quiet -lt 30 ]; do
    if pgrep -f "$HOLD" >/dev/null || pgrep -f "clang.*aadjit|clang.*(coarse|ir|branch|guardtrace2?)\.cpp" >/dev/null; then quiet=0; else quiet=$((quiet + 10)); fi
    sleep 10
  done
}
build() {
  local out=$1 src=$2; shift 2
  wait_steps
  echo "[build $out $(date +%T) load $(sysctl -n vm.loadavg)]"
  nice -n 19 xcrun clang++ "${F[@]}" "$@" "$src" -o "$out" 2>&1 | grep -E "error" -A4 | head -30
  [ -x "$out" ] && echo "[built $out]" || echo "[BUILD FAILED $out]"
}
rm -f guardtrace_coarse_counts_nofma guardtrace_coarse_counts_fma guardtrace_ir_counts
build guardtrace_coarse_counts_nofma coarse.cpp -ffp-contract=off
build guardtrace_coarse_counts_fma coarse.cpp
build guardtrace_ir_counts ir.cpp -ffp-contract=off
for b in guardtrace_coarse_counts_nofma guardtrace_coarse_counts_fma guardtrace_ir_counts; do
  [ -x "$b" ] || continue
  wait_steps
  echo "=== $b (counts only) start $(date +%T) load $(sysctl -n vm.loadavg)"
  AADJIT_NO_TIMING=1 nice -n 10 ./$b
  echo "=== $b end $(date +%T) exit $?"
done
echo "COUNTS DONE"
