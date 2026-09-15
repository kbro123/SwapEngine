#!/bin/bash
# Part 3 probes. Builds only after chain.sh has finished ITS builds (one compile at a time), never during a benchmark/test step,
# niced. guardtrace is counts-only; branchprobe times under util.hpp's hold (it also holds on chain.sh's probes: ./coarse, ./ir).
cd "$(dirname "$0")"
E="/Users/kevinbroughton/Desktop/Claude Projects/SwapEngine"
SDK=$(xcrun --show-sdk-path)
F=(-std=c++20 -O3 -DNDEBUG -fno-math-errno -march=x86-64-v3 -isysroot "$SDK" "-DSWAPS_CONVENTIONS_JSON=\"$E/conventions/conventions.json\"" -I. "-I$E/include" "-I$E/build/generated" "-I$E/bench/fixtures" "-I$E/third_party/eigen")
HOLD='shape_ladder_bench|check_perf\.py|verify\.sh|ctest|mutate\.py|probe_dm_stall'
wait_steps() {
  local quiet=0
  while [ $quiet -lt 60 ]; do
    if pgrep -f "$HOLD" >/dev/null || pgrep -f "clang.*(coarse|ir)\.cpp" >/dev/null; then quiet=0; else quiet=$((quiet + 10)); fi
    sleep 10
  done
}
# wait until chain.sh has built all three of its binaries (or has exited)
while pgrep -f "bash chain.sh" >/dev/null && ! grep -q "^\[built ir\]\|BUILD FAILED ir" run_chain.txt 2>/dev/null; do sleep 20; done
build() {
  local out=$1 src=$2; shift 2
  wait_steps
  echo "[build $out $(date +%T) load $(sysctl -n vm.loadavg)]"
  nice -n 19 xcrun clang++ "${F[@]}" "$@" "$src" -o "$out" 2>&1 | grep -E "error" -A4 | head -30
  [ -x "$out" ] && echo "[built $out]" || echo "[BUILD FAILED $out]"
}
rm -f guardtrace branchprobe
build guardtrace guardtrace.cpp
build branchprobe branch.cpp -ffp-contract=off
if [ -x guardtrace ]; then
  wait_steps
  echo "=== guardtrace (counts only) start $(date +%T) load $(sysctl -n vm.loadavg)"
  nice -n 10 ./guardtrace
  echo "=== guardtrace end $(date +%T) exit $?"
fi
if [ -x branchprobe ]; then
  echo "=== branchprobe start $(date +%T) load $(sysctl -n vm.loadavg)"
  AADJIT_EXTRA_HOLD="./coarse,./ir" ./branchprobe
  echo "=== branchprobe end $(date +%T) exit $?"
fi
echo "PART3 DONE"
