#!/bin/bash
# guardtrace2: counts only. Build niced, never during a benchmark/test step and never alongside another of our compiles.
cd "$(dirname "$0")"
E="/Users/kevinbroughton/Desktop/Claude Projects/SwapEngine"
SDK=$(xcrun --show-sdk-path)
F=(-std=c++20 -O3 -DNDEBUG -fno-math-errno -march=x86-64-v3 -isysroot "$SDK" "-DSWAPS_CONVENTIONS_JSON=\"$E/conventions/conventions.json\"" -I. "-I$E/include" "-I$E/build/generated" "-I$E/bench/fixtures" "-I$E/third_party/eigen")
HOLD='shape_ladder_bench|check_perf\.py|verify\.sh|ctest|mutate\.py|probe_dm_stall'
wait_steps() {
  local quiet=0
  while [ $quiet -lt 60 ]; do
    if pgrep -f "$HOLD" >/dev/null || pgrep -f "clang.*aadjit|clang.*(coarse|ir|branch|guardtrace)\.cpp" >/dev/null; then quiet=0; else quiet=$((quiet + 10)); fi
    sleep 10
  done
}
rm -f guardtrace2
wait_steps
echo "[build guardtrace2 $(date +%T) load $(sysctl -n vm.loadavg)]"
nice -n 19 xcrun clang++ "${F[@]}" guardtrace2.cpp -o guardtrace2 2>&1 | grep -E "error" -A4 | head -30
[ -x guardtrace2 ] && echo "[built guardtrace2]" || { echo "[BUILD FAILED guardtrace2]"; echo "PART3B DONE"; exit 1; }
wait_steps
echo "=== guardtrace2 (counts only) start $(date +%T) load $(sysctl -n vm.loadavg)"
nice -n 10 ./guardtrace2
echo "=== guardtrace2 end $(date +%T) exit $?"
echo "PART3B DONE"
