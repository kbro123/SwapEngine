#!/bin/bash
# Build (niced, one compile at a time, never during a benchmark/test step) then run the three probes. The probes hold their own
# timing loops (util.hpp: extended hold list + 120 s quiet window + per-block watcher).
cd "$(dirname "$0")"
E="/Users/kevinbroughton/Desktop/Claude Projects/SwapEngine"
SDK=$(xcrun --show-sdk-path)
F=(-std=c++20 -O3 -DNDEBUG -fno-math-errno -march=x86-64-v3 -isysroot "$SDK" "-DSWAPS_CONVENTIONS_JSON=\"$E/conventions/conventions.json\"" -I. "-I$E/include" "-I$E/build/generated" "-I$E/bench/fixtures" "-I$E/third_party/eigen")
HOLD='shape_ladder_bench|check_perf\.py|verify\.sh|ctest|mutate\.py|probe_dm_stall'
wait_steps() {
  local quiet=0
  while [ $quiet -lt 60 ]; do
    if pgrep -f "$HOLD" >/dev/null; then quiet=0; else quiet=$((quiet + 10)); fi
    sleep 10
  done
}
build() {  # $1 out, $2 src, rest extra flags
  local out=$1 src=$2; shift 2
  wait_steps
  echo "[build $out $(date +%T) load $(sysctl -n vm.loadavg)]"
  nice -n 19 xcrun clang++ "${F[@]}" "$@" "$src" -o "$out" 2>&1 | grep -E "error" -A4 | head -30
  [ -x "$out" ] && echo "[built $out]" || echo "[BUILD FAILED $out]"
}
rm -f coarse coarse_nofma ir
build coarse coarse.cpp
build coarse_nofma coarse.cpp -ffp-contract=off
build ir ir.cpp -ffp-contract=off
for b in coarse coarse_nofma ir; do
  echo "=== $b  start $(date +%T)  load $(sysctl -n vm.loadavg)"
  ./$b
  echo "=== $b  end $(date +%T)  load $(sysctl -n vm.loadavg)  exit $?"
done
echo "CHAIN DONE"
