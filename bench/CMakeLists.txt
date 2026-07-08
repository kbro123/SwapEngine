# Performance gate — Google Benchmark.
# Benchmarks emit JSON consumed by tools/check_perf.py, which compares against
# baselines/baselines.json and fails if we are not faster than the QuantLib baseline.
# Phase 0 ships only a smoke benchmark to prove the wiring.

find_package(benchmark REQUIRED)

add_executable(swaps_bench
  smoke_bench.cpp
)
target_link_libraries(swaps_bench PRIVATE swaps benchmark::benchmark_main)

# Convenience target: run the perf gate.
add_custom_target(bench
  COMMAND swaps_bench --benchmark_format=json --benchmark_out=${CMAKE_BINARY_DIR}/bench.json
  DEPENDS swaps_bench
  COMMENT "Running performance benchmarks -> ${CMAKE_BINARY_DIR}/bench.json"
)
