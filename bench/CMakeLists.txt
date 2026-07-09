# Performance gate — Google Benchmark.
# Benchmarks emit JSON consumed by tools/check_perf.py, which compares against
# baselines/baselines.json and fails if we are not faster than the QuantLib baseline.
# Phase 0 ships only a smoke benchmark to prove the wiring.

find_package(benchmark REQUIRED)

add_executable(swaps_bench
  smoke_bench.cpp
)
target_link_libraries(swaps_bench PRIVATE swaps benchmark::benchmark_main)

# AAD vs bump / calibration benchmarks (link the QuantLib oracle for problem setup).
if(TARGET quantlib_oracle)
  add_executable(calibration_bench calibration_bench.cpp)
  target_link_libraries(calibration_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(calibration_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)
endif()

# Convenience target: run the perf gate.
add_custom_target(bench
  COMMAND swaps_bench --benchmark_format=json --benchmark_out=${CMAKE_BINARY_DIR}/bench.json
  DEPENDS swaps_bench
  COMMENT "Running performance benchmarks -> ${CMAKE_BINARY_DIR}/bench.json"
)
