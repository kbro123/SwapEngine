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

  # The committed curve-build baseline: ours vs QuantLib GlobalBootstrap (same algorithm class).
  add_executable(curve_build_bench curve_build_bench.cpp)
  target_link_libraries(curve_build_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(curve_build_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  # The risk-ladder baseline: our AAD + IFT vs QuantLib bump-and-reprice.
  add_executable(risk_bench risk_bench.cpp)
  target_link_libraries(risk_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(risk_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  # The portfolio-analytics baseline: batched reprice vs QuantLib per-swap NPV loop.
  add_executable(portfolio_bench portfolio_bench.cpp)
  target_link_libraries(portfolio_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(portfolio_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  # Stage 2: warm re-calibration (cached-Jacobian) vs QuantLib re-bootstrap.
  add_executable(warm_bench warm_bench.cpp)
  target_link_libraries(warm_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(warm_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  # Stage 3: curve-bundle build -- our joint LM vs QuantLib sequential multi-curve GlobalBootstrap.
  add_executable(bundle_build_bench bundle_build_bench.cpp)
  target_link_libraries(bundle_build_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(bundle_build_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)
endif()

# Convenience target: run the perf gate.
add_custom_target(bench
  COMMAND swaps_bench --benchmark_format=json --benchmark_out=${CMAKE_BINARY_DIR}/bench.json
  DEPENDS swaps_bench
  COMMENT "Running performance benchmarks -> ${CMAKE_BINARY_DIR}/bench.json"
)
