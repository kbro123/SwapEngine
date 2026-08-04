# Performance gate — Google Benchmark.
# Benchmarks emit JSON consumed by tools/check_perf.py, which compares against
# baselines/baselines.json and fails if we are not faster than the QuantLib baseline.
# Phase 0 ships only a smoke benchmark to prove the wiring.

find_package(benchmark REQUIRED)

add_executable(swaps_bench
  smoke_bench.cpp
)
target_link_libraries(swaps_bench PRIVATE swaps benchmark::benchmark_main)

# Large-scale spread-chain bundle -- ours-only scaling probe for the compiled-engine hot-path
# optimizations (no QuantLib needed; not a fingerprint gate metric).
add_executable(bundle_scale_bench bundle_scale_bench.cpp)
target_link_libraries(bundle_scale_bench PRIVATE swaps benchmark::benchmark_main)

# Coherent parallel portfolio reprice: serial full-book vs N-slice split off one curve (ours-only).
add_executable(portfolio_parallel_bench portfolio_parallel_bench.cpp)
target_link_libraries(portfolio_parallel_bench PRIVATE swaps benchmark::benchmark_main)

# Swaption vol-cube reprice: warm (streaming, no recalibration) vs cold (one-shot verb). Ours-only probe;
# links swaps_api for BundleSession + bundle_from_json, reads the SOFR bundle fixture by compiled-in path.
add_executable(vol_cube_bench vol_cube_bench.cpp)
target_link_libraries(vol_cube_bench PRIVATE swaps swaps_api benchmark::benchmark_main)
target_compile_definitions(vol_cube_bench PRIVATE
  SOFR_BUNDLE_JSON="${CMAKE_SOURCE_DIR}/bench/fixtures/sofr_bundle.json")


# AAD vs bump / calibration benchmarks (link the QuantLib oracle for problem setup).
if(TARGET quantlib_oracle)
  # Swaption vol-surface reprice: ours (batched SoA) vs QuantLib's per-cell analytic path. QuantLib-paired
  # speedup metric (vol_cube_warm); links swaps_api for BundleSession + the SOFR bundle fixture.
  add_executable(vol_cube_ql_bench vol_cube_ql_bench.cpp)
  target_link_libraries(vol_cube_ql_bench PRIVATE swaps swaps_api quantlib_oracle benchmark::benchmark_main)
  target_include_directories(vol_cube_ql_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_compile_definitions(vol_cube_ql_bench PRIVATE
    SOFR_BUNDLE_JSON="${CMAKE_SOURCE_DIR}/bench/fixtures/sofr_bundle.json")

  # Desk-scale swaption portfolio (thousands of random swaptions), ours vs QuantLib, swept over book size.
  add_executable(swaption_portfolio_bench swaption_portfolio_bench.cpp)
  target_link_libraries(swaption_portfolio_bench PRIVATE swaps swaps_api quantlib_oracle benchmark::benchmark_main)
  target_compile_definitions(swaption_portfolio_bench PRIVATE
    SOFR_BUNDLE_JSON="${CMAKE_SOURCE_DIR}/bench/fixtures/sofr_bundle.json")

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

  # Branch-parallel staged bundle calibrate on the REALISTIC star (serial vs thread-per-SCC; ours-only).
  add_executable(bundle_parallel_bench bundle_parallel_bench.cpp)
  target_link_libraries(bundle_parallel_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(bundle_parallel_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  # Jacobian-recompute vs tick cost -- profiling probe for the speculative background-Jacobian research.
  add_executable(jacobian_cost_bench jacobian_cost_bench.cpp)
  target_link_libraries(jacobian_cost_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(jacobian_cost_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)
endif()

# Convenience target: run the perf gate.
add_custom_target(bench
  COMMAND swaps_bench --benchmark_format=json --benchmark_out=${CMAKE_BINARY_DIR}/bench.json
  DEPENDS swaps_bench
  COMMENT "Running performance benchmarks -> ${CMAKE_BINARY_DIR}/bench.json"
)
