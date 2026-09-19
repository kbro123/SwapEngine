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

# Compiled multi-curve book reprice vs the templated CurveHandle path (audit U2; ours-only).
add_executable(compiled_multi_bench compiled_multi_bench.cpp)
target_link_libraries(compiled_multi_bench PRIVATE swaps benchmark::benchmark_main)

# FX/MtM hybrid hot path: the AadBlock width-reduced AAD sweep, pooled vs the heap fallback, plus a
# streaming tick on a mixed bundle (ours-only, R11 pooled-dual retype; not a fingerprint gate metric).
add_executable(fx_stream_bench fx_stream_bench.cpp)
target_link_libraries(fx_stream_bench PRIVATE swaps benchmark::benchmark_main)

# Curve-space gamma via the reverse-mode AAD tape: 2nd-order Hessian vs the 1st-order delta ladder.
add_executable(gamma_bench gamma_bench.cpp)
target_link_libraries(gamma_bench PRIVATE swaps benchmark::benchmark_main)
target_include_directories(gamma_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)  # research/gamma.hpp (E6.1)

# Coherent parallel portfolio reprice: serial full-book vs N-slice split off one curve (ours-only).

add_executable(exposure_grid_bench exposure_grid_bench.cpp)
target_link_libraries(exposure_grid_bench PRIVATE swaps benchmark::benchmark_main)

# CMS static-replication hot path: Hagan G-derivative fan + Bachelier smile integrand (ours-only, R7).

# Caplet-vol stripping: full cap-curve bootstrap (ours-only, R9 O(n^2)->O(n) prefix).

# BundleSession warm paths at desk scale: rebind (plain + tension-regularized) and the streaming tick.
# Ours-only; session_rebind + stream_tick are fingerprint-GATED metrics (tools/check_perf.py).
add_executable(session_warm_bench session_warm_bench.cpp)
target_link_libraries(session_warm_bench PRIVATE swaps swaps_api benchmark::benchmark_main)

# Swaption vol-cube reprice: warm (streaming, no recalibration) vs cold (one-shot verb). Ours-only probe;
# links swaps_api for BundleSession + bundle_from_json, reads the SOFR bundle fixture by compiled-in path.
add_executable(vol_cube_bench vol_cube_bench.cpp)
target_link_libraries(vol_cube_bench PRIVATE swaps swaps_api benchmark::benchmark_main)
target_compile_definitions(vol_cube_bench PRIVATE
  SOFR_BUNDLE_JSON="${CMAKE_SOURCE_DIR}/bench/fixtures/sofr_bundle.json")


# AAD vs bump / calibration benchmarks (link the QuantLib oracle for problem setup).
if(TARGET quantlib_oracle)
  # The PRODUCT bundle (SOFR + ESTR + EUR-in-USD xccy, real conventions): cold/tension/warm/stream probe.
  add_executable(xccy_bundle_bench xccy_bundle_bench.cpp)
  target_link_libraries(xccy_bundle_bench PRIVATE swaps swaps_api quantlib_oracle benchmark::benchmark_main)
  target_include_directories(xccy_bundle_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  # Swaption vol-surface reprice: ours (batched SoA) vs QuantLib's per-cell analytic path. QuantLib-paired
  # reference bench (NOT the gated sofr_swaption_cube17_* metrics, which are ours-only in vol_cube_bench); links swaps_api for BundleSession + the SOFR bundle fixture.
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

  # THE MULTI-CURVE QuantLib reference: four curves, both engines, cold. (Its own main() asserts the two
  # arms agree at the pillars before timing them, so `benchmark` not `benchmark_main`.)
  add_executable(multicurve_ql_bench multicurve_ql_bench.cpp)
  target_link_libraries(multicurve_ql_bench PRIVATE swaps swaps_api quantlib_oracle benchmark::benchmark)
  target_include_directories(curve_build_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  # The risk-ladder baseline: our AAD + IFT vs QuantLib bump-and-reprice.
  add_executable(risk_bench risk_bench.cpp)
  target_link_libraries(risk_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(risk_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  # The portfolio-analytics baseline: batched reprice vs QuantLib per-swap NPV loop.
  add_executable(portfolio_bench portfolio_bench.cpp)
  target_link_libraries(portfolio_bench PRIVATE swaps swaps_api quantlib_oracle benchmark::benchmark_main)
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

  # The bond baseline: our batched universe sweep + curve-space book vs QuantLib's per-bond loops.
  add_executable(bond_sweep_bench bond_sweep_bench.cpp)
  target_link_libraries(bond_sweep_bench PRIVATE swaps quantlib_oracle benchmark::benchmark_main)
  target_include_directories(bond_sweep_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

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

# E3.1 hot-path census: per production path, allocations/bytes/frees per call via libmalloc's logger hook
# (every allocation in the process, not Eigen-only) + time. Plain executable (no google-benchmark), ours-only.
add_executable(hotpath_census hotpath_census.cpp)
target_link_libraries(hotpath_census PRIVATE swaps swaps_api)
target_compile_definitions(hotpath_census PRIVATE SOFR_BUNDLE_JSON="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/sofr_bundle.json")

# THE SHAPE LADDER (E4.D acceptance): per-shape stream tick / refresh tick / Jacobian, gated per rung.
add_executable(shape_ladder_bench shape_ladder_bench.cpp)
target_link_libraries(shape_ladder_bench PRIVATE swaps swaps_api benchmark::benchmark)
target_include_directories(shape_ladder_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

# The router's per-row partition vs the whole-bundle veto it replaced, on every rung of the same ladder.
add_executable(routing_bench routing_bench.cpp)
target_link_libraries(routing_bench PRIVATE swaps swaps_api benchmark::benchmark)
target_include_directories(routing_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

# EXPERIMENT (exp/piecewise-linear-w): one-pass coefficient map vs region-by-region set_forwards.
add_executable(exp_onepass_bench exp_onepass_bench.cpp)
target_link_libraries(exp_onepass_bench PRIVATE swaps benchmark::benchmark)

# EXPERIMENT (exp/piecewise-linear-w): pattern-flip frequency + per-tick cost of the PWL tier under a streaming feed.
add_executable(exp_stream_flip exp_stream_flip.cpp)
target_link_libraries(exp_stream_flip PRIVATE swaps swaps_api)
target_include_directories(exp_stream_flip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

# EXPERIMENT (exp/piecewise-linear-w): differential check of the analytic re-take against a fresh engine.
add_executable(exp_pwl_check exp_pwl_check.cpp)
target_link_libraries(exp_pwl_check PRIVATE swaps swaps_api)
target_include_directories(exp_pwl_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

# EXPERIMENT (exp/piecewise-linear-w): step-by-step repro of the gate-tick convergence test on mixed_scheme.
add_executable(exp_session_repro exp_session_repro.cpp)
target_link_libraries(exp_session_repro PRIVATE swaps swaps_api)
target_include_directories(exp_session_repro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)
