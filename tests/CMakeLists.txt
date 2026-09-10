# Correctness gate — GoogleTest.
# Tests compare the engine against the QuantLib oracle within tests/tolerances.hpp.

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(swaps_tests
  smoke_test.cpp
  multi_region_test.cpp
  region_combinatorial_test.cpp
  bspline_test.cpp
  monotone_cubic_test.cpp
  live_curve_test.cpp
  parallel_portfolio_test.cpp
  compiled_generic_test.cpp
  generic_instrument_test.cpp
  portfolio_instrument_test.cpp
  fixings_test.cpp
  tension_regularizer_test.cpp
  tension_test.cpp
  turns_test.cpp
  build_calendar_test.cpp
  calendar_data_test.cpp
  build_instruments_test.cpp
  instrument_params_test.cpp
  structure_fingerprint_test.cpp
  market_currency_test.cpp
  market_fx_test.cpp
  market_quote_test.cpp
  build_tenor_test.cpp
  market_env_test.cpp
  trade_test.cpp
  market_scenario_test.cpp
  market_fixing_series_test.cpp
  market_vol_surface_test.cpp
  trade_book_test.cpp
  compiled_multi_test.cpp
  schedule_stub_eom_test.cpp
  rfr_lag_test.cpp
  stepped_principal_test.cpp
  aad_block_pooled_test.cpp
  asset_swap_spread_test.cpp
  derive_asset_swap_test.cpp
  bond_yield_test.cpp
  bond_reference_test.cpp
  vol_bachelier_test.cpp
  vol_sabr_test.cpp
  vol_sabr_calibration_test.cpp
  vol_sabr_beta_test.cpp
  fx_black_test.cpp
  inflation_test.cpp
  bond_future_test.cpp
  ndf_test.cpp
  credit_test.cpp
  bug_hunt_2026_09_test.cpp
  streaming_band_test.cpp
  streaming_contract_test.cpp
  calibration_status_test.cpp
  kernel_edges_test.cpp
  moment_quadrature_test.cpp
  rank_safety_test.cpp
  bus252_daycount_test.cpp
  zero_coupon_swap_test.cpp
  conventions_property_test.cpp
  curve_property_test.cpp
  fx_black_grid_test.cpp
  rates_fx_grid_test.cpp
  vol_cap_stripping_test.cpp
  vol_cube_interp_test.cpp
  vol_swaption_test.cpp
  vega_ladder_test.cpp
  vol_cms_test.cpp
  xva_exposure_test.cpp
  gamma_test.cpp
  kernel_pins_test.cpp
  cds_closed_form_test.cpp
)
target_link_libraries(swaps_tests PRIVATE swaps GTest::gtest_main)
target_include_directories(swaps_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Golden data lives in the source tree and is committed.
target_compile_definitions(swaps_tests PRIVATE
  SWAPS_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")

gtest_discover_tests(swaps_tests)

# Allocation-free hot-path proof -- SEPARATE executable: this TU compiles Eigen with
# EIGEN_RUNTIME_NO_MALLOC, which must not ODR-clash with the normally-compiled Eigen in swaps_tests.
add_executable(swaps_allocfree_tests alloc_free_test.cpp)
target_link_libraries(swaps_allocfree_tests PRIVATE swaps GTest::gtest_main)
target_include_directories(swaps_allocfree_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(swaps_allocfree_tests)

# ---------------------------------------------------------------------------------------------------
# QuantLib-linked binaries. QuantLib is REQUIRED for the gate (root CMakeLists.txt makes its absence a
# configure error unless -DSWAPS_ALLOW_NO_ORACLE=ON), so these targets always exist in a gated build.
#
# swaps_oracle_tests  — ORACLE tests: every file compares an ENGINE number to a QUANTLIB number
#                       (PRINCIPLES.md P8). The list is the registry enforced by tools/check_oracle_tests.sh
#                       (banner + QuantLib reference + committed assertion counts in oracle_assertions.lock).
# swaps_consistency_tests — QuantLib-LINKED SELF-CONSISTENCY tests: they use QuantLib only to BUILD the
#                       reference market/schedules and then compare the engine to ITSELF (cross-path parity,
#                       solver recovery of a self-generated market, warm vs cold LM, hand formulas). Valuable,
#                       protected by the same lock, but NOT oracles — until 2026-09-08 they carried the
#                       @oracle-test banner and ORACLE_TESTS.md described them as "vs QuantLib".
# ---------------------------------------------------------------------------------------------------
if(TARGET quantlib_oracle)
  add_executable(swaps_oracle_tests
    ql_test_isolation.cpp
    convexity_test.cpp
    pricing_test.cpp
    modular_curve_test.cpp
    bond_oracle_test.cpp
    bond_asset_swap_oracle.cpp
    extract_test.cpp
    bundle_test.cpp
    bspline_oracle_test.cpp
    vol_bachelier_ql_oracle.cpp
    vol_cms_replication_oracle.cpp
    monotone_cubic_oracle_test.cpp
    rfr_coupon_oracle_test.cpp
    ois_weekend_test.cpp
    multicurrency_test.cpp
    conventions_test.cpp
    tension_oracle_test.cpp
    turns_oracle_test.cpp
    scheme_value_oracle_test.cpp
    sabr_ql_oracle_test.cpp
    calendar_ql_oracle_test.cpp
  )
  target_link_libraries(swaps_oracle_tests PRIVATE swaps quantlib_oracle GTest::gtest_main)
  target_include_directories(swaps_oracle_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  gtest_discover_tests(swaps_oracle_tests)

  add_executable(swaps_consistency_tests
    ql_test_isolation.cpp
    generic_cashflow_test.cpp
    calibration_test.cpp
    aad_test.cpp
    risk_test.cpp
    portfolio_test.cpp
    spread_test.cpp
    warm_test.cpp
    streaming_test.cpp
    compiled_residual_test.cpp
    migration_guard_test.cpp
    tension_regularizer_oracle_test.cpp
  )
  target_link_libraries(swaps_consistency_tests PRIVATE swaps quantlib_oracle GTest::gtest_main)
  target_include_directories(swaps_consistency_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  gtest_discover_tests(swaps_consistency_tests)
endif()
