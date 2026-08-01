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
  build_instruments_test.cpp
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

# Oracle-linked tests: these self-check the reference market against QuantLib itself. Kept in a
# SEPARATE binary so the engine's own gate (swaps_tests) never links QuantLib.
if(TARGET quantlib_oracle)
  add_executable(swaps_oracle_tests
    convexity_test.cpp
    pricing_test.cpp
    generic_cashflow_test.cpp
    extract_test.cpp
    calibration_test.cpp
    aad_test.cpp
    risk_test.cpp
    portfolio_test.cpp
    spread_test.cpp
    warm_test.cpp
    streaming_test.cpp
    bundle_test.cpp
    compiled_residual_test.cpp
    bspline_oracle_test.cpp
    monotone_cubic_oracle_test.cpp
    rfr_coupon_oracle_test.cpp
    ois_weekend_test.cpp
    migration_guard_test.cpp
    multicurrency_test.cpp
    conventions_test.cpp
    tension_regularizer_oracle_test.cpp
    tension_oracle_test.cpp
    turns_oracle_test.cpp
  )
  target_link_libraries(swaps_oracle_tests PRIVATE swaps quantlib_oracle GTest::gtest_main)
  target_include_directories(swaps_oracle_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  gtest_discover_tests(swaps_oracle_tests)
endif()
