# Correctness gate — GoogleTest.
# Tests compare the engine against the QuantLib oracle within tests/tolerances.hpp.
# Phase 0 ships only a smoke test that proves the engine + Eigen + GTest wiring builds.

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(swaps_tests
  smoke_test.cpp
)
target_link_libraries(swaps_tests PRIVATE swaps GTest::gtest_main)

# Later phases add QuantLib-backed oracle tests; link QuantLib only where needed:
#   target_link_libraries(swaps_oracle_tests PRIVATE swaps GTest::gtest_main ${QUANTLIB_LIB})

gtest_discover_tests(swaps_tests)
