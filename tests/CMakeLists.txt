# Correctness gate — GoogleTest.
# Tests compare the engine against the QuantLib oracle within tests/tolerances.hpp.

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(swaps_tests
  smoke_test.cpp
  curve_test.cpp
)
target_link_libraries(swaps_tests PRIVATE swaps GTest::gtest_main)
target_include_directories(swaps_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Golden data lives in the source tree and is committed.
target_compile_definitions(swaps_tests PRIVATE
  SWAPS_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")

gtest_discover_tests(swaps_tests)

# Oracle-linked tests: these self-check the reference market against QuantLib itself. Kept in a
# SEPARATE binary so the engine's own gate (swaps_tests) never links QuantLib.
if(TARGET quantlib_oracle)
  add_executable(swaps_oracle_tests convexity_test.cpp)
  target_link_libraries(swaps_oracle_tests PRIVATE swaps quantlib_oracle GTest::gtest_main)
  target_include_directories(swaps_oracle_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  gtest_discover_tests(swaps_oracle_tests)
endif()
