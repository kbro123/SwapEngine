# Public API facade: JSON <-> engine object graph + the BundleSession (calibrate / sample / price /
# risk / stream). QuantLib-FREE (it consumes the generic instrument model), but needs the vendored
# Boost headers for Boost.JSON.

add_library(swaps_api STATIC bundle_api.cpp)
target_link_libraries(swaps_api PUBLIC swaps)
target_include_directories(swaps_api PUBLIC ${SWAPS_THIRD_PARTY}/boost)
target_compile_features(swaps_api PUBLIC cxx_std_20)

# CLI driver (file/stdin -> JSON response).
add_executable(swaps_api_cli swaps_api_cli.cpp)
target_link_libraries(swaps_api_cli PRIVATE swaps_api)

# Self-contained round-trip test (no QuantLib): hand-built bundle -> JSON -> back -> calibrate.
if(SWAPS_BUILD_TESTS AND GTest_FOUND)
  add_executable(swaps_api_tests ${CMAKE_CURRENT_SOURCE_DIR}/../tests/api_test.cpp)
  target_link_libraries(swaps_api_tests PRIVATE swaps_api GTest::gtest_main)
  target_include_directories(swaps_api_tests PRIVATE ${SWAPS_THIRD_PARTY}/boost)
  gtest_discover_tests(swaps_api_tests)
endif()
