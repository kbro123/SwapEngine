# Public API facade: JSON <-> engine object graph + the BundleSession (calibrate / sample / price /
# risk / stream). QuantLib-FREE (it consumes the generic instrument model), but needs the vendored
# Boost headers for Boost.JSON.

add_library(swaps_api STATIC bundle_api.cpp compile.cpp generate_risk.cpp options.cpp bond.cpp rv.cpp exposure.cpp scenario.cpp scenario_grid.cpp var.cpp pnl.cpp vega.cpp inflation.cpp fx_option.cpp bond_future.cpp ndf.cpp calib_report.cpp credit.cpp conventions.cpp swaps_capi.cpp)
target_link_libraries(swaps_api PUBLIC swaps)
target_include_directories(swaps_api PUBLIC ${SWAPS_THIRD_PARTY}/boost)
target_compile_features(swaps_api PUBLIC cxx_std_20)

# CLI driver (file/stdin -> JSON response).
add_executable(swaps_api_cli swaps_api_cli.cpp)
target_link_libraries(swaps_api_cli PRIVATE swaps_api)

# C-ABI SHARED library: the single symbol (swaps_run_json, capi.h) a NON-C++ host loads -- an Excel XLL,
# a .NET P/Invoke wrapper, or Python ctypes. All the risk machinery lives behind the JSON seam it forwards
# to. Optional (SWAPS_BUILD_CAPI_SHARED, default ON); it links the same swaps_api the pybind module uses.
option(SWAPS_BUILD_CAPI_SHARED "Build the C-ABI shared library (libswaps_capi) for non-C++ hosts" ON)
if(SWAPS_BUILD_CAPI_SHARED)
  add_library(swaps_capi_shared SHARED swaps_capi.cpp)
  target_link_libraries(swaps_capi_shared PRIVATE swaps_api)
  set_target_properties(swaps_capi_shared PROPERTIES OUTPUT_NAME swaps_capi)
endif()

# Self-contained round-trip test (no QuantLib): hand-built bundle -> JSON -> back -> calibrate.
if(SWAPS_BUILD_TESTS AND GTest_FOUND)
  add_executable(swaps_api_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/api_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/api_fixings_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/api_status_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/api_state_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/portfolio_reprice_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/vol_cube_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/compile_parity_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/exposure_netting_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/scenario_verb_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/scenario_grid_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/var_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/calib_report_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/pnl_explain_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/portfolio_compiled_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/hotpath_shapes_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/compiled_mtm_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/conventions_registry_test.cpp)
  target_link_libraries(swaps_api_tests PRIVATE swaps_api GTest::gtest_main)
  target_include_directories(swaps_api_tests PRIVATE ${SWAPS_THIRD_PARTY}/boost ${CMAKE_SOURCE_DIR}/bench/fixtures)
  # Golden compiler-parity data (server/compile.py output), committed under tests/golden/compile/; and the
  # SOFR bundle fixture the vol-cube test calibrates against (also used by bench/vol_cube_bench).
  target_compile_definitions(swaps_api_tests PRIVATE
    SWAPS_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/golden"
    SOFR_BUNDLE_JSON="${CMAKE_CURRENT_SOURCE_DIR}/../bench/fixtures/sofr_bundle.json")
  gtest_discover_tests(swaps_api_tests)
endif()
