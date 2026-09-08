# archprobe — the ONE way to learn the engine's arch flag outside the main build.
# tools/bootstrap_deps.sh and tools/fingerprint.sh configure this tiny project so that QuantLib (the perf
# reference) and the perf-baseline fingerprint use EXACTLY the flag cmake/DetectISA.cmake gives the engine
# (script-mode cmake cannot try_compile, so a real configure is the only faithful probe).
cmake_minimum_required(VERSION 3.20)
project(archprobe LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)
include(DetectISA)
file(WRITE ${CMAKE_BINARY_DIR}/arch_flags.txt "${SWAPS_ARCH_FLAGS}\n")
file(WRITE ${CMAKE_BINARY_DIR}/isa_name.txt "${SWAPS_ISA_NAME}\n")
file(WRITE ${CMAKE_BINARY_DIR}/packet.txt "${SWAPS_DOUBLE_PACKET_SIZE}\n")
message(STATUS "archprobe: ${SWAPS_ARCH_FLAGS} (${SWAPS_ISA_NAME}, ${SWAPS_DOUBLE_PACKET_SIZE} doubles/reg)")
