// E5 taxonomy: T6 regression (fails on the reverted bug)
// FLK1 REPRODUCTION (2026-09-14). bench/fixtures/malloc_count.hpp's AllocScope armed libmalloc's PROCESS-WIDE logger, so
// the hot-path allocation pins (tests/hotpath_shapes_test.cpp ShapeLadder desk_mixed: 3600 small-tick allocations) also
// counted the streaming calibrator's BackgroundJacobian worker thread (calibration/background_jacobian.hpp: J + complete
// orthogonal decomposition, ~15k Eigen allocations) whenever a loaded machine delayed that off-thread refresh into the
// 20-tick window -- four sightings in one session's ctest -j 8 runs, exact 3600 in every serial run. The pin measures the
// TICK; the refresh is off the critical thread by design. So a scope must count only the thread that armed it.
//
// Deterministic: the worker is started BEFORE the scope (a thread start allocates), and the handshake uses atomics and
// std::this_thread::yield, neither of which allocates on the arming thread. Every allocation goes through std::malloc via a
// VOLATILE function pointer into a kept buffer: at -O3 clang may elide a paired new/delete or an unused std::vector
// (heap-allocation elision), which made a first draft of this test count nothing on either thread.
#include <array>
#include <atomic>
#include <cstdlib>
#include <thread>

#include <gtest/gtest.h>

#include "malloc_count.hpp"  // bench/fixtures (an include dir of swaps_tests, mirrored by tools/mutate.py)

namespace {
using alloc_fn = void* (*)(std::size_t);
using free_fn = void (*)(void*);
alloc_fn volatile g_alloc = std::malloc;  // volatile: the compiler must call through it (no elision)
free_fn volatile g_free = std::free;
constexpr std::size_t kN = 64;
}  // namespace

TEST(AllocScopeThreadRepro, AnotherThreadsAllocationsAreNotCountedInTheScope) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  std::array<void*, kN> kept{};
  std::atomic<bool> go{false}, done{false};
  std::thread worker([&] {
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    for (std::size_t i = 0; i < kN; ++i) kept[i] = g_alloc(4096 + i);  // heap allocations on the WORKER thread
    done.store(true, std::memory_order_release);
  });
  unsigned long counted = 0;
  {
    swaps::testing::AllocScope scope;
    go.store(true, std::memory_order_release);
    while (!done.load(std::memory_order_acquire)) std::this_thread::yield();
    counted = scope.allocs();
  }
  worker.join();
  for (void* p : kept) g_free(p);
  ASSERT_NE(kept[kN - 1], nullptr) << "premise: the worker really allocated";
  EXPECT_EQ(counted, 0u) << "the scope counted another thread's allocations (FLK1)";
}

TEST(AllocScopeThreadRepro, ControlTheArmingThreadsOwnAllocationsAreStillCounted) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  std::array<void*, kN> kept{};
  unsigned long counted = 0;
  {
    swaps::testing::AllocScope scope;
    for (std::size_t i = 0; i < kN; ++i) kept[i] = g_alloc(4096 + i);  // on THIS (the arming) thread
    counted = scope.allocs();
  }
  for (void* p : kept) g_free(p);
  EXPECT_GE(counted, kN) << "the arming thread's own allocations must be counted";
}
