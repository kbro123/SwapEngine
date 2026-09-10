// @oracle-test (support fixture) — clears QuantLib's process-global IndexManager fixing histories
// E5 taxonomy: T4 hot-path invariant (allocation / determinism / structure)
// between oracle tests so the single-process swaps_oracle_tests binary is order-independent. DO NOT DELETE.
// Per-test isolation of QuantLib's PROCESS-GLOBAL state for the single-process oracle binary.
//
// QuantLib keeps index fixing histories in a global singleton (IndexManager). A test that seeds SOFR/OIS
// fixings leaves them there, so a LATER test that adds a fixing in the same span hits addFixing's
// "duplicate with a different value" guard and throws at setup — the monolithic swaps_oracle_tests binary
// failed ~16 tests purely by ordering, even though every one passed in isolation and under ctest
// (gtest_discover_tests runs each in its own process, so the perf/verify gate never saw this). A few tests
// already worked around it by calling clearHistory() defensively (see tests/ois_weekend_test.cpp); this makes
// the isolation universal.
//
// A gtest listener clears ALL fixing histories before every test — restoring per-test isolation without
// editing any test body. Registered at static-init (before gtest_main's RUN_ALL_TESTS), which is the
// supported way to attach a listener under the default main.
#include <gtest/gtest.h>

#include <ql/indexes/indexmanager.hpp>

namespace {

struct QLGlobalReset : ::testing::EmptyTestEventListener {
  void OnTestStart(const ::testing::TestInfo& /*info*/) override {
    QuantLib::IndexManager::instance().clearHistories();
  }
};

const bool kQLGlobalResetRegistered = [] {
  ::testing::UnitTest::GetInstance()->listeners().Append(new QLGlobalReset);
  return true;
}();

}  // namespace
