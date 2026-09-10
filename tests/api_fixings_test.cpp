// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// E2: fixings wired into the BundleSession pricing context (the binding/web will drive this exact flow).
// A Rate observation carries a fixing SCHEDULE (index + per-day dates/accruals); the session resolves it
// against its evaluation date + fixing table — past days -> realized, future days -> forecast subs — and
// re-resolves in place on set_fixings, with NO recompile (realized is a residual constant). Proves:
//   * build the session with NO table -> the schedule-carrying obs is un-priceable (n_unresolved > 0);
//   * set_fixings for the past days -> resolves; realized == Σ fixing·accrual; future days stay subs;
//   * a later fixing for TODAY -> the same obs re-resolves, pulling today into realized.

#include <boost/json.hpp>

#include <gtest/gtest.h>

#include "swaps/api/bundle_api.hpp"

namespace js = boost::json;
using swaps::api::BundleSession;
using swaps::api::bundle_from_json;

namespace {
// One curve + one Rate instrument whose observation carries a 5-day schedule (serials 97..101, accrual
// 1/360 each). Everything before the eval date (100) is a past fixing; 100/101 are forecast until fixed.
const char* kBundle = R"({
  "curves":[{"meeting":[],"back":[1.0,2.0],"base":-1,"currency":0}],
  "instruments":[{"quote":"Rate","forecast":0,"market":0.04,
    "obs":{"tau_index":0.0138889,"fixing_index":"USD-SOFR","fixing_schedule":[
      {"fixing_date":97, "accrual":0.0027778,"t_start":0.000,"t_end":0.001,"weight":1.0},
      {"fixing_date":98, "accrual":0.0027778,"t_start":0.001,"t_end":0.002,"weight":1.0},
      {"fixing_date":99, "accrual":0.0027778,"t_start":0.002,"t_end":0.003,"weight":1.0},
      {"fixing_date":100,"accrual":0.0027778,"t_start":0.003,"t_end":0.004,"weight":1.0},
      {"fixing_date":101,"accrual":0.0027778,"t_start":0.004,"t_end":0.005,"weight":1.0}]}}]})";
}  // namespace

TEST(ApiFixings, SessionResolvesFromTableAndReResolvesOnUpdate) {
  BundleSession s(bundle_from_json(js::parse(kBundle)));

  // Eval date set, but no fixings yet -> the past days (97,98,99) can't resolve -> un-priceable.
  s.set_evaluation_date(100);
  EXPECT_EQ(s.n_unresolved(), 1);

  const double r = 0.043, acc = 0.0027778;
  EXPECT_EQ(s.set_fixings("USD-SOFR", {{97, r}, {98, r}, {99, r}}), 0);  // all resolved now
  EXPECT_EQ(s.n_unresolved(), 0);

  const auto& obs = s.problem().instruments[0].obs;
  EXPECT_NEAR(obs.realized, 3 * r * acc, 1e-12);   // 97,98,99 summed into realized
  EXPECT_EQ(obs.sub_start.size(), 2u);             // 100 (today, unfixed) and 101 still forecast

  // Today's fixing arrives -> the SAME observation re-resolves, pulling day 100 into realized.
  EXPECT_EQ(s.set_fixings("USD-SOFR", {{100, r}}), 0);
  const auto& obs2 = s.problem().instruments[0].obs;
  EXPECT_NEAR(obs2.realized, 4 * r * acc, 1e-12);
  EXPECT_EQ(obs2.sub_start.size(), 1u);            // only day 101 remains forecast
}

TEST(ApiFixings, NoScheduleBundleIsUnaffected) {
  // A bundle whose observation carries NO schedule (the legacy baked path) reports nothing to resolve.
  const char* baked = R"({
    "curves":[{"meeting":[],"back":[1.0],"base":-1,"currency":0}],
    "instruments":[{"quote":"Rate","forecast":0,"market":0.04,
      "obs":{"tau_index":0.25,"realized":0.01,"sub_start":[0.1],"sub_end":[0.25]}}]})";
  BundleSession s(bundle_from_json(js::parse(baked)));
  s.set_evaluation_date(100);
  EXPECT_EQ(s.n_unresolved(), 0);
  EXPECT_DOUBLE_EQ(s.problem().instruments[0].obs.realized, 0.01);  // untouched
}
