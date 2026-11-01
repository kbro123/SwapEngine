// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Fixing tables as part of the pricing context (design: attach-and-observe). Proves the E1 engine core:
//   * resolve() splits an observation at the eval date, sums PAST fixings from the table into `realized`
//     (compounded -> `realized_factor`) and leaves FUTURE days as forecast sub-periods — bit-identical to
//     the hand-baked observation the market layer builds today (tests/reference_curve.hpp);
//   * a missing PAST fixing THROWS MissingFixing (pricing can't silently proceed);
//   * the schedule survives resolution, so a later table or evaluation date re-resolves the same observation
//     (the session's resolve_fixings; the attach-and-observe BoundObservation design that used to sit beside it
//     had no production reader and was deleted on 2026-11-01, review item 1.5).
//
// Pure engine (no QuantLib): dates are integer serials.

#include <gtest/gtest.h>

#include "swaps/pricing/fixings.hpp"

using namespace swaps::pricing;

namespace {

// A 5-day averaged RFR observation carrying its fixing schedule: fixing serials 97,98,99,100,101 (eval = 100),
// accrual 1/360 each, forecast sub-periods for the future days. Weights are 1.0 here (so resolution clears the
// weight vector).
RateObservation make_obs(bool compounded = false) {
  RateObservation o;
  o.fixing_index = "USD-SOFR";
  o.tau_index = 5.0 / 360.0;
  o.compounded = compounded;
  for (int i = 0; i < 5; ++i) {
    FixingDay d;
    d.fixing_date = 97 + i;
    d.accrual = 1.0 / 360.0;
    d.t_start = 0.001 * i;           // arbitrary but ordered curve times for the forecast subs
    d.t_end = 0.001 * (i + 1);
    d.weight = 1.0;
    o.fixing_schedule.push_back(d);
  }
  return o;
}
// resolve_into on a copy: the resolved observation.
RateObservation resolved(RateObservation o, const PricingContext& ctx) {
  resolve_into(o, ctx);
  return o;
}

const double kR97 = 0.043, kR98 = 0.044, kR99 = 0.045, kR100 = 0.046;

void seed_past(FixingTable& t) {  // 97,98,99 are strictly before eval=100
  t.bulk_set("USD-SOFR", {{97, kR97}, {98, kR98}, {99, kR99}});
}

}  // namespace

TEST(Fixings, AveragedResolvesPastIntoRealizedFutureIntoSubs) {
  FixingTable table;
  seed_past(table);
  PricingContext ctx{/*eval=*/100, &table};

  const RateObservation obs = resolved(make_obs(), ctx);

  // 97,98,99 are past -> realized; 100 (today, no fixing) and 101 are future -> two sub-periods.
  EXPECT_DOUBLE_EQ(obs.realized, (kR97 + kR98 + kR99) / 360.0);
  ASSERT_EQ(obs.sub_start.size(), 2u);
  EXPECT_DOUBLE_EQ(obs.sub_start[0], 0.003);   // the day-100 sub-period (i=3)
  EXPECT_DOUBLE_EQ(obs.sub_end[1], 0.005);     // the day-101 sub-period end (i=4)
  EXPECT_TRUE(obs.weight.empty());             // all-ones -> cleared (bit-exact standard reduction)
  EXPECT_EQ(obs.tau_index, 5.0 / 360.0);
  EXPECT_FALSE(obs.compounded);
}

TEST(Fixings, TodayCountsAsPastOnlyWhenAFixingExists) {
  FixingTable table;
  seed_past(table);
  table.set("USD-SOFR", 100, kR100);              // now day 100 has a fixing -> it's realized
  PricingContext ctx{100, &table};

  const RateObservation obs = resolved(make_obs(), ctx);
  EXPECT_DOUBLE_EQ(obs.realized, (kR97 + kR98 + kR99 + kR100) / 360.0);
  ASSERT_EQ(obs.sub_start.size(), 1u);            // only day 101 remains forecast
}

TEST(Fixings, MissingPastFixingThrows) {
  FixingTable table;
  table.bulk_set("USD-SOFR", {{97, kR97}, {99, kR99}});  // 98 missing
  PricingContext ctx{100, &table};
  try {
    resolved(make_obs(), ctx);
    FAIL() << "expected MissingFixing";
  } catch (const MissingFixing& e) {
    EXPECT_EQ(e.index, "USD-SOFR");
    EXPECT_EQ(e.date, 98);
  }
}

TEST(Fixings, NoTableAttachedIsMissingForPastDays) {
  PricingContext ctx{100, nullptr};               // built WITHOUT a fixing table
  EXPECT_THROW(resolved(make_obs(), ctx), MissingFixing);
}

TEST(Fixings, CompoundedUsesRealizedFactorProduct) {
  FixingTable table;
  seed_past(table);
  PricingContext ctx{100, &table};
  const RateObservation obs = resolved(make_obs(/*compounded=*/true), ctx);
  const double expect_rf = (1 + kR97 / 360.0) * (1 + kR98 / 360.0) * (1 + kR99 / 360.0);
  EXPECT_TRUE(obs.compounded);
  EXPECT_DOUBLE_EQ(obs.realized_factor, expect_rf);
  EXPECT_DOUBLE_EQ(obs.realized, 0.0);
}

TEST(Fixings, ResolvedEqualsBakedObservation) {
  // The whole point: a resolved observation is bit-identical to the one the market layer bakes by hand.
  FixingTable table;
  seed_past(table);
  PricingContext ctx{100, &table};
  const RateObservation got = resolved(make_obs(), ctx);

  RateObservation baked;                          // built directly, the "old" way
  baked.tau_index = 5.0 / 360.0;
  baked.realized = (kR97 + kR98 + kR99) / 360.0;
  baked.sub_start = {0.003, 0.004};
  baked.sub_end = {0.004, 0.005};

  EXPECT_DOUBLE_EQ(got.realized, baked.realized);
  EXPECT_EQ(got.sub_start, baked.sub_start);
  EXPECT_EQ(got.sub_end, baked.sub_end);
  EXPECT_EQ(got.weight, baked.weight);            // both empty
  EXPECT_EQ(got.tau_index, baked.tau_index);
}

// The schedule SURVIVES resolution: the same observation re-resolves against a later table / date (what the
// session's resolve_fixings relies on), and a missing fixing names the first past day it could not resolve.
TEST(Fixings, AResolvedObservationReResolvesAgainstALaterTable) {
  RateObservation o = make_obs();
  FixingTable table;
  table.bulk_set("USD-SOFR", {{97, kR97}, {99, kR99}});  // 98 missing
  try {
    resolve_into(o, PricingContext{100, &table});
    FAIL() << "expected MissingFixing";
  } catch (const MissingFixing& e) {
    EXPECT_EQ(e.date, 98);
  }
  table.set("USD-SOFR", 98, kR98);
  resolve_into(o, PricingContext{100, &table});
  EXPECT_TRUE(o.resolved);
  EXPECT_DOUBLE_EQ(o.realized, (kR97 + kR98 + kR99) / 360.0);
  ASSERT_EQ(o.sub_start.size(), 2u);
  table.set("USD-SOFR", 100, kR100);  // a later fixing for TODAY: re-resolving pulls day 100 into realized too
  resolve_into(o, PricingContext{100, &table});
  EXPECT_DOUBLE_EQ(o.realized, (kR97 + kR98 + kR99 + kR100) / 360.0);
  EXPECT_EQ(o.sub_start.size(), 1u);  // only day 101 still forecast
  EXPECT_EQ(o.fixing_schedule.size(), 5u) << "the schedule is preserved";
}

// The engine's ONE integer date convention: Unix-day serials (days since 1970-01-01). Until 2026-09-09 the
// builders emitted Python ordinals (+719163) into FixingDay while FixingSeries used serials; a table now
// refuses anything outside the plausible market-date range so the two stores can never disagree silently.
TEST(Fixings, TableRejectsDatesThatAreNotUnixDaySerials) {
  FixingTable t;
  EXPECT_NO_THROW(t.set("USD-SOFR", 20454, 0.04));                         // 2026-01-01
  EXPECT_THROW(t.set("USD-SOFR", 739617, 0.04), std::invalid_argument);    // a Python date.toordinal()
  EXPECT_THROW(t.set("USD-SOFR", 20260101, 0.04), std::invalid_argument);  // a YYYYMMDD integer
  EXPECT_THROW(t.bulk_set("USD-SOFR", {{20454, 0.04}, {739617, 0.04}}), std::invalid_argument);
  EXPECT_EQ(t.size("USD-SOFR"), 1u);
}

// 2026-09-14 (SW1): a day that STARTS before curve time 0 -- the value date its schedule was built on -- is in the past
// whatever the evaluation date says. With no evaluation date and no table it must throw, never be forecast; a day after
// curve time 0 still forecasts. tests/seasoned_eval_date_repro_test.cpp reproduces the silent forecast end to end.
TEST(FixingsResolve, ADayBeforeCurveTimeZeroIsPastWithoutAnEvaluationDate) {
  swaps::pricing::RateObservation o;
  o.fixing_index = "USD-SOFR";
  o.tau_index = 1.0;
  o.compounded = true;
  o.fixing_schedule.push_back(swaps::pricing::FixingDay{20279, 1.0 / 360.0, -0.99, -0.987, 1.0});
  EXPECT_THROW(swaps::pricing::resolve_into(o, swaps::pricing::PricingContext{}), swaps::pricing::MissingFixing);
  o.fixing_schedule[0].t_start = 0.01;
  o.fixing_schedule[0].t_end = 0.013;
  EXPECT_NO_THROW(swaps::pricing::resolve_into(o, swaps::pricing::PricingContext{})) << "a future day forecasts";
}
