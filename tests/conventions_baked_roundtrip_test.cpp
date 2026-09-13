// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// PIN BEFORE THE E7 STAGE-4 LIFT of the `conventions` verb. The baked conventions DB reaches the engine by TWO
// independent paths: tools/gen_conventions_hpp.py (Python) turns conventions/conventions.json into the generated
// kProducts / kIndices / ... arrays at build time, and the runtime `conventions` verb (api/conventions.cpp) parses the
// SAME rows in C++. Posting conventions.json verbatim through the verb must give overlay rows EQUAL, field for field,
// to the generated arrays -- a cross-check of the generator against the runtime parser that neither can pass alone.
// Every family first proves its ids really landed in the OVERLAY (the lookups fall back to the baked row, so a row
// the verb silently dropped would otherwise compare equal to itself). Doubles compare exactly: both paths parse the
// same decimal text with correct rounding.
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "swaps/api/conventions.hpp"
#include "swaps/conventions_data.hpp"

namespace api = swaps::api;
namespace cvd = swaps::conventions;

namespace {
struct BakedRoundTrip : ::testing::Test {
  void SetUp() override { cvd::Registry::instance().clear_overlay(); }
  void TearDown() override { cvd::Registry::instance().clear_overlay(); }
};

std::string read_file(const char* path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool in_overlay(const cvd::Registry::Listing& l, std::string_view id) {
  return std::find(l.overlay.begin(), l.overlay.end(), std::string(id)) != l.overlay.end();
}

#define EQ(f) EXPECT_EQ(got.f, want.f) << where << " ." #f

void eq_leg(const cvd::LegConv& got, const cvd::LegConv& want, const std::string& where) {
  EQ(index);
  EQ(day_count);
  EQ(frequency);
  EQ(compounding);
  EQ(carries_spread);
  EQ(notional_resets);
  EQ(flat);
}
}  // namespace

TEST_F(BakedRoundTrip, TheBakedDatabasePostedThroughTheVerbEqualsTheGeneratedArrays) {
  const std::string doc = read_file(SWAPS_CONVENTIONS_JSON);
  ASSERT_FALSE(doc.empty()) << SWAPS_CONVENTIONS_JSON;
  api::conventions_json(doc);  // the whole file: its top-level keys are the families
  const cvd::Registry& R = cvd::Registry::instance();

  {
    const auto l = R.list_products();
    ASSERT_EQ(l.overlay.size(), cvd::kProducts.size());
    for (const cvd::ProductConv& want : cvd::kProducts) {
      const std::string where = "product " + std::string(want.id);
      ASSERT_TRUE(in_overlay(l, want.id)) << where;
      const cvd::ProductConv got = *R.product(want.id);
      EQ(id); EQ(type); EQ(currency); EQ(calendar); EQ(bdc); EQ(frequency); EQ(discount_index); EQ(pair);
      EQ(base_currency); EQ(spot_lag); EQ(payment_lag); EQ(zero_coupon);
      eq_leg(got.fixed, want.fixed, where + " fixed");
      eq_leg(got.floating, want.floating, where + " floating");
      eq_leg(got.other, want.other, where + " other");
    }
  }
  {
    const auto l = R.list_indices();
    ASSERT_EQ(l.overlay.size(), cvd::kIndices.size());
    for (const cvd::IndexConv& want : cvd::kIndices) {
      const std::string where = "index " + std::string(want.id);
      ASSERT_TRUE(in_overlay(l, want.id)) << where;
      const cvd::IndexConv got = *R.index(want.id);
      EQ(id); EQ(currency); EQ(type); EQ(day_count); EQ(calendar); EQ(par_product); EQ(tenor); EQ(fixing_lag);
      EQ(publication_lag);
    }
  }
  {
    const auto l = R.list_currencies();
    ASSERT_EQ(l.overlay.size(), cvd::kCurrencies.size());
    for (const cvd::CurrencyConv& want : cvd::kCurrencies) {
      const std::string where = "currency " + std::string(want.code);
      ASSERT_TRUE(in_overlay(l, want.code)) << where;
      const cvd::CurrencyConv got = *R.currency(want.code);
      EQ(code); EQ(name); EQ(settlement_calendar); EQ(discount_index); EQ(default_swap_product); EQ(repo_day_count);
      EQ(minor_units);
    }
  }
  {
    const auto l = R.list_credit_products();
    ASSERT_EQ(l.overlay.size(), cvd::kCredit.size());
    for (const cvd::CreditConv& want : cvd::kCredit) {
      const std::string where = "cds product " + std::string(want.id);
      ASSERT_TRUE(in_overlay(l, want.id)) << where;
      const cvd::CreditConv got = *R.credit_product(want.id);
      EQ(id); EQ(currency); EQ(calendar); EQ(day_count); EQ(frequency); EQ(roll); EQ(recovery_default);
      EQ(settlement_lag); EQ(protection_steps);
    }
  }
  {
    const auto l = R.list_bond_futures();
    ASSERT_EQ(l.overlay.size(), cvd::kBondFutures.size());
    for (const cvd::BondFutureConv& want : cvd::kBondFutures) {
      const std::string where = "bond future " + std::string(want.id);
      ASSERT_TRUE(in_overlay(l, want.id)) << where;
      const cvd::BondFutureConv got = *R.bond_future(want.id);
      EQ(id); EQ(currency); EQ(exchange_calendar); EQ(deliverable_convention); EQ(repo_day_count); EQ(delivery);
      EQ(notional_coupon); EQ(basket_min_years); EQ(basket_max_years); EQ(maturity_rounding_months);
      EQ(conversion_factor_decimals);
    }
  }
  {
    const auto l = R.list_fx_pairs();
    ASSERT_EQ(l.overlay.size(), cvd::kFxPairs.size());
    for (const cvd::FxPairConv& want : cvd::kFxPairs) {
      const std::string where = "fx pair " + std::string(want.id);
      ASSERT_TRUE(in_overlay(l, want.id)) << where;
      const cvd::FxPairConv got = *R.fx_pair(want.id);
      EQ(id); EQ(base); EQ(quote); EQ(calendar); EQ(premium_currency); EQ(delta_convention); EQ(atm_convention);
      EQ(xccy_product); EQ(forward_product); EQ(spot_lag); EQ(smile_pillar_lo); EQ(smile_pillar_hi);
    }
  }
  {
    const auto l = R.list_fixing_sources();
    ASSERT_EQ(l.overlay.size(), cvd::kFixingSources.size());
    for (const cvd::FixingSourceConv& want : cvd::kFixingSources) {
      const std::string where = "fixing source " + std::string(want.id);
      ASSERT_TRUE(in_overlay(l, want.id)) << where;
      const cvd::FixingSourceConv got = *R.fixing_source(want.id);
      EQ(id); EQ(provider); EQ(series); EQ(start); EQ(granularity);
    }
  }
  {
    const auto l = R.list_inflation_indices();
    ASSERT_EQ(l.overlay.size(), cvd::kInflationIndices.size());
    for (const cvd::InflationIndexConv& want : cvd::kInflationIndices) {
      const std::string where = "inflation index " + std::string(want.id);
      ASSERT_TRUE(in_overlay(l, want.id)) << where;
      const cvd::InflationIndexConv got = *R.inflation_index(want.id);
      EQ(id); EQ(label); EQ(currency); EQ(calendar); EQ(interpolation); EQ(frequency); EQ(observation_lag_months);
    }
  }
  {
    const auto l = R.list_bonds();
    ASSERT_EQ(l.overlay.size(), cvd::kBonds.size());
    for (const cvd::BondConv& want : cvd::kBonds) {
      const std::string where = "bond " + std::string(want.id);
      ASSERT_TRUE(in_overlay(l, want.id)) << where;
      const cvd::BondConv got = *R.bond(want.id);
      EQ(id); EQ(currency); EQ(calendar); EQ(day_count); EQ(frequency); EQ(stub_discount); EQ(settle_lag);
      EQ(final_period_simple);
    }
  }
  {
    const auto l = R.list_calendars();
    ASSERT_EQ(l.overlay.size(), cvd::kCalendars.size());
    for (const cvd::CalendarConv& want_row : cvd::kCalendars) {
      const std::string where = "calendar " + std::string(want_row.id);
      ASSERT_TRUE(in_overlay(l, want_row.id)) << where;
      const cvd::CalendarView view = *R.calendar(want_row.id);
      {
        const cvd::CalendarConv& got = view.row;
        const cvd::CalendarConv& want = want_row;
        EQ(id); EQ(name); EQ(observance); EQ(weekend_mask); EQ(sandwich); EQ(rule_count); EQ(join_count);
      }
      for (std::size_t i = 0; i < std::min(view.row.rule_count, want_row.rule_count); ++i) {
        const cvd::HolidayRule& got = view.rules[i];
        const cvd::HolidayRule& want = cvd::kHolidayRules[want_row.rule_begin + i];
        const std::string rw = where + " rule " + std::to_string(i);
        EXPECT_EQ(got.kind, want.kind) << rw;
        EXPECT_EQ(got.month, want.month) << rw;
        EXPECT_EQ(got.day, want.day) << rw;
        EXPECT_EQ(got.weekday, want.weekday) << rw;
        EXPECT_EQ(got.n, want.n) << rw;
        EXPECT_EQ(got.days, want.days) << rw;
        EXPECT_EQ(got.from_year, want.from_year) << rw;
        EXPECT_EQ(got.to_year, want.to_year) << rw;
        EXPECT_EQ(got.observance, want.observance) << rw;
        EXPECT_EQ(got.except_first_friday, want.except_first_friday) << rw;
      }
      for (std::size_t i = 0; i < std::min(view.row.join_count, want_row.join_count); ++i)
        EXPECT_EQ(view.joins[i], cvd::kCalendarJoins[want_row.join_begin + i]) << where << " join " << i;
    }
  }
  {
    const auto l = R.list_cb_schedules();
    ASSERT_EQ(l.overlay.size(), cvd::kCbSchedules.size());
    for (const cvd::CbScheduleConv& want : cvd::kCbSchedules) {
      const std::string where = "cb schedule " + std::string(want.currency);
      ASSERT_TRUE(in_overlay(l, want.currency)) << where;
      const std::vector<long> got = *R.cb_meetings(want.currency);
      const std::vector<long> expected(cvd::kCbMeetings.begin() + static_cast<std::ptrdiff_t>(want.begin),
                                       cvd::kCbMeetings.begin() + static_cast<std::ptrdiff_t>(want.begin + want.count));
      EXPECT_EQ(got, expected) << where;
    }
  }
}
