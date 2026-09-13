// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// The conventions Registry's row rules and all-or-nothing batches (E7 stage 4.2), header-only so tools/mutate.py
// reaches them. Every baked row passes the rules (so they are never stricter than the data check_schema.py accepts),
// and each rule is shown refusing the one field it guards on an otherwise-baked row. The JSON verb's side of the same
// fixes is tests/conventions_repro_test.cpp.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "swaps/conventions_data.hpp"

namespace cvd = swaps::conventions;

namespace {

struct ConventionsRules : ::testing::Test {
  void SetUp() override { cvd::Registry::instance().clear_overlay(); }
  void TearDown() override { cvd::Registry::instance().clear_overlay(); }
};

// `f` must throw std::invalid_argument whose message names `needle`.
template <class F>
void rejects(F&& f, const std::string& needle) {
  try {
    f();
    ADD_FAILURE() << "accepted; expected a rejection naming '" << needle << "'";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
  }
}

cvd::ProductConv baked_product(std::string_view type, bool zero_coupon) {
  for (const auto& p : cvd::kProducts)
    if (p.type == type && p.zero_coupon == zero_coupon) return p;
  throw std::logic_error("no baked product of type " + std::string(type));
}

cvd::IndexConv test_index(std::string_view id) {
  cvd::IndexConv i = cvd::kIndices[0];
  i.id = id;
  return i;
}

std::vector<cvd::HolidayRule> rules_of(const cvd::CalendarConv& c) {
  return {cvd::kHolidayRules.data() + c.rule_begin, cvd::kHolidayRules.data() + c.rule_begin + c.rule_count};
}
std::vector<std::string_view> joins_of(const cvd::CalendarConv& c) {
  return {cvd::kCalendarJoins.data() + c.join_begin, cvd::kCalendarJoins.data() + c.join_begin + c.join_count};
}

}  // namespace

TEST(ConventionsRulesBaked, EveryBakedRowPassesTheRules) {
  for (const auto& x : cvd::kProducts) EXPECT_NO_THROW(cvd::validate(x)) << x.id;
  for (const auto& x : cvd::kIndices) EXPECT_NO_THROW(cvd::validate(x)) << x.id;
  for (const auto& x : cvd::kBonds) EXPECT_NO_THROW(cvd::validate(x)) << x.id;
  for (const auto& x : cvd::kCurrencies) EXPECT_NO_THROW(cvd::validate(x)) << x.code;
  for (const auto& x : cvd::kCredit) EXPECT_NO_THROW(cvd::validate(x)) << x.id;
  for (const auto& x : cvd::kBondFutures) EXPECT_NO_THROW(cvd::validate(x)) << x.id;
  for (const auto& x : cvd::kFxPairs) EXPECT_NO_THROW(cvd::validate(x)) << x.id;
  for (const auto& x : cvd::kFixingSources) EXPECT_NO_THROW(cvd::validate(x)) << x.id;
  for (const auto& x : cvd::kInflationIndices) EXPECT_NO_THROW(cvd::validate(x)) << x.id;
  for (const auto& c : cvd::kCalendars) EXPECT_NO_THROW(cvd::validate(c, rules_of(c), joins_of(c))) << c.id;
  for (const auto& c : cvd::kCbSchedules) {
    const std::vector<long> m(cvd::kCbMeetings.data() + c.begin, cvd::kCbMeetings.data() + c.begin + c.count);
    EXPECT_NO_THROW(cvd::validate(c, m)) << c.currency;
  }
}

TEST(ConventionsRulesBaked, ACouponSwapNeedsBothLegsInFull) {
  const cvd::ProductConv ois = baked_product("ois", false);
  rejects([&] { auto p = ois; p.fixed.frequency = {}; cvd::validate(p); }, "fixed_leg.frequency");
  rejects([&] { auto p = ois; p.fixed.day_count = {}; cvd::validate(p); }, "fixed_leg.day_count");
  rejects([&] { auto p = ois; p.floating.index = {}; cvd::validate(p); }, "float_leg.index");
  rejects([&] { auto p = ois; p.floating.frequency = {}; cvd::validate(p); }, "float_leg.frequency");
  rejects([&] { auto p = ois; p.floating.compounding = "simple"; cvd::validate(p); }, "float_leg.compounding");
  rejects([&] { auto p = ois; p.floating.frequency = "3m"; cvd::validate(p); }, "float_leg.frequency");
  rejects([&] { auto p = ois; p.bdc = {}; cvd::validate(p); }, "bdc");
  rejects([&] { auto p = ois; p.bdc = "Modified"; cvd::validate(p); }, "bdc");
  rejects([&] { auto p = ois; p.spot_lag = -1; cvd::validate(p); }, "spot_lag");
  rejects([&] { auto p = ois; p.payment_lag = -1; cvd::validate(p); }, "payment_lag");
  rejects([&] { auto p = ois; p.currency = {}; cvd::validate(p); }, "currency");
  rejects([&] { auto p = ois; p.calendar = {}; cvd::validate(p); }, "calendar");
  rejects([&] { auto p = ois; p.type = "swap"; cvd::validate(p); }, "type");
  rejects([&] { auto p = ois; p.frequency = "Q"; cvd::validate(p); }, "frequency");
}

TEST(ConventionsRulesBaked, ZeroCouponBasisAndXccyShapes) {
  const cvd::ProductConv zc = baked_product("irs", true);
  rejects([&] { auto p = zc; p.fixed.frequency = "1Y"; cvd::validate(p); }, "zero_coupon");
  rejects([&] { auto p = zc; p.floating.day_count = {}; cvd::validate(p); }, "float_leg.day_count");
  rejects([&] { auto p = baked_product("basis", false); p.zero_coupon = true; cvd::validate(p); }, "zero_coupon");

  const cvd::ProductConv basis = baked_product("basis", false);
  rejects([&] { auto p = basis; p.floating.index = {}; cvd::validate(p); }, "spread_leg.index");
  rejects([&] { auto p = basis; p.other.frequency = {}; cvd::validate(p); }, "flat_leg.frequency");
  rejects([&] { auto p = basis; p.discount_index = {}; cvd::validate(p); }, "discount_index");

  const cvd::ProductConv xccy = baked_product("xccy_mtm", false);
  rejects([&] { auto p = xccy; p.other = cvd::LegConv{}; cvd::validate(p); }, "eur_leg");
  rejects([&] { auto p = xccy; p.pair = {}; cvd::validate(p); }, "pair");
  rejects([&] { auto p = xccy; p.frequency = {}; cvd::validate(p); }, "frequency");

  const cvd::ProductConv administered = baked_product("administered-basis", false);
  EXPECT_EQ(administered.bdc, "") << "the fixture: a type the schema only asks for a calendar";
  EXPECT_NO_THROW(cvd::validate(administered));
  rejects([&] { auto p = administered; p.calendar = {}; cvd::validate(p); }, "calendar");
}

TEST(ConventionsRulesBaked, EveryOtherFamilysEnumsBoundsAndRequiredFields) {
  const cvd::IndexConv index = cvd::kIndices[0];
  rejects([&] { auto x = index; x.type = "term"; cvd::validate(x); }, "type");
  rejects([&] { auto x = index; x.tenor = "M3"; cvd::validate(x); }, "tenor");
  rejects([&] { auto x = index; x.tenor = "XM"; cvd::validate(x); }, "tenor");
  EXPECT_NO_THROW([&] { auto x = index; x.tenor = "12M"; cvd::validate(x); }()) << "a count of any length";
  rejects([&] { auto x = index; x.fixing_lag = -2; cvd::validate(x); }, "fixing_lag");
  rejects([&] { auto x = index; x.day_count = {}; cvd::validate(x); }, "day_count");

  const cvd::BondConv bond = cvd::kBonds[0];
  rejects([&] { auto x = bond; x.stub_discount = "Simple"; cvd::validate(x); }, "stub_discount");
  rejects([&] { auto x = bond; x.stub_discount = {}; cvd::validate(x); }, "stub_discount");
  rejects([&] { auto x = bond; x.settle_lag = -1; cvd::validate(x); }, "settle_lag");
  rejects([&] { auto x = bond; x.frequency = {}; cvd::validate(x); }, "frequency");

  const cvd::CurrencyConv ccy = cvd::kCurrencies[0];
  rejects([&] { auto x = ccy; x.minor_units = 5; cvd::validate(x); }, "minor_units");
  rejects([&] { auto x = ccy; x.minor_units = -1; cvd::validate(x); }, "minor_units");
  rejects([&] { auto x = ccy; x.repo_day_count = {}; cvd::validate(x); }, "repo_day_count");

  const cvd::CreditConv cds = cvd::kCredit[0];
  rejects([&] { auto x = cds; x.recovery_default = 1.5; cvd::validate(x); }, "recovery_default");
  rejects([&] { auto x = cds; x.recovery_default = std::nan(""); cvd::validate(x); }, "recovery_default");
  rejects([&] { auto x = cds; x.protection_steps = 0; cvd::validate(x); }, "protection_steps");
  rejects([&] { auto x = cds; x.frequency = "Q"; cvd::validate(x); }, "frequency");
  EXPECT_NO_THROW([&] { auto x = cds; x.recovery_default = 1.0; cvd::validate(x); }()) << "the bound is inclusive";

  const cvd::BondFutureConv fut = cvd::kBondFutures[0];
  rejects([&] { auto x = fut; x.conversion_factor_decimals = -1; cvd::validate(x); }, "conversion_factor_decimals");
  rejects([&] { auto x = fut; x.maturity_rounding_months = 0; cvd::validate(x); }, "maturity_rounding_months");
  rejects([&] { auto x = fut; x.notional_coupon = std::nan(""); cvd::validate(x); }, "notional_coupon");
  rejects([&] { auto x = fut; x.deliverable_convention = {}; cvd::validate(x); }, "deliverable_convention");

  const cvd::FxPairConv fx = cvd::kFxPairs[0];
  rejects([&] { auto x = fx; x.id = "XXXYYY"; cvd::validate(x); }, "base + quote");
  rejects([&] { auto x = fx; x.delta_convention = "spot_premium"; cvd::validate(x); }, "delta_convention");
  rejects([&] { auto x = fx; x.atm_convention = "atmf"; cvd::validate(x); }, "atm_convention");
  rejects([&] { auto x = fx; x.spot_lag = -1; cvd::validate(x); }, "spot_lag");

  const cvd::FixingSourceConv src = cvd::kFixingSources[0];
  rejects([&] { auto x = src; x.provider = "bloomberg"; cvd::validate(x); }, "provider");
  rejects([&] { auto x = src; x.granularity = "weekly"; cvd::validate(x); }, "granularity");
  rejects([&] { auto x = src; x.start = "2018-4-01"; cvd::validate(x); }, "start");

  const cvd::InflationIndexConv cpi = cvd::kInflationIndices[0];
  rejects([&] { auto x = cpi; x.interpolation = "cubic"; cvd::validate(x); }, "interpolation");
  rejects([&] { auto x = cpi; x.observation_lag_months = -1; cvd::validate(x); }, "observation_lag_months");
  rejects([&] { auto x = cpi; x.frequency = "monthly"; cvd::validate(x); }, "frequency");

  const cvd::CbScheduleConv cb = cvd::kCbSchedules[0];
  rejects([&] { cvd::validate(cb, {10, 10}); }, "ascending");
  rejects([&] { cvd::validate(cb, {10, 9}); }, "ascending");
  rejects([&] { auto x = cb; x.as_of = "2026/09/01"; cvd::validate(x, {}); }, "as_of");
  rejects([&] { auto x = cb; x.bank = {}; cvd::validate(x, {}); }, "bank");
}

TEST(ConventionsRulesBaked, CalendarsWeekendsObservancesAndRules) {
  const cvd::CalendarConv* ruled = nullptr;
  for (const auto& c : cvd::kCalendars)
    if (c.rule_count > 0) { ruled = &c; break; }
  ASSERT_NE(ruled, nullptr);
  const auto rules = rules_of(*ruled);
  const auto joins = joins_of(*ruled);
  rejects([&] { auto x = *ruled; x.weekend_mask |= 1 << 7; cvd::validate(x, rules, joins); }, "weekend");
  rejects([&] { auto x = *ruled; x.observance = "sometimes"; cvd::validate(x, rules, joins); }, "observance");
  rejects([&] { auto x = *ruled; x.name = {}; cvd::validate(x, rules, joins); }, "name");
  rejects([&] { auto r = rules; r[0].kind = "moon"; cvd::validate(*ruled, r, joins); }, "holiday rule");
  rejects([&] { auto r = rules; r[0].observance = "never"; cvd::validate(*ruled, r, joins); }, "holiday observance");
  rejects([&] { auto j = joins; j.push_back({}); cvd::validate(*ruled, rules, j); }, "join");
}

TEST_F(ConventionsRules, ABatchCommitsAllOrNothingInOneGeneration) {
  auto& R = cvd::Registry::instance();
  cvd::OverlayBatch b;
  b.indices = {test_index("TEST-A"), test_index("TEST-B")};
  cvd::BondConv bond = cvd::kBonds[0];
  bond.id = "TEST-BOND";
  bond.stub_discount = "Simple";
  b.bonds = {bond};
  const unsigned long g0 = R.generation();
  rejects([&] { R.apply(b); }, "stub_discount");
  EXPECT_FALSE(cvd::index("TEST-A").has_value()) << "the rows before the bad one are not committed";
  EXPECT_EQ(R.overlay_size(), 0);
  EXPECT_EQ(R.generation(), g0) << "a refused batch is not a mutation";

  b.bonds[0].stub_discount = "simple";
  R.apply(b);
  EXPECT_TRUE(cvd::index("TEST-A").has_value());
  EXPECT_TRUE(cvd::index("TEST-B").has_value());
  EXPECT_EQ(cvd::require_bond("TEST-BOND").stub_discount, "simple");
  EXPECT_EQ(R.overlay_size(), 3);
  EXPECT_EQ(R.generation(), g0 + 1) << "one batch, one generation";

  rejects([&] { auto x = bond; x.id = "TEST-BOND-2"; R.add_bond(x); }, "stub_discount");  // add_* is apply of one row
  EXPECT_FALSE(cvd::bond("TEST-BOND-2").has_value());
}

TEST_F(ConventionsRules, ClearFirstIsPartOfTheSameCommit) {
  auto& R = cvd::Registry::instance();
  R.add_index(test_index("TEST-KEEP"));
  cvd::OverlayBatch b;
  b.clear_first = true;
  cvd::IndexConv bad = test_index("TEST-BAD");
  bad.calendar = {};
  b.indices = {bad};
  rejects([&] { R.apply(b); }, "calendar");
  EXPECT_TRUE(cvd::index("TEST-KEEP").has_value()) << "a refused batch does not clear";

  b.indices = {test_index("TEST-NEW")};
  R.apply(b);
  EXPECT_FALSE(cvd::index("TEST-KEEP").has_value());
  EXPECT_TRUE(cvd::index("TEST-NEW").has_value());
  EXPECT_EQ(R.overlay_size(), 1);
}

TEST_F(ConventionsRules, AFixingSourceNeedsItsIndexOnceTheBatchIsIn) {
  auto& R = cvd::Registry::instance();
  cvd::FixingSourceConv s = cvd::kFixingSources[0];
  s.id = "TEST-IDX";
  cvd::OverlayBatch alone;
  alone.fixing_sources = {s};
  rejects([&] { R.apply(alone); }, "does not know");
  EXPECT_EQ(R.overlay_size(), 0);

  cvd::OverlayBatch with = alone;
  with.indices = {test_index("TEST-IDX")};
  R.apply(with);  // the index arrives in the same batch
  EXPECT_TRUE(cvd::fixing_source("TEST-IDX").has_value());
  R.apply(alone);  // and now it is in the overlay
  EXPECT_EQ(R.overlay_size(), 3);

  cvd::OverlayBatch cleared = alone;
  cleared.clear_first = true;
  rejects([&] { R.apply(cleared); }, "does not know");  // the clear would take the index with it
  EXPECT_EQ(R.overlay_size(), 3);

  cvd::OverlayBatch baked;
  baked.clear_first = true;
  baked.fixing_sources = {cvd::kFixingSources[0]};  // a baked index survives any clear
  R.apply(baked);
  EXPECT_EQ(R.overlay_size(), 1);
}

TEST_F(ConventionsRules, TheListingIsOneSnapshotOfBakedAndOverlayIds) {
  auto& R = cvd::Registry::instance();
  cvd::OverlayBatch b;
  b.indices = {test_index("TEST-L")};
  cvd::CurrencyConv c = cvd::kCurrencies[0];
  c.code = "TST";
  b.currencies = {c};
  R.apply(b);
  const cvd::Registry::Listings L = R.listing();
  EXPECT_EQ(L.indices.baked.size(), cvd::kIndices.size());
  EXPECT_EQ(L.indices.overlay, std::vector<std::string>{"TEST-L"});
  EXPECT_EQ(L.currencies.baked.front(), std::string(cvd::kCurrencies[0].code));
  EXPECT_EQ(L.currencies.overlay, std::vector<std::string>{"TST"});
  EXPECT_TRUE(L.products.overlay.empty());
  EXPECT_EQ(L.cb_schedules.baked.size(), cvd::kCbSchedules.size());
  EXPECT_EQ(L.overlay_size, 2);
}
