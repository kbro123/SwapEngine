// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// @regression-test — the runtime conventions registry (PRINCIPLES.md P2): the baked JSON is the DEFAULT set,
// any API can ADD entries through the `conventions` verb, a lookup miss THROWS (no silent USD/ACT/360/SIFMA
// fallbacks), and an added product is used by the very next build. Class T6 (proven to fail without the
// registry: before 2026-09-08 `swap_conv("XXX", "")` returned a USD-SOFR-OIS convention and
// `is_business_day("NOPE", d)` used the SIFMA calendar).
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include <boost/json.hpp>

#include "swaps/api/conventions.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/credit_instruments.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/conventions_data.hpp"

namespace api = swaps::api;
namespace b = swaps::build;
namespace cvd = swaps::conventions;
namespace json = boost::json;

namespace {
struct RegistryFixture : ::testing::Test {
  void SetUp() override { cvd::Registry::instance().clear_overlay(); }
  void TearDown() override { cvd::Registry::instance().clear_overlay(); }
};
}  // namespace

TEST_F(RegistryFixture, UnknownIdsThrowInsteadOfDefaulting) {
  EXPECT_THROW(b::swap_conv("XXX", ""), std::invalid_argument);            // no currency row -> no product guess
  EXPECT_THROW(b::swap_conv("USD", "NOT-AN-INDEX"), std::invalid_argument);
  EXPECT_THROW(b::xccy_conv("GBPJPY"), std::invalid_argument);            // only XCCY-MTM-EURUSD is baked
  EXPECT_THROW(b::index_day_count(""), std::invalid_argument);
  EXPECT_THROW(b::index_calendar("NOT-AN-INDEX"), std::invalid_argument);
  EXPECT_THROW(b::is_business_day("NOPE", b::Date::from_iso("2026-01-01")), std::invalid_argument);
  EXPECT_THROW(b::is_business_day("", b::Date::from_iso("2026-01-01")), std::invalid_argument);
  EXPECT_THROW(cvd::require_currency("ZZZ"), std::invalid_argument);
  EXPECT_FALSE(cvd::product("NOT-A-PRODUCT").has_value());
}

TEST_F(RegistryFixture, NoIndexResolvesThroughTheCurrencyRowNotAHeuristic) {
  // currencies[USD].default_swap_product == USD-SOFR-OIS is DATA; the old code guessed EUR-EURIBOR-*-IRS
  // from a float frequency and fell back to USD-SOFR-OIS for every currency.
  const b::SwapConv usd = b::swap_conv("USD", "");
  EXPECT_EQ(usd.product_id, std::string(cvd::require_currency("USD").default_swap_product));
  const b::SwapConv aud = b::swap_conv("aud", "");  // case-insensitive code
  EXPECT_EQ(aud.product_id, "AUD-AONIA-OIS");
  EXPECT_EQ(aud.calendar, "AUD");
  EXPECT_EQ(aud.spot_lag, 1);
}

TEST_F(RegistryFixture, BasisAnnuityFollowsTheQuotedLegNotAnAnnualLiteral) {
  const b::SwapConv sar = b::swap_conv("SAR", "SAR-SAIBOR-3M");
  EXPECT_EQ(sar.fixed_freq_tok, "6M");
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const b::Date mat = b::resolve("2Y", vd, "NONE", "Unadjusted", 0);
  const auto par = b::par_swap(vd, sar, mat, 0, 0, 0.04);
  const auto bas = b::basis_swap(vd, sar, mat, 0, 1, 0, 0.0010);
  EXPECT_EQ(par.fixed.coupons.size(), 4u);                       // 6M fixed leg over 2y
  EXPECT_EQ(bas.fixed.coupons.size(), bas.fwd.coupons.size());   // annuity on the quoted (3M) leg: 8, not 2
  EXPECT_EQ(bas.fixed.coupons.size(), 8u);
}

TEST_F(RegistryFixture, ConventionsVerbAddsACurrencyCalendarIndexAndProductUsedByTheNextBuild) {
  // A brand-new market (fictional "XYZ") added through the JSON seam only — no rebuild, no code.
  const std::string req = R"({"conventions": {
    "currencies": {"XYZ": {"name": "Test dollar", "minor_units": 3, "settlement_calendar": "XYZ",
                           "discount_index": "XYZ-ONIA", "default_swap_product": "XYZ-ONIA-OIS"}},
    "calendars": {"XYZ": {"name": "Test", "weekend": [4, 5], "observance": "none",
                          "holidays": [{"rule": "fixed", "month": 7, "day": 9}]}},
    "indices": {"XYZ-ONIA": {"currency": "XYZ", "type": "overnight", "day_count": "ACT/365F",
                             "calendar": "XYZ", "publication_lag": 1, "par_product": "XYZ-ONIA-OIS"}},
    "products": {"XYZ-ONIA-OIS": {"description": "test", "type": "ois", "currency": "XYZ", "calendar": "XYZ",
                                  "bdc": "Following", "spot_lag": 0, "payment_lag": 1,
                                  "fixed_leg": {"day_count": "ACT/365F", "frequency": "3M"},
                                  "float_leg": {"index": "XYZ-ONIA", "compounding": "compounded",
                                                "day_count": "ACT/365F", "frequency": "3M"}}}}})";
  const json::object resp = json::parse(api::conventions_json(req)).as_object();
  EXPECT_EQ(resp.at("conventions").as_object().at("overlay_size").to_number<int>(), 4);

  // Consumed by every layer: currency row, calendar interpreter, index accessors, the swap builder.
  EXPECT_EQ(cvd::require_currency("XYZ").minor_units, 3);
  EXPECT_FALSE(b::is_business_day("XYZ", b::Date::from_iso("2026-07-09")));   // the added fixed holiday
  EXPECT_FALSE(b::is_business_day("XYZ", b::Date::from_iso("2026-07-10")));   // Friday = weekend [4,5]
  EXPECT_TRUE(b::is_business_day("XYZ", b::Date::from_iso("2026-07-12")));    // Sunday is a business day here
  EXPECT_EQ(b::index_day_count("XYZ-ONIA"), "ACT/365F");
  const b::SwapConv c = b::swap_conv("XYZ", "");
  EXPECT_EQ(c.product_id, "XYZ-ONIA-OIS");
  EXPECT_EQ(c.bdc, "Following");
  EXPECT_EQ(c.spot_lag, 0);
  EXPECT_EQ(c.pay_lag, 1);
  EXPECT_EQ(c.fixed_freq_tok, "3M");
  const b::Date vd = b::Date::from_iso("2026-07-08");
  const auto ins = b::par_swap(vd, c, b::resolve("1Y", vd, "NONE", "Unadjusted", 0), 0, 0, 0.03);
  EXPECT_EQ(ins.fixed.coupons.size(), 4u);  // quarterly, from the added product

  // Listed, and distinguishable from the baked defaults.
  const json::object L = json::parse(api::list_conventions_json("{\"list_conventions\": true}")).as_object();
  const auto& prods = L.at("conventions").as_object().at("products").as_object();
  bool in_overlay = false, in_baked = false;
  for (const auto& v : prods.at("overlay").as_array()) in_overlay |= (v.as_string() == "XYZ-ONIA-OIS");
  for (const auto& v : prods.at("baked").as_array()) in_baked |= (v.as_string() == "USD-SOFR-OIS");
  EXPECT_TRUE(in_overlay);
  EXPECT_TRUE(in_baked);

  // Overlay wins over baked for the SAME id (an override), and clear_overlay restores the default.
  const std::string over = R"({"conventions": {"products": {"USD-SOFR-OIS": {"description": "override", "type": "ois",
    "currency": "USD", "calendar": "USD-SOFR", "bdc": "ModifiedFollowing", "spot_lag": 2, "payment_lag": 2,
    "fixed_leg": {"day_count": "ACT/360", "frequency": "6M"},
    "float_leg": {"index": "USD-SOFR", "compounding": "compounded", "day_count": "ACT/360", "frequency": "6M"}}}}})";
  api::conventions_json(over);
  EXPECT_EQ(b::swap_conv("USD", "USD-SOFR").fixed_freq_tok, "6M");
  api::conventions_json(R"({"conventions": {"clear_overlay": true}})");
  EXPECT_EQ(b::swap_conv("USD", "USD-SOFR").fixed_freq_tok, "1Y");
  EXPECT_THROW(cvd::require_currency("XYZ"), std::invalid_argument);
}

TEST_F(RegistryFixture, MalformedEntriesAreRejectedLoudly) {
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"products": {"P": {"description": "no type"}}}})"),
               std::invalid_argument);
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"calendars": {"C": {"name": "no weekend"}}}})"),
               std::invalid_argument);
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"indices": {"I": {"currency": "USD"}}}})"),
               std::invalid_argument);
  // A product row that lacks a field the builders need is rejected at USE, with the row id in the message.
  api::conventions_json(R"({"conventions": {"products": {"BAD-OIS": {"description": "x", "type": "ois",
    "currency": "USD", "calendar": "USD-SOFR", "bdc": "ModifiedFollowing", "spot_lag": 2, "payment_lag": 2,
    "fixed_leg": {"day_count": "ACT/360"}, "float_leg": {"index": "USD-SOFR", "day_count": "ACT/360", "frequency": "1Y"}}}}})");
  try {
    b::conv_from_product(cvd::require_product("BAD-OIS"));
    FAIL() << "missing fixed_leg frequency must throw";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("BAD-OIS"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("frequency"), std::string::npos);
  }
}

TEST_F(RegistryFixture, NewFamiliesAreDataAndRuntimeExtensible) {
  // credit / bond futures / fx pairs / central-bank schedules: baked rows replace what were C++ literals
  // (recovery 0.40, CF 6% + 3-month rounding + repo /360, USD pivot, a web-side meeting table).
  const cvd::CreditConv cds = cvd::require_credit_product("CDS-USD-SNAC");
  EXPECT_DOUBLE_EQ(cds.recovery_default, 0.40);
  EXPECT_EQ(cds.day_count, "ACT/360");
  EXPECT_DOUBLE_EQ(b::credit_accrual_ratio(std::string(cds.day_count)), 365.0 / 360.0);
  const cvd::BondFutureConv ty = cvd::require_bond_future("CME-TY");
  EXPECT_DOUBLE_EQ(ty.notional_coupon, 0.06);
  EXPECT_EQ(ty.maturity_rounding_months, 3);
  EXPECT_EQ(cvd::require_bond_future("CME-FV").maturity_rounding_months, 1);
  EXPECT_EQ(ty.deliverable_convention, "US-TREASURY");
  EXPECT_DOUBLE_EQ(b::day_count_basis(std::string(ty.repo_day_count)), 360.0);
  EXPECT_DOUBLE_EQ(b::day_count_basis(std::string(cvd::require_currency("GBP").repo_day_count)), 365.0);
  const cvd::FxPairConv eu = cvd::require_fx_pair("EURUSD");
  EXPECT_EQ(eu.base, "EUR"); EXPECT_EQ(eu.quote, "USD"); EXPECT_EQ(eu.spot_lag, 2);
  const std::vector<long> fomc = cvd::require_cb_meetings("USD");
  EXPECT_EQ(fomc.size(), 16u);
  EXPECT_EQ(b::iso(b::Date::from_iso("1970-01-01").plus_days(int(fomc.front()))), "2026-01-28");
  EXPECT_THROW(cvd::require_cb_meetings("NZD"), std::invalid_argument);   // not a DB currency
  EXPECT_THROW(cvd::require_bond_future("EUREX-FGBL"), std::invalid_argument);

  // Runtime overlay through the verb: a new contract, and a REPLACEMENT meeting schedule for GBP.
  api::conventions_json(R"({"conventions": {
    "bond_futures": {"TEST-XX": {"currency": "USD", "exchange_calendar": "USD", "deliverable_convention": "US-TREASURY",
                                 "notional_coupon": 0.04, "maturity_rounding_months": 1, "conversion_factor_decimals": 6,
                                 "repo_day_count": "ACT/365F"}},
    "cb_schedules": {"GBP": {"bank": "BoE", "source": "test", "as_of": "2026-09-09",
                             "meetings": ["2027-02-04", "2027-03-18"]}}}})");
  EXPECT_DOUBLE_EQ(cvd::require_bond_future("TEST-XX").notional_coupon, 0.04);
  EXPECT_EQ(cvd::require_bond_future("TEST-XX").conversion_factor_decimals, 6);  // an exchange that rounds to 6 dp
  EXPECT_DOUBLE_EQ(b::day_count_basis(std::string(cvd::require_bond_future("TEST-XX").repo_day_count)), 365.0);
  const std::vector<long> boe = cvd::require_cb_meetings("GBP");
  ASSERT_EQ(boe.size(), 2u);
  EXPECT_EQ(b::iso(b::Date::from_iso("1970-01-01").plus_days(int(boe[1]))), "2027-03-18");
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"cb_schedules": {"GBP": {"meetings": ["2027-03-18", "2027-02-04"]}}}})"),
               std::invalid_argument);  // must be ascending
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"fx_pairs": {"USDEUR": {"base": "EUR", "quote": "USD", "spot_lag": 2,
    "calendar": "EURUSD", "premium_currency": "USD", "delta_convention": "spot", "atm_convention": "delta_neutral"}}}})"),
               std::invalid_argument);  // id must equal base+quote
}

TEST_F(RegistryFixture, FixingSourcesAndInflationIndicesAreDataAndRuntimeExtensible) {
  // Baked: the fixing-source metadata every API used to carry its own copy of, and the ZCIS reference indices.
  const auto sofr = cvd::require_fixing_source("USD-SOFR");
  EXPECT_EQ(sofr.provider, std::string_view("nyfed"));
  EXPECT_EQ(sofr.granularity, std::string_view("daily"));
  const auto rpi = cvd::require_inflation_index("UK-RPI");
  EXPECT_EQ(rpi.observation_lag_months, 2);
  EXPECT_EQ(rpi.interpolation, std::string_view("flat"));
  EXPECT_EQ(cvd::require_inflation_index("US-CPI-U").interpolation, std::string_view("linear"));
  EXPECT_THROW(cvd::require_fixing_source("GBP-SONIA"), std::invalid_argument);   // no source baked
  EXPECT_THROW(cvd::require_inflation_index("CA-CPI"), std::invalid_argument);
  // Runtime: the verb adds both families; the next lookup sees them; malformed rows are rejected.
  api::conventions_json(R"({"conventions": {
    "fixing_sources": {"GBP-SONIA": {"provider": "ecb", "series": "TEST/SONIA", "granularity": "daily"}},
    "inflation": {"CA-CPI": {"currency": "CAD", "calendar": "CAD", "observation_lag_months": 3, "interpolation": "linear", "frequency": "1M"}}}})");
  EXPECT_EQ(cvd::require_fixing_source("GBP-SONIA").series, std::string_view("TEST/SONIA"));
  EXPECT_EQ(cvd::require_inflation_index("CA-CPI").observation_lag_months, 3);
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"fixing_sources": {"NOT-AN-INDEX": {"provider": "ecb", "series": "x", "granularity": "daily"}}}})"), std::invalid_argument);
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"inflation": {"XX": {"currency": "USD", "calendar": "USD", "observation_lag_months": 3, "interpolation": "cubic", "frequency": "1M"}}}})"), std::invalid_argument);
  const auto listed = json::parse(api::list_conventions_json("{}")).as_object().at("conventions").as_object();
  EXPECT_EQ(listed.at("inflation").as_object().at("overlay").as_array().size(), 1u);
  EXPECT_GE(listed.at("fixing_sources").as_object().at("baked").as_array().size(), 5u);
}

TEST_F(RegistryFixture, CalendarOverrideInvalidatesTheHolidayCacheImmediately) {
  // is_business_day caches each (calendar, year) closed-day bitmap keyed by the registry generation; a runtime
  // override of an EXISTING calendar must be seen by the very next query, and clear_overlay must restore it.
  const b::Date d = b::Date::from_iso("2026-07-09");  // a Thursday, open on TARGET
  EXPECT_TRUE(b::is_business_day("EUR", d));
  api::conventions_json(R"({"conventions": {"calendars": {"EUR": {"name": "override", "weekend": [5, 6], "observance": "none",
    "holidays": [{"rule": "fixed", "month": 7, "day": 9}]}}}})");
  EXPECT_FALSE(b::is_business_day("EUR", d));                              // the override, not the cached bitmap
  EXPECT_TRUE(b::is_business_day("EUR", b::Date::from_iso("2026-12-25")));  // the override REPLACES the row
  api::conventions_json(R"({"conventions": {"clear_overlay": true}})");
  EXPECT_TRUE(b::is_business_day("EUR", d));
  EXPECT_FALSE(b::is_business_day("EUR", b::Date::from_iso("2026-12-25")));
}
