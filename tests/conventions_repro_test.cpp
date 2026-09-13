// E5 taxonomy: T6 regression (fails on the reverted bug) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// P12 REPRODUCTIONS for E7 stage 4.2: the runtime conventions registry accepted rows its own schema rejects, and a
// failing request kept the rows before the failure. Every test here FAILED before the fix (the registry's add_* validated
// nothing and committed row by row); each row is otherwise schema-complete, so it fails for the one field under test.
// The reference is conventions/conventions.schema.json -- the rules the baked DB is already held to by check_schema.py.
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "swaps/api/conventions.hpp"
#include "swaps/conventions_data.hpp"

namespace api = swaps::api;
namespace cvd = swaps::conventions;

namespace {
struct ConventionsRepro : ::testing::Test {
  void SetUp() override { cvd::Registry::instance().clear_overlay(); }
  void TearDown() override { cvd::Registry::instance().clear_overlay(); }
};
}  // namespace

// A failing request must leave the registry exactly as it found it: before the fix, AAA-GOOD stayed committed.
TEST_F(ConventionsRepro, ABatchWithABadLastRowAddsNothing) {
  const std::string doc = R"({"conventions": {"indices": {
      "AAA-GOOD": {"currency": "USD", "type": "overnight", "day_count": "ACT/360", "calendar": "USD"},
      "ZZZ-BAD": {"currency": "USD", "type": "overnight", "day_count": "ACT/360"}}}})";
  EXPECT_THROW(api::conventions_json(doc), std::invalid_argument) << "ZZZ-BAD has no calendar";
  EXPECT_FALSE(cvd::index("AAA-GOOD").has_value()) << "the good row before the bad one must not be committed";
  EXPECT_EQ(cvd::Registry::instance().overlay_size(), 0);
}

// build/bond.hpp reads stub_discount == "simple" ? Simple : Compound, so a missing or misspelt value silently priced as
// Compound. The schema requires it and enumerates compound | simple.
TEST_F(ConventionsRepro, ABondRowWithoutAValidStubDiscountIsRejected) {
  const std::string row = R"("currency": "USD", "calendar": "USD", "day_count": "ACT/ACT", "frequency": "6M", "settle_lag": 1)";
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"bonds": {"TEST-BOND": {)" + row + "}}}}"),
               std::invalid_argument) << "missing stub_discount";
  EXPECT_THROW(api::conventions_json(R"({"conventions": {"bonds": {"TEST-BOND": {)" + row + R"(, "stub_discount": "Simple"}}}})"),
               std::invalid_argument) << "stub_discount is case-sensitive";
  EXPECT_FALSE(cvd::bond("TEST-BOND").has_value());
}

// weekend days are Mon=0..Sun=6: `1 << 7` set a bit on_weekend never reads, silently dropping the day.
TEST_F(ConventionsRepro, AWeekendDayOutsideMondayToSundayIsRejected) {
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"calendars": {"TEST-CAL":
                   {"name": "test", "weekend": [5, 7], "observance": "none", "holidays": []}}}})")),
               std::invalid_argument);
  EXPECT_FALSE(cvd::calendar("TEST-CAL").has_value());
}

// Enums and bounds the schema declares: each row is otherwise complete, so each throw is for the one field named.
TEST_F(ConventionsRepro, TheSchemasEnumsAndBoundsAreEnforced) {
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"credit": {"cds_products": {"TEST-CDS":
                   {"currency": "USD", "calendar": "USD", "day_count": "ACT/360", "frequency": "3M",
                    "recovery_default": 1.5, "settlement_lag": 3, "protection_steps": 12}}}}})")),
               std::invalid_argument) << "recovery_default is a fraction in [0, 1]";
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"fixing_sources": {"USD-SOFR":
                   {"provider": "bloomberg", "series": "SOFR", "granularity": "daily"}}}})")),
               std::invalid_argument) << "provider is nyfed | ecb";
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"fx_pairs": {"EURGBP":
                   {"base": "EUR", "quote": "GBP", "calendar": "EUR", "spot_lag": 2, "premium_currency": "EUR",
                    "delta_convention": "spot_premium", "atm_convention": "delta_neutral"}}}})")),
               std::invalid_argument) << "delta_convention is spot | forward | spot_pa | forward_pa";
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"bond_futures": {"TEST-FUT":
                   {"currency": "USD", "exchange_calendar": "USD", "deliverable_convention": "US-TREASURY",
                    "notional_coupon": 0.06, "maturity_rounding_months": 3, "conversion_factor_decimals": -1,
                    "repo_day_count": "ACT/360"}}}})")),
               std::invalid_argument) << "conversion_factor_decimals >= 0";
  EXPECT_EQ(cvd::Registry::instance().overlay_size(), 0);
}

// The rules belong to the REGISTRY, not the JSON verb: a C++ caller adding a row directly gets the same refusal.
TEST_F(ConventionsRepro, TheRegistryItselfRejectsAnInvalidRow) {
  cvd::BondConv b;
  b.id = "TEST-BOND";
  b.currency = "USD";
  b.calendar = "USD";
  b.day_count = "ACT/ACT";
  b.frequency = "6M";
  b.settle_lag = 1;
  b.stub_discount = "Simple";
  EXPECT_THROW(cvd::Registry::instance().add_bond(b), std::invalid_argument);
  EXPECT_FALSE(cvd::bond("TEST-BOND").has_value());
}

// A lag is a non-negative integer: 2.5 was truncated to 2, -1 read as "unset", and "2" escaped as a
// boost::system::system_error rather than the std::invalid_argument every other malformed row throws.
TEST_F(ConventionsRepro, ALagMustBeANonNegativeInteger) {
  const auto index = [](const std::string& lag) {
    return R"({"conventions": {"indices": {"TEST-IDX": {"currency": "USD", "type": "overnight", "day_count": "ACT/360",
        "calendar": "USD", "fixing_lag": )" + lag + "}}}}";
  };
  EXPECT_THROW(api::conventions_json(index("2.5")), std::invalid_argument) << "fractional";
  EXPECT_THROW(api::conventions_json(index("-1")), std::invalid_argument) << "negative";
  EXPECT_THROW(api::conventions_json(index("\"2\"")), std::invalid_argument) << "a string";
  EXPECT_FALSE(cvd::index("TEST-IDX").has_value());
  api::conventions_json(index("2"));
  EXPECT_EQ(cvd::require_index("TEST-IDX").fixing_lag, 2);
}

// Meeting dates are real calendar dates: serial_of_iso turned 2027-02-30 into 2 March.
TEST_F(ConventionsRepro, AMeetingThatIsNotACalendarDateIsRejected) {
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"cb_schedules": {"USD": {"bank": "Fed",
                   "source": "test", "as_of": "2026-09-01", "meetings": ["2027-01-27", "2027-02-30"]}}}})")),
               std::invalid_argument);
  EXPECT_EQ(cvd::Registry::instance().overlay_size(), 0);
}

// The floating leg has three spellings (float_leg | spread_leg | usd_leg): a row giving two silently kept the first.
TEST_F(ConventionsRepro, AProductGivingOneLegTwoNamesIsRejected) {
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"products": {"TEST-OIS": {"description": "x",
                   "type": "ois", "currency": "USD", "calendar": "USD", "bdc": "ModifiedFollowing", "spot_lag": 2,
                   "payment_lag": 2, "fixed_leg": {"day_count": "ACT/360", "frequency": "1Y"},
                   "float_leg": {"index": "USD-SOFR", "day_count": "ACT/360", "frequency": "1Y"},
                   "spread_leg": {"index": "USD-FEDFUNDS", "day_count": "ACT/360", "frequency": "3M"}}}}})")),
               std::invalid_argument);
  EXPECT_FALSE(cvd::product("TEST-OIS").has_value());
}

// A misspelt family was silently ignored, so the caller's rows never arrived and nothing said so.
TEST_F(ConventionsRepro, AnUnknownFamilyIsRejected) {
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"indexes": {"TEST-IDX": {"currency": "USD",
                   "type": "overnight", "day_count": "ACT/360", "calendar": "USD"}}}})")),
               std::invalid_argument);
}

// clear_overlay is part of the request's batch: it used to run first and stay done when a later row failed.
TEST_F(ConventionsRepro, ClearingInAFailingRequestKeepsTheOverlay) {
  api::conventions_json(std::string(R"({"conventions": {"indices": {"TEST-KEEP": {"currency": "USD",
      "type": "overnight", "day_count": "ACT/360", "calendar": "USD"}}}})"));
  ASSERT_TRUE(cvd::index("TEST-KEEP").has_value());
  EXPECT_THROW(api::conventions_json(std::string(R"({"conventions": {"clear_overlay": true, "indices": {"ZZZ-BAD":
                   {"currency": "USD", "type": "overnight", "day_count": "ACT/360"}}}})")),
               std::invalid_argument);
  EXPECT_TRUE(cvd::index("TEST-KEEP").has_value()) << "the failed request must not have cleared the overlay";
  EXPECT_EQ(cvd::Registry::instance().overlay_size(), 1);
}
