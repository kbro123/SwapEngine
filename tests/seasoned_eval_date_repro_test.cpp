// E5 taxonomy: T6 regression (fails on the reverted bug)
// P12 REPRODUCTION (2026-09-14, found by portfolio_verb_oracle_test.cpp's seasoned-trade control): a SEASONED coupon was
// silently FORECAST when the session had no evaluation date. A typed trade effective before the book's value date
// builds its accruing coupon with a fixing schedule (build::float_leg_from); BundleSession resolves it against
// {eval_date_, fixings_}, but a stateless session's eval_date_ is 0 (never set from the request), and
// pricing::resolve called a day past only when fixing_date < evaluation_date -- so every 2025 day looked like the
// future and the realized year was forecast off a curve that starts at the value date, at negative curve times.
// The `portfolio` verb returned NPV -284854 / PV01 3827 for a trade it cannot value. A day that starts before curve
// time 0 is in the past by construction; without an evaluation date and a fixing for it, pricing must refuse.
#include <string>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include "swaps/api/bundle_api.hpp"
#include "swaps/build/date.hpp"
#include "swaps/pricing/fixings.hpp"

namespace api = swaps::api;
namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace px = swaps::pricing;
namespace json = boost::json;

TEST(SeasonedEvalDateRepro, ADayBeforeCurveTimeZeroIsPastEvenWithNoEvaluationDate) {
  px::FixingSchedule sch;
  sch.index = "USD-SOFR";
  sch.tau_index = 1.0;
  sch.compounded = true;
  sch.days.push_back(px::FixingDay{int(b::Date::from_iso("2025-07-10").serial()), 1.0 / 360.0, -0.99, -0.987, 1.0});
  const px::PricingContext no_date{};  // no evaluation date, no fixings table
  EXPECT_THROW((void)px::resolve(sch, no_date), px::MissingFixing)
      << "a day starting at curve time -0.99 was forecast as if it were in the future";
}

TEST(SeasonedEvalDateRepro, APortfolioRequestWithASeasonedTradeIsRefusedNotForecast) {
  cal::BundleProblem p;
  crv::CurveModule flat;
  flat.scheme = crv::Scheme::Flat;
  flat.knots = {1.0};
  cal::BundleCurveSpec sofr;
  sofr.regions = {flat};
  p.curves = {sofr};
  cal::Instrument pin;
  pin.quote = cal::QuoteKind::Rate;
  pin.forecast = 0;
  pin.obs.sub_start = {0.0};
  pin.obs.sub_end = {1.0};
  pin.obs.tau_index = 1.0;
  pin.market = std::exp(0.03) - 1.0;
  p.instruments = {pin};

  const json::object trade{{"id", "SEASONED"},       {"index", "USD-SOFR"},     {"currency", "USD"},
                           {"notional", 1e6},        {"pay", "fixed"},          {"fixed_rate", 0.03},
                           {"effective", "2025-07-10"}, {"maturity", "2030-07-10"}, {"discount_index", "USD-SOFR"}};
  const json::object req{{"bundle", api::bundle_to_json(p)},
                         {"portfolio", json::object{{"value_date", "2026-07-08"},
                                                    {"curve_roles", json::object{{"USD-SOFR", 0}}},
                                                    {"trades", json::array{trade}}}}};
  const json::object out = json::parse(api::run_json(req)).as_object();
  ASSERT_TRUE(out.contains("error")) << "a seasoned trade was priced: " << json::serialize(out.at("portfolio"));
  EXPECT_NE(std::string(out.at("error").as_string().c_str()).find("fixing"), std::string::npos) << json::serialize(out);
}
