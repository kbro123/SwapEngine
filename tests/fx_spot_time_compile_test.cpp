// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T6 regression (fails on the reverted bug)
// O-X3 fx_spot_time, piece 2, through the compile verb and the codec (2026-09-14; owner answers). The verb quotes an FX forward's
// fx_spot for the PAIR's spot date (fx_pairs spot lag 2 on the EURUSD joint calendar) and refuses a row without fx_spot (no silent
// 1.0). Expected spot dates are hand-derived from the named holidays.
#include <fstream>
#include <sstream>
#include <string>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/compile.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/date.hpp"

namespace api = swaps::api;
namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace json = boost::json;

namespace {
json::value golden_spec() {
  std::ifstream f(std::string(SWAPS_GOLDEN_DIR) + "/compile/usd_eur_xccy.spec.json");
  EXPECT_TRUE(f.good());
  std::stringstream ss;
  ss << f.rdbuf();
  return json::parse(ss.str());
}
const cal::Instrument* first_of(const cal::BundleProblem& p, cal::QuoteKind k) {
  for (const auto& i : p.instruments) if (i.quote == k) return &i;
  return nullptr;
}
json::object* first_row(json::value& spec, const char* kind) {
  for (auto& c : spec.as_object()["curves"].as_array())
    for (auto& i : c.as_object()["instruments"].as_array())
      if (i.as_object().contains("quote_kind") && i.as_object()["quote_kind"] == kind) return &i.as_object();
  return nullptr;
}
}  // namespace

TEST(FxSpotTimeCompile, TheVerbQuotesTheFxForwardForThePairsSpotDate) {
  const json::value spec = golden_spec();
  ASSERT_EQ(spec.as_object().at("value_date").as_string(), "2026-07-08") << "premise";
  ASSERT_TRUE(b::is_business_day("EURUSD", b::Date::from_iso("2026-07-09")) && b::is_business_day("EURUSD", b::Date::from_iso("2026-07-10")))
      << "premise: Thu 07-09 and Fri 07-10 are open on SIFMA + TARGET (Independence Day observed Fri 07-03)";
  const api::CompileResult r = api::compile_spec(spec, "");
  const cal::Instrument* fx = first_of(r.bundle, cal::QuoteKind::FxForward);
  ASSERT_NE(fx, nullptr);
  EXPECT_DOUBLE_EQ(fx->fx_spot_time, 2.0 / 365.0) << "Wed 07-08 -> spot Fri 07-10";
  const cal::Instrument* xb = first_of(r.bundle, cal::QuoteKind::XccyMtmBasis);
  if (xb) EXPECT_DOUBLE_EQ(xb->mtm.fx_spot_time, 2.0 / 365.0) << "the product's spot date";
}

TEST(FxSpotTimeCompile, TheSpotDateIsCountedOnThePairCalendarNotTheCurvesIndexCalendar) {
  json::value spec = golden_spec();
  spec.as_object()["value_date"] = "2026-11-25";
  ASSERT_FALSE(b::is_business_day("EURUSD", b::Date::from_iso("2026-11-26"))) << "premise: US Thanksgiving closes the joint calendar";
  ASSERT_TRUE(b::is_business_day("EUR", b::Date::from_iso("2026-11-26"))) << "premise: TARGET is open (an index calendar would count it)";
  const api::CompileResult r = api::compile_spec(spec, "");
  const cal::Instrument* fx = first_of(r.bundle, cal::QuoteKind::FxForward);
  ASSERT_NE(fx, nullptr);
  EXPECT_DOUBLE_EQ(fx->fx_spot_time, 5.0 / 365.0) << "Wed 11-25: Fri 11-27 is day 1, Mon 11-30 day 2";
}

TEST(FxSpotTimeCompile, ARowWithoutFxSpotIsRefused) {
  for (const char* kind : {"FxForward", "XccyMtmBasis"}) {
    json::value spec = golden_spec();
    json::object* row = first_row(spec, kind);
    if (!row) continue;
    ASSERT_TRUE(row->contains("fx_spot")) << "premise: the golden row carries fx_spot";
    row->erase("fx_spot");
    EXPECT_ANY_THROW(api::compile_spec(spec, "")) << kind << ": no silent 1.0";
  }
}

TEST(FxSpotTimeCodec, TheSpotAndFixingTimesRoundTrip) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::XccyMtmBasis;
  ins.fx_spot_time = 2.0 / 365.0;
  ins.mtm.fx_spot_time = 5.0 / 365.0;
  swaps::pricing::FloatCoupon c;
  c.obs.sub_start = {0.1};
  c.obs.sub_end = {0.35};
  c.obs.tau_index = 0.25;
  c.fx_fixing_set = true;
  c.fx_fixing_time = 0.09;
  ins.mtm.coupons = {c};
  const cal::Instrument back = api::instrument_from_json(api::instrument_to_json(ins));
  EXPECT_EQ(back.fx_spot_time, ins.fx_spot_time);
  EXPECT_EQ(back.mtm.fx_spot_time, ins.mtm.fx_spot_time);
  ASSERT_EQ(back.mtm.coupons.size(), 1u);
  EXPECT_TRUE(back.mtm.coupons[0].fx_fixing_set);
  EXPECT_EQ(back.mtm.coupons[0].fx_fixing_time, 0.09);
  cal::Instrument plain;
  plain.quote = cal::QuoteKind::FxForward;
  const json::value j = api::instrument_to_json(plain);
  EXPECT_FALSE(j.as_object().contains("fx_spot_time")) << "an unset spot time is not emitted (documents stay byte-identical)";
}
