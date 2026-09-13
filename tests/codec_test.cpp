// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// The api codecs (include/swaps/api/codec.hpp, E7 stage 2) tested DIRECTLY: build JSON, call the decoder,
// inspect the struct -- no session, no verb, no run_json. Each TEST pins one clause of the header's contract:
// an absent field takes the library struct's own default (compared against a default-constructed struct,
// never a literal), a present field is carried exactly, the wrong type or an unknown name throws, and a field
// the library cannot default is required.
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/trade/csa.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;
namespace crv = swaps::curve;
namespace tr = swaps::trade;
namespace json = boost::json;

namespace {

// Every scalar the DEFAULT document carries must differ in the rich one, so the round trip that follows
// really exercises every field: a field the decoder dropped would come back as its default and could not hide.
void expect_every_scalar_differs(const json::object& rich, const json::object& dflt, const std::string& path) {
  for (const auto& kv : dflt) {
    const std::string at = path + "." + std::string(kv.key());
    ASSERT_TRUE(rich.contains(kv.key())) << at;
    const json::value& r = rich.at(kv.key());
    const json::value& d = kv.value();
    if (kv.key() == "instrument") continue;  // a Portfolio component is a whole instrument; its weight is checked
    if (d.is_object()) {
      expect_every_scalar_differs(r.as_object(), d.as_object(), at);
    } else if (d.is_array()) {
      if (!d.as_array().empty() && d.as_array()[0].is_object())
        expect_every_scalar_differs(r.as_array()[0].as_object(), d.as_array()[0].as_object(), at + "[0]");
    } else {
      EXPECT_NE(json::serialize(r), json::serialize(d)) << at << " holds its default in the fixture";
    }
  }
}

px::RateObservation rich_obs() {
  px::RateObservation o;
  o.sub_start = {0.25, 0.5};
  o.sub_end = {0.5, 0.75};
  o.weight = {0.4, 0.6};
  o.realized = 0.0012;
  o.tau_index = 0.26;
  o.fixing_step = 0.003;
  o.fixing_step3 = 0.004;
  o.compounded = true;
  o.realized_factor = 1.0007;
  o.fixing_index = "IDX";
  px::FixingDay d;
  d.fixing_date = 20300;
  d.accrual = 0.0028;
  d.t_start = 0.1;
  d.t_end = 0.11;
  d.weight = 0.5;
  o.fixing_schedule = {d};
  return o;
}

cal::FloatLeg rich_leg(int base_role) {
  px::FloatCoupon c;
  c.obs = rich_obs();
  c.pay = 0.76;
  c.tau_pay = 0.255;
  c.spread = 0.0015;
  c.scale = 0.9;
  c.reset_time = 0.2;
  c.accrual_set = true;
  c.accrual_start = 0.2;  // differs from the observation window, so it rides the wire
  c.accrual_end = 0.8;
  c.reset_fx = 1.1;
  cal::FloatLeg L;
  L.coupons = {c};
  L.forecast = base_role + 1;
  L.discount = base_role + 2;
  L.reset_num = base_role + 3;
  L.reset_den = base_role + 4;
  L.fx_spot = 1.2;
  return L;
}

cal::Instrument rich_instrument() {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::XccyMtmBasis;
  ins.fwd = rich_leg(0);
  ins.bench = rich_leg(10);
  ins.mtm = rich_leg(20);
  px::FixedCoupon x;
  x.pay = 1.0;
  x.tau = 1.01;
  x.scale = 0.95;
  ins.fixed.coupons = {x};
  ins.fixed.discount = 7;
  ins.obs = rich_obs();
  ins.forecast = 1;
  ins.convexity = -0.0002;
  ins.market = 0.0123;
  ins.pv_currency = 1;
  ins.fx_num = 2;
  ins.fx_den = 3;
  ins.fx_spot = 1.08;
  ins.fx_time = 0.5;
  ins.band_lower = 0.01;
  ins.band_upper = 0.02;
  ins.band_decay = 0.3;
  ins.turn_curve = 1;
  ins.turn_index = 2;
  cal::Instrument component;
  component.market = 0.5;
  ins.combination = {cal::WeightedInstrument{-0.5, component}};
  return ins;
}

// The default document with one default element in every array, so every nested object has keys to compare.
cal::Instrument default_shaped_instrument() {
  cal::Instrument d;
  px::FloatCoupon c;
  c.obs.fixing_schedule = {px::FixingDay{}};
  for (cal::FloatLeg* L : {&d.fwd, &d.bench, &d.mtm}) L->coupons = {c};
  d.fixed.coupons = {px::FixedCoupon{}};
  d.obs.fixing_schedule = {px::FixingDay{}};
  d.combination = {cal::WeightedInstrument{}};
  return d;
}

const char* kTrade = R"({"notional": 1000000, "pay": "fixed", "fixed_rate": 0.03, "index": "USD-SOFR",
    "effective": "2026-09-08", "maturity": "2027-09-08", "csa": {"collateral_currency": "USD"}})";
std::string trade_book(const std::string& trade) {
  return R"({"value_date": "2026-09-04", "curve_roles": {"USD-SOFR": 0}, "trades": [)" + trade + "]}";
}

}  // namespace

TEST(Codec, AnAbsentInstrumentFieldTakesTheStructsOwnDefault) {
  const cal::Instrument ins = api::instrument_from_json(json::parse(
      R"({"fwd": {"coupons": [{"obs": {"fixing_schedule": [{}]}}]}, "fixed": {"coupons": [{}]},
          "combination": [{"instrument": {}}]})"));
  const cal::Instrument di;
  EXPECT_EQ(ins.quote, di.quote);
  EXPECT_EQ(ins.forecast, di.forecast);
  EXPECT_EQ(ins.convexity, di.convexity);
  EXPECT_EQ(ins.market, di.market);
  EXPECT_EQ(ins.pv_currency, di.pv_currency);
  EXPECT_EQ(ins.fx_num, di.fx_num);
  EXPECT_EQ(ins.fx_den, di.fx_den);
  EXPECT_EQ(ins.fx_spot, di.fx_spot);
  EXPECT_EQ(ins.fx_time, di.fx_time);
  EXPECT_EQ(ins.band_lower, di.band_lower);
  EXPECT_EQ(ins.band_upper, di.band_upper);
  EXPECT_EQ(ins.band_decay, di.band_decay);
  EXPECT_EQ(ins.turn_curve, di.turn_curve);
  EXPECT_EQ(ins.turn_index, di.turn_index);

  const cal::FloatLeg dl;
  EXPECT_EQ(ins.fwd.forecast, dl.forecast);
  EXPECT_EQ(ins.fwd.discount, dl.discount);
  EXPECT_EQ(ins.fwd.reset_num, dl.reset_num);
  EXPECT_EQ(ins.fwd.reset_den, dl.reset_den);
  EXPECT_EQ(ins.fwd.fx_spot, dl.fx_spot);
  EXPECT_EQ(ins.fixed.discount, cal::FixedLeg{}.discount);

  ASSERT_EQ(ins.fwd.coupons.size(), 1u);
  const px::FloatCoupon& c = ins.fwd.coupons[0];
  const px::FloatCoupon dc;
  EXPECT_EQ(c.pay, dc.pay);
  EXPECT_EQ(c.tau_pay, dc.tau_pay);
  EXPECT_EQ(c.spread, dc.spread);
  EXPECT_EQ(c.scale, dc.scale);
  EXPECT_EQ(c.reset_time, dc.reset_time);
  EXPECT_EQ(c.accrual_set, dc.accrual_set);
  EXPECT_EQ(c.reset_fx, dc.reset_fx);

  const px::RateObservation dobs;
  EXPECT_EQ(c.obs.realized, dobs.realized);
  EXPECT_EQ(c.obs.tau_index, dobs.tau_index);
  EXPECT_EQ(c.obs.fixing_step, dobs.fixing_step);
  EXPECT_EQ(c.obs.fixing_step3, dobs.fixing_step3);
  EXPECT_EQ(c.obs.compounded, dobs.compounded);
  EXPECT_EQ(c.obs.realized_factor, dobs.realized_factor);
  EXPECT_EQ(c.obs.fixing_index, dobs.fixing_index);
  ASSERT_EQ(c.obs.fixing_schedule.size(), 1u);
  const px::FixingDay dd;
  EXPECT_EQ(c.obs.fixing_schedule[0].fixing_date, dd.fixing_date);
  EXPECT_EQ(c.obs.fixing_schedule[0].accrual, dd.accrual);
  EXPECT_EQ(c.obs.fixing_schedule[0].t_start, dd.t_start);
  EXPECT_EQ(c.obs.fixing_schedule[0].t_end, dd.t_end);
  EXPECT_EQ(c.obs.fixing_schedule[0].weight, dd.weight);

  ASSERT_EQ(ins.fixed.coupons.size(), 1u);
  const px::FixedCoupon dx;
  EXPECT_EQ(ins.fixed.coupons[0].pay, dx.pay);
  EXPECT_EQ(ins.fixed.coupons[0].tau, dx.tau);
  EXPECT_EQ(ins.fixed.coupons[0].scale, dx.scale);

  ASSERT_EQ(ins.combination.size(), 1u);
  EXPECT_EQ(ins.combination[0].weight, cal::WeightedInstrument{}.weight);
}

TEST(Codec, AnAbsentCurveOrRegFieldTakesTheStructsOwnDefault) {
  const cal::BundleProblem p =
      api::bundle_from_json(json::parse(R"({"curves": [{"regions": [{"knots": [1.0]}], "turns": [{}]}]})"));
  ASSERT_EQ(p.curves.size(), 1u);
  const cal::BundleCurveSpec ds;
  EXPECT_EQ(p.curves[0].base, ds.base);
  EXPECT_EQ(p.curves[0].currency, ds.currency);
  const crv::CurveModule dm;
  ASSERT_EQ(p.curves[0].regions.size(), 1u);
  EXPECT_EQ(p.curves[0].regions[0].scheme, dm.scheme);  // the struct now HAS a default (it was indeterminate)
  EXPECT_EQ(p.curves[0].regions[0].sigma, dm.sigma);
  EXPECT_EQ(p.curves[0].regions[0].reg_lambda, dm.reg_lambda);
  EXPECT_EQ(p.curves[0].regions[0].reg_sigma, dm.reg_sigma);
  ASSERT_EQ(p.curves[0].turns.size(), 1u);
  EXPECT_EQ(p.curves[0].turns[0].start, px::Turn{}.start);
  EXPECT_EQ(p.curves[0].turns[0].end, px::Turn{}.end);

  const api::RegSpec dr;
  for (const char* doc : {R"({})", R"({"regularize": {}})", R"({"regularize": null})"}) {
    const api::RegSpec r = api::reg_from_json(json::parse(doc).as_object());
    EXPECT_EQ(r.lambda, dr.lambda) << doc;
    EXPECT_EQ(r.curves, dr.curves) << doc;
    EXPECT_EQ(r.tension, dr.tension) << doc;
    EXPECT_EQ(r.sigma, dr.sigma) << doc;
  }
}

TEST(Codec, AnAbsentPositionFieldTakesTheStructsOwnDefault) {
  const pf::MultiCurveBook b = api::book_from_json(json::parse(R"({"positions": [{"float_coupons": [{}]}]})"));
  ASSERT_EQ(b.positions.size(), 1u);
  const pf::MultiCurveBook::Position& q = b.positions[0];
  const pf::MultiCurveBook::Position d;
  EXPECT_EQ(q.kind, d.kind);
  EXPECT_EQ(q.notional, d.notional);
  EXPECT_EQ(q.fixed_rate, d.fixed_rate);
  EXPECT_EQ(q.fwd_curve, d.fwd_curve);
  EXPECT_EQ(q.disc_curve, d.disc_curve);
  EXPECT_EQ(q.fixed_curve, d.fixed_curve);
  EXPECT_EQ(q.fx_spot, d.fx_spot);
  EXPECT_EQ(q.mtm_fwd_curve, d.mtm_fwd_curve);
  EXPECT_EQ(q.mtm_disc_curve, d.mtm_disc_curve);
  EXPECT_EQ(q.mtm_reset_num, d.mtm_reset_num);
  EXPECT_EQ(q.mtm_reset_den, d.mtm_reset_den);
}

TEST(Codec, APresentFieldIsCarriedExactly) {
  const cal::Instrument rich = rich_instrument();
  const json::value rj = api::instrument_to_json(rich);
  expect_every_scalar_differs(rj.as_object(), api::instrument_to_json(default_shaped_instrument()).as_object(),
                              "instrument");
  EXPECT_EQ(json::serialize(api::instrument_to_json(api::instrument_from_json(rj))), json::serialize(rj));

  // A bundle with every scheme, the Tension-only sigma, per-region smoothing, a turn and a spread curve.
  cal::BundleProblem p;
  cal::BundleCurveSpec outright;
  outright.currency = 1;
  double t = 0.1;
  for (crv::Scheme s : {crv::Scheme::Flat, crv::Scheme::Linear, crv::Scheme::NaturalCubic, crv::Scheme::Hermite,
                        crv::Scheme::MonotoneCubic, crv::Scheme::BSpline, crv::Scheme::Tension}) {
    crv::CurveModule m;
    m.scheme = s;
    m.knots = {t, t + 0.5, t + 1.0};
    m.reg_lambda = 0.1;
    m.reg_sigma = 0.2;
    if (s == crv::Scheme::Tension) m.sigma = 1.5;
    outright.regions.push_back(m);
    t += 2.0;
  }
  outright.turns = {px::Turn{0.3, 0.35}};
  cal::BundleCurveSpec spread;
  spread.base = 0;
  crv::CurveModule sm;
  sm.scheme = crv::Scheme::Linear;
  sm.knots = {1.0, 5.0};
  spread.regions = {sm};
  p.curves = {outright, spread};
  p.instruments = {rich};
  const json::value pj = api::bundle_to_json(p);
  EXPECT_EQ(json::serialize(api::bundle_to_json(api::bundle_from_json(pj))), json::serialize(pj));

  const pf::MultiCurveBook b = api::book_from_json(json::parse(R"({"positions": [
      {"kind": "xccy", "notional": -2.5, "fixed_rate": 0.031, "fwd_curve": 1, "disc_curve": 2, "fixed_curve": 3,
       "float_coupons": [{"pay": 0.5}], "fixed_coupons": [{"pay": 1.0, "tau": 1.0}], "fx_spot": 1.09,
       "mtm_fwd_curve": 4, "mtm_disc_curve": 5, "mtm_reset_num": 6, "mtm_reset_den": 7,
       "mtm_coupons": [{"pay": 0.25}]}]})"));
  ASSERT_EQ(b.positions.size(), 1u);
  const auto& q = b.positions[0];
  EXPECT_EQ(q.kind, pf::MultiCurveBook::Kind::Xccy);
  EXPECT_EQ(q.notional, -2.5);
  EXPECT_EQ(q.fixed_rate, 0.031);
  EXPECT_EQ(q.fwd_curve, 1);
  EXPECT_EQ(q.disc_curve, 2);
  EXPECT_EQ(q.fixed_curve, 3);
  EXPECT_EQ(q.fx_spot, 1.09);
  EXPECT_EQ(q.mtm_fwd_curve, 4);
  EXPECT_EQ(q.mtm_disc_curve, 5);
  EXPECT_EQ(q.mtm_reset_num, 6);
  EXPECT_EQ(q.mtm_reset_den, 7);
  ASSERT_EQ(q.float_coupons.size(), 1u);
  ASSERT_EQ(q.fixed_coupons.size(), 1u);
  ASSERT_EQ(q.mtm_coupons.size(), 1u);
  EXPECT_EQ(q.mtm_coupons[0].pay, 0.25);

  const api::RegSpec r = api::reg_from_json(
      json::parse(R"({"regularize": {"lambda": 0.02, "curves": [0, 2], "tension": true, "sigma": 0.7}})").as_object());
  EXPECT_EQ(r.lambda, 0.02);
  EXPECT_EQ(r.curves, (std::vector<int>{0, 2}));
  EXPECT_TRUE(r.tension);
  EXPECT_EQ(r.sigma, 0.7);
}

// The six verbs that decoded "regularize" by hand disagreed on exactly these inputs until 2026-09-13: five
// ignored a non-object, two truncated a curve index of 1.5 to 1, two read the string "true" as false.
TEST(Codec, TheWrongTypeOrAnUnknownNameThrowsInsteadOfBeingIgnoredOrTruncated) {
  const auto reg = [](const char* doc) { return api::reg_from_json(json::parse(doc).as_object()); };
  EXPECT_THROW(reg(R"({"regularize": "light"})"), std::invalid_argument);
  EXPECT_ANY_THROW(reg(R"({"regularize": {"curves": [1.5]}})"));
  EXPECT_ANY_THROW(reg(R"({"regularize": {"tension": "true"}})"));
  EXPECT_ANY_THROW(reg(R"({"regularize": {"lambda": "0.02"}})"));

  EXPECT_THROW(api::instrument_from_json(json::parse(R"({"quote": "Parrate"})")), std::invalid_argument);
  EXPECT_THROW(api::bundle_from_json(json::parse(R"({"curves": [{"regions": [{"scheme": "Cubic", "knots": [1]}]}]})")),
               std::invalid_argument);
  EXPECT_THROW(api::book_from_json(json::parse(R"({"positions": [{"kind": "fra"}]})")), std::invalid_argument);
  EXPECT_ANY_THROW(api::book_from_json(json::parse(R"({"positions": [{"notional": "1e6"}]})")));

  json::object t = json::parse(kTrade).as_object();
  t["pay"] = "receive";
  EXPECT_THROW(api::book_from_json(json::parse(trade_book(json::serialize(t)))), std::invalid_argument);
}

TEST(Codec, AFieldTheLibraryCannotDefaultIsRequired) {
  // A position's fixed-leg discount curve: it used to fall back to disc_curve in JSON and to 0 in C++.
  EXPECT_THROW(api::book_from_json(json::parse(R"({"positions": [{"disc_curve": 2, "fixed_coupons": [{"pay": 1.0}]}]})")),
               std::invalid_argument);
  EXPECT_EQ(api::book_from_json(json::parse(
                R"({"positions": [{"disc_curve": 2, "fixed_curve": 2, "fixed_coupons": [{"pay": 1.0}]}]})"))
                .positions[0].fixed_curve,
            2);

  // A booked trade's economic terms: a deal with no notional / direction / rate / dates is an error, not 1 / fixed / 0.
  ASSERT_EQ(api::book_from_json(json::parse(trade_book(kTrade))).positions.size(), 1u);
  for (const char* k : {"notional", "pay", "fixed_rate", "index", "effective", "maturity"}) {
    json::object t = json::parse(kTrade).as_object();
    t.erase(k);
    EXPECT_THROW(api::book_from_json(json::parse(trade_book(json::serialize(t)))), std::invalid_argument) << k;
  }
}

// The discount-index rule a typed trade uses lives in the trade library now, so a C++ caller gets the same rule.
TEST(Codec, TheTradeDiscountIndexRuleIsALibraryFunction) {
  const std::string usd_ois = tr::CSA::cash("USD").discount_index_id();
  ASSERT_FALSE(usd_ois.empty());
  EXPECT_EQ(tr::discount_index_for(tr::CSA::cash("USD"), ""), usd_ois);
  EXPECT_EQ(tr::discount_index_for(tr::CSA::cash("USD"), "EUR-ESTR"), usd_ois);  // the CSA decides when present
  EXPECT_EQ(tr::discount_index_for(std::nullopt, "EUR-ESTR"), "EUR-ESTR");
  EXPECT_THROW(tr::discount_index_for(std::nullopt, ""), std::invalid_argument);
}

TEST(Codec, AStreetBondRequestDecodesTermsAndTreatsNullAsAbsent) {
  const auto r = api::street_bond_request_from_json(json::parse(R"({"bonds": [
      {"convention": "US-TREASURY-TSY", "settle": "2024-06-15", "maturity": "2034-11-15", "coupon": 0.045,
       "dated": "2024-06-15", "first_coupon": "2024-11-15", "freq": 2, "clean": null, "yield": 0.047}]})").as_object());
  EXPECT_FALSE(r.value_date.has_value());
  ASSERT_EQ(r.bonds.size(), 1u);
  const auto& q = r.bonds[0];
  EXPECT_EQ(q.terms.convention, "US-TREASURY-TSY");
  EXPECT_EQ(q.terms.coupon, 0.045);
  EXPECT_FALSE(q.terms.issue.has_value());
  ASSERT_TRUE(q.terms.dated.has_value());
  ASSERT_TRUE(q.terms.first_coupon.has_value());
  EXPECT_EQ(q.terms.freq, std::optional<int>(2));
  EXPECT_FALSE(q.clean.has_value()) << "an explicit null is absent";
  EXPECT_EQ(q.yield, std::optional<double>(0.047));
  for (const char* k : {"settle", "maturity", "coupon"}) {
    json::object bond = json::parse(R"({"settle": "2024-01-16", "maturity": "2029-08-15", "coupon": 0.025})").as_object();
    bond.erase(k);
    json::object req;
    req["bonds"] = json::array{bond};
    EXPECT_THROW(api::street_bond_request_from_json(req), std::invalid_argument) << k;
  }
}

TEST(Codec, ADeliveryBasketRequiresItsMarketInputsAndDefersContractFieldsToTheLibrary) {
  const char* doc = R"({"contract": "CME-TY", "value_date": "2008-11-20", "first_delivery": "2008-12-01",
      "futures_price": 1.24, "repo": 0.005,
      "basket": [{"issue": "2008-11-15", "maturity": "2018-11-15", "coupon": 0.0375, "clean": 1.03}]})";
  const auto r = api::delivery_basket_request_from_json(json::parse(doc).as_object());
  EXPECT_FALSE(r.delivery.has_value());
  EXPECT_FALSE(r.notional_coupon.has_value()) << "the contract row supplies it, in the library";
  EXPECT_FALSE(r.round_months.has_value());
  ASSERT_EQ(r.basket.size(), 1u);
  EXPECT_TRUE(r.basket[0].convention.empty()) << "empty = the contract's deliverable convention";
  EXPECT_FALSE(r.basket[0].settle.has_value());
  // Until 2026-09-13 a missing futures_price or repo silently priced at 0.
  for (const char* k : {"contract", "value_date", "first_delivery", "futures_price", "repo", "basket"}) {
    json::object o = json::parse(doc).as_object();
    o.erase(k);
    EXPECT_THROW(api::delivery_basket_request_from_json(o), std::invalid_argument) << k;
  }
}
