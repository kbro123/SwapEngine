// E5 taxonomy: T6 regression (fails on the reverted bug)
// P12 REPRODUCTION (SC2, owner decision 2026-09-14: build exact per-pair FX scaling).
//
// Drives the three verbs through JSON only; the one field the fix adds (bundle.currency_codes) is a JSON key the decoder
// ignored when this was written. Run against the unfixed code, every TEST below failed except where noted; after the
// SC2 fix the file passes unchanged.
//
// THE BUG. A MultiCurveBook position carries no pair, so `scenario` (derive/scenario.hpp resolve_scenario_move), `scenario_grid`
// (derive/scenario_grid.hpp add_axis_shock) and `var` (derive/var.hpp var_reval) compound EVERY bump_fx in a move into
// ONE factor and multiply EVERY xccy position's fx_spot by it (portfolio/xccy_fx_scaled.hpp). With two xccy
// positions on two pairs, EURUSD +2% and GBPUSD -1% give BOTH positions 1.02 * 0.99 = 1.0098. The GBPUSD position then
// GAINS where it should lose, and the EURUSD one gains about half what it should. A single-pair bump still reaches the
// other pair's position, and a bump named in the inverse orientation (USD/EUR) scales an EURUSD spot the wrong way.
//
// THE BOOK. Three flat one-knot curves (tag 0 USD, 1 EUR, 2 GBP), each pinned by one [0,1] Rate quote. Two one-period
// xccy positions, domestic leg on USD (curve 0), resetting foreign leg forecast + discounted on its own curve, spread
// 5 bp on the foreign leg:
//   A: EURUSD (num = curve 1, den = curve 0), fx_spot 1.10, notional 3e7
//   B: GBPUSD (num = curve 2, den = curve 0), fx_spot 1.30, notional 2e7
// For this shape the value is CLOSED FORM (pricing/cashflows.hpp float_coupon_pv + xccy_mtm_leg_pv, portfolio.hpp
// :130-162): the domestic leg is (1 - DF0(1)) + DF0(1) - DF0(0) = 0, and the foreign bracket is
// DF(1)*((DF(0)/DF(1) - 1) + spread) + DF(1) - DF(0) = spread*DF(1), with reset notional fx_spot * DF_num(0)/DF_den(0) =
// fx_spot. So value = notional * fx_spot * spread * DF_foreign(1), LINEAR in fx_spot.
//
// TWO INDEPENDENT ARMS PER CHECK:
//   identity   : a SINGLE-position book under a move naming only its own pair is priced exactly today (one position,
//                one factor, same orientation), and a book's value is the plain sum of its positions
//                (MultiCurveBook::value), so the two-pair book under the joint move is the sum of the one-position
//                books, bit for bit on the templated `scenario` path.
//   closed form: notional * (fx_spot * factor) * spread * DF_foreign(1), with DF read from the verb's OWN base sample
//                at t = 1 (so the calibration is not under test here).
// Hand numbers at the construction state (DF_EUR(1) = e^-0.02, DF_GBP(1) = e^-0.04; computed in python, not the engine):
//   base 286635.408; exact joint-move delta +1985.629; TODAY +2809.027 (error +823.40, 41 % of the move).
//   EURUSD +2% alone: exact +3234.656; TODAY +5732.708 (error +2498.05: the GBPUSD position moved too).
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <boost/json.hpp>
#include <gtest/gtest.h>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/api/scenario.hpp"
#include "swaps/api/scenario_grid.hpp"
#include "swaps/api/var.hpp"
#include "tolerances.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace json = boost::json;

namespace {

constexpr double kSpread = 0.005;
constexpr double kSpotA = 1.10, kNotionalA = 3e7;  // EURUSD
constexpr double kSpotB = 1.30, kNotionalB = 2e7;  // GBPUSD
constexpr double kGross = kNotionalA * kSpotA + kNotionalB * kSpotB;

cal::BundleProblem three_currency_bundle() {
  cal::BundleProblem p;
  for (int ccy : {0, 1, 2}) {
    crv::CurveModule flat;
    flat.scheme = crv::Scheme::Flat;
    flat.knots = {1.0};
    cal::BundleCurveSpec s;
    s.currency = ccy;
    s.regions = {flat};
    p.curves.push_back(s);
  }
  for (int c = 0; c < 3; ++c) {
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::Rate;
    ins.forecast = c;
    ins.obs.sub_start = {0.0};
    ins.obs.sub_end = {1.0};
    ins.obs.tau_index = 1.0;
    p.instruments.push_back(ins);
  }
  const Eigen::Vector3d state(0.03, 0.02, 0.04);  // USD, EUR, GBP flat forwards
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return state[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return p;
}

json::value bundle_json() {
  json::value b = api::bundle_to_json(three_currency_bundle());
  // The ISO code of each curve currency tag. Ignored by today's decoder; read by the SC2 fix
  // (BundleProblem::currency_codes). Order matches the tags above.
  b.as_object()["currency_codes"] = json::array{"USD", "EUR", "GBP"};
  return b;
}

json::array one_period_leg(double spread) {
  return json::array{json::object{
      {"obs", json::object{{"sub_start", json::array{0.0}}, {"sub_end", json::array{1.0}}, {"tau_index", 1.0}}},
      {"pay", 1.0},
      {"tau_pay", 1.0},
      {"spread", spread}}};
}

json::object xccy(int foreign, double fx_spot, double notional) {
  return json::object{{"kind", "xccy"},          {"notional", notional},          {"fwd_curve", 0},
                      {"disc_curve", 0},         {"float_coupons", one_period_leg(0.0)},
                      {"mtm_coupons", one_period_leg(kSpread)},                   {"mtm_fwd_curve", foreign},
                      {"mtm_disc_curve", foreign}, {"mtm_reset_num", foreign},    {"mtm_reset_den", 0},
                      {"fx_spot", fx_spot}};
}
json::object eurusd() { return xccy(1, kSpotA, kNotionalA); }
json::object gbpusd() { return xccy(2, kSpotB, kNotionalB); }

json::object bump(const char* base, const char* quote, double rel) {
  return json::object{{"base", base}, {"quote", quote}, {"rel", rel}};
}

struct Scen {
  double base_npv = 0.0, npv = 0.0, df_eur = 0.0, df_gbp = 0.0;
};

// One `scenario` call with a single move {bump_fx: bumps}; the verb's own base DFs at t = 1 are read back.
Scen scenario(json::array positions, json::array bumps, const char* fx_pivot = nullptr) {
  json::object req{{"bundle", bundle_json()},
                   {"book", json::object{{"positions", std::move(positions)}}},
                   {"sample_times", json::array{1.0}},
                   {"scenarios", json::array{json::object{{"name", "fx"}, {"bump_fx", std::move(bumps)}}}}};
  if (fx_pivot) req["fx_pivot"] = fx_pivot;
  const json::value v = json::parse(api::scenario_json(json::object{{"scenario", std::move(req)}}));
  const json::object& out = v.as_object().at("scenario").as_object();
  const json::object& base = out.at("base").as_object();
  const auto df = [&base](int c) {
    return base.at("curves").as_array()[static_cast<std::size_t>(c)].as_object().at("discount").as_array()[0].to_number<double>();
  };
  Scen s;
  s.base_npv = base.at("npv").to_number<double>();
  s.npv = out.at("scenarios").as_array()[0].as_object().at("npv").to_number<double>();
  s.df_eur = df(1);
  s.df_gbp = df(2);
  return s;
}

// The closed-form value of a one-period position of this shape at fx_spot * factor.
double closed_form(double notional, double fx_spot, double factor, double df_foreign) {
  return notional * (fx_spot * factor) * kSpread * df_foreign;
}

}  // namespace

TEST(ScenarioFxPairsRepro, EachXccyPositionMovesWithItsOwnPair) {
  const Scen both = scenario(json::array{eurusd(), gbpusd()},
                             json::array{bump("EUR", "USD", 0.02), bump("GBP", "USD", -0.01)});
  const Scen a = scenario(json::array{eurusd()}, json::array{bump("EUR", "USD", 0.02)});
  const Scen b = scenario(json::array{gbpusd()}, json::array{bump("GBP", "USD", -0.01)});

  EXPECT_EQ(both.npv, a.npv + b.npv)
      << "identity: the two-pair book is the sum of each position under its OWN pair (TODAY both take 1.02 * 0.99)";
  const double want = closed_form(kNotionalA, kSpotA, 1.02, both.df_eur) + closed_form(kNotionalB, kSpotB, 0.99, both.df_gbp);
  EXPECT_NEAR(both.npv, want, swaps::tol::analytics_rel * kGross) << "closed form (TODAY off by about +823)";
  EXPECT_NEAR(both.base_npv,
              closed_form(kNotionalA, kSpotA, 1.0, both.df_eur) + closed_form(kNotionalB, kSpotB, 1.0, both.df_gbp),
              swaps::tol::analytics_rel * kGross)
      << "the fixture is the closed form it claims (passes today: no FX move on the base)";
}

TEST(ScenarioFxPairsRepro, ABumpOnOnePairLeavesAnotherPairsPositionAlone) {
  // Owner rule (2026-09-14): under an EURUSD-only bump, GBPUSD is set by what the unbumped currencies hold against.
  // With fx_pivot USD, GBP holds against USD, so GBPUSD does not move; with no pivot the move is refused.
  const Scen both = scenario(json::array{eurusd(), gbpusd()}, json::array{bump("EUR", "USD", 0.02)}, "USD");
  const Scen a = scenario(json::array{eurusd()}, json::array{bump("EUR", "USD", 0.02)});
  const Scen b = scenario(json::array{gbpusd()}, json::array{});
  EXPECT_EQ(both.npv, a.npv + b.npv) << "GBPUSD is not bumped (before SC2 it took 1.02 too: +2498 too much)";
  EXPECT_THROW((void)scenario(json::array{eurusd(), gbpusd()}, json::array{bump("EUR", "USD", 0.02)}),
               std::invalid_argument)
      << "no fx_pivot: the move does not determine GBPUSD";
}

TEST(ScenarioFxPairsRepro, ABumpNamedInTheInverseOrientationDividesTheSpot) {
  // USD/EUR +2% is EUR/USD * 1/1.02. TODAY the pair is ignored and the EURUSD spot is multiplied by 1.02.
  const Scen inv = scenario(json::array{eurusd()}, json::array{bump("USD", "EUR", 0.02)});
  EXPECT_NEAR(inv.npv, closed_form(kNotionalA, kSpotA, 1.0 / 1.02, inv.df_eur), swaps::tol::analytics_rel * kGross);
}

TEST(ScenarioFxPairsRepro, ACrossPairTriangulatesThroughTheBumpedPairs) {
  // An EURGBP position (num EUR curve 1, den GBP curve 2) under EURUSD +2% and GBPUSD -1%: EURGBP = EURUSD / GBPUSD moves
  // by 1.02 / 0.99. Its domestic leg is on GBP (curve 2), so the closed form is unchanged in shape.
  json::object cross = xccy(1, 0.85, 1e7);
  cross["fwd_curve"] = 2;
  cross["disc_curve"] = 2;
  cross["mtm_reset_den"] = 2;
  const Scen s = scenario(json::array{cross}, json::array{bump("EUR", "USD", 0.02), bump("GBP", "USD", -0.01)});
  EXPECT_NEAR(s.npv, closed_form(1e7, 0.85, 1.02 / 0.99, s.df_eur), swaps::tol::analytics_rel * kGross)
      << "TODAY the factor is 1.02 * 0.99";
}

TEST(ScenarioFxPairsRepro, AGridOfTwoFxAxesMovesEachPositionWithItsAxis) {
  const auto grid = [](json::array positions, json::array axes) {
    json::object req{{"bundle", bundle_json()},
                     {"book", json::object{{"positions", std::move(positions)}}},
                     {"axes", std::move(axes)}};
    return json::parse(api::scenario_grid_json(json::object{{"scenario_grid", std::move(req)}}))
        .as_object().at("scenario_grid").as_object().at("npv").as_array();
  };
  const json::object eur_axis{{"label", "eurusd"}, {"kind", "fx"}, {"base", "EUR"}, {"quote", "USD"},
                              {"values", json::array{-0.05, 0.05}}};
  const json::object gbp_axis{{"label", "gbpusd"}, {"kind", "fx"}, {"base", "GBP"}, {"quote", "USD"},
                              {"values", json::array{-0.03, 0.03}}};
  const json::array both = grid(json::array{eurusd(), gbpusd()}, json::array{eur_axis, gbp_axis});
  const json::array a = grid(json::array{eurusd()}, json::array{eur_axis});
  const json::array b = grid(json::array{gbpusd()}, json::array{gbp_axis});
  const auto at = [](const json::array& g, int i, int j) {
    return g[static_cast<std::size_t>(i)].as_array()[static_cast<std::size_t>(j)].to_number<double>();
  };
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      EXPECT_NEAR(at(both, i, j), at(a, i, 0) + at(b, j, 0), swaps::tol::parity_value * kGross)
          << "cell " << i << "," << j << " (compiled rows sum in a different grouping: rounding only)";
}

TEST(ScenarioFxPairsRepro, VarRevaluesATwoPairMoveExactly) {
  const Scen s = scenario(json::array{eurusd(), gbpusd()}, json::array{});  // the base DFs
  json::object req{{"bundle", bundle_json()},
                   {"book", json::object{{"positions", json::array{eurusd(), gbpusd()}}}},
                   {"scenarios", json::array{json::object{{"bump_fx", json::array{bump("EUR", "USD", 0.02),
                                                                                   bump("GBP", "USD", -0.01)}}},
                                             json::object{}}}};
  const json::object out =
      json::parse(api::var_json(json::object{{"var", std::move(req)}})).as_object().at("var").as_object();
  const json::array& sorted = out.at("pnl_sorted").as_array();
  ASSERT_EQ(sorted.size(), 2u);
  const double pnl = sorted[0].to_number<double>() == 0.0 ? sorted[1].to_number<double>() : sorted[0].to_number<double>();
  const double want = (closed_form(kNotionalA, kSpotA, 1.02, s.df_eur) - closed_form(kNotionalA, kSpotA, 1.0, s.df_eur)) +
                      (closed_form(kNotionalB, kSpotB, 0.99, s.df_gbp) - closed_form(kNotionalB, kSpotB, 1.0, s.df_gbp));
  EXPECT_NEAR(pnl, want, swaps::tol::analytics_rel * kGross) << "exact +1985.63 at the construction state; TODAY +2809.03";
}

// The owner rule (2026-09-14): an FX bump on a book with cross-currency positions needs bundle.currency_codes -- without
// them a position's pair orientation is unknowable, so the request is refused rather than approximated. A request with
// no FX bump needs no codes.
TEST(ScenarioFxPairsRepro, AnFxMoveOnAnXccyBookNeedsTheBundlesCurrencyCodes) {
  json::object req{{"bundle", api::bundle_to_json(three_currency_bundle())},
                   {"book", json::object{{"positions", json::array{eurusd()}}}},
                   {"scenarios", json::array{json::object{{"bump_fx", json::array{bump("EUR", "USD", 0.02)}}}}}};
  EXPECT_THROW((void)api::scenario_json(json::object{{"scenario", req}}), std::invalid_argument);
  req["scenarios"] = json::array{json::object{{"parallel_bp", 1.0}}};
  EXPECT_NO_THROW((void)api::scenario_json(json::object{{"scenario", req}})) << "no FX move: no codes needed";
}
