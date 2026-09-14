// @oracle-test — FX forward OUTRIGHT (QuoteKind::FxForward) vs QuantLib FxSwapRateHelper with the spot quoted for the SPOT DATE
// DO NOT DELETE OR WEAKEN without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5 taxonomy: T1 oracle (engine number vs an independent number) | T6 regression (fails on the reverted bug)
//
// O-X3 (owner 2026-09-14): fx_spot is the SPOT-DATE quote and fx_spot_time the curve time of that spot date. QuantLib core
// (ql/termstructures/yield/ratehelpers.hpp -- NOT experimental) FxSwapRateHelper::impliedQuote() returns forward POINTS for
// delivery latestDate against the spot for earliestDate = calendar.advance(adjust(eval), fixingDays):
//   points = (collRatio/ratio - 1) * spot,  ratio = DF_eur(spot)/DF_eur(T), collRatio = DF_usd(spot)/DF_usd(T)
// (isFxBaseCurrencyCollateralCurrency = false: EUR base, USD collateral). Outright = spot + points
//   = S * DF_eur(T)/DF_usd(T) * DF_usd(t_s)/DF_eur(t_s).
// Both sides read the SAME discount factors (engine CurveHandles through swaps::qlx::CurveTermStructure, ACT/365F == the
// engine's curve_time) and QuantLib's own spot/delivery dates, fed to the engine as curve times. Only the outright formula's
// treatment of the spot date is compared.
//
// ASSERTIONS                                                                        TOLERANCE
//   engine outright (fx_spot S, fx_time T, fx_spot_time t_s) vs S + impliedQuote()   tol::curve_rel (identical DFs)
//   engine FX residual at the QuantLib outright                                       tol::curve_rel / T
//   premise: QuantLib's spot date == engine spot_date on the DB EURUSD calendar      exact
// NEGATIVE CONTROLS
//   N1 t_s = 0 (today's t = 0 reading) misses by > half the predicted DF_eur(t_s)/DF_usd(t_s) - 1
//   N2 t_s = T (spot date taken as the delivery date) misses
// REPRODUCED pre-field on 8770c32: the t = 0 reading missed QuantLib by exactly the predicted 1.15e-4 (T+2) and 2.88e-4
// (the Thanksgiving T+5 spot) relative; with fx_spot_time the two agree to ~2e-16.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "conventions_ql.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "tolerances.hpp"

namespace {

namespace ql = QuantLib;
namespace bld = swaps::build;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace qconv = swaps::refbuild::conv;
namespace tol = swaps::tol;

ql::Date qd(const bld::Date& d) { return ql::Date(ql::Day(d.day()), ql::Month(d.month()), ql::Year(d.year())); }

constexpr double kS = 1.10;
constexpr int kEur = 0, kUsd = 1;

// Two engine curves with non-flat shape: 0 = EUR discounted under USD collateral, 1 = USD SOFR.
struct Curves {
  std::vector<cal::BundleCurveSpec> specs;
  Eigen::VectorXd x;
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> C;
  Curves() {
    const std::vector<double> knots{7.0 / 365.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0};
    specs.push_back({.base = -1, .currency = 1, .regions = cv::flat_hermite({}, knots)});
    specs.push_back({.base = -1, .currency = 0, .regions = cv::flat_hermite({}, knots)});
    const int n = specs[0].n_interp_knots();
    x.resize(2 * n);
    for (int i = 0; i < n; ++i) {
      x[i] = 0.0195 + 0.0004 * i;
      x[n + i] = 0.0405 - 0.0006 * i;
    }
    C = cal::build_bundle_curves<double>(specs, [&](int c, int i) { return x[c * n + i]; });
  }
  const cal::CurveHandle<double>& operator()(int i) const { return *C[static_cast<std::size_t>(i)]; }
};
struct HandleCurve {
  const cal::CurveHandle<double>* h = nullptr;
  double discount(double t) const { return h->discount(t); }
};

struct Case { const char* value_date; const char* tenor; const char* why; };
const std::vector<Case>& cases() {
  static const std::vector<Case> c = {
      {"2026-09-15", "1W", "T+2 = Thu 09-17; a 1W outright carries the largest zero error (~eps/T)"},
      {"2026-09-15", "1M", "T+2, 1M"},
      {"2026-09-15", "3M", "T+2, 3M"},
      {"2026-09-15", "1Y", "T+2, 1Y"},
      {"2026-11-25", "1W", "US Thanksgiving on T+1: joint SIFMA+TARGET spot Mon 11-30, t_s = 5/365"},
      {"2026-11-25", "1Y", "Thanksgiving spot, 1Y"},
  };
  return c;
}

}  // namespace

TEST(FxForwardSpotDateOracle, OutrightMatchesFxSwapRateHelperWithTheSpotForTheSpotDate) {
  const bld::XccyConv x = bld::xccy_conv("EURUSD");  // calendar EURUSD, spot_lag 2 (the fx_pairs row agrees)
  for (const Case& cs : cases()) {
    SCOPED_TRACE(std::string(cs.value_date) + " " + cs.tenor);
    const bld::Date vd = bld::Date::from_iso(cs.value_date);
    const ql::Date today = qd(vd);
    ql::Settings::instance().evaluationDate() = today;
    const Curves cur;
    HandleCurve heur{cur.C[kEur].get()}, husd{cur.C[kUsd].get()};
    auto eur_ts = ql::ext::make_shared<swaps::qlx::CurveTermStructure<HandleCurve>>(today, ql::Actual365Fixed(), &heur);
    auto usd_ts = ql::ext::make_shared<swaps::qlx::CurveTermStructure<HandleCurve>>(today, ql::Actual365Fixed(), &husd);
    ql::Handle<ql::YieldTermStructure> usd_h(usd_ts);
    ql::FxSwapRateHelper helper(ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(0.0)),
                                ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(kS)),
                                qconv::period(cs.tenor), static_cast<ql::Natural>(x.spot_lag),
                                qconv::calendar(x.calendar), ql::Following, /*endOfMonth=*/false,
                                /*isFxBaseCurrencyCollateralCurrency=*/false, usd_h);
    helper.setTermStructure(eur_ts.get());

    const ql::Date spot = helper.earliestDate(), delivery = helper.latestDate();
    ASSERT_EQ(spot, qd(bld::spot_date(vd, x.calendar, x.spot_lag))) << "premise: QuantLib and the DB agree on the spot date";
    const double t_s = static_cast<double>(spot - today) / 365.0;
    const double T = static_cast<double>(delivery - today) / 365.0;
    ASSERT_GT(t_s, 0.0) << "premise: a lagged spot";
    const double F_ql = kS + helper.impliedQuote();
    const double predicted = cur(kEur).discount(t_s) / cur(kUsd).discount(t_s) - 1.0;  // the t = 0 reading's gap
    ASSERT_GT(std::abs(predicted), 1e3 * tol::curve_rel) << "premise: the rate gap makes the defect visible";

    cal::Instrument ins = bld::fx_forward(kEur, kUsd, kS, T, F_ql);
    ins.fx_spot_time = t_s;
    const double F = cal::instrument_model_quote<double>(ins, cur);
    const double rel = F / F_ql - 1.0;
    std::cout << "  [fx-spot] " << cs.value_date << " " << cs.tenor << " t_s=" << t_s * 365 << "d T=" << T * 365
              << "d rel=" << rel << " (t=0 reading predicts " << predicted << ")\n";
    EXPECT_LE(std::abs(rel), tol::curve_rel) << cs.why << ": off QuantLib by " << rel << " relative = "
                                            << rel * F_ql * 1e4 << " pips";
    EXPECT_LE(std::abs(cal::instrument_residual<double>(ins, cur)), tol::curve_rel / T);

    cal::Instrument n1 = bld::fx_forward(kEur, kUsd, kS, T, F_ql);  // N1: t_s left at 0
    EXPECT_GT(std::abs(cal::instrument_model_quote<double>(n1, cur) / F_ql - 1.0), 0.5 * std::abs(predicted));
    cal::Instrument n2 = bld::fx_forward(kEur, kUsd, kS, T, F_ql);  // N2: spot date == delivery date
    n2.fx_spot_time = T;
    EXPECT_GT(std::abs(cal::instrument_model_quote<double>(n2, cur) / F_ql - 1.0), 1e3 * tol::curve_rel);
  }
}
