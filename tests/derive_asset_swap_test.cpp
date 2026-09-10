// E5 taxonomy: T2 calibration (optimum / stationarity / recovery)
// derive/asset_swap.hpp — the market->calibration bridge for asset swaps, and its RV tie-in. QuantLib-free.
//
// Two workflows over ONE govvie-curve substrate:
//   * HEADLINE swap spread: derive_asset_swap reads the benchmark bond's market clean price from the Market,
//     inverts it to a street yield, and emits the {pin, asw} calibration rows — the yield precomputed off
//     the Market, never on the calibration hot path.
//   * RV / MINIMUM PRICING ERROR: make_govvie_fit builds a GovvieBondFit over a bond UNIVERSE priced from
//     the Market; cal::calibrate fits the govvie curve by least squares, and CompiledBondBook::z_spreads on
//     the SAME curve is the per-bond richness/cheapness ladder. Proves the substrate is RV-ready today.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "swaps/build/bond.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/date.hpp"
#include "swaps/calibration/bond_fit.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/parametric.hpp"
#include "swaps/derive/asset_swap.hpp"
#include "swaps/market/market.hpp"
#include "swaps/market/quote.hpp"
#include "swaps/portfolio/bond_universe.hpp"

namespace cal = swaps::calibration;
namespace bd = swaps::build;
namespace cv = swaps::curve;
namespace der = swaps::derive;
namespace mkt = swaps::market;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

bd::Date iso(const char* s) { return bd::Date::from_iso(s); }

// A local par swap on `curve` (annual, year-fraction grid) — the matched swap the spread is quoted against.
cal::Instrument par_swap(double T, int curve) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = curve;
  ins.fwd.discount = curve;
  ins.fixed.discount = curve;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    px::FloatCoupon fc;
    fc.obs.sub_start = {prev};
    fc.obs.sub_end = {t};
    fc.obs.tau_index = t - prev;
    fc.pay = t;
    fc.tau_pay = t - prev;
    ins.fwd.coupons.push_back(fc);
    px::FixedCoupon xc;
    xc.pay = t;
    xc.tau = 1.0;
    ins.fixed.coupons.push_back(xc);
    prev = t;
  }
  return ins;
}

der::AssetSwapConvention usd_headline() {
  return {.currency = "USD",
          .type = der::SwapSpreadType::HeadlineYield,
          .govvie_curve = "USD-GOVVIE",
          .swap_index = "USD-SOFR",
          .settle_calendar = "USD",
          .settle_lag = 1};
}

}  // namespace

TEST(DeriveAssetSwap, HeadlineSpreadDerivesTheBenchmarkYieldAndRowsFromTheMarket) {
  const bd::Date today = iso("2026-09-04");
  const der::AssetSwapConvention conv = usd_headline();
  const bd::BondId otr5{.id = "UST-5Y",
                          .yield_conv = "US-TREASURY",
                          .issue = iso("2026-08-15"),
                          .maturity = iso("2031-08-15"),
                          .coupon = 0.04};
  const double clean = 0.991;      // the benchmark's market clean price (per unit notional)
  const double spread = -0.0032;   // the quoted 5y swap spread (−32bp)

  mkt::Market m;
  m.as_of(today).add_quote(otr5.id, mkt::Quote::mid(clean)).add_quote("USD-SS-5Y", mkt::Quote::mid(spread));

  // Direct yield via the SAME settlement + convention, to cross-check the derivation.
  const bd::Date settle = bd::advance_bd(conv.settle_calendar, today, conv.settle_lag);
  const bd::BuiltBond bb =
      bd::bond_from_convention(otr5.yield_conv, today, settle, otr5.issue, otr5.maturity, otr5.coupon);
  const double y_direct = px::bond_yield_from_clean(bb.yield, clean);

  const cal::Instrument spot = par_swap(5.0, /*swap curve*/ 0);
  const der::DerivedAssetSwap d = der::derive_asset_swap(conv, otr5, spot, /*factor_curve*/ 1, /*anchor*/ 5.0,
                                                         m, "USD-SS-5Y");

  EXPECT_NEAR(d.bond_yield, y_direct, 1e-12) << "benchmark_yield must invert the market clean price";
  EXPECT_GT(d.bond_yield, otr5.coupon) << "a sub-par bond yields above its coupon";
  EXPECT_NEAR(d.spread, spread, 1e-15);

  // The rows: pin = Rate on the govvie factor at the derived yield; asw = Portfolio{+1 swap, −1 Rate} at the
  // spread. This is what makes the asset swap a first-class basis (see asset_swap_spread_test).
  EXPECT_EQ(d.rows.pin.quote, cal::QuoteKind::Rate);
  EXPECT_EQ(d.rows.pin.forecast, 1);
  EXPECT_NEAR(d.rows.pin.market, y_direct, 1e-12);
  ASSERT_EQ(d.rows.asw.quote, cal::QuoteKind::Portfolio);
  ASSERT_EQ(d.rows.asw.combination.size(), 2u);
  EXPECT_EQ(d.rows.asw.combination[0].weight, +1.0);
  EXPECT_EQ(d.rows.asw.combination[1].weight, -1.0);
  EXPECT_NEAR(d.rows.asw.market, spread, 1e-15);
}

TEST(DeriveAssetSwap, RvMinimumPricingErrorFitAndZSpreadSignal) {
  const bd::Date today = iso("2026-09-04");
  const der::AssetSwapConvention conv = usd_headline();

  // An 8-bond universe (maturities 1..10y, mixed coupons) — over-determined vs a 4-knot curve.
  struct U {
    const char* id;
    const char* mat;
    double coupon;
  };
  const std::vector<U> defs = {{"B1", "2027-09-15", 0.030}, {"B2", "2028-09-15", 0.035},
                               {"B3", "2029-09-15", 0.040}, {"B4", "2030-09-15", 0.038},
                               {"B5", "2031-09-15", 0.042}, {"B6", "2033-09-15", 0.045},
                               {"B7", "2036-09-15", 0.043}, {"B8", "2036-09-15", 0.050}};
  std::vector<bd::BondId> universe;
  for (const auto& u : defs)
    universe.push_back({.id = u.id,
                        .yield_conv = "US-TREASURY",
                        .issue = iso("2026-08-15"),
                        .maturity = iso(u.mat),
                        .coupon = u.coupon});

  // A known TRUE govvie curve; price every bond ON it to get self-consistent market clean prices.
  const std::vector<double> meeting{0.5}, back{2.0, 5.0, 10.0};
  Eigen::VectorXd x_true(4);
  x_true << 0.030, 0.033, 0.038, 0.043;

  const bd::Date settle = bd::advance_bd(conv.settle_calendar, today, conv.settle_lag);
  std::vector<px::Bond> bonds;
  std::vector<double> accrued;
  for (const auto& br : universe) {
    const bd::BuiltBond bb =
        bd::bond_from_convention(br.yield_conv, today, settle, br.issue, br.maturity, br.coupon);
    bonds.push_back(bb.curve);
    accrued.push_back(bb.accrued);
  }
  pf::CompiledBondBook book(meeting, back, bonds);
  const Eigen::VectorXd clean_true = book.clean_prices(x_true);

  // A Market carrying those on-curve clean prices, then the minimum-pricing-error fit.
  mkt::Market m;
  m.as_of(today);
  for (int i = 0; i < static_cast<int>(universe.size()); ++i)
    m.add_quote(universe[i].id, mkt::Quote::mid(clean_true[i]));

  cal::GovvieBondFit fit = der::make_govvie_fit(conv, universe, m, meeting, back);
  ASSERT_EQ(fit.n_residuals(), 8);
  ASSERT_EQ(fit.n_knots(), 4);
  const cal::CalibrationResult res = cal::calibrate(fit, Eigen::VectorXd::Constant(4, 0.03));
  // On-curve prices => the fit reprices every bond and recovers the true curve (min error ~ 0).
  EXPECT_LT(fit.residuals<double>(res.x).cwiseAbs().maxCoeff(), 1e-9)
      << "minimum pricing error over an on-curve universe must be ~0";
  EXPECT_LT((res.x - x_true).cwiseAbs().maxCoeff(), 1e-6);

  // RV signal: cheapen ONE bond (B5), refit, and read z-spreads off the fitted curve. The cheap bond must
  // stand out as the richest positive spread — that is the bond-RV richness/cheapness ladder.
  const int k = 4;  // B5
  Eigen::VectorXd clean_mkt = clean_true;
  clean_mkt[k] -= 0.010;  // 1 point cheap
  mkt::Market m2;
  m2.as_of(today);
  for (int i = 0; i < static_cast<int>(universe.size()); ++i)
    m2.add_quote(universe[i].id, mkt::Quote::mid(clean_mkt[i]));

  cal::GovvieBondFit fit2 = der::make_govvie_fit(conv, universe, m2, meeting, back);
  const cal::CalibrationResult res2 = cal::calibrate(fit2, Eigen::VectorXd::Constant(4, 0.03));

  Eigen::VectorXd target_dirty(universe.size());
  for (int i = 0; i < static_cast<int>(universe.size()); ++i) target_dirty[i] = clean_mkt[i] + accrued[i];
  const Eigen::VectorXd z = book.z_spreads(res2.x, target_dirty);  // per-bond spread to the fitted curve

  int argmax = 0;
  for (int i = 1; i < z.size(); ++i)
    if (z[i] > z[argmax]) argmax = i;
  EXPECT_EQ(argmax, k) << "the cheapened bond must be the richest z-spread (the RV pick)";
  EXPECT_GT(z[k], 5e-4) << "a 1-point-cheap bond should carry a clearly positive z-spread";
}

TEST(DeriveAssetSwap, ParametricNelsonSiegelFitRecoversStableParamsAndFlagsRv) {
  const bd::Date today = iso("2026-09-04");
  const der::AssetSwapConvention conv = usd_headline();

  struct U {
    const char* id;
    const char* mat;
    double coupon;
  };
  const std::vector<U> defs = {{"B1", "2027-09-15", 0.030}, {"B2", "2028-09-15", 0.035},
                               {"B3", "2029-09-15", 0.040}, {"B4", "2030-09-15", 0.038},
                               {"B5", "2031-09-15", 0.042}, {"B6", "2033-09-15", 0.045},
                               {"B7", "2036-09-15", 0.043}, {"B8", "2041-09-15", 0.050}};
  std::vector<bd::BondId> universe;
  for (const auto& u : defs)
    universe.push_back({.id = u.id,
                        .yield_conv = "US-TREASURY",
                        .issue = iso("2026-08-15"),
                        .maturity = iso(u.mat),
                        .coupon = u.coupon});

  // A TRUE Nelson-Siegel curve (level/slope/curvature, fixed decay τ); price every bond ON it.
  const double tau = 2.0;
  Eigen::VectorXd theta_true(3);
  theta_true << 0.045, -0.015, -0.020;  // β0 (long level), β1 (slope), β2 (curvature)
  const bd::Date settle = bd::advance_bd(conv.settle_calendar, today, conv.settle_lag);
  cv::NelsonSiegel<double> truth(tau);
  truth.set_params(theta_true);

  std::vector<px::Bond> bonds;
  Eigen::VectorXd clean_true(universe.size());
  for (int i = 0; i < static_cast<int>(universe.size()); ++i) {
    const auto& br = universe[i];
    const bd::BuiltBond bb =
        bd::bond_from_convention(br.yield_conv, today, settle, br.issue, br.maturity, br.coupon);
    bonds.push_back(bb.curve);
    clean_true[i] = px::bond_clean_price<double>(bb.curve, truth);
  }

  mkt::Market m;
  m.as_of(today);
  for (int i = 0; i < static_cast<int>(universe.size()); ++i)
    m.add_quote(universe[i].id, mkt::Quote::mid(clean_true[i]));

  // Fit the 3 Nelson-Siegel parameters to the 8-bond universe by minimum pricing error.
  auto fit = der::make_parametric_fit<cv::NelsonSiegel>(conv, universe, m, tau);
  ASSERT_EQ(fit.n_knots(), 3);
  ASSERT_EQ(fit.n_residuals(), 8);
  Eigen::VectorXd seed(3);
  seed << 0.04, 0.0, 0.0;
  const cal::CalibrationResult res = cal::calibrate(fit, seed);
  EXPECT_LT(fit.residuals<double>(res.x).cwiseAbs().maxCoeff(), 1e-9)
      << "minimum pricing error over an on-model universe must be ~0";
  EXPECT_LT((res.x - theta_true).cwiseAbs().maxCoeff(), 1e-7) << "the fit must recover the true parameters";

  // RV + STABILITY: cheapen ONE bond (B5), refit. The few global parameters barely move (time-stable), yet
  // the cheap bond stands out as the largest positive model−market residual — the RV pick.
  const int k = 4;
  Eigen::VectorXd clean_mkt = clean_true;
  clean_mkt[k] -= 0.010;  // 1 point cheap
  mkt::Market m2;
  m2.as_of(today);
  for (int i = 0; i < static_cast<int>(universe.size()); ++i)
    m2.add_quote(universe[i].id, mkt::Quote::mid(clean_mkt[i]));

  auto fit2 = der::make_parametric_fit<cv::NelsonSiegel>(conv, universe, m2, tau);
  const cal::CalibrationResult res2 = cal::calibrate(fit2, res.x);

  // The RV signal: model − market per bond off the fitted fair-value curve. The cheapened bond reads CHEAP —
  // the largest positive residual (a few-parameter curve smears an outlier into neighbours, unlike the
  // free-knot spline above which isolates it; both are valid RV lenses — the parametric one trades locality
  // for stable, interpretable parameters).
  const Eigen::VectorXd rv = fit2.residuals<double>(res2.x);
  int argmax = 0;
  for (int i = 1; i < rv.size(); ++i)
    if (rv[i] > rv[argmax]) argmax = i;
  EXPECT_EQ(argmax, k) << "the cheapened bond must be the largest positive fair-value residual (the RV pick)";
  EXPECT_GT(rv[k], 3e-3) << "a 1-point-cheap bond must read clearly cheap to the fitted curve";
}
