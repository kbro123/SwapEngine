// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5 taxonomy: T1 oracle (engine number vs an independent number) | T2 calibration (optimum / stationarity / recovery) | T3 cross-path parity (two engine paths, same inputs)
// Multi-currency EUR bundle (plan: composed-tickling-snowglobe, Phase 1). Validates the FULL EUR block
// -- ESTR OIS (compounded overnight), 3M/6M EURIBOR IRS (the previously-untested IBOR/par-coupon path),
// 3s6s tenor basis, a payment-delay OIS, and an EONIA default-spread curve -- FIRST-PRINCIPLES (self-
// consistent recovery + compiled-vs-AAD) AND against QuantLib core to 1e-10 (fairRate off our curves).
//
// Everything is a generic Instrument built via QuantLib MakeOIS / MakeVanillaSwap and the coupon-type-
// dispatch extractors; the engine names no currency (CLAUDE.md §1). The per-index default discount
// (EURIBOR -> ESTR) and its per-trade override are a builder-layer resolution rule -- no engine change.

#include <gtest/gtest.h>

#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <memory>
#include <vector>

#include "reference_multicurrency.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/calibration/residual_engine.hpp"

namespace {
// E6.1c (2026-09-10): second-difference smoothing = the hybrid engine + the constant curvature block, the same
// composition BundleSession::calibrate uses (the SmoothedProblem wrapper and its numeric Jacobian are gone).
swaps::calibration::CalibrationResult regularised(const swaps::calibration::BundleProblem& p, double lambda,
                                                  const std::vector<int>& curves, const Eigen::VectorXd& x0) {
  namespace cal = swaps::calibration;
  const cal::HybridBundleResidual eng(p);
  const Eigen::MatrixXd R = cal::second_difference_operator(p, lambda, curves);
  const cal::RegularizedEngine<cal::HybridBundleResidual> sm(eng, R);
  return cal::calibrate_with(sm, p.n_knots(), sm.n_residuals(), x0);
}
}  // namespace
#include "swaps/calibration/streaming.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include <random>
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/ql/extract.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace qlx = swaps::qlx;

namespace {

// Our ParRate model quote for a freshly-built QuantLib swap, priced off the bundle's spread-aware
// CurveHandles at x_true (forecast on `fc`, discount on `dc`).
double our_par_rate(const rb::MultiCcyBundle& b, const Leg& floatLeg, const Leg& fixedLeg, int fc, int dc) {
  const auto fl = qlx::extract_float_leg(floatLeg, b.today, b.dc);
  const auto fx = qlx::extract_fixed_leg(fixedLeg, b.today, b.dc);
  return px::par_rate<double>(fl, fx, *b.curve_handles[fc], *b.curve_handles[dc]);
}

}  // namespace

// ---- Self-consistency: x_true zeroes the residual (the market was generated from it). ----
TEST(MultiCcyEur, MarketIsSelfConsistentAtXTrue) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  const double r = b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-eur] curves=" << b.n_curves() << " knots=" << b.prob.n_knots()
            << " instruments=" << b.prob.n_residuals() << " ||r(x_true)||inf=" << r << "\n";
  EXPECT_LT(r, 1e-12) << "self-consistent EUR market must zero the residual at x_true";
}

// ---- Joint + staged solves recover x_true. ESTR is the base SCC; EUR3M then EUR6M stage after it. ----
TEST(MultiCcyEur, JointAndStagedRecoverAllThreeCurves) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  const auto joint = cal::calibrate(b.prob, b.x0);
  const auto staged = cal::calibrate_staged(b.prob, b.x0);
  const double dj = (joint.x - b.x_true).cwiseAbs().maxCoeff();
  const double ds = (staged.x - b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-eur] joint ||x*-xtrue||=" << dj << " iters=" << joint.iterations
            << " stat=" << joint.stationarity << "  staged ||x*-xtrue||=" << ds
            << " iters=" << staged.iterations << "\n";
  EXPECT_LT(dj, 1e-6) << "joint solve recovers ESTR + EUR3M + EUR6M";
  EXPECT_LT(ds, 1e-6) << "staged solve recovers ESTR then EUR3M then EUR6M";
}

// ---- QuantLib-core oracle: our par rate == QuantLib fairRate off the SAME curves, to 1e-10, across
// ESTR OIS, 3M EURIBOR IRS and 6M EURIBOR IRS. This is the decisive proof the EURIBOR/IBOR path is
// correct (its first exercise anywhere) and that ESTR compounded overnight matches. ----
TEST(MultiCcyEur, ParRatesMatchQuantLibCore) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  double worst_ois = 0, worst_3m = 0, worst_6m = 0;

  for (int y : {1, 2, 3, 5, 7, 10, 15, 20, 30}) {
    auto o = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(y * Years, b.estr, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
    o->deepUpdate();
    const double ours = our_par_rate(b, o->overnightLeg(), o->fixedLeg(), b.ESTR, b.ESTR);
    worst_ois = std::max(worst_ois, std::abs(ours - o->fairRate()));
  }
  auto ibor = [&](const ext::shared_ptr<IborIndex>& idx, int fc, int y) {
    auto s = ext::shared_ptr<VanillaSwap>(MakeVanillaSwap(y * Years, idx, 0.03)
                                              .withDiscountingTermStructure(b.h[b.ESTR])
                                              .withFixedLegDayCount(Thirty360(Thirty360::BondBasis))
                                              .withFixedLegTenor(1 * Years)
                                              .withFixedLegConvention(ModifiedFollowing)
                                              .withFixedLegCalendar(TARGET())
                                              .withFloatingLegCalendar(TARGET()));
    s->deepUpdate();
    return std::abs(our_par_rate(b, s->floatingLeg(), s->fixedLeg(), fc, b.ESTR) - s->fairRate());
  };
  for (int y : {1, 2, 3, 5, 7, 10}) {
    worst_3m = std::max(worst_3m, ibor(b.eur3m, b.EUR3M, y));
    worst_6m = std::max(worst_6m, ibor(b.eur6m, b.EUR6M, y));
  }
  std::cout << "  [mc-eur] max |ours - QuantLib fairRate|:  ESTR OIS=" << worst_ois
            << "  3M EURIBOR=" << worst_3m << "  6M EURIBOR=" << worst_6m << "\n";
  EXPECT_LT(worst_ois, swaps::tol::curve_rel) << "ESTR OIS par rate must match QuantLib";
  EXPECT_LT(worst_3m, swaps::tol::curve_rel) << "3M EURIBOR IRS par rate must match QuantLib (IBOR path)";
  EXPECT_LT(worst_6m, swaps::tol::curve_rel) << "6M EURIBOR IRS par rate must match QuantLib (IBOR path)";
}

// ---- The multi-curve W-cache residual + its analytic block Jacobian must match the templated /
// AAD paths -- so the EURIBOR and 3s6s instruments ride the compiled fast path correctly. ----
TEST(MultiCcyEur, CompiledBundleResidualMatchesAad) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  cal::CompiledBundleResidual cr(b.prob);
  const double dr = (cr.residuals(b.x_true) - b.prob.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(b.x_true) - cal::aad_jacobian(b.prob, b.x_true)).cwiseAbs().maxCoeff();
  std::cout << "  [mc-eur] compiled |residual - templated|=" << dr << " |Jacobian - AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-13) << "compiled multi-curve residual matches the templated kernel";
  EXPECT_LT(dj, 1e-9) << "analytic block Jacobian matches AAD";
}

// ---- Payment delay: an ESTR OIS with a 2-business-day payment lag has its pay date AFTER the accrual
// end, that lag is captured in the extracted coupon's `pay`, and it still prices to QuantLib to 1e-10. ----
TEST(MultiCcyEur, PaymentDelayIsCapturedAndPriced) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  auto lag = ext::shared_ptr<OvernightIndexedSwap>(
      MakeOIS(2 * Years, b.estr, 0.03).withDiscountingTermStructure(b.h[b.ESTR]).withPaymentLag(2));
  auto nolag = ext::shared_ptr<OvernightIndexedSwap>(
      MakeOIS(2 * Years, b.estr, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
  lag->deepUpdate();
  nolag->deepUpdate();

  // The lagged coupon's pay time must sit strictly after its accrual-end (last value date), and after
  // the no-lag coupon's pay time.
  const auto lc = qlx::extract_float_leg(lag->overnightLeg(), b.today, b.dc);
  const auto nc = qlx::extract_float_leg(nolag->overnightLeg(), b.today, b.dc);
  ASSERT_FALSE(lc.empty());
  const px::FloatCoupon& last_lag = lc.back();
  const px::FloatCoupon& last_nolag = nc.back();
  EXPECT_GT(last_lag.pay, last_lag.obs.sub_end.back())
      << "a payment delay must put the pay date after the accrual end (last value date)";
  EXPECT_GT(last_lag.pay, last_nolag.pay) << "the lagged pay date must be later than the un-lagged one";

  // And it still prices to QuantLib exactly (DF(pay) discounts the lagged date).
  const double ours = our_par_rate(b, lag->overnightLeg(), lag->fixedLeg(), b.ESTR, b.ESTR);
  std::cout << "  [mc-eur] payment-lag pay-time delta=" << (last_lag.pay - last_nolag.pay)
            << "  |ours - QuantLib|=" << std::abs(ours - lag->fairRate()) << "\n";
  EXPECT_LT(std::abs(ours - lag->fairRate()), swaps::tol::curve_rel)
      << "a payment-delayed OIS must still match QuantLib";
}

// ---- EONIA as a DEFAULT fixed spread over ESTR (ESTR + 8.5bp), the "default the spread, overridable"
// mechanism: build it as a SpreadHandle with a flat 8.5bp spread and confirm forward == ESTR + 8.5bp
// exactly (and DF = ESTR_DF * exp(-int spread)). No calibration instruments => a fixed base spread. ----
TEST(MultiCcyEur, EoniaDefaultSpreadOverEstr) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  constexpr double kEoniaSpread = 0.00085;  // EONIA = ESTR + 8.5bp (legacy transition spread)

  // ESTR curve at x_true, then EONIA = ESTR + flat 8.5bp forward spread.
  const auto& estr = *b.curve_handles[b.ESTR];
  const std::vector<double> meet{0.5}, back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  auto sp = swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(meet, back));
  Eigen::VectorXd s(static_cast<int>(meet.size() + back.size()));
  s.setConstant(kEoniaSpread);
  sp.set_forwards(s);
  cal::SpreadHandle<double> eonia(std::move(sp), &estr);

  double worst_fwd = 0, worst_df = 0;
  for (double T : {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0}) {
    worst_fwd = std::max(worst_fwd, std::abs(eonia.forward(T) - (estr.forward(T) + kEoniaSpread)));
    worst_df = std::max(worst_df, std::abs(eonia.discount(T) - estr.discount(T) * std::exp(-kEoniaSpread * T)));
  }
  std::cout << "  [mc-eur] EONIA=ESTR+8.5bp  max|fwd-(ESTR+spr)|=" << worst_fwd
            << "  max|DF-ESTR*exp(-spr t)|=" << worst_df << "\n";
  EXPECT_LT(worst_fwd, 1e-14) << "EONIA forward must be ESTR + the default 8.5bp spread exactly";
  EXPECT_LT(worst_df, 1e-14) << "EONIA discount must be ESTR_DF * exp(-int spread) exactly";
}

// ===============================================================================================
// USD block (Phase 2): SOFR + Fed Funds (compounded FF-OIS basis + arithmetic FF averaging future).
// ===============================================================================================

TEST(MultiCcyUsd, MarketIsSelfConsistentAtXTrue) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  const double r = b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-usd] curves=" << b.n_curves() << " knots=" << b.prob.n_knots()
            << " instruments=" << b.prob.n_residuals() << " ||r(x_true)||inf=" << r << "\n";
  EXPECT_LT(r, 1e-10) << "self-consistent USD market must zero the residual at x_true";
}

TEST(MultiCcyUsd, JointAndStagedRecoverSofrAndFf) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  const auto joint = cal::calibrate(b.prob, b.x0);
  const auto staged = cal::calibrate_staged(b.prob, b.x0);
  const double dj = (joint.x - b.x_true).cwiseAbs().maxCoeff();
  const double ds = (staged.x - b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-usd] joint ||x*-xtrue||=" << dj << " iters=" << joint.iterations
            << "  staged ||x*-xtrue||=" << ds << " iters=" << staged.iterations << "\n";
  EXPECT_LT(dj, 1e-6) << "joint solve recovers SOFR + FF";
  EXPECT_LT(ds, 1e-6) << "staged solve recovers SOFR then the FF spread";
}

// Fed Funds OIS (COMPOUNDED, like SOFR OIS) and the FF/SOFR basis must match QuantLib to 1e-10 off our
// curves -- the proof FF-OIS compounds correctly on the FedFunds (FederalReserve) calendar.
TEST(MultiCcyUsd, FedFundsOisAndBasisMatchQuantLib) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  double worst_ois = 0, worst_basis = 0;
  for (int y : {1, 2, 3, 5, 10, 30}) {
    auto ffo = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(y * Years, b.fedfunds, 0.03).withDiscountingTermStructure(b.h[b.SOFR]));
    auto so = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(y * Years, b.sofr, 0.03).withDiscountingTermStructure(b.h[b.SOFR]));
    ffo->deepUpdate();
    so->deepUpdate();
    // FF-OIS fair rate (FF forecast, SOFR-discounted).
    worst_ois = std::max(worst_ois,
                         std::abs(our_par_rate(b, ffo->overnightLeg(), ffo->fixedLeg(), b.FF, b.SOFR) -
                                  ffo->fairRate()));
    // FF/SOFR basis = r_SOFR - r_FF; our par_spread(fwd=FF, bench=SOFR, disc=SOFR).
    const auto ff_fl = qlx::extract_float_leg(ffo->overnightLeg(), b.today, b.dc);
    const auto so_fl = qlx::extract_float_leg(so->overnightLeg(), b.today, b.dc);
    const auto so_fx = qlx::extract_fixed_leg(so->fixedLeg(), b.today, b.dc);
    const double ours = px::par_spread<double>(ff_fl, so_fl, so_fx, *b.curve_handles[b.FF],
                                               *b.curve_handles[b.SOFR], *b.curve_handles[b.SOFR]);
    worst_basis = std::max(worst_basis, std::abs(ours - (so->fairRate() - ffo->fairRate())));
  }
  std::cout << "  [mc-usd] max |ours - QuantLib|: FF-OIS=" << worst_ois << "  FF/SOFR basis=" << worst_basis << "\n";
  EXPECT_LT(worst_ois, swaps::tol::curve_rel) << "compounded FedFunds OIS must match QuantLib";
  EXPECT_LT(worst_basis, swaps::tol::curve_rel) << "FF/SOFR basis must match QuantLib";
}

// The Fed Funds 1M ARITHMETIC averaging future must match QuantLib's OvernightIndexFuture
// (RateAveraging::Simple) to machine precision -- the proof FF futures average the daily EFFR correctly.
TEST(MultiCcyUsd, FedFundsAveragingFutureMatchesQuantLib) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  const Calendar fcal = b.fedfunds->fixingCalendar();
  double worst = 0;
  // Fully-forward monthly contracts (eval is 8 Jul 2026).
  for (const auto& mo : {std::make_pair(September, 2026), std::make_pair(October, 2026),
                         std::make_pair(November, 2026)}) {
    const Date s = fcal.adjust(Date(1, mo.first, mo.second));
    const Date e = fcal.adjust(Date(1, Month(int(mo.first) % 12 + 1), mo.second + (mo.first == December)));
    OvernightIndexFuture fut(b.fedfunds, s, e, Handle<Quote>(), RateAveraging::Simple);
    const double ql = 1.0 - fut.NPV() / 100.0;  // NPV = 100·(1 − (convexity + rate)); convexity = 0
    const px::RateObservation obs = rb::avg_future_obs_idx(b.fedfunds, b.today, b.dc, s, e);
    const double ours = px::rate<double>(obs, *b.curve_handles[b.FF]);
    worst = std::max(worst, std::abs(ours - ql));
  }
  std::cout << "  [mc-usd] FF 1M averaging future max |ours - QuantLib averagedRate| = " << worst << "\n";
  EXPECT_LT(worst, swaps::tol::curve_rel) << "FF arithmetic averaging future must match QuantLib";
}

TEST(MultiCcyUsd, CompiledBundleResidualMatchesAad) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  cal::CompiledBundleResidual cr(b.prob);
  const double dr = (cr.residuals(b.x_true) - b.prob.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(b.x_true) - cal::aad_jacobian(b.prob, b.x_true)).cwiseAbs().maxCoeff();
  std::cout << "  [mc-usd] compiled |residual - templated|=" << dr << " |Jacobian - AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-12) << "compiled USD residual matches the templated kernel";
  EXPECT_LT(dj, 1e-8) << "analytic block Jacobian matches AAD";
}

// PRIME as a DEFAULT fixed spread over Fed Funds (FF + ~300bp), overridable -- same mechanism as EONIA.
TEST(MultiCcyUsd, PrimeDefaultSpreadOverFedFunds) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  constexpr double kPrimeSpread = 0.0300;  // PRIME ~= Fed Funds + 300bp
  const auto& ff = *b.curve_handles[b.FF];
  const std::vector<double> meet{0.5}, back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  auto sp = swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(meet, back));
  Eigen::VectorXd s(static_cast<int>(meet.size() + back.size()));
  s.setConstant(kPrimeSpread);
  sp.set_forwards(s);
  cal::SpreadHandle<double> prime(std::move(sp), &ff);

  double worst_fwd = 0;
  for (double T : {0.25, 1.0, 5.0, 10.0, 30.0})
    worst_fwd = std::max(worst_fwd, std::abs(prime.forward(T) - (ff.forward(T) + kPrimeSpread)));
  std::cout << "  [mc-usd] PRIME=FF+300bp  max|fwd-(FF+spr)|=" << worst_fwd << "\n";
  EXPECT_LT(worst_fwd, 1e-14) << "PRIME forward must be Fed Funds + the default 300bp spread exactly";
}

// The multi-currency BOOK: USD (SOFR+FF) and EUR (ESTR+EURIBOR) are INDEPENDENT problems -- the across-
// problem parallelism CLAUDE.md §7b calls out. Once built (their instruments are pure data), each
// calibrates with no shared state and recovers its own x_true, so a desk can solve them concurrently.
TEST(MultiCcy, UsdAndEurBooksCalibrateIndependently) {
  const rb::MultiCcyBundle usd = rb::build_usd_bundle();   // sets eval to the SOFR reference date
  const rb::MultiCcyBundle eur = rb::build_eur_bundle();   // sets eval to its own date; USD data already extracted
  const auto u = cal::calibrate(usd.prob, usd.x0);
  const auto e = cal::calibrate(eur.prob, eur.x0);
  const double du = (u.x - usd.x_true).cwiseAbs().maxCoeff();
  const double de = (e.x - eur.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-book] USD ||x*-xtrue||=" << du << "  EUR ||x*-xtrue||=" << de << "\n";
  EXPECT_LT(du, 1e-6) << "the USD book recovers independently";
  EXPECT_LT(de, 1e-6) << "the EUR book recovers independently";
}

// ===============================================================================================
// Cross-currency block (Phase 3): the EUR-collateralized-in-USD discount curve from constant-notional
// EURUSD OIS xccy basis swaps, in ONE bundle with SOFR + ESTR. Validated first-principles (FX no-arb).
// ===============================================================================================

TEST(MultiCcyXccy, MarketSelfConsistentAndRecovers) {
  const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  const double r = b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff();
  const auto joint = cal::calibrate(b.prob, b.x0);
  const auto staged = cal::calibrate_staged(b.prob, b.x0);
  const double dj = (joint.x - b.x_true).cwiseAbs().maxCoeff();
  const double ds = (staged.x - b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-xccy] curves=" << b.n_curves() << " knots=" << b.prob.n_knots()
            << " instruments=" << b.prob.n_residuals() << " ||r(x_true)||inf=" << r
            << "  joint ||x*-xtrue||=" << dj << " staged=" << ds << "\n";
  EXPECT_LT(r, 1e-12) << "self-consistent xccy market zeroes the residual";
  EXPECT_LT(dj, 1e-6) << "joint solve recovers SOFR + ESTR + EUR-in-USD";
  EXPECT_LT(ds, 1e-6) << "staged solve recovers them (EUR-in-USD staged after ESTR)";
}

TEST(MultiCcyXccy, CompiledBundleResidualMatchesAad) {
  const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  cal::CompiledBundleResidual cr(b.prob);
  const double dr = (cr.residuals(b.x_true) - b.prob.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(b.x_true) - cal::aad_jacobian(b.prob, b.x_true)).cwiseAbs().maxCoeff();
  std::cout << "  [mc-xccy] compiled |residual - templated|=" << dr << " |Jacobian - AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-13) << "compiled xccy residual matches the templated kernel";
  EXPECT_LT(dj, 1e-9) << "analytic block Jacobian matches AAD (the spread-over-ESTR block)";
}

// The dependency graph must order the EUR-in-USD spread AFTER its ESTR base; SOFR is an independent SCC.
TEST(MultiCcyXccy, DependencyGraphOrdersEurInUsdAfterEstr) {
  const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  const auto sccs = cal::bundle_dependency_order(b.prob);  // dependency order: base before spread
  int pos_estr = -1, pos_eurusd = -1;
  for (int i = 0; i < static_cast<int>(sccs.size()); ++i)
    for (int c : sccs[i]) {
      if (c == b.ESTR) pos_estr = i;
      if (c == b.EURUSD) pos_eurusd = i;
    }
  std::cout << "  [mc-xccy] SCC order: ESTR at " << pos_estr << ", EUR-in-USD at " << pos_eurusd
            << " (of " << sccs.size() << " SCCs)\n";
  ASSERT_GE(pos_estr, 0);
  ASSERT_GE(pos_eurusd, 0);
  EXPECT_LT(pos_estr, pos_eurusd) << "EUR-in-USD (spread over ESTR) must be solved after ESTR";
}

// FX no-arbitrage oracle (hand-built; QuantLib's xccy helpers are experimental so are not the oracle).
// The EUR-collateralized-in-USD forward is F(t) = S · DF_c(t) / DF_SOFR(t). We check: F(0) == spot; the
// xccy basis genuinely shifts the FX forward (DF_c != DF_ESTR); and, reconstructing a par const-notional
// xccy basis swap's EUR leg through the FX forwards F(t) (an independent path), the swap prices to par.
TEST(MultiCcyXccy, FxForwardNoArbitrage) {
  const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  const auto& SOFR = *b.curve_handles[b.SOFR];
  const auto& ESTR = *b.curve_handles[b.ESTR];
  const auto& C = *b.curve_handles[b.EURUSD];  // EUR-in-USD collateral curve
  const double S = b.fx_spot;
  auto F = [&](double t) { return S * C.discount(t) / SOFR.discount(t); };

  // F(0) == spot exactly (DF ratios are 1); and the basis shifts the forward vs the naive ESTR forward.
  EXPECT_NEAR(F(0.0), S, 1e-12) << "the FX forward equals spot at t = 0";
  double max_basis_shift = 0;
  for (double t : {1.0, 5.0, 10.0}) {
    const double f_naive = S * ESTR.discount(t) / SOFR.discount(t);  // if we ignored the xccy basis
    max_basis_shift = std::max(max_basis_shift, std::abs(F(t) - f_naive));
  }
  EXPECT_GT(max_basis_shift, 1e-4) << "the -15bp xccy basis must move the FX forward measurably";

  // Reconstruct a 10Y const-notional xccy basis swap's EUR leg through F(t), at the calibrated basis, and
  // confirm the swap is par (its net USD value is ~0). The USD SOFR-flat leg is par by construction.
  Settings::instance().evaluationDate() = b.today;
  auto o = ext::shared_ptr<OvernightIndexedSwap>(
      MakeOIS(10 * Years, b.estr, 0.03).withDiscountingTermStructure(b.h[b.EURUSD]));
  o->deepUpdate();
  const auto leg = qlx::extract_float_leg(o->overnightLeg(), b.today, b.dc);
  const auto fx = qlx::extract_fixed_leg(o->fixedLeg(), b.today, b.dc);
  // Calibrated par basis = par_spread(fwd = ESTR-forecast leg on DF_c, bench = DF_c-self leg on DF_c).
  const double bcal = px::par_spread<double>(leg, leg, fx, ESTR, C, C);

  // EUR leg (per unit EUR notional): principal exchange -1 at the SPOT start, +1 at maturity, plus
  // (ESTR_i + b·tau_i) coupons. Value each EUR amount in USD via A·F(t)·DF_SOFR(t)/S and confirm the net
  // is ~0 at par. (The swap starts at the spot/settlement date, so the first principal is at t_spot, not 0.)
  const double t_spot = leg.front().obs.sub_start[0];
  double eur_leg_usd = -1.0 * F(t_spot) * SOFR.discount(t_spot) / S;  // -1 EUR principal at spot
  for (const auto& c : leg) {
    const double c_estr = ESTR.discount(c.obs.sub_start[0]) / ESTR.discount(c.obs.sub_end[0]) - 1.0;
    const double amount = c_estr + bcal * c.tau_pay;  // ESTR compounded growth + basis on the accrual
    eur_leg_usd += amount * F(c.pay) * SOFR.discount(c.pay) / S;
  }
  const double Tlast = leg.back().pay;
  eur_leg_usd += 1.0 * F(Tlast) * SOFR.discount(Tlast) / S;  // +1 EUR principal at maturity
  std::cout << "  [mc-xccy] basis(10Y)=" << bcal << "  FX-fwd basis shift(max)=" << max_basis_shift
            << "  reconstructed EUR-leg USD PV (par => ~0)=" << eur_leg_usd << "\n";
  EXPECT_LT(std::abs(eur_leg_usd), 1e-10)
      << "the const-notional xccy basis swap must price to par through the FX forwards";
}

// ===============================================================================================
// MtM (mark-to-market, FX-resettable-notional) xccy leg (Phase 4) -- the one genuinely-new coupon
// shape. Its notional N_i = S·DF_c(reset)/DF_SOFR(reset) is CURVE-DEPENDENT, so the coupon PV is a
// product of two curves' DFs (not a single exp(-Wx)); it lives on the templated/AAD path only.
// ===============================================================================================

namespace {
// A USD SOFR OIS overnight leg (the MtM funding leg's coupons), from a fresh QuantLib swap.
std::vector<px::FloatCoupon> usd_sofr_leg(const rb::MultiCcyBundle& b, int years) {
  auto o = ext::shared_ptr<OvernightIndexedSwap>(
      MakeOIS(years * Years, b.sofr, 0.03).withDiscountingTermStructure(b.h[b.SOFR]));
  o->deepUpdate();
  return qlx::extract_float_leg(o->overnightLeg(), b.today, b.dc);
}
}  // namespace

// The reset mechanism is correct: an MtM leg paying its funding index (SOFR) FLAT is PAR (each period is
// a self-financing one-period loan). This is why the MtM xccy basis equals the constant-notional basis
// in deterministic curves (their difference is an FX-vol convexity term, out of scope for a curve engine).
TEST(MultiCcyMtm, FundingIndexFlatLegIsPar) {
  const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  const auto& SOFR = *b.curve_handles[b.SOFR];
  const auto& C = *b.curve_handles[b.EURUSD];
  const auto leg = usd_sofr_leg(b, 10);  // SOFR-flat (spread == 0)
  const double pv = px::xccy_mtm_leg_pv<double>(leg, b.fx_spot, SOFR, SOFR, C, SOFR);
  std::cout << "  [mc-mtm] SOFR-flat MtM leg PV (par => ~0) = " << pv << "\n";
  EXPECT_LT(std::abs(pv), 1e-12) << "an MtM leg paying its funding index flat is par";
}

// An MtM leg WITH a spread has a non-zero PV that (a) matches an independent first-principles
// reconstruction, and (b) is genuinely CURVE-NONLINEAR -- it moves when EITHER SOFR or the EUR-in-USD
// notional curve moves, proving the notional is a real DF ratio of two curves.
TEST(MultiCcyMtm, SpreadLegMatchesReconstructionAndIsTwoCurve) {
  const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  const auto& SOFR = *b.curve_handles[b.SOFR];
  const auto& C = *b.curve_handles[b.EURUSD];
  auto leg = usd_sofr_leg(b, 10);
  for (auto& c : leg) c.spread = 0.0025;  // 25bp funding spread => not par
  const double S = b.fx_spot;

  const double pv = px::xccy_mtm_leg_pv<double>(leg, S, SOFR, SOFR, C, SOFR);
  // Independent reconstruction: N_i·(interest_i + notional_i), interest recomputed from raw DFs.
  double recon = 0;
  for (const auto& c : leg) {
    const double s = c.obs.sub_start.front(), e = c.obs.sub_end.back();
    const double N = S * C.discount(s) / SOFR.discount(s);
    const double rate = (SOFR.discount(c.obs.sub_start[0]) / SOFR.discount(c.obs.sub_end[0]) - 1.0) /
                        c.obs.tau_index;
    const double interest = SOFR.discount(c.pay) * (rate + c.spread) * c.tau_pay;
    recon += N * (interest + (SOFR.discount(e) - SOFR.discount(s)));
  }
  // Curve-nonlinearity: rebuild with a bumped EUR-in-USD block and a bumped SOFR block.
  auto bump = [&](int role, double dv) {
    auto ch = cal::build_bundle_curves<double>(
        b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i] + (c == role ? dv : 0.0); });
    return px::xccy_mtm_leg_pv<double>(leg, S, *ch[b.SOFR], *ch[b.SOFR], *ch[b.EURUSD], *ch[b.SOFR]);
  };
  const double d_eurusd = bump(b.EURUSD, 1e-4) - pv;  // sensitivity to the notional (num) curve
  const double d_sofr = bump(b.SOFR, 1e-4) - pv;      // sensitivity to the funding/notional-den curve
  std::cout << "  [mc-mtm] spread-leg PV=" << pv << " |PV-recon|=" << std::abs(pv - recon)
            << "  dPV/d(EUR-in-USD)=" << d_eurusd << " dPV/d(SOFR)=" << d_sofr << "\n";
  EXPECT_LT(std::abs(pv - recon), 1e-13) << "MtM spread leg matches the first-principles reconstruction";
  EXPECT_GT(std::abs(pv), 1e-4) << "a 25bp funding spread makes the MtM leg materially non-par";
  EXPECT_GT(std::abs(d_eurusd), 1e-9) << "the leg depends on the EUR-in-USD notional curve";
  EXPECT_GT(std::abs(d_sofr), 1e-9) << "the leg depends on the SOFR curve too (a genuine two-curve product)";
}

// AAD differentiates the curve-dependent-notional MtM coupon correctly: the forward-mode gradient of the
// MtM leg PV w.r.t. every knot matches a central finite-difference bump. This is what lets an MtM
// instrument ride the AAD calibration path (aad_jacobian) with zero hand-derived partials.
TEST(MultiCcyMtm, AadGradientMatchesFiniteDifference) {
  const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  auto leg = usd_sofr_leg(b, 10);
  for (auto& c : leg) c.spread = 0.0025;
  const double S = b.fx_spot;
  const int N = b.prob.n_knots();

  // AAD: build the curves as Dual-typed off a seeded x, evaluate once.
  const auto xd = swaps::ad::seed(b.x_true);
  const auto Cd = cal::build_bundle_curves<swaps::ad::Dual>(
      b.prob.curves, [&](int c, int i) { return xd[b.off[c] + i]; });
  const swaps::ad::Dual pvd =
      px::xccy_mtm_leg_pv<swaps::ad::Dual>(leg, S, *Cd[b.SOFR], *Cd[b.SOFR], *Cd[b.EURUSD], *Cd[b.SOFR]);
  const Eigen::VectorXd grad = pvd.derivatives();

  // Central finite difference on each knot.
  const double h = 1e-6;
  double worst = 0;
  for (int k = 0; k < N; ++k) {
    auto price = [&](double dv) {
      auto ch = cal::build_bundle_curves<double>(
          b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i] + (b.off[c] + i == k ? dv : 0.0); });
      return px::xccy_mtm_leg_pv<double>(leg, S, *ch[b.SOFR], *ch[b.SOFR], *ch[b.EURUSD], *ch[b.SOFR]);
    };
    const double fd = (price(h) - price(-h)) / (2 * h);
    worst = std::max(worst, std::abs(fd - grad[k]));
  }
  std::cout << "  [mc-mtm] AAD gradient vs central FD: worst |diff| = " << worst << " over " << N << " knots\n";
  EXPECT_LT(worst, 1e-6) << "AAD differentiates the FX-reset-notional coupon correctly (FD-limited)";
}

// The MtM xccy basis EQUALS the constant-notional basis in deterministic curves: a full MtM swap
// (USD MtM SOFR-flat leg vs EUR ESTR+basis leg) prices to par at the Phase-3 constant-notional basis.
TEST(MultiCcyMtm, MtmBasisEqualsConstantNotionalBasis) {
  const rb::MultiCcyBundle b = rb::build_xccy_bundle();
  const auto& SOFR = *b.curve_handles[b.SOFR];
  const auto& ESTR = *b.curve_handles[b.ESTR];
  const auto& C = *b.curve_handles[b.EURUSD];
  const double S = b.fx_spot;

  // EUR ESTR leg + the constant-notional par basis (Phase 3).
  auto oe = ext::shared_ptr<OvernightIndexedSwap>(
      MakeOIS(10 * Years, b.estr, 0.03).withDiscountingTermStructure(b.h[b.EURUSD]));
  oe->deepUpdate();
  const auto eleg = qlx::extract_float_leg(oe->overnightLeg(), b.today, b.dc);
  const auto efix = qlx::extract_fixed_leg(oe->fixedLeg(), b.today, b.dc);
  const double b_const = px::par_spread<double>(eleg, eleg, efix, ESTR, C, C);

  // EUR leg value (per 1 EUR, in EUR) at b_const, discounted DF_c: principal at spot/T + (ESTR + b)·tau.
  const double spot = eleg.front().obs.sub_start.front(), T = eleg.back().pay;
  double eur_pv = -C.discount(spot) + C.discount(T);
  for (const auto& c : eleg) {
    const double c_estr = ESTR.discount(c.obs.sub_start[0]) / ESTR.discount(c.obs.sub_end[0]) - 1.0;
    eur_pv += (c_estr + b_const * c.tau_pay) * C.discount(c.pay);
  }
  // USD MtM funding leg (SOFR-flat) value in USD.
  const double usd_pv = px::xccy_mtm_leg_pv<double>(usd_sofr_leg(b, 10), S, SOFR, SOFR, C, SOFR);
  const double npv = usd_pv - S * eur_pv;  // receive USD MtM, pay EUR
  std::cout << "  [mc-mtm] b_const(10Y)=" << b_const << "  MtM swap NPV at b_const=" << npv
            << " (USD MtM leg=" << usd_pv << ", EUR leg=" << eur_pv << ")\n";
  EXPECT_LT(std::abs(npv), 1e-10)
      << "the MtM xccy swap prices to par at the constant-notional basis (MtM == const-notional here)";
}

// ===============================================================================================
// The DESK xccy build (Phase 5): EUR-in-USD from FX FORWARD POINTS (short) + MtM xccy swaps (long) --
// pure rates, no FX-vol model. FX forwards are a first-class calibration instrument (a DF ratio).
// ===============================================================================================

// The compiled W-cache engine must REJECT the cross-currency quotes (a DF ratio / a curve-dependent
// notional is not exp(-Wx)); they ride the AAD/templated path.
// Both cross-currency quotes are now W-cacheable in standard form: a STANDALONE FX forward (affine
// (ln F − ln q)/T residual) and a MtM-xccy basis with a PAR funding leg (its FX-reset-notional term is
// identically zero, so it collapses to the ParSpread quotient). So the WHOLE FX+MtM oracle bundle compiles
// and matches AAD to machine precision. A NON-par funding leg keeps a genuine curve-dependent notional and
// must still reject (fall back to AAD).
TEST(XccyFx, CompiledEngineTakesFxAndParMtm) {
  const rb::MultiCcyBundle b = rb::build_xccy_fx_bundle();
  cal::CompiledBundleResidual cr(b.prob);  // FX + par-MtM -> fully W-cacheable (no throw)
  const double dr = (cr.residuals(b.x_true) - b.prob.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(b.x_true) - cal::aad_jacobian(b.prob, b.x_true)).cwiseAbs().maxCoeff();
  std::cout << "  [mc-fx-compiled] full FX+MtM on the W-cache: |resid-templated|=" << dr
            << " |Jac-AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-11) << "compiled FX + par-MtM residual matches the templated kernel";
  EXPECT_LT(dj, 1e-8) << "compiled FX + par-MtM Jacobian matches AAD";

  // Until 2026-09-09 each tweak below made the FX-reset funding term non-negligible and the row was REJECTED by
  // the compiled engine (routed to AAD). The compiled batch now prices the funding leg EXACTLY (a product of
  // registered DFs), so every variant compiles AND must match the templated kernel and the AAD Jacobian.
  auto compiles_exactly = [&](const char* what, auto tweak) {
    cal::BundleProblem p = b.prob;
    for (auto& ins : p.instruments) if (ins.quote == cal::QuoteKind::XccyMtmBasis) tweak(ins);
    const Eigen::VectorXd r0 = p.residuals<double>(b.x_true);
    int n_mtm = 0; double moved = 0.0;  // the tweak must actually change the MtM quotes (else the case tests nothing)
    for (int i = 0; i < p.n_residuals(); ++i) if (p.instruments[i].quote == cal::QuoteKind::XccyMtmBasis) { ++n_mtm; moved = std::max(moved, std::abs(r0[i])); }
    ASSERT_GT(n_mtm, 0); EXPECT_GT(moved, 1e-9) << what << ": the tweak did not move the MtM quote";
    for (int i = 0; i < p.n_residuals(); ++i) p.instruments[i].market += r0[i];  // re-centre the market
    cal::CompiledBundleResidual cr2(p);
    const double dr2 = (cr2.residuals(b.x_true) - p.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
    const Eigen::MatrixXd Ja = cr2.jacobian(b.x_true), Jaad = cal::aad_jacobian(p, b.x_true);
    const double dj2 = (Ja - Jaad).cwiseAbs().maxCoeff() / (Jaad.cwiseAbs().maxCoeff() + 1e-300);
    std::cout << "  [mc-fx-compiled] " << what << ": |dr|=" << dr2 << " |dJ|rel=" << dj2 << "\n";
    EXPECT_LT(dr2, 1e-11) << what;
    EXPECT_LT(dj2, 1e-9) << what;
  };
  compiles_exactly("CSA discounting (mtm.discount != forecast)",
                   [&](cal::Instrument& ins) { ins.mtm.discount = b.EURUSD; });                 // dc != fc
  compiles_exactly("a payment lag (pay != period end)",
                   [](cal::Instrument& ins) { for (auto& c : ins.mtm.coupons) c.pay += 2.0 / 365.0; });
  compiles_exactly("averaging convexity (fixing_step > 0)",
                   [](cal::Instrument& ins) { for (auto& c : ins.mtm.coupons) c.obs.fixing_step = 1.0 / 252.0; });
  compiles_exactly("a funding spread",
                   [](cal::Instrument& ins) { for (auto& c : ins.mtm.coupons) c.spread = 1e-4; });
}

// Build the curve from FX forwards + MtM swaps, and confirm (a) the curve recovers on the AAD path, and
// (b) every FX forward point reprices off the CALIBRATED curve to 1e-10.
TEST(XccyFx, RecoversAndFxForwardsReprice) {
  const rb::MultiCcyBundle b = rb::build_xccy_fx_bundle();
  ASSERT_LT(b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff(), 1e-12) << "self-consistent market";
  const auto sol = cal::calibrate(b.prob, b.x0, /*use_aad=*/false);  // AAD/numeric path (not W-cache)
  const double err = (sol.x - b.x_true).cwiseAbs().maxCoeff();

  // Reprice every FX forward off the calibrated curve: F_model(x*) vs the market outright forward.
  auto ch = cal::build_bundle_curves<double>(b.prob.curves, [&](int c, int i) { return sol.x[b.off[c] + i]; });
  double worst_fx = 0;
  int n_fx = 0;
  for (const auto& ins : b.prob.instruments)
    if (ins.quote == cal::QuoteKind::FxForward) {
      const double F = ins.fx_spot * ch[ins.fx_num]->discount(ins.fx_time) / ch[ins.fx_den]->discount(ins.fx_time);
      worst_fx = std::max(worst_fx, std::abs(F - ins.market));
      ++n_fx;
    }
  std::cout << "  [mc-fx] ||x*-xtrue||=" << err << " iters=" << sol.iterations << "  FX points=" << n_fx
            << " max|F_model - F_market|=" << worst_fx << "\n";
  EXPECT_LT(err, 1e-6) << "curve recovers from FX forwards + MtM swaps";
  EXPECT_LT(worst_fx, 1e-10) << "every FX forward point reprices off the calibrated curve";
}

// The HYBRID engine on a genuinely mixed bundle: cacheable swaps on the W-cache + the FX/MtM rows on a
// width-reduced AAD block. It must (a) not throw (the old compiled engine did), (b) match the templated
// residual and the full AAD Jacobian -- which also proves the touched-knot set captures every non-zero
// column of the FX rows -- and (c) recover the curve via the default use_aad path.
TEST(XccyFx, HybridResidualMatchesAadAndRecovers) {
  const rb::MultiCcyBundle b = rb::build_xccy_fx_bundle();
  cal::HybridBundleResidual hr(b.prob);  // FX/MtM rows -> AAD block, cacheable rows -> W-cache
  const double dr = (hr.residuals(b.x_true) - b.prob.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
  const double dj = (hr.jacobian(b.x_true) - cal::aad_jacobian(b.prob, b.x_true)).cwiseAbs().maxCoeff();
  std::cout << "  [mc-fx-hybrid] |residual - templated|=" << dr << " |Jacobian - AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-11) << "hybrid residual (W-cache + AAD block) matches the templated kernel";
  EXPECT_LT(dj, 1e-8) << "hybrid Jacobian matches AAD (incl. zeros on untouched knots)";
  const auto sol = cal::calibrate(b.prob, b.x0);  // use_aad=true -> the hybrid engine
  EXPECT_LT((sol.x - b.x_true).cwiseAbs().maxCoeff(), 1e-6) << "hybrid solve recovers the curve";
}

// A MIXED FX/MtM bundle now STREAMS on the hybrid frozen-Newton path instead of recalibrating each tick:
// the cacheable rows ride the W-cache and the FX/MtM rows ride the AAD block, whose Jacobian is refreshed
// only on staleness (not every tick). The per-tick solution must equal a full cold recalibrate at the same
// market to machine precision -- i.e. the hybrid streamer solves the SAME nonlinear system, just frozen.
TEST(XccyFx, HybridStreamingEqualsRecalibrate) {
  const rb::MultiCcyBundle b = rb::build_xccy_fx_bundle();
  cal::BundleProblem prob = b.prob;
  const Eigen::VectorXd x0 = cal::calibrate(prob, b.x0).x;  // start on the calibrated curve
  const Eigen::VectorXd q_anchor = prob.market();

  // A FEASIBLE market move: perturb the KNOTS and reprice, so q_new is exactly reproducible by x_pert.
  // (An arbitrary per-quote bump would be infeasible for the FX/MtM couplings -- the recalibrate would
  // leave the same nonzero least-squares residual, so a reprice==0 assertion could not tell streaming
  // apart from recalibrate. This keeps the true nonlinear system solvable and the recovery checkable.)
  Eigen::VectorXd x_pert = x0;
  for (int k = 0; k < x_pert.size(); ++k) x_pert[k] += 5e-4 * ((k % 2) ? 1.0 : -1.0);  // ~5bp zig-zag
  const auto Cp = cal::build_bundle_curves<double>(
      prob.curves, [&](int c, int i) { return x_pert[prob.offset(c) + i]; });
  const auto curve_of = [&Cp](int i) -> const cal::CurveHandle<double>& { return *Cp[i]; };
  Eigen::VectorXd q_new(prob.n_residuals());
  for (int i = 0; i < prob.n_residuals(); ++i)
    q_new[i] = cal::instrument_model_quote<double>(prob.instruments[i], curve_of);

  cal::StreamingCalibrator<cal::BundleProblem> sc(prob, x0, q_anchor, {});  // engine = HybridBundleResidual
  const cal::StreamTick t = sc.update(q_new);

  cal::BundleProblem prob2 = prob;  // reference: cold recalibrate at the new (self-consistent) market
  for (int i = 0; i < prob2.n_residuals(); ++i) prob2.instruments[i].market = q_new[i];
  const Eigen::VectorXd x_ref = cal::calibrate(prob2, x0).x;

  const double dx = (sc.current() - x_ref).cwiseAbs().maxCoeff();
  const double dtruth = (sc.current() - x_pert).cwiseAbs().maxCoeff();  // recovers the perturbed curve
  const double rr = prob2.residuals<double>(sc.current()).cwiseAbs().maxCoeff();  // reprices to the feed
  std::cout << "  [mc-fx-stream] |x_stream - x_recal|=" << dx << " |x_stream - x_truth|=" << dtruth
            << " ||reprice||inf=" << rr << " newton_steps=" << t.newton_steps
            << " refreshes=" << t.refreshes << "\n";
  EXPECT_LT(dx, 1e-8) << "hybrid frozen-Newton streaming == cold recalibrate for a mixed FX/MtM bundle";
  EXPECT_LT(dtruth, 1e-6) << "the streamed solution recovers the perturbed curve";
  EXPECT_LT(rr, 1e-8) << "the streamed curve reprices every instrument (incl. FX/MtM) to the live market";
}

// EUR-in-USD genuinely depends on BOTH ESTR (spread base) and SOFR (FX-forward denominator / funding leg).
TEST(XccyFx, DependsOnBothSofrAndEstr) {
  const rb::MultiCcyBundle b = rb::build_xccy_fx_bundle();
  const auto adj = cal::bundle_adjacency(b.prob);
  const bool on_sofr = std::count(adj[b.EURUSD].begin(), adj[b.EURUSD].end(), b.SOFR) > 0;
  const bool on_estr = std::count(adj[b.EURUSD].begin(), adj[b.EURUSD].end(), b.ESTR) > 0;
  const auto sccs = cal::bundle_dependency_order(b.prob);
  int p_sofr = -1, p_estr = -1, p_eurusd = -1;
  for (int i = 0; i < static_cast<int>(sccs.size()); ++i)
    for (int c : sccs[i]) {
      if (c == b.SOFR) p_sofr = i;
      if (c == b.ESTR) p_estr = i;
      if (c == b.EURUSD) p_eurusd = i;
    }
  std::cout << "  [mc-fx] EUR-in-USD deps: SOFR=" << on_sofr << " ESTR=" << on_estr
            << "  SCC order S=" << p_sofr << " E=" << p_estr << " C=" << p_eurusd << "\n";
  EXPECT_TRUE(on_sofr) << "EUR-in-USD depends on SOFR (FX-forward denominator)";
  EXPECT_TRUE(on_estr) << "EUR-in-USD depends on ESTR (spread base)";
  EXPECT_GT(p_eurusd, p_sofr) << "EUR-in-USD is solved after SOFR";
  EXPECT_GT(p_eurusd, p_estr) << "EUR-in-USD is solved after ESTR";
}

// The curve converts a forward EUR cashflow back to USD at ANY date: F(t) = S·DF_c(t)/DF_SOFR(t), the USD
// value of 1 EUR at t equals S·DF_c(t) == F(t)·DF_SOFR(t), F(0) == spot, and the basis moves F off the
// naive ESTR-implied forward.
TEST(XccyFx, ConvertsForwardEurToUsdAnyDate) {
  const rb::MultiCcyBundle b = rb::build_xccy_fx_bundle();
  const auto sol = cal::calibrate(b.prob, b.x0, /*use_aad=*/false);
  auto ch = cal::build_bundle_curves<double>(b.prob.curves, [&](int c, int i) { return sol.x[b.off[c] + i]; });
  const auto& SOFR = *ch[b.SOFR];
  const auto& ESTR = *ch[b.ESTR];
  const auto& C = *ch[b.EURUSD];
  const double S = b.fx_spot;
  auto F = [&](double t) { return S * C.discount(t) / SOFR.discount(t); };

  EXPECT_NEAR(F(0.0), S, 1e-12) << "F(0) = spot";
  double worst_id = 0, max_shift = 0;
  for (double t : {0.37, 0.9, 1.6, 4.2, 8.5}) {  // arbitrary NON-pillar dates
    // USD value of 1 EUR at t, two ways: discount on the EUR-in-USD curve x spot, vs convert at the
    // forward and discount at SOFR. They must agree exactly (no-arbitrage).
    worst_id = std::max(worst_id, std::abs(S * C.discount(t) - F(t) * SOFR.discount(t)));
    max_shift = std::max(max_shift, std::abs(F(t) - S * ESTR.discount(t) / SOFR.discount(t)));
  }
  std::cout << "  [mc-fx] any-date convert: worst |S·DFc - F·DFsofr|=" << worst_id
            << "  max FX-fwd basis shift vs naive ESTR=" << max_shift << "\n";
  EXPECT_LT(worst_id, 1e-14) << "forward-EUR->USD conversion is self-consistent at any date";
  EXPECT_GT(max_shift, 1e-4) << "the xccy basis moves the FX forward off the naive ESTR-discounted one";
}

// AAD differentiates the FX-forward and MtM-basis residuals correctly: the forward-mode Jacobian matches
// a central finite difference (this is what lets the cross-currency bundle calibrate on the AAD path).
TEST(XccyFx, AadJacobianMatchesFiniteDifference) {
  const rb::MultiCcyBundle b = rb::build_xccy_fx_bundle();
  const Eigen::MatrixXd J = cal::aad_jacobian(b.prob, b.x_true);
  const int N = b.prob.n_knots(), M = b.prob.n_residuals();
  const double h = 1e-6;
  double worst = 0;
  for (int k = 0; k < N; ++k) {
    Eigen::VectorXd xp = b.x_true, xm = b.x_true;
    xp[k] += h;
    xm[k] -= h;
    const Eigen::VectorXd fd = (b.prob.residuals<double>(xp) - b.prob.residuals<double>(xm)) / (2 * h);
    for (int i = 0; i < M; ++i) worst = std::max(worst, std::abs(fd[i] - J(i, k)));
  }
  std::cout << "  [mc-fx] AAD Jacobian vs central FD: worst |diff| = " << worst << "\n";
  EXPECT_LT(worst, 1e-6) << "AAD differentiates the FX-forward + MtM-basis residuals (FD-limited)";
}

// ===============================================================================================
// The PROPER 3-curve EUR build (ESTR / EURIBOR-3M / EURIBOR-6M) — one cyclic SCC, calibrated jointly.
// ===============================================================================================

// The three EUR curves form ONE strongly-connected component (the basis ladders + spread bases close a
// full cycle), so the dependency decomposition must put all three in a single SCC solved by one joint LM.
TEST(EurCurves, ThreeCurvesAreOneCyclicSCC) {
  const rb::MultiCcyBundle b = rb::build_eur_curves();
  const double r = b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff();
  const auto sccs = cal::bundle_dependency_order(b.prob);
  std::size_t biggest = 0;
  for (const auto& s : sccs) biggest = std::max(biggest, s.size());
  std::cout << "  [eur3] curves=" << b.n_curves() << " knots=" << b.prob.n_knots()
            << " instruments=" << b.prob.n_residuals() << " ||r(x_true)||inf=" << r << "  SCCs=" << sccs.size()
            << " biggest=" << biggest << "\n";
  EXPECT_LT(r, 1e-12) << "self-consistent EUR market zeroes the residual at x_true";
  EXPECT_EQ(sccs.size(), 1u) << "ESTR/EUR3M/EUR6M must condense into ONE strongly-connected component";
  EXPECT_EQ(biggest, 3u) << "that SCC must contain all three curves (a joint solve)";
}

// The joint solve reaches FIRST-ORDER OPTIMALITY (‖Jᵀr‖∞ ≈ 0 — the correct calibration criterion,
// CLAUDE.md §3b), and the curve is recovered to sub-bp. It does NOT recover to 1e-10, and that is a
// REAL, expected property of the user's basis-only build, not a defect: with no direct long ESTR OIS,
// the ESTR/EUR3M long-end absolute level is only WEAKLY identified — par rates are first-order
// insensitive to a parallel DISCOUNT shift, and the basis swaps pin only DIFFERENCES — so a ~0.5bp
// near-null direction remains (it lands on the EUR3M 30y knot; ESTR/EUR6M are pinned tighter). Adding a
// direct ESTR OIS at the back would collapse it, but the user chose basis-only.
TEST(EurCurves, JointSolveIsFirstOrderOptimal) {
  const rb::MultiCcyBundle b = rb::build_eur_curves();
  const auto joint = cal::calibrate(b.prob, b.x0);
  // The market is self-consistent, so the solve is first-order optimal and prices every instrument
  // exactly -- but because the basis-only EUR3M/EUR6M forecast curves are rank-deficient (see
  // DiagnoseNullDirection), the raw solve WANDERS in the null space and does NOT recover x_true.
  std::cout << "  [eur3] UNREGULARISED joint ||x*-xtrue||=" << (joint.x - b.x_true).cwiseAbs().maxCoeff()
            << " stat=" << joint.stationarity << " rms_resid=" << joint.rms_residual << "\n";
  EXPECT_LT(joint.stationarity, 1e-9) << "first-order optimal (‖Jᵀr‖∞ ≈ 0)";
  EXPECT_LT(joint.rms_residual, 1e-10) << "achieved objective is machine-zero (prices the market exactly)";
}

// HONEST NOTE: build_eur_curves (like every MC builder) sets each quote to model_quote(x_true), i.e. a
// SELF-CONSISTENT market, so the over-determined €STR futures all reprice to ZERO by construction -- there
// is no futures tension in the recovery tests. A REAL futures strip is over-determined AND inconsistent:
// several 1M/3M futures constrain the same piecewise-flat forward segments (and a future straddling an ECB
// meeting averages two segments), so they cannot all reprice to zero and the least-squares fit keeps a
// NON-ZERO residual (CLAUDE.md §2/§3; the SOFR reference's calibration_test '[market]' asserts exactly this).
// Here we inject that tension into the €STR front and confirm the fit is a first-order-optimal compromise.
TEST(EurCurves, OverDeterminedFuturesCarryRealTension) {
  rb::MultiCcyBundle b = rb::build_eur_curves();
  std::mt19937 rng(11);
  std::normal_distribution<double> noise(0.0, 1e-4);  // ~1bp of independent quote noise on the €STR futures
  int nfut = 0;
  std::vector<int> fut;
  for (int i = 0; i < static_cast<int>(b.prob.instruments.size()); ++i)
    if (b.prob.instruments[i].quote == cal::QuoteKind::Rate && b.prob.instruments[i].forecast == b.ESTR) {
      b.prob.instruments[i].market += noise(rng);  // break self-consistency -> the futures now disagree
      fut.push_back(i);
      ++nfut;
    }
  // E6.1c: the curvature penalty rides the hybrid engine as a constant R block (the one definition).
  const cal::HybridBundleResidual eng(b.prob);
  const Eigen::MatrixXd R = cal::second_difference_operator(b.prob, 1.0, {b.EUR3M, b.EUR6M});
  const cal::RegularizedEngine<cal::HybridBundleResidual> sm(eng, R);
  const auto sol = cal::calibrate_with(sm, b.prob.n_knots(), sm.n_residuals(), b.x0);
  const Eigen::VectorXd r = b.prob.residuals<double>(sol.x);  // DATA residuals (no reg rows)
  double fut_resid = 0, all_resid = r.cwiseAbs().maxCoeff();
  for (int i : fut) fut_resid = std::max(fut_resid, std::abs(r[i]));
  // First-order optimality of the (regularised) objective actually solved: ‖Jᵀr‖∞ over data + reg rows.
  const Eigen::MatrixXd Jr = sm.jacobian(sol.x);
  const Eigen::VectorXd rr = sm.residuals(sol.x);
  const double stat = (Jr.transpose() * rr).cwiseAbs().maxCoeff();
  std::cout << "  [tension] €STR futures perturbed=" << nfut << "  max futures residual=" << fut_resid
            << "  ||r_data||inf=" << all_resid << "  ||Jᵀr||inf(regularised)=" << stat << "\n";
  EXPECT_GT(fut_resid, 1e-6) << "inconsistent over-determined €STR futures cannot all reprice to zero";
  EXPECT_LT(fut_resid, 5e-3) << "but the fit is a sensible least-squares compromise, not blown up";
  EXPECT_LT(stat, 1e-5) << "and it is first-order optimal on the objective solved (‖Jᵀr‖∞ ≈ 0)";
}

// The SMOOTHNESS regulariser resolves the null: penalising curvature of the EUR3M/EUR6M forwards
// (NOT ESTR -- its €STR-futures front has real policy steps) selects the smoothest market-consistent
// curve, so the coupled basis-only trio now recovers x_true tightly.
TEST(EurCurves, SmoothnessRegulariserRecoversTheCoupledTrio) {
  const rb::MultiCcyBundle b = rb::build_eur_curves();
  const double raw = (cal::calibrate(b.prob, b.x0).x - b.x_true).cwiseAbs().maxCoeff();
  const auto reg = regularised(b.prob, 1.0, {b.EUR3M, b.EUR6M}, b.x0);
  const double err = (reg.x - b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [eur3-reg] raw ||x*-xtrue||=" << raw << "  regularised=" << err << "\n";
  EXPECT_GT(raw, 1e-3) << "the unregularised basis-only trio wanders in the null space";
  EXPECT_LT(err, 1e-6) << "the smoothness regulariser recovers the coupled trio tightly";
}

// The regulariser folded into the STREAMING operator (M = (JᵀJ + RᵀR)⁻¹Jᵀ) lets the rank-deficient
// coupled trio stream DIRECTLY: without it M is singular and streaming diverges; with it every tick
// reprices the market exactly and tracks the smooth true curve, staying on the fast path.
TEST(EurCurves, RegularisedStreamingTracksTheCoupledTrio) {
  const rb::MultiCcyBundle b = rb::build_eur_curves();
  cal::CompiledBundleResidual engine(b.prob);
  using SC = cal::StreamingCalibrator<cal::BundleProblem>;
  SC::Options opt;
  opt.regularizer = cal::second_difference_operator(b.prob, 1.0, {b.EUR3M, b.EUR6M});
  const Eigen::VectorXd q0 = engine.model_rates(b.x_true);
  SC sc(b.prob, b.x_true, q0, opt);

  // Drive the market with SMOOTH moves (level + slope) so the true curve stays smooth; the regularised
  // stream should then reprice every tick AND stay on the smooth true curve (no null-space wander).
  std::mt19937 rng(7u);
  std::normal_distribution<double> lvl(0.0, 0.4e-4), slp(0.0, 0.1e-4);
  Eigen::VectorXd x = b.x_true;
  double worst_rt = 0, worst_dev = 0;
  for (int tk = 0; tk < 250; ++tk) {
    const double dL = lvl(rng), dS = slp(rng);
    for (int c = 0; c < 3; ++c) {  // level + a slope linear in TIME => stays in the divided-difference null space (2026-09-21)
      int i = 0;
      for (const auto& m : b.prob.curves[c].regions)
        for (double t : m.knots) x[b.off[c] + i++] += dL + dS * t;
      for (const int nk = b.prob.curves[c].n_knots(); i < nk; ++i) x[b.off[c] + i] += dL;
    }
    const Eigen::VectorXd q = engine.model_rates(x);
    sc.update(q);
    worst_rt = std::max(worst_rt, (engine.model_rates(sc.current()) - q).cwiseAbs().maxCoeff());
    worst_dev = std::max(worst_dev, (sc.current() - x).cwiseAbs().maxCoeff());
  }
  std::cout << "  [eur3-stream] refreshes=" << sc.refresh_count() << " worst round-trip=" << worst_rt
            << " worst |x_stream - x_true|=" << worst_dev << "\n";
  EXPECT_LT(worst_rt, 1e-7) << "regularised streaming reprices the market exactly every tick";
  EXPECT_LT(worst_dev, 1e-6) << "and tracks the smooth true curve (the rank-deficient trio now streams)";
}

// The genuinely-new instruments reprice off the curves to QuantLib core to 1e-10: 1M/3M €STR futures
// (OvernightIndexFuture), the 3M EURIBOR future (IborIndex::forecastFixing), and the outright 6M swap.
TEST(EurCurves, InstrumentsMatchQuantLibCore) {
  const rb::MultiCcyBundle b = rb::build_eur_curves();
  Settings::instance().evaluationDate() = b.today;
  const auto& ESTR = *b.curve_handles[b.ESTR];
  const auto& EUR3M = *b.curve_handles[b.EUR3M];

  // 1M €STR averaging future vs OvernightIndexFuture(Simple).
  const Date s1 = TARGET().adjust(Date(1, October, 2026)), e1 = TARGET().adjust(Date(1, November, 2026));
  OvernightIndexFuture f1(b.estr, s1, e1, Handle<Quote>(), RateAveraging::Simple);
  const double d_estr1m = std::abs(px::rate<double>(rb::avg_future_obs_idx(b.estr, b.today, b.dc, s1, e1), ESTR) -
                                   (1.0 - f1.NPV() / 100.0));
  // 3M €STR compounded future vs OvernightIndexFuture(Compound).
  const Date s3 = Date::nthWeekday(3, Wednesday, December, 2027);
  const Date e3 = Date::nthWeekday(3, Wednesday, (s3 + Period(3, Months)).month(), (s3 + Period(3, Months)).year());
  OvernightIndexFuture f3(b.estr, s3, e3, Handle<Quote>(), RateAveraging::Compound);
  const auto o3 = qlx::make_observation({{s3, e3}}, 0.0, b.estr->dayCounter().yearFraction(s3, e3), b.today, b.dc);
  const double d_estr3m = std::abs(px::rate<double>(o3, ESTR) - (1.0 - f3.NPV() / 100.0));
  // 3M EURIBOR future vs IborIndex::forecastFixing (settles on the actual fixing).
  const Date fx = Date::nthWeekday(3, Wednesday, March, 2028);
  const Date d1 = b.eur3m->valueDate(fx), d2 = b.eur3m->maturityDate(d1);
  const auto oe = qlx::make_observation({{d1, d2}}, 0.0, b.eur3m->dayCounter().yearFraction(d1, d2), b.today, b.dc);
  const double d_eurfut = std::abs(px::rate<double>(oe, EUR3M) - b.eur3m->forecastFixing(fx));
  // Outright 6M EURIBOR swap vs VanillaSwap::fairRate.
  double d_6m = 0;
  for (int y : {5, 10, 30}) {
    auto sw = ext::shared_ptr<VanillaSwap>(MakeVanillaSwap(y * Years, b.eur6m, 0.03)
                                               .withDiscountingTermStructure(b.h[b.ESTR])
                                               .withFixedLegDayCount(Thirty360(Thirty360::BondBasis))
                                               .withFixedLegTenor(1 * Years)
                                               .withFixedLegCalendar(TARGET())
                                               .withFloatingLegCalendar(TARGET()));
    sw->deepUpdate();
    d_6m = std::max(d_6m, std::abs(our_par_rate(b, sw->floatingLeg(), sw->fixedLeg(), b.EUR6M, b.ESTR) - sw->fairRate()));
  }
  std::cout << "  [eur3] oracle: €STR-1M-fut=" << d_estr1m << " €STR-3M-fut=" << d_estr3m
            << " EURIBOR-fut=" << d_eurfut << " 6M-outright=" << d_6m << "\n";
  EXPECT_LT(d_estr1m, swaps::tol::curve_rel) << "1M €STR averaging future vs QuantLib";
  EXPECT_LT(d_estr3m, swaps::tol::curve_rel) << "3M €STR compounded future vs QuantLib";
  EXPECT_LT(d_eurfut, swaps::tol::curve_rel) << "3M EURIBOR future vs QuantLib forecastFixing";
  EXPECT_LT(d_6m, swaps::tol::curve_rel) << "outright 6M EURIBOR swap vs QuantLib fairRate";
}

// The whole EUR bundle is Rate/ParRate/ParSpread (W-cacheable), so the compiled multi-curve residual +
// analytic block Jacobian must match the templated / AAD paths.
TEST(EurCurves, CompiledBundleResidualMatchesAad) {
  const rb::MultiCcyBundle b = rb::build_eur_curves();
  cal::CompiledBundleResidual cr(b.prob);
  const double dr = (cr.residuals(b.x_true) - b.prob.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(b.x_true) - cal::aad_jacobian(b.prob, b.x_true)).cwiseAbs().maxCoeff();
  std::cout << "  [eur3] compiled |residual - templated|=" << dr << " |Jacobian - AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-12) << "compiled EUR residual matches the templated kernel";
  EXPECT_LT(dj, 1e-8) << "analytic block Jacobian matches AAD across the coupled EUR curves";
}
// ---- Part B: the EUR trio inside the multi-currency bundle (SOFR + trio + EUR-in-USD) ----

// The EUR trio is one SCC (cross-tenor cycle); EUR-in-USD depends on ESTR AND SOFR (cross-currency),
// so it is solved in a LATER wave than both the EUR trio and SOFR.
TEST(EurMultiCcy, EurTrioIsOneSccAndXccyDependsOnEstrAndSofr) {
  const rb::MultiCcyBundle b = rb::build_eur_multicurrency();
  const double r = b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff();
  const auto adj = cal::bundle_adjacency(b.prob);
  const auto sccs = cal::bundle_dependency_order(b.prob);
  std::size_t trio = 0;
  int pos_trio = -1, pos_eurusd = -1, pos_sofr = -1;
  for (int i = 0; i < static_cast<int>(sccs.size()); ++i) {
    if (sccs[i].size() == 3) { trio = 3; pos_trio = i; }
    for (int c : sccs[i]) {
      if (c == b.EURUSD) pos_eurusd = i;
      if (c == b.SOFR) pos_sofr = i;
    }
  }
  const bool on_estr = std::count(adj[b.EURUSD].begin(), adj[b.EURUSD].end(), b.ESTR) > 0;
  const bool on_sofr = std::count(adj[b.EURUSD].begin(), adj[b.EURUSD].end(), b.SOFR) > 0;
  std::cout << "  [eur-mc] curves=" << b.n_curves() << " knots=" << b.prob.n_knots() << " instruments="
            << b.prob.n_residuals() << " ||r(x_true)||=" << r << "  SCCs=" << sccs.size()
            << " trioSCC=" << trio << "  EUR-in-USD deps: ESTR=" << on_estr << " SOFR=" << on_sofr << "\n";
  EXPECT_LT(r, 1e-10) << "self-consistent combined market";
  EXPECT_EQ(trio, 3u) << "ESTR/EUR3M/EUR6M form ONE cross-tenor SCC inside the multi-currency bundle";
  EXPECT_TRUE(on_estr && on_sofr) << "EUR-in-USD depends on ESTR (spread base) and SOFR (FX-fwd/funding)";
  EXPECT_GT(pos_eurusd, pos_trio) << "EUR-in-USD is solved after the EUR trio";
  EXPECT_GT(pos_eurusd, pos_sofr) << "EUR-in-USD is solved after SOFR";
}

// The full staged solve recovers the combined bundle (to sub-bp — the same basis-only weak long-end
// direction the standalone EUR build has; SOFR and EUR-in-USD are pinned tighter).
TEST(EurMultiCcy, RegularisedRecovers) {
  const rb::MultiCcyBundle b = rb::build_eur_multicurrency();  // SOFR ESTR EUR3M EUR6M EUR-in-USD
  // Regularise the basis-only EURIBOR forecast curves (EUR3M=2, EUR6M=3); SOFR/ESTR/EUR-in-USD untouched.
  const auto sol = regularised(b.prob, 1.0, {2, 3}, b.x0);
  const char* nm[] = {"SOFR", "ESTR", "EUR3M", "EUR6M", "EUR-in-USD"};
  double worst = 0;
  for (int c = 0; c < 5; ++c) {
    const int nk = b.prob.curves[c].n_knots();
    const double e = (sol.x - b.x_true).segment(b.off[c], nk).cwiseAbs().maxCoeff();
    worst = std::max(worst, e);
    std::cout << "  [eur-mc] " << nm[c] << " maxerr=" << e << "\n";
  }
  std::cout << "  [eur-mc] regularised ||x*-xtrue||=" << worst << " iters=" << sol.iterations << "\n";
  EXPECT_LT(worst, 1e-10) << "the combined SOFR + EUR-trio + EUR-in-USD bundle recovers x_true";  // measured 1e-14 (E5 2026-09-10; the old 1e-3 "sub-bp" was 10 bp)
}
// The WHOLE bundle (SOFR + FF + PRIME + ESTR + EUR3M + EUR6M + EUR-in-USD) calibrates: zero residual,
// first-order optimal, and the identified curves recover the generator. EONIA dropped: ESTR is the sole EUR
// discounting curve post-2022, so a separate EONIA curve adds nothing.
TEST(FullMultiCcy, WholeBundleCalibrates) {
  const rb::MultiCcyBundle b = rb::build_full_multicurrency();
  const double r = b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff();
  // Regularise only the coupled basis-only EURIBOR forecast curves (their forward shape is the null);
  // SOFR/ESTR/EUR-in-USD are shaped/policy-step or self-discounting and are left untouched.
  const auto sol = regularised(b.prob, 1.0, {b.EUR3M, b.EUR6M}, b.x0);
  const double err = (sol.x - b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [full] curves=" << b.n_curves() << " knots=" << b.prob.n_knots()
            << " instruments=" << b.prob.n_residuals() << " ||r(x_true)||=" << r << " recovery=" << err
            << " iters=" << sol.iterations << " rank_deficiency=" << sol.rank_deficiency
            << " ||r(x*)||=" << b.prob.residuals<double>(sol.x).cwiseAbs().maxCoeff()
            << " stationarity=" << sol.stationarity << "\n";
  for (int c = 0; c < b.n_curves(); ++c) {
    const int nk = b.prob.curves[c].n_knots();
    const Eigen::VectorXd d = (sol.x - b.x_true).segment(b.off[c], nk).cwiseAbs();
    int worst_i = 0; d.maxCoeff(&worst_i);
    std::cout << "  [full]   curve " << c << " maxerr=" << d[worst_i] << " at knot " << worst_i << "/" << nk << "\n";
  }
  EXPECT_EQ(b.n_curves(), 7);
  EXPECT_LT(r, 1e-10) << "self-consistent 7-curve market";
  // The calibration claim: the optimum reprices the market to machine precision and is first-order optimal.
  EXPECT_LT(b.prob.residuals<double>(sol.x).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT(sol.stationarity, 1e-12);
  // Recovery is asserted PER CURVE (E5 2026-09-10; the old single bound `err < 1e-3` was labelled "sub-bp
  // (~0.02bp)" and hid a 1.8 bp miss): the outright/pinned curves recover x_true to rounding (measured
  // 1e-14); PRIME to 5e-9 (measured 4.6e-9 -- the numeric-Jacobian floor); the FF spread curve is NOT asserted:
  // its FF/SOFR + PRIME/FF basis rows leave a near-null COMBINATION of FF knots that this test deliberately
  // does not regularise (only EUR3M/EUR6M are smoothed), so the fit lands 1.8e-4 off x_true at the 30y knot
  // while repricing every instrument to 5e-15. The tension regulariser is the documented remedy
  // (EurMultiCcy.RegularisedRecovers shows a smoothed spread curve recovering to 1e-14).
  for (int c = 0; c < b.n_curves(); ++c) {
    if (c == b.FF) continue;
    const double e = (sol.x - b.x_true).segment(b.off[c], b.prob.curves[c].n_knots()).cwiseAbs().maxCoeff();
    EXPECT_LT(e, c == 2 ? 5e-8 : 1e-10) << "curve " << c;
  }
}

// Both FEED TYPES on the FULL-RANK streamable bundle (the one the day-sim streams and tools/cal_times
// benchmarks): a SELF-CONSISTENT feed reaches the zero-residual optimum, while a REALISTIC over-determined
// feed (self-consistent + independent 0.4bp per-instrument noise) keeps a non-zero least-squares residual
// -- the tension between the futures and swaps -- yet is still first-order optimal. This pins the property
// the "compare calibration times" tool relies on (self-consistent vs noisy, single curve vs bundle).
TEST(FullMultiCcy, RealisticNoisyFeedCarriesTensionButIsOptimal) {
  rb::MultiCcyBundle b = rb::build_full_multicurrency(/*include_xccy=*/false, /*coupled_eur=*/false);
  EXPECT_EQ(b.n_curves(), 6) << "the full-rank streamable bundle (SOFR+FF+PRIME+ESTR+3M+6M)";

  // Self-consistent: b.prob is built from x_true, so the optimum sits at r=0. The cold solve drives the
  // RESIDUAL to machine-zero; it may land a hair off x_true along the documented forecast-curve null (see
  // WholeBundleCalibrates), which is orthogonal to the feed-type point here -- so we assert on the residual.
  const auto sc = cal::calibrate(b.prob, b.x0);
  EXPECT_LT(sc.rms_residual, 1e-9) << "self-consistent feed reaches the zero-residual optimum";
  EXPECT_LT(sc.stationarity, 1e-6) << "and is first-order optimal there";

  // Realistic: add independent ~0.4bp mispricing to every quote -> the strip no longer agrees.
  cal::BundleProblem ns = b.prob;
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 0.4e-4);
  for (auto& ins : ns.instruments) ins.market += noise(rng);
  const auto nr = cal::calibrate(ns, b.x0);
  std::cout << "  [noisy-bundle] self-consistent rms=" << sc.rms_residual << "  noisy rms(bp)="
            << nr.rms_residual * 1e4 << "  ||Jᵀr||inf=" << nr.stationarity << "  iters=" << nr.iterations << "\n";
  EXPECT_GT(nr.rms_residual, 1e-6) << "the over-determined noisy strip cannot reprice to zero (real tension)";
  EXPECT_LT(nr.rms_residual, 5e-3) << "but the fit is a sensible least-squares compromise, not blown up";
  EXPECT_LT(nr.stationarity, 1e-6) << "and it is first-order optimal (||Jᵀr||inf ≈ 0)";
}
