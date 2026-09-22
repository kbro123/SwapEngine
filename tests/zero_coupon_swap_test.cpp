// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// @regression-test — QuoteKind::ZeroCouponRate: the BRL DI×Pre swap as a ZERO-COUPON, annually-compounded
// (exponential, BUS/252) quote — a nonlinear transform of the ParRate quotient of the same two legs. Until
// 2026-09-09 the DB row was a "schedule placeholder" (annual coupons, ParRate), i.e. the wrong instrument.
// Proves: (1) the DB row drives the shape (no frequencies, one period, one settlement) and the builder
// dispatches on it; (2) the transform is exactly r = (1+τq)^(1/τ) − 1 of the ParRate of the same legs;
// (3) CLOSED FORM: on a single self-discounted curve the calibrated DF satisfies DF(T)/DF(spot) == (1+r)^(−τ)
// to 1e-12 — an analytic identity, not a self-consistency; (4) the compiled W-cache path (value AND analytic
// Jacobian, with and without a band) matches the templated/AAD path; (5) a Portfolio of zero-coupon rates is a
// compiled row too (the transform is applied per term, 2026-09-22).
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "swaps/build/conventions.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/pricing/compiled_book.hpp"
#include "swaps/pricing/curve_spec.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {
struct ZcFixture : ::testing::Test {
  const b::Date vd = b::Date::from_iso("2025-01-02");
  const b::SwapConv conv = b::swap_conv("BRL", "BRL-CDI");
  const std::vector<std::string> tenors{"1Y", "2Y", "3Y", "5Y", "10Y"};
  const std::vector<double> quotes{0.1300, 0.1250, 0.1200, 0.1180, 0.1150};  // DI rates, annually compounded
  cal::BundleProblem p;
  std::vector<b::Date> mats;

  void SetUp() override {
    std::vector<double> knots;
    for (std::size_t i = 0; i < tenors.size(); ++i) {
      mats.push_back(b::resolve(tenors[i], vd, conv.calendar, conv.bdc, conv.spot_lag));
      p.instruments.push_back(b::par_swap(vd, conv, mats[i], 0, 0, quotes[i]));  // the DB row dispatches
      knots.push_back(p.instruments.back().fixed.coupons.front().pay);
    }
    p.curves = {{.base = -1, .regions = swaps::curve::flat_hermite({}, knots)}};
  }
  Eigen::VectorXd x0() const { return Eigen::VectorXd::Constant(p.n_knots(), 0.12); }
};
}  // namespace

TEST_F(ZcFixture, DbRowIsZeroCouponAndTheBuilderDispatchesOnIt) {
  EXPECT_TRUE(conv.zero_coupon);
  EXPECT_TRUE(conv.fixed_freq_tok.empty());
  EXPECT_TRUE(conv.float_freq_tok.empty());
  EXPECT_EQ(conv.fixed_dc, "BUS/252");
  for (const auto& ins : p.instruments) {
    EXPECT_EQ(ins.quote, cal::QuoteKind::ZeroCouponRate);
    ASSERT_EQ(ins.fixed.coupons.size(), 1u);
    ASSERT_EQ(ins.fwd.coupons.size(), 1u);
    const double bd = ins.fixed.coupons.front().tau * 252.0;   // BUS/252 => an integer business-day count
    EXPECT_NEAR(std::round(bd), bd, 1e-9);
    EXPECT_DOUBLE_EQ(ins.fwd.coupons.front().tau_pay, ins.fixed.coupons.front().tau);  // same span, same day count
    EXPECT_DOUBLE_EQ(ins.fwd.coupons.front().pay, ins.fixed.coupons.front().pay);      // one settlement
  }
  EXPECT_GT(p.instruments.back().fixed.coupons.front().tau, 9.5);  // a 10Y DI accrues ~10 x 252 business days
  // No coupon schedule exists for a zero_coupon product: the schedule builders refuse rather than invent one.
  EXPECT_THROW(b::fixed_coupons(vd, conv, mats[0], 0), std::invalid_argument);
  EXPECT_THROW(b::par_swap(vd, conv, mats[0], 0, 0, 0.12, 0.001), std::invalid_argument);  // no spread on a ZC
  // ... and a non-zero-coupon product cannot be built as one.
  EXPECT_THROW(b::zero_coupon_swap(vd, b::swap_conv("USD", "USD-SOFR"), mats[0], 0, 0, 0.04), std::invalid_argument);
}

TEST_F(ZcFixture, TransformIsExactlyTheAnnualCompoundingOfTheParRateOfTheSameLegs) {
  const Eigen::VectorXd x = x0();
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int, int i) { return x[i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (const auto& ins : p.instruments) {
    cal::Instrument lin = ins;
    lin.quote = cal::QuoteKind::ParRate;
    const double q = cal::instrument_model_quote<double>(lin, curve_of);
    const double tau = ins.fixed.coupons.front().tau;
    const double want = std::pow(1.0 + tau * q, 1.0 / tau) - 1.0;
    EXPECT_NEAR(cal::instrument_model_quote<double>(ins, curve_of), want, 1e-15);
    // (1+r)^τ − 1 == τ·q: the fixed leg's one payment equals the linear annuity times the par rate.
    const double r = cal::instrument_model_quote<double>(ins, curve_of);
    EXPECT_NEAR(std::pow(1.0 + r, tau) - 1.0, tau * q, 1e-15);
  }
}

TEST_F(ZcFixture, CalibratedDiscountFactorsSatisfyTheClosedForm) {
  const cal::CalibrationResult res = cal::calibrate(p, x0());
  ASSERT_LT(p.residuals<double>(res.x).cwiseAbs().maxCoeff(), 1e-12) << "every DI quote repriced exactly";
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int, int i) { return res.x[i]; });
  const double t_spot = b::curve_time(vd, b::spot_date(vd, conv.calendar, conv.spot_lag));
  const double df_spot = C[0]->discount(t_spot);
  for (std::size_t i = 0; i < p.instruments.size(); ++i) {
    // Fixed pays (1+r)^τ − 1 at T; the CDI leg pays DF(spot)/DF(T) − 1 (telescoped compounding, same span,
    // BUS/252 both). Par => DF(T)/DF(spot) = (1+r)^(−τ). Closed form, no solver tolerance hides in it.
    const double tau = p.instruments[i].fixed.coupons.front().tau;
    const double T = p.instruments[i].fixed.coupons.front().pay;
    const double want = std::pow(1.0 + quotes[i], -tau);
    EXPECT_NEAR(C[0]->discount(T) / df_spot, want, 1e-12) << tenors[i];
  }
  std::cout << "  [zero-coupon] BRL DI curve: iters=" << res.iterations << " rms=" << res.rms_residual << "\n";
}

TEST_F(ZcFixture, CompiledPathMatchesTemplatedAndAadWithAndWithoutABand) {
  const Eigen::VectorXd xs = cal::calibrate(p, x0()).x;
  for (int banded = 0; banded < 2; ++banded) {
    cal::BundleProblem q = p;
    if (banded) {  // a bid/offer band on the transformed quote: the band must see r, not the ParRate q
      auto& ins = q.instruments[2];
      ins.band_lower = ins.market - 3e-4; ins.band_upper = ins.market + 3e-4; ins.band_decay = 0.25;
      ins.market += 1e-4;  // off-mid so the band pull is active
    }
    const cal::CompiledBundleResidual cr(q);
    double worst_r = 0, worst_j = 0, scale = 0;
    for (double bump : {0.0, 8e-4, -1.5e-3}) {
      Eigen::VectorXd x = xs;
      for (int i = 0; i < x.size(); ++i) x[i] += bump * std::sin(0.7 * i + 0.3);
      worst_r = std::max(worst_r, (cr.residuals(x) - q.residuals<double>(x)).cwiseAbs().maxCoeff());
      const Eigen::MatrixXd Ja = cr.jacobian(x), Jaad = cal::aad_jacobian(q, x);
      worst_j = std::max(worst_j, (Ja - Jaad).cwiseAbs().maxCoeff());
      scale = std::max(scale, Jaad.cwiseAbs().maxCoeff());
    }
    std::cout << "  [zero-coupon] compiled vs templated (banded=" << banded << ") |dr|=" << worst_r
              << " |dJ|/scale=" << worst_j / scale << "\n";
    EXPECT_LT(worst_r, 1e-13) << "banded=" << banded;
    EXPECT_LT(worst_j / scale, 1e-10) << "banded=" << banded;
  }
}

TEST_F(ZcFixture, PortfolioOfZeroCouponRatesRidesTheCompiledPath) {
  // Until 2026-09-22 this was REFUSED here (and routed to AAD): the compiled engine applied the zero-coupon
  // transform to the accumulated row, so a Σ could not be one transformed quotient. The row model applies the
  // transform per TERM before accumulating, so a DI curve spread (a Σ of two transformed quotients, exactly
  // instrument_model_quote's definition) is a compiled row like any other: value vs the templated quote,
  // Jacobian vs AAD, and the router keeps it on the W-cache.
  cal::BundleProblem q = p;
  cal::Instrument fly;
  fly.quote = cal::QuoteKind::Portfolio;
  fly.combination.push_back({1.0, p.instruments[1]});
  fly.combination.push_back({-1.0, p.instruments[3]});
  fly.market = quotes[1] - quotes[3];
  q.instruments.push_back(fly);
  EXPECT_FALSE(cal::instrument_is_noncacheable(fly, q.curves));
  const cal::CompiledBundleResidual cr(q);
  EXPECT_EQ(cr.n_terms(), static_cast<int>(p.instruments.size()) + 2);  // five standalone rows + the two components
  const cal::HybridBundleResidual hy(q);
  EXPECT_GE(hy.compiled_row(q.n_residuals() - 1), 0) << "the router keeps the portfolio on the W-cache";
  const Eigen::VectorXd xs = cal::calibrate(p, x0()).x;
  for (double bump : {0.0, 8e-4, -1.5e-3}) {
    Eigen::VectorXd x = xs;
    for (int i = 0; i < x.size(); ++i) x[i] += bump * std::sin(0.7 * i + 0.3);
    const auto C = cal::build_bundle_curves<double>(q.curves, [&](int, int i) { return x[i]; });
    const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
    const int r = q.n_residuals() - 1;
    EXPECT_NEAR(cr.model_rates(x)[r], cal::instrument_model_quote<double>(fly, curve_of), 1e-15) << bump;
    EXPECT_NEAR(cr.residuals(x)[r], q.residuals<double>(x)[r], 1e-15) << bump;
    const Eigen::MatrixXd Ja = cr.jacobian(x), Jaad = cal::aad_jacobian(q, x);
    const double scale = Jaad.cwiseAbs().maxCoeff();
    EXPECT_LT((Ja - Jaad).cwiseAbs().maxCoeff() / scale, 1e-10) << bump;
    EXPECT_LT((hy.jacobian(x) - Jaad).cwiseAbs().maxCoeff() / scale, 1e-10) << bump;
  }
}
