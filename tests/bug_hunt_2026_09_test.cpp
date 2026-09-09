// Regression gate for the 2026-09 engine bug hunt: every test here is a bug that was REPRODUCED with a
// standalone probe (crash, UB, or a wrong number on an ordinary input) and then fixed. Each test is the
// probe, distilled -- if a fix regresses, the same input fails here.
//
//   1. Bond coupon grids anchored at maturity (month-end dates no longer drift through February).
//   2. when_issued_bond accepts a month-end WI note (same drift used to throw "not on a common grid").
//   3. ACT/ACT ISDA of a zero-length span is 0 (was infinite recursion -> stack overflow).
//   4. ThreadPool::parallel_for completion handshake cannot lock a destroyed mutex.
//   5. A single-knot LEADING Linear/NaturalCubic/Hermite/Tension region throws (was OOB read / SEGV).
//   6. MonotoneCubic<Dual> survives the Hyman filter zeroing node 0's tangent (was SEGV: empty dual).
//   7. Two-node NaturalCubic/Tension regions carry AAD derivatives (were silently all-zero).
//   8. Tension's coshm2 series is accurate to ~1e-15 (was 1e-7: a missing term + a wrong coefficient).
//   9. Premium-adjusted CALL strike-from-delta returns the OTM root at ordinary vols (was deep-ITM K=|Δ|F).
//  10. sabr_vol_gradient honours beta (was the beta=0 gradient for every beta).
//  11. Piecewise-flat hazard: a humped CDS strip gives h >= 0 and a monotone survival (Hermite went to -20%).
//  12. Empty float legs / annuities price to zero (was a null dereference in release).
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <atomic>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/build/credit_instruments.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/calibration/credit_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/hazard.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/observations.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/conventions_data.hpp"
#include "swaps/parallel/thread_pool.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/fixings.hpp"
#include "swaps/trade/trade.hpp"
#include "swaps/vol/fx_vol_surface.hpp"
#include "swaps/vol/sabr.hpp"
#include "swaps/vol/sabr_calibration.hpp"

namespace bld = swaps::build;
namespace cv = swaps::curve;
namespace cal = swaps::calibration;
namespace ad = swaps::ad;
namespace v = swaps::vol;
namespace px = swaps::pricing;

namespace {

cv::CurveModule mod(cv::Scheme s, std::vector<double> knots, double sigma = 1.5) {
  cv::CurveModule m;
  m.knots = std::move(knots);
  m.scheme = s;
  m.sigma = sigma;
  return m;
}

struct FlatDf {
  double r;
  double operator()(double t) const { return std::exp(-r * t); }
};

}  // namespace

// 1. Every coupon date is an offset from MATURITY: Aug-31 stays Aug-31, Feb is 28/29, never 28 forever.
TEST(BugHunt, BondCouponGridAnchoredAtMaturity) {
  bld::Date ref_start;
  const auto cpn = bld::coupon_dates_backward(bld::Date::ymd(2025, 8, 31), bld::Date::ymd(2030, 8, 31), 2, ref_start);
  const std::vector<bld::Date> expect{
      bld::Date::ymd(2026, 2, 28), bld::Date::ymd(2026, 8, 31), bld::Date::ymd(2027, 2, 28), bld::Date::ymd(2027, 8, 31),
      bld::Date::ymd(2028, 2, 29), bld::Date::ymd(2028, 8, 31), bld::Date::ymd(2029, 2, 28), bld::Date::ymd(2029, 8, 31),
      bld::Date::ymd(2030, 2, 28), bld::Date::ymd(2030, 8, 31)};
  ASSERT_EQ(cpn.size(), expect.size());
  for (std::size_t i = 0; i < expect.size(); ++i) EXPECT_EQ(cpn[i], expect[i]) << "coupon " << i;
  EXPECT_EQ(ref_start, bld::Date::ymd(2025, 8, 31));  // dated on a coupon date: ref start == issue
}

// 2. A month-end when-issued note builds (first_coupon + k periods lands exactly on maturity).
TEST(BugHunt, WhenIssuedMonthEndBuilds) {
  const bld::Date vd = bld::Date::ymd(2026, 9, 7);
  EXPECT_NO_THROW(bld::when_issued_bond(vd, bld::Date::ymd(2026, 8, 31), bld::Date::ymd(2027, 2, 28),
                                        bld::Date::ymd(2028, 8, 31), 0.04, 2, vd));
}

// 3. Zero-length ACT/ACT span.
TEST(BugHunt, ActActIsdaZeroSpanIsZero) {
  const bld::Date d = bld::Date::ymd(2026, 3, 15);
  EXPECT_EQ(bld::act_act_isda(d, d), 0.0);
  EXPECT_DOUBLE_EQ(bld::act_act_isda(d, d.plus_days(1)), 1.0 / 365.0);
  EXPECT_DOUBLE_EQ(bld::act_act_isda(d.plus_days(1), d), -1.0 / 365.0);
}

// 4. The completion handshake: many short parallel_for calls, each destroying its stack mutex on return.
// With the decrement outside the lock this terminated with "mutex lock failed: Invalid argument".
TEST(BugHunt, ThreadPoolParallelForHandshakeIsSafe) {
  swaps::parallel::ThreadPool pool(4);
  std::atomic<long> sum{0};
  for (int it = 0; it < 20000; ++it) pool.parallel_for(8, [&](int i) { sum.fetch_add(i, std::memory_order_relaxed); });
  EXPECT_EQ(sum.load(), 20000L * 28L);
}

// 5. A leading region with ONE knot has no segment to build; every scheme must refuse it loudly.
TEST(BugHunt, SingleKnotLeadingRegionThrows) {
  for (cv::Scheme s : {cv::Scheme::Linear, cv::Scheme::NaturalCubic, cv::Scheme::Hermite, cv::Scheme::Tension}) {
    EXPECT_THROW(
        {
          auto c = cv::make_modular_curve<double>({mod(s, {2.0})});
          c.set_forwards(Eigen::VectorXd::Constant(1, 0.03));
        },
        std::invalid_argument)
        << "scheme " << static_cast<int>(s);
  }
}

// 6. Flat front, near-flat first back segment, then a steep rise: the natural-spline tangent at node 0
// overshoots negative while S0 > 0, so Hyman zeroes it. The zero must carry the derivative width.
TEST(BugHunt, MonotoneCubicDualSurvivesZeroedFirstTangent) {
  const auto spec = cv::flat_monotone({0.5, 1.0}, {2, 3, 4, 5, 6});
  Eigen::VectorXd x(7);
  x << 0.030, 0.030, 0.0301, 0.045, 0.046, 0.047, 0.048;
  auto cd = cv::make_modular_curve<ad::Dual>(spec);
  ASSERT_NO_THROW(cd.set_forwards(ad::seed(x)));
  const ad::Dual df = cd.discount(4.0);
  ASSERT_EQ(df.derivatives().size(), 7);
  auto c = cv::make_modular_curve<double>(spec);
  for (int k = 0; k < 7; ++k) {
    Eigen::VectorXd xp = x, xm = x;
    xp[k] += 1e-6;
    xm[k] -= 1e-6;
    c.set_forwards(xp);
    const double up = c.discount(4.0);
    c.set_forwards(xm);
    const double dn = c.discount(4.0);
    EXPECT_NEAR(df.derivatives()[k], (up - dn) / 2e-6, 1e-7) << "knot " << k;
  }
}

// 7. N == 2 (a following spline region with one knot): the boundary curvatures are the only nodes, so a
// width-less zero silently dropped every derivative. Compare the AAD gradient of the integral to FD.
TEST(BugHunt, TwoNodeSplineRegionsCarryDerivatives) {
  for (cv::Scheme s : {cv::Scheme::NaturalCubic, cv::Scheme::Tension}) {
    const std::vector<cv::CurveModule> spec{mod(cv::Scheme::Flat, {0.5}), mod(s, {3.0})};
    auto c = cv::make_modular_curve<double>(spec);
    const int m = c.n_knots();
    Eigen::VectorXd x(m);
    for (int i = 0; i < m; ++i) x[i] = 0.03 + 0.01 * i;
    auto cd = cv::make_modular_curve<ad::Dual>(spec);
    cd.set_forwards(ad::seed(x));
    for (double t : {2.0, 3.0, 4.0}) {
      const ad::Dual I = cd.integral(t);
      ASSERT_EQ(I.derivatives().size(), m) << "scheme " << static_cast<int>(s) << " t=" << t;
      for (int k = 0; k < m; ++k) {
        Eigen::VectorXd xp = x, xm = x;
        xp[k] += 1e-6;
        xm[k] -= 1e-6;
        c.set_forwards(xp);
        const double up = c.integral(t);
        c.set_forwards(xm);
        const double dn = c.integral(t);
        EXPECT_NEAR(I.derivatives()[k], (up - dn) / 2e-6, 1e-7) << "scheme " << static_cast<int>(s) << " t=" << t;
      }
    }
  }
}

// 8. cosh(x) - 1 - x²/2 by series on |x| < 0.5, against a long-double Taylor series carried to x²⁴/24!
// (a direct long-double cosh(x) - 1 - x²/2 cancels to ~1e-13 relative at x = 0.05 and is NOT a reference).
TEST(BugHunt, TensionCoshm2SeriesIsAccurate) {
  for (double x = 0.05; x < 0.5; x += 0.025) {
    const long double x2 = static_cast<long double>(x) * static_cast<long double>(x);
    long double ref = 0.0L, term = x2 * x2 / 24.0L;  // x⁴/4!
    for (int k = 4; k <= 24; k += 2) {
      ref += term;
      term *= x2 / (static_cast<long double>(k + 1) * static_cast<long double>(k + 2));
    }
    EXPECT_NEAR(cv::tension_detail::coshm2(x) / static_cast<double>(ref), 1.0, 1e-14) << "x=" << x;
  }
}

// 9. A 25Δ / 10Δ premium-adjusted CALL at low vol must be the OTM strike (above the forward, near it), not
// the deep-ITM root K = |Δ|·F. Both PA conventions, vols on either side of the old σ√T ≈ 0.078 cliff.
TEST(BugHunt, PremiumAdjustedCallStrikeIsTheOtmRoot) {
  const double F = 1.10, T = 0.25;
  for (v::DeltaConv conv : {v::DeltaConv::SpotPA, v::DeltaConv::FwdPA}) {
    for (double vol : {0.04, 0.07, 0.10, 0.14, 0.20}) {
      for (double delta : {0.25, 0.10}) {
        const double K = v::fx_strike_from_delta(F, T, 1.0, vol, delta, v::CallPut::Call, conv);
        EXPECT_GT(K, F) << "conv " << static_cast<int>(conv) << " vol " << vol << " delta " << delta;
        EXPECT_LT(K, F * std::exp(3.0 * vol * std::sqrt(T))) << "conv " << static_cast<int>(conv) << " vol " << vol;
        // The unadjusted strike is the nearby reference: PA moves a call strike by O(σ√T), never to 0.25.
        const double Ku = v::fx_strike_from_delta(F, T, 1.0, vol, delta, v::CallPut::Call, v::DeltaConv::SpotUnadj);
        EXPECT_NEAR(K / Ku, 1.0, 0.06) << "conv " << static_cast<int>(conv) << " vol " << vol << " delta " << delta;
      }
    }
  }
}

// 10. beta = 0.5 gradient vs central FD of the same beta = 0.5 smile.
TEST(BugHunt, SabrVolGradientHonoursBeta) {
  v::SabrParams p;
  p.alpha = 0.035;
  p.rho = -0.2;
  p.nu = 0.4;
  p.beta = 0.5;
  const double F = 0.04, T = 2.0, h = 1e-6;
  const auto vol = [&](double a, double r, double n) { return v::sabr_normal_vol<double>(F, 0.03, T, a, r, n, p.beta); };
  const v::SabrVolGrad g = v::sabr_vol_gradient(F, 0.03, T, p);
  EXPECT_NEAR(g.d_alpha, (vol(p.alpha + h, p.rho, p.nu) - vol(p.alpha - h, p.rho, p.nu)) / (2 * h), 1e-6);
  EXPECT_NEAR(g.d_rho, (vol(p.alpha, p.rho + h, p.nu) - vol(p.alpha, p.rho - h, p.nu)) / (2 * h), 1e-6);
  EXPECT_NEAR(g.d_nu, (vol(p.alpha, p.rho, p.nu + h) - vol(p.alpha, p.rho, p.nu - h)) / (2 * h), 1e-6);
}

// 11. A mildly INVERTED strip (1y300/3y200/5y150/10y120) whose implied forward hazards are all positive
// (≈2.5%, 1.25%, 1.5%): the Hermite hazard overshot to -0.6% between knots; with the piecewise-flat layout
// the strip still reprices exactly, the hazard is non-negative everywhere and the survival is monotone.
TEST(BugHunt, InvertedCdsStripHazardNonNegativeSurvivalMonotone) {
  const double r = 0.02, R = 0.40;
  const std::vector<double> mats{1.0, 3.0, 5.0, 10.0};
  const std::vector<double> tgt{0.0300, 0.0200, 0.0150, 0.0120};
  cal::CreditProblem prob;
  prob.back_times = mats;
  for (std::size_t i = 0; i < mats.size(); ++i) prob.instruments.push_back(bld::make_cds(mats[i], tgt[i], R, FlatDf{r}, 4, 4, 365.0 / 360.0));
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(prob.n_knots(), tgt.front() / (1.0 - R));
  const cal::CalibrationResult res = cal::calibrate(prob, x0);
  EXPECT_LT(res.stationarity, 1e-8);
  auto hz = cv::make_modular_curve<double>(prob.hazard_layout());
  hz.set_forwards(res.x);
  const cv::SurvivalCurve<double> surv{&hz};
  for (std::size_t i = 0; i < mats.size(); ++i) EXPECT_NEAR(prob.instruments[i].model_quote<double>(surv), tgt[i], 1e-8);
  double prev = 1.0;
  for (double t = 0.05; t <= 12.0; t += 0.05) {
    EXPECT_GE(surv.hazard(t), 0.0) << "t=" << t;
    const double q = surv.survival(t);
    EXPECT_LT(q, prev) << "t=" << t;
    prev = q;
  }
}

// 13. The product's FIXED-leg frequency reaches the built swap (it used to be dropped: every fixed leg annual).
TEST(BugHunt, FixedLegFrequencyComesFromTheProduct) {
  const auto p = swaps::conventions::product("SAR-SAIBOR-3M-IRS");
  ASSERT_TRUE(p.has_value());
  const bld::SwapConv conv = bld::conv_from_product(*p);
  EXPECT_EQ(conv.fixed_freq_tok, "6M");
  const bld::Date vd = bld::Date::ymd(2026, 9, 9);
  const cal::Instrument sw = bld::par_swap(vd, conv, bld::resolve("5Y", vd, "NONE", "Following", 0), 0, 0, 0.04);
  EXPECT_EQ(sw.fixed.coupons.size(), 10u);  // semi-annual over 5y
  EXPECT_EQ(sw.fwd.coupons.size(), 20u);    // quarterly float
  // An explicit override still wins.
  EXPECT_EQ(bld::par_swap(vd, conv, bld::resolve("5Y", vd, "NONE", "Following", 0), 0, 0, 0.04, 0.0, {}, "1Y").fixed.coupons.size(), 5u);
}

// 14. A seasoned trade rolls from its EFFECTIVE date. With a zero payment lag, a 7y trade struck two years
// ago has exactly the coupons of a 5y trade struck today (the elapsed periods are settled and dropped).
TEST(BugHunt, SeasonedTradeRollsFromEffective) {
  bld::SwapConv conv = bld::swap_conv("USD", "USD-SOFR");
  conv.pay_lag = 0;
  const bld::Date vd = bld::Date::ymd(2026, 9, 9), mat = bld::Date::ymd(2031, 9, 9);
  const auto seasoned = swaps::trade::Trade::vanilla_swap("S", 1e6, swaps::trade::Pay::Fixed, 0.03, "USD", "USD-SOFR",
                                                         bld::Date::ymd(2024, 9, 9), mat, 0, 0);
  const auto fresh = swaps::trade::Trade::vanilla_swap("F", 1e6, swaps::trade::Pay::Fixed, 0.03, "USD", "USD-SOFR",
                                                      vd, mat, 0, 0);
  const auto ps = seasoned.to_position(vd, conv), pf = fresh.to_position(vd, conv);
  ASSERT_EQ(ps.float_coupons.size(), pf.float_coupons.size());
  ASSERT_EQ(ps.fixed_coupons.size(), pf.fixed_coupons.size());
  EXPECT_EQ(ps.float_coupons.size(), 5u);
  for (std::size_t i = 0; i < ps.float_coupons.size(); ++i) {
    EXPECT_DOUBLE_EQ(ps.float_coupons[i].pay, pf.float_coupons[i].pay);
    EXPECT_DOUBLE_EQ(ps.float_coupons[i].tau_pay, pf.float_coupons[i].tau_pay);
    ASSERT_EQ(ps.float_coupons[i].obs.sub_start.size(), 1u);
    EXPECT_DOUBLE_EQ(ps.float_coupons[i].obs.sub_start[0], pf.float_coupons[i].obs.sub_start[0]);
    EXPECT_DOUBLE_EQ(ps.fixed_coupons[i].pay, pf.fixed_coupons[i].pay);
  }
  EXPECT_DOUBLE_EQ(ps.float_coupons.front().obs.sub_start[0], 0.0);  // the first remaining period starts today
}

// 15. A trade in the MIDDLE of a coupon period carries a fixings-resolvable current coupon: pricing it
// unresolved is refused (it would price the elapsed part at zero), and resolving it against a fixing table
// puts the realized compounding into realized_factor and starts the forecast at today.
TEST(BugHunt, SeasonedTradeCurrentPeriodResolvesFromFixings) {
  const bld::SwapConv conv = bld::swap_conv("USD", "USD-SOFR");
  const bld::Date vd = bld::Date::ymd(2026, 9, 9);
  const auto t = swaps::trade::Trade::vanilla_swap("S", 1e6, swaps::trade::Pay::Fixed, 0.03, "USD", "USD-SOFR",
                                                  bld::Date::ymd(2026, 3, 9), bld::Date::ymd(2031, 3, 9), 0, 0);
  auto p = t.to_position(vd, conv);
  ASSERT_FALSE(p.float_coupons.empty());
  px::FloatCoupon& cur = p.float_coupons.front();
  ASSERT_FALSE(cur.obs.fixing_schedule.empty());
  EXPECT_FALSE(cur.obs.resolved);
  auto c = cv::make_modular_curve<double>(cv::flat_hermite({0.5}, {1, 2, 5, 10}));
  c.set_forwards(Eigen::VectorXd::Constant(5, 0.03));
  EXPECT_THROW(px::float_coupon_pv<double>(cur, c, c), std::runtime_error);

  px::FixingTable table;
  double expect_rf = 1.0;
  for (const px::FixingDay& d : cur.obs.fixing_schedule)
    if (d.fixing_date < int(vd.serial())) {
      table.set("USD-SOFR", d.fixing_date, 0.03);
      expect_rf *= 1.0 + 0.03 * d.accrual;
    }
  px::resolve_into(cur.obs, px::PricingContext{int(vd.serial()), &table});
  EXPECT_TRUE(cur.obs.resolved);
  EXPECT_NEAR(cur.obs.realized_factor, expect_rf, 1e-14);
  EXPECT_GT(cur.obs.realized_factor, 1.01);
  ASSERT_FALSE(cur.obs.sub_start.empty());
  EXPECT_NEAR(cur.obs.sub_start.front(), 0.0, 1e-12);
  EXPECT_NO_THROW(px::float_coupon_pv<double>(cur, c, c));
}

// 16. A zero-basis, funding-flat xccy position is PAR: both legs carry their notional exchanges.
TEST(BugHunt, XccyZeroBasisFundingFlatIsPar) {
  auto c = cv::make_modular_curve<double>(cv::flat_hermite({0.5}, {1, 2, 5, 10}));
  c.set_forwards(Eigen::VectorXd::Constant(5, 0.03));
  const auto C = [&c](int) -> const cv::ModularCurve<double>& { return c; };
  swaps::portfolio::MultiCurveBook::Position p;
  p.kind = swaps::portfolio::MultiCurveBook::Kind::Xccy;
  p.notional = 1.0;
  double prev = 0.0;
  for (double u = 1.0; u <= 5.0 + 1e-9; u += 1.0) {
    px::FloatCoupon fc;
    fc.obs.sub_start = {prev};
    fc.obs.sub_end = {u};
    fc.obs.tau_index = u - prev;
    fc.pay = u;
    fc.tau_pay = u - prev;
    p.float_coupons.push_back(fc);
    p.mtm_coupons.push_back(fc);
    prev = u;
  }
  EXPECT_NEAR(swaps::portfolio::MultiCurveBook::position_value<double>(p, C), 0.0, 1e-12);
}

// 17. The compiled (W-cache) book refuses a moment-path coupon instead of dropping its averaging term.
TEST(BugHunt, CompiledBookRefusesMomentPathCoupons) {
  swaps::portfolio::Portfolio pf;
  swaps::portfolio::Portfolio::Position pos;
  px::FloatCoupon fc;
  fc.obs.sub_start = {0.0};
  fc.obs.sub_end = {1.0};
  fc.obs.tau_index = 1.0;
  fc.obs.fixing_step = 1.0 / 360.0;  // averaged coupon on the moment path
  fc.pay = 1.0;
  fc.tau_pay = 1.0;
  pos.float_coupons.push_back(fc);
  pos.fixed_rate = 0.03;
  pos.notional = 1.0;
  pf.positions.push_back(pos);
  EXPECT_THROW(swaps::portfolio::CompiledPortfolio(cv::flat_hermite({0.5}, {1, 2, 5, 10}), pf), std::invalid_argument);
}

// 12. Empty legs are worth nothing and must not touch leg[0].
TEST(BugHunt, EmptyLegsPriceToZero) {
  auto c = cv::make_modular_curve<double>(cv::flat_hermite({0.5}, {1, 2, 5, 10}));
  c.set_forwards(Eigen::VectorXd::Constant(5, 0.03));
  const std::vector<px::FloatCoupon> no_float;
  const std::vector<px::FixedCoupon> no_fixed;
  EXPECT_EQ(px::float_leg_pv<double>(no_float, c, c), 0.0);
  EXPECT_EQ(px::annuity<double>(no_fixed, c), 0.0);
}
