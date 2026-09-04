// The asset-swap / swap-spread BASIS instrument (build/swap_spread.hpp). QuantLib-free: a small 2-curve
// bundle (SOFR swap curve + a govvie yield factor) is hand-built self-consistent from a known x_true and
// calibrated, with one swap pillar (5y) pinned by a HEADLINE SWAP SPREAD instead of an outright.
//
// The point under test is RISK, not just the curve: modelling the bond as a live factor + the spread as its
// own calibration row makes the asset-swap basis a FIRST-CLASS, ORTHOGONAL axis. We prove
//   (1) the bundle calibrates and recovers x_true;
//   (2) the decomposition is exact and orthogonal — at the solution govvie_yield == q_pin and
//       par5 == q_pin + q_spread, so a bond-yield shock (q_pin) moves BOTH the bond and the 5y swap by the
//       same amount (spread held) while a spread shock (q_spread) moves the swap but NOT the bond;
//   (3) the analytic bucketed-delta ladder reprojects a 5y SOFR swap's risk onto BOTH the bond-pin and the
//       ASW-spread buckets (its outright delta is re-expressed as bond + basis), matching bump-recalibrate.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/swap_spread.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace cal = swaps::calibration;
namespace b = swaps::build;
namespace cv = swaps::curve;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

constexpr int kSwap = 0;    // SOFR swap curve  (bundle index 0): flat_hermite -> 4 knots
constexpr int kGovvie = 1;  // govvie yield factor (bundle index 1): 1 flat knot
constexpr double kAnchor = 5.0;

// --- local, calendar-free coupon/leg builders (annual on a year-fraction grid) -----------------------
px::FloatCoupon ois_coupon(double s, double e) {
  px::FloatCoupon c;
  c.obs.sub_start = {s};
  c.obs.sub_end = {e};
  c.obs.tau_index = e - s;
  c.pay = e;
  c.tau_pay = e - s;
  return c;
}
cal::FloatLeg float_leg(double T, int fc, int disc) {
  cal::FloatLeg leg;
  leg.forecast = fc;
  leg.discount = disc;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    leg.coupons.push_back(ois_coupon(prev, t));
    prev = t;
  }
  return leg;
}
cal::FixedLeg fixed_leg(double T, int disc) {
  cal::FixedLeg leg;
  leg.discount = disc;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    px::FixedCoupon fc;
    fc.pay = t;
    fc.tau = 1.0;
    leg.coupons.push_back(fc);
  }
  return leg;
}
cal::Instrument par_swap(double T, int curve, double market = 0.0) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd = float_leg(T, curve, curve);
  ins.fixed = fixed_leg(T, curve);
  ins.market = market;
  return ins;
}

// A swap position for the reprice book (payer-of-fixed), forecasting+discounting one bundle curve.
pf::MultiCurveBook::Position swap_position(double T, int curve, double rate, double notional) {
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Swap;
  p.notional = notional;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    p.float_coupons.push_back(ois_coupon(t - 1.0, t));
    px::FixedCoupon fc;
    fc.pay = t;
    fc.tau = 1.0;
    p.fixed_coupons.push_back(fc);
  }
  p.fwd_curve = curve;
  p.disc_curve = curve;
  p.fixed_curve = curve;
  p.fixed_rate = rate;
  return p;
}

Eigen::VectorXd x_true() {
  Eigen::VectorXd x(5);
  x << 0.030, 0.032, 0.036, 0.040,  // swap curve: flat(0.5) + hermite(2,5,10)
      0.0365;                        // govvie yield factor
  return x;
}

// Build the bundle. Residual order (5 rows for 5 knots):
//   0: front rate (pins swap flat knot)   1: 2y par swap   2: 10y par swap
//   3: bond-yield PIN (govvie factor)      4: ASW spread (Portfolio) — pins the swap 5y pillar
// `dpin`/`dspread` perturb the two spread-related quotes off their x_true-consistent base.
cal::BundleProblem make_problem(double dpin = 0.0, double dspread = 0.0) {
  cal::BundleProblem p;
  const std::vector<double> meeting{0.5}, back{2.0, 5.0, 10.0};
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite(meeting, back)});
  p.curves.push_back({.base = -1, .regions = b::govvie_factor_regions(kAnchor)});

  // Curves at x_true, to make every quote self-consistent (residual 0 at x_true).
  const Eigen::VectorXd xt = x_true();
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return xt[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };

  cal::Instrument front = b::rate_instrument(kSwap, b::plain_rate_obs(0.0, 0.5), 0.0);
  cal::Instrument s2 = par_swap(2.0, kSwap);
  cal::Instrument s10 = par_swap(10.0, kSwap);
  front.market = cal::instrument_model_quote<double>(front, curve_of);
  s2.market = cal::instrument_model_quote<double>(s2, curve_of);
  s10.market = cal::instrument_model_quote<double>(s10, curve_of);

  // The headline swap spread: the benchmark yield and the spread READ OFF x_true (self-consistent).
  const double bond_yield = cal::instrument_model_quote<double>(
      b::rate_instrument(kGovvie, b::asw_factor_obs(kAnchor), 0.0), curve_of);
  const cal::Instrument swap5 = par_swap(kAnchor, kSwap);
  const double par5 = cal::instrument_model_quote<double>(swap5, curve_of);
  const double spread = par5 - bond_yield;
  b::AssetSwapSpread asw = b::asset_swap_spread(swap5, kGovvie, kAnchor, bond_yield, spread);

  asw.pin.market += dpin;      // perturb the bond-yield quote
  asw.asw.market += dspread;   // perturb the spread quote

  p.instruments = {front, s2, s10, asw.pin, asw.asw};
  return p;
}

// govvie yield and 5y par swap rate implied by a calibrated state x.
double govvie_yield_at(const cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  return cal::instrument_model_quote<double>(b::rate_instrument(kGovvie, b::asw_factor_obs(kAnchor), 0.0),
                                             [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; });
}
double par5_at(const cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  return cal::instrument_model_quote<double>(par_swap(kAnchor, kSwap),
                                             [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; });
}

// Analytic bucketed delta over the BUNDLE (risk.hpp's bucketed_delta is single-curve): d(NPV)/dq per row.
Eigen::VectorXd bundle_bucketed_delta(const cal::BundleProblem& p, const Eigen::VectorXd& x,
                                      const pf::MultiCurveBook& book) {
  const int nk = p.n_knots();
  Eigen::Matrix<swaps::ad::Dual, Eigen::Dynamic, 1> xd(nk);
  for (int k = 0; k < nk; ++k) {
    xd[k].value() = x[k];
    xd[k].derivatives() = Eigen::VectorXd::Unit(nk, k);
  }
  const auto C = cal::build_bundle_curves<swaps::ad::Dual>(p.curves, [&](int c, int i) { return xd[p.offset(c) + i]; });
  const swaps::ad::Dual npv =
      book.value<swaps::ad::Dual>([&C](int i) -> const cal::CurveHandle<swaps::ad::Dual>& { return *C[i]; });
  const Eigen::VectorXd g = npv.derivatives();       // d(NPV)/dx
  const Eigen::MatrixXd J = cal::aad_jacobian(p, x);  // n_resid x n_knots
  const Eigen::VectorXd a = (J.transpose() * J).ldlt().solve(g);
  return J * a;                                       // length n_resid
}

Eigen::VectorXd solve(const cal::BundleProblem& p) {
  return cal::calibrate(p, Eigen::VectorXd::Constant(p.n_knots(), 0.035)).x;
}

}  // namespace

TEST(AssetSwapSpread, CalibratesAndRecoversTheBundle) {
  const cal::BundleProblem p = make_problem();
  const Eigen::VectorXd x = solve(p);
  EXPECT_LT((x - x_true()).cwiseAbs().maxCoeff(), 1e-8)
      << "a bundle with a 5y pillar pinned by the asset-swap spread must recover the true state";
  // Every residual is satisfied (the ASW Portfolio row included).
  EXPECT_LT(p.residuals<double>(x).cwiseAbs().maxCoeff(), 1e-9);
}

TEST(AssetSwapSpread, BasisDecomposesIntoOrthogonalBondAndSpreadAxes) {
  const double d = 1e-4;  // 1bp
  const cal::BundleProblem p0 = make_problem();
  const cal::BundleProblem pp = make_problem(/*dpin=*/d, 0.0);       // shock the BOND yield
  const cal::BundleProblem ps = make_problem(0.0, /*dspread=*/d);    // shock the SPREAD

  const Eigen::VectorXd x0 = solve(p0), xp = solve(pp), xs = solve(ps);
  const double g0 = govvie_yield_at(p0, x0), gp = govvie_yield_at(pp, xp), gs = govvie_yield_at(ps, xs);
  const double q0 = par5_at(p0, x0), qp = par5_at(pp, xp), qs = par5_at(ps, xs);

  // A BOND-yield shock moves the bond AND the 5y swap together (spread held) — a basis, not an outright.
  EXPECT_NEAR(gp - g0, d, 1e-7) << "bond yield must track its own quote 1:1";
  EXPECT_NEAR(qp - q0, d, 1e-7) << "at constant spread, the 5y swap follows the bond yield 1:1";

  // A SPREAD shock moves the swap but leaves the bond UNCHANGED — the two axes are orthogonal.
  EXPECT_NEAR(qs - q0, d, 1e-7) << "widening the spread moves the 5y swap";
  EXPECT_NEAR(gs - g0, 0.0, 1e-9) << "the spread axis must NOT move the bond yield";
}

TEST(AssetSwapSpread, RiskBucketsTheAssetSwapBasisSeparately) {
  const cal::BundleProblem p = make_problem();
  const Eigen::VectorXd x = solve(p);

  // A plain 5y SOFR swap book (the 5y pillar is calibrated via the spread).
  const pf::MultiCurveBook book{{swap_position(kAnchor, kSwap, 0.03, 1.0)}};
  const Eigen::VectorXd d = bundle_bucketed_delta(p, x, book);
  ASSERT_EQ(d.size(), p.n_residuals());

  // Rows: 0 front, 1 s2, 2 s10, 3 bond-PIN, 4 ASW-spread. The 5y swap's risk is RE-EXPRESSED as
  // bond + basis: both the pin (bond) and the ASW (spread) buckets carry material risk.
  const double scale = d.cwiseAbs().maxCoeff();
  EXPECT_GT(std::abs(d[3]) / scale, 1e-3) << "a 5y swap must carry bond-yield (pin) risk";
  EXPECT_GT(std::abs(d[4]) / scale, 1e-3) << "a 5y swap must carry asset-swap basis (spread) risk";

  // The analytic ladder equals a bump-and-recalibrate ladder (validates the bundle IFT reprojection).
  const double eps = 1e-6;
  for (int j = 0; j < p.n_residuals(); ++j) {
    cal::BundleProblem pu = p, pd = p;
    pu.instruments[j].market += eps;
    pd.instruments[j].market -= eps;
    const auto price = [&](const cal::BundleProblem& pr, const Eigen::VectorXd& xx) {
      const auto C = cal::build_bundle_curves<double>(pr.curves, [&](int c, int i) { return xx[pr.offset(c) + i]; });
      return book.value<double>([&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; });
    };
    const double fd = (price(pu, solve(pu)) - price(pd, solve(pd))) / (2 * eps);
    EXPECT_NEAR(d[j], fd, 1e-4 * scale) << "bucket " << j << " analytic vs bump";
  }
}
