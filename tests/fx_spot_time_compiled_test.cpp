// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// O-X3 fx_spot_time, piece 1 (2026-09-14). The compiled FX row carries the spot-date roll-back EXACTLY (value vs the templated
// quote, Jacobian vs AAD and a central finite difference, and the size of the move vs its closed form); fx_spot_time = 0 is
// the t = 0 expression bit for bit; the MtM basis quote is invariant to the spot level and the spot time; an xccy position
// with a spot time leaves the compiled book for the templated fallback and prices identically; validation refuses a
// negative or non-finite spot time (a tom-next forward delivering before spot stays valid); structure equality sees it.
// Also (2026-09-22, the row model): a Portfolio of FX forwards is a compiled row with a PLAIN residual.
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/structure_fingerprint.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/compiled_multi.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;
namespace tol = swaps::tol;

namespace {

constexpr double kS = 1.10;
constexpr int kNum = 0, kForeign = 1, kDen = 2;  // 0 EUR-in-USD, 1 ESTR, 2 SOFR
constexpr double kTs2 = 2.0 / 365.0, kTs5 = 5.0 / 365.0;

struct World {
  cal::BundleProblem p;
  Eigen::VectorXd x;
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> C;
  explicit World(const Eigen::VectorXd* state = nullptr) {
    const std::vector<double> knots{7.0 / 365.0, 0.5, 1.0, 2.0, 3.0, 5.0};
    for (int c = 0; c < 3; ++c) p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, knots)});
    const double level[] = {0.022, 0.020, 0.040}, slope[] = {0.0010, 0.0008, -0.0005};
    x.resize(p.n_knots());
    for (int c = 0; c < 3; ++c)
      for (int i = 0; i < p.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i)
        x[p.offset(c) + i] = level[c] + slope[c] * i;
    if (state) x = *state;
    C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  }
  const cal::CurveHandle<double>& operator()(int i) const { return *C[static_cast<std::size_t>(i)]; }
};

Eigen::VectorXd steep(const Eigen::VectorXd& x) {
  Eigen::VectorXd s = x;
  for (int i = 0; i < s.size(); ++i) s[i] += 2e-3 * std::sin(0.9 * i + 0.2);
  return s;
}

cal::BundleProblem fx_rows(bool with_spot_time) {
  World w;
  struct Row { double T, ts; };
  const Row rows[] = {{1.0 / 365.0, kTs2}, {kTs2, kTs2}, {9.0 / 365.0, kTs2}, {0.5, kTs2}, {1.0, kTs5}, {3.0, kTs2}};
  for (const Row& r : rows) {
    cal::Instrument f = b::fx_forward(kNum, kDen, kS, r.T, 1.0);
    if (with_spot_time) f.fx_spot_time = r.ts;
    w.p.instruments.push_back(f);
  }
  for (auto& ins : w.p.instruments) ins.market = cal::instrument_model_quote<double>(ins, w);
  return w.p;
}

}  // namespace

TEST(FxSpotTimeCompiled, ZeroSpotTimeIsTheTZeroExpressionBitForBit) {
  const World w;
  EXPECT_EQ(cal::Instrument{}.fx_spot_time, 0.0);
  EXPECT_EQ(cal::FloatLeg{}.fx_spot_time, 0.0);
  EXPECT_EQ(pf::MultiCurveBook::Position{}.fx_spot_time, 0.0);
  for (double T : {1.0 / 365.0, 0.5, 3.0}) {
    const cal::Instrument f = b::fx_forward(kNum, kDen, kS, T, 1.1);
    EXPECT_EQ(cal::instrument_model_quote<double>(f, w), kS * (w(kNum).discount(T) / w(kDen).discount(T))) << T;
  }
}

TEST(FxSpotTimeCompiled, CompiledFxRowsCarryTheSpotTimeExactly) {
  const cal::BundleProblem p = fx_rows(true);
  cal::BundleProblem zero = p;  // the SAME markets: the residual difference is then the spot factor alone
  for (auto& ins : zero.instruments) ins.fx_spot_time = 0.0;
  const World w0;
  const Eigen::VectorXd xs = steep(w0.x);
  const cal::CompiledBundleResidual cr(p), cr0(zero);
  const Eigen::VectorXd rc = cr.residuals(xs), rc0 = cr0.residuals(xs), rt = p.residuals<double>(xs);
  // compiled vs templated: same DFs through two paths (~1e-15 relative), magnified by 1/T (T >= 1/365)
  EXPECT_LE((rc - rt).cwiseAbs().maxCoeff(), 1e-11);
  // the move is LIVE on the compiled row and has the closed-form size (ln DF_den(t_s) − ln DF_num(t_s)) / T
  const World ws(&xs);
  for (int i = 0; i < p.n_residuals(); ++i) {
    const cal::Instrument& f = p.instruments[static_cast<std::size_t>(i)];
    const double want = (std::log(ws(kDen).discount(f.fx_spot_time)) - std::log(ws(kNum).discount(f.fx_spot_time))) / f.fx_time;
    ASSERT_GT(std::abs(want), 1e-6) << "premise: the spot time moves row " << i;
    EXPECT_NEAR(rc[i] - rc0[i], want, 1e-11) << "row " << i;
  }
  const Eigen::MatrixXd J = cr.jacobian(xs), Jaad = cal::aad_jacobian(p, xs);
  const double scale = std::max(1.0, Jaad.cwiseAbs().maxCoeff());
  EXPECT_LE((J - Jaad).cwiseAbs().maxCoeff() / scale, tol::jacobian_rel);
  Eigen::MatrixXd Jfd(J.rows(), J.cols());
  const double h = 1e-6;
  for (int j = 0; j < xs.size(); ++j) {
    Eigen::VectorXd xp = xs, xm = xs;
    xp[j] += h;
    xm[j] -= h;
    Jfd.col(j) = (p.residuals<double>(xp) - p.residuals<double>(xm)) / (2 * h);
  }
  EXPECT_LE((J - Jfd).cwiseAbs().maxCoeff() / scale, 1e-6);  // central FD, h = 1e-6: truncation + roundoff floor
}

TEST(FxSpotTimeCompiled, PortfolioOfFxForwardsIsACompiledRowWithAPlainResidual) {
  // A forward-forward (F(6M) − F(9d), a Σ of two outrights) as ONE Portfolio row. Until 2026-09-22 an FX forward
  // inside a Portfolio was refused on the compiled path (the FX row ASSIGNED its outright; a Σ of log-residuals
  // is not the log map). Under the row model each forward is an FxRatio TERM accumulating onto the row, and the
  // row's residual map is decided by the TOP-LEVEL kind -- a Portfolio: PLAIN q − market, exactly what the
  // templated instrument_residual computes -- while the standalone rows keep their log-basis map. Value vs the
  // templated quote, Jacobian vs AAD, and the router keeps the row on the W-cache.
  cal::BundleProblem p = fx_rows(true);
  cal::Instrument ff;
  ff.quote = cal::QuoteKind::Portfolio;
  ff.combination.push_back({1.0, p.instruments[3]});   // 6M
  ff.combination.push_back({-1.0, p.instruments[2]});  // 9d
  const World w0;
  ff.market = cal::instrument_model_quote<double>(ff, w0) + 2e-4;  // off-market so the residual is not 0
  p.instruments.push_back(ff);
  const int r = p.n_residuals() - 1;
  EXPECT_FALSE((ff).noncacheable());
  const cal::CompiledBundleResidual cr(p);
  EXPECT_EQ(cr.n_terms(), p.n_residuals() + 1);
  const cal::HybridBundleResidual hy(p);
  EXPECT_GE(hy.compiled_row(r), 0) << "the router keeps the portfolio on the W-cache";
  const Eigen::VectorXd xs = steep(w0.x);
  const World ws(&xs);
  const double want = cal::instrument_model_quote<double>(ff, ws);
  EXPECT_NEAR(cr.model_rates(xs)[r], want, 1e-14);
  EXPECT_NEAR(cr.residuals(xs)[r], want - ff.market, 1e-14) << "plain residual, not a log map";
  EXPECT_LE((cr.residuals(xs) - p.residuals<double>(xs)).cwiseAbs().maxCoeff(), 1e-11) << "every row, incl. the standalone log rows";
  const Eigen::MatrixXd J = cr.jacobian(xs), Jaad = cal::aad_jacobian(p, xs);
  const double scale = std::max(1.0, Jaad.cwiseAbs().maxCoeff());
  EXPECT_LE((J - Jaad).cwiseAbs().maxCoeff() / scale, tol::jacobian_rel);
  EXPECT_LE((hy.jacobian(xs) - Jaad).cwiseAbs().maxCoeff() / scale, tol::jacobian_rel);
  // The band applies to the portfolio row (a plain row) but never to a standalone FX row.
  cal::BundleProblem pb = p;
  pb.instruments[static_cast<std::size_t>(r)].band_lower = ff.market - 5e-4;
  pb.instruments[static_cast<std::size_t>(r)].band_upper = ff.market + 5e-4;
  pb.instruments[static_cast<std::size_t>(r)].band_decay = 0.25;
  const cal::CompiledBundleResidual crb(pb);
  // The Huber band residual of the PORTFOLIO quote (problem.hpp band_residual), whichever side of the band it sits.
  EXPECT_NEAR(crb.residuals(xs)[r], cal::band_residual_d(want, ff.market, ff.market - 5e-4, ff.market + 5e-4, 0.25).first, 1e-14);
  EXPECT_NEAR(crb.residuals(xs)[r], pb.residuals<double>(xs)[r], 1e-14) << "== the templated banded residual";
  const Eigen::MatrixXd Jb = crb.jacobian(xs), Jbaad = cal::aad_jacobian(pb, xs);
  EXPECT_LE((Jb - Jbaad).cwiseAbs().maxCoeff() / scale, tol::jacobian_rel);
}

TEST(FxSpotTimeCompiled, MtmBasisQuoteIsInvariantToTheSpotLevelAndTheSpotTime) {
  const World w;
  const b::XccyConv xc = b::xccy_conv("EURUSD");
  const b::Date vd = b::Date::from_iso("2026-09-15");
  cal::Instrument base = b::xccy_mtm_basis(vd, xc, b::resolve("2Y", vd, xc.calendar, xc.bdc, xc.spot_lag), kNum, kForeign, kDen, kS, 0.0);
  for (auto& c : base.mtm.coupons) c.spread = 0.01;  // a non-par funding leg: the MtM term is not ~0
  const auto quote = [&](double S, double ts) {
    cal::Instrument i = base;
    i.mtm.fx_spot = S;
    i.mtm.fx_spot_time = ts;
    return cal::instrument_model_quote<double>(i, w);
  };
  const double q0 = quote(kS, 0.0);
  const double ann = px::annuity<double>(base.fixed.coupons, w(base.fixed.discount));
  const double pv_self = px::float_leg_pv<double>(base.fwd.coupons, w(base.fwd.forecast), w(base.fwd.discount));
  const double pv_fx = px::float_leg_pv<double>(base.bench.coupons, w(base.bench.forecast), w(base.bench.discount));
  const double mtm_term = q0 - (pv_self - pv_fx) / ann;
  ASSERT_GT(std::abs(mtm_term * (w(kDen).discount(kTs5) / w(kNum).discount(kTs5) - 1.0)), 1e3 * tol::literal)
      << "premise: a divisor ignoring the spot time would be visible";
  for (const auto& [S, ts] : std::vector<std::pair<double, double>>{{1.25, 0.0}, {kS, kTs2}, {kS, kTs5}, {0.9, kTs5}})
    EXPECT_NEAR(quote(S, ts), q0, tol::literal) << "S=" << S << " t_s=" << ts * 365 << "d";
}

TEST(FxSpotTimeCompiled, AnXccyPositionWithASpotTimeLeavesTheCompiledBookAndPricesTheSame) {
  const World w;
  const b::XccyConv xc = b::xccy_conv("EURUSD");
  const b::Date vd = b::Date::from_iso("2026-09-15");
  const cal::Instrument legs = b::xccy_mtm_basis(vd, xc, b::resolve("2Y", vd, xc.calendar, xc.bdc, xc.spot_lag), kNum, kForeign, kDen, kS, 0.0);
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  p.notional = 1e6;
  p.float_coupons = legs.bench.coupons; p.fwd_curve = kForeign; p.disc_curve = kNum;
  p.mtm_coupons = legs.mtm.coupons; p.mtm_fwd_curve = kDen; p.mtm_disc_curve = kDen;
  p.mtm_reset_num = kNum; p.mtm_reset_den = kDen; p.fx_spot = kS;
  for (auto& c : p.mtm_coupons) c.spread = 0.01;  // a non-par funding leg, so the resetting notional's level is visible
  pf::MultiCurveBook::Position p0 = p;
  p.fx_spot_time = kTs2;
  const double v = pf::MultiCurveBook::position_value<double>(p, w), v0 = pf::MultiCurveBook::position_value<double>(p0, w);
  ASSERT_GT(std::abs(v - v0), 1e3 * 1e-9 * std::abs(v)) << "premise: the spot time moves the position far beyond the tolerance";
  pf::MultiCurveBook book, book0;
  book.positions = {p};
  book0.positions = {p0};
  const pf::CompiledMultiCurveBook cb(w.p.curves, book), cb0(w.p.curves, book0);
  EXPECT_EQ(cb0.n_compiled(), 1) << "control: without a spot time the position compiles";
  EXPECT_EQ(cb.n_compiled(), 0) << "a spot-time position rides the templated fallback";
  EXPECT_NEAR(cb.npv(w.x), v, 1e-9 * std::abs(v));
}

TEST(FxSpotTimeCompiled, ValidationRefusesANegativeOrNonFiniteSpotTime) {
  const b::XccyConv xc = b::xccy_conv("EURUSD");
  const b::Date vd = b::Date::from_iso("2026-09-15");
  const cal::Instrument basis = b::xccy_mtm_basis(vd, xc, b::resolve("2Y", vd, xc.calendar, xc.bdc, xc.spot_lag), kNum, kForeign, kDen, kS, 0.0);
  for (double bad : {-1.0 / 365.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    cal::Instrument f = b::fx_forward(kNum, kDen, kS, 1.0, 1.1);
    f.fx_spot_time = bad;
    EXPECT_THROW(cal::validate_instrument(f, "fx"), std::invalid_argument) << bad;
    cal::Instrument m = basis;
    m.mtm.fx_spot_time = bad;
    EXPECT_THROW(cal::validate_instrument(m, "basis"), std::invalid_argument) << bad;
  }
  cal::Instrument tn = b::fx_forward(kNum, kDen, kS, 1.0 / 365.0, 1.1);
  tn.fx_spot_time = kTs2;
  EXPECT_NO_THROW(cal::validate_instrument(tn, "tn")) << "delivery before spot (TN) is valid";
}

TEST(FxSpotTimeCompiled, StructureEqualitySeesTheSpotTime) {
  cal::BundleProblem a = fx_rows(true);
  cal::BundleProblem s = a, q = a;
  s.instruments[0].fx_spot_time = kTs5;
  q.instruments[0].market *= 1.001;  // control: a re-quote is not structural
  EXPECT_FALSE(cal::structure_equal(a, s));
  EXPECT_NE(cal::structure_fingerprint(a), cal::structure_fingerprint(s));
  EXPECT_TRUE(cal::structure_equal(a, q));
}
