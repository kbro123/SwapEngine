// @regression-test — the compiled (W-cache) MtM cross-currency leg (2026-09-09). Until now an XccyMtmBasis row
// rode the W-cache only when its FX-reset funding term was numerically negligible (and was then DROPPED); a
// payment lag, CSA discounting, a funding spread or averaging sent it to the AAD engine — on the ladder's desk
// rung 8 such rows cost 6x the other 62. The batch now prices the resetting notional EXACTLY as a product of
// registered DFs. These tests pin the partials against an oracle that is NOT the code under test (central finite
// differences of the templated kernel), then every adversarial configuration against templated + AAD + FD.
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <string>

#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"

namespace cal = swaps::calibration;
namespace api = swaps::api;
namespace b = swaps::build;
using swaps::shapes::Shape;

namespace {
struct Parity { double dr, dj_aad, dj_fd; };
// compiled vs templated (value), compiled vs AAD (Jacobian), compiled vs central FD of the TEMPLATED residual.
Parity parity(const cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const cal::CompiledBundleResidual cr(p);
  const Eigen::VectorXd rc = cr.residuals(x), rt = p.residuals<double>(x);
  const Eigen::MatrixXd J = cr.jacobian(x), Jaad = cal::aad_jacobian(p, x);
  Eigen::MatrixXd Jfd(J.rows(), J.cols());
  const double h = 1e-6;
  for (int j = 0; j < x.size(); ++j) {
    Eigen::VectorXd xp = x, xm = x; xp[j] += h; xm[j] -= h;
    Jfd.col(j) = (p.residuals<double>(xp) - p.residuals<double>(xm)) / (2 * h);
  }
  const double scale = Jaad.cwiseAbs().maxCoeff() + 1e-300;
  return {(rc - rt).cwiseAbs().maxCoeff(), (J - Jaad).cwiseAbs().maxCoeff() / scale, (J - Jfd).cwiseAbs().maxCoeff() / scale};
}
void recentre(cal::BundleProblem& p, const Eigen::VectorXd& x) {  // markets := model quotes at x (FX rows kept)
  const Eigen::VectorXd r = p.residuals<double>(x);
  for (int i = 0; i < p.n_residuals(); ++i) if (p.instruments[i].quote != cal::QuoteKind::FxForward) p.instruments[i].market += r[i];
}
std::vector<Eigen::VectorXd> states(const Shape& s) {  // calibrated, steep, negative
  std::vector<Eigen::VectorXd> out{s.x_true};
  Eigen::VectorXd steep = s.x_true; for (int i = 0; i < steep.size(); ++i) steep[i] += 2e-3 * std::sin(0.9 * i + 0.2) + 1e-3;
  Eigen::VectorXd neg = s.x_true; neg.array() -= 0.045;
  out.push_back(steep); out.push_back(neg);
  return out;
}
}  // namespace

// 1. Partials vs an oracle that is not AAD: central finite differences of the templated residual, on the real
//    EURUSD MtM rows (2-day pay lag, 3M legs) of the ladder's fx_xccy rung, at three curve states.
TEST(CompiledMtm, PartialsMatchFiniteDifferencesOfTheTemplatedKernel) {
  const Shape s = swaps::shapes::fx_xccy();
  for (const auto& x : states(s)) {
    const Parity pr = parity(s.prob, x);
    std::cout << "  [mtm] fx_xccy |dr|=" << pr.dr << " |dJ-AAD|rel=" << pr.dj_aad << " |dJ-FD|rel=" << pr.dj_fd << "\n";
    EXPECT_LT(pr.dr, 1e-12);
    EXPECT_LT(pr.dj_aad, 1e-9);
    EXPECT_LT(pr.dj_fd, 1e-6);
  }
}

// 2. Adversarial MtM configurations, each checked three ways at three states.
TEST(CompiledMtm, AdversarialConfigurationsCompileExactly) {
  const Shape base = swaps::shapes::fx_xccy();
  struct Case { const char* name; void (*tweak)(cal::Instrument&); };
  static const Case cases[] = {
      {"explicit reset date 2 days before the period start", [](cal::Instrument& in) { for (auto& c : in.mtm.coupons) c.reset_time = c.obs.sub_start.front() - 2.0 / 365.0; }},
      {"reset curves aliased to the forecast/discount curves", [](cal::Instrument& in) { in.mtm.reset_num = in.mtm.forecast; in.mtm.reset_den = in.mtm.discount; }},
      {"CSA discounting: funding leg discounted on the collateral curve", [](cal::Instrument& in) { in.mtm.discount = in.fwd.discount; }},
      {"reset ratio inverted (other pair direction)", [](cal::Instrument& in) { std::swap(in.mtm.reset_num, in.mtm.reset_den); }},
      {"fx_spot 1.25", [](cal::Instrument& in) { in.mtm.fx_spot = 1.25; }},
      {"leg scale 1.1 on every funding coupon", [](cal::Instrument& in) { for (auto& c : in.mtm.coupons) c.scale = 1.1; }},
      {"funding spread +10bp", [](cal::Instrument& in) { for (auto& c : in.mtm.coupons) c.spread = 1e-3; }},
      {"realized part on the first funding coupon", [](cal::Instrument& in) { in.mtm.coupons.front().obs.realized = 1e-3; }},
      {"pay lag 0 (pay == accrual end)", [](cal::Instrument& in) { for (auto& c : in.mtm.coupons) c.pay = c.obs.sub_end.back(); }},
      {"moment-path (averaged) funding coupons", [](cal::Instrument& in) { for (auto& c : in.mtm.coupons) { c.obs.fixing_step = 1.0 / 252.0; } }},
      {"band on the MtM row", [](cal::Instrument& in) { in.band_lower = in.market - 2e-4; in.band_upper = in.market + 2e-4; in.band_decay = 0.5; }},
  };
  for (const auto& cs : cases) {
    cal::BundleProblem p = base.prob;
    for (auto& in : p.instruments) if (in.quote == cal::QuoteKind::XccyMtmBasis) cs.tweak(in);
    recentre(p, base.x_true);
    for (const auto& x : states(base)) {
      const Parity pr = parity(p, x);
      EXPECT_LT(pr.dr, 1e-12) << cs.name;
      EXPECT_LT(pr.dj_aad, 1e-9) << cs.name;
      EXPECT_LT(pr.dj_fd, 1e-6) << cs.name;
    }
    std::cout << "  [mtm] " << cs.name << ": ok\n";
  }
  // an MtM row INSIDE a Portfolio (2x one row minus another) composes on the compiled path too
  {
    cal::BundleProblem p = base.prob;
    int a = -1, c = -1;
    for (int i = 0; i < p.n_residuals(); ++i) if (p.instruments[i].quote == cal::QuoteKind::XccyMtmBasis) { if (a < 0) a = i; else c = i; }
    cal::Instrument fly; fly.quote = cal::QuoteKind::Portfolio;
    fly.combination = {{2.0, p.instruments[a]}, {-1.0, p.instruments[c]}};
    for (auto& w : fly.combination) w.instrument.market = 0.0;
    p.instruments.push_back(fly);
    recentre(p, base.x_true);
    const Parity pr = parity(p, base.x_true);
    EXPECT_LT(pr.dr, 1e-12); EXPECT_LT(pr.dj_aad, 1e-9); EXPECT_LT(pr.dj_fd, 1e-6);
  }
  // a fully-fixed MtM coupon has nowhere to place its notional exchanges: both paths refuse identically
  {
    cal::BundleProblem p = base.prob;
    for (auto& in : p.instruments) if (in.quote == cal::QuoteKind::XccyMtmBasis) { in.mtm.coupons.front().obs.sub_start.clear(); in.mtm.coupons.front().obs.sub_end.clear(); break; }
    EXPECT_THROW(p.residuals<double>(base.x_true), std::runtime_error);
    EXPECT_THROW(cal::CompiledBundleResidual{p}, std::runtime_error);
  }
}

// 3. Routing is explicit: no MtM row is on the AAD route any more; the things that genuinely need AAD still are.
TEST(CompiledMtm, RoutingIsExplicit) {
  for (const Shape& s : {swaps::shapes::fx_xccy(), swaps::shapes::desk()}) {
    int aad = 0;
    for (const auto& in : s.prob.instruments) if (cal::instrument_is_noncacheable(in, s.prob.curves)) ++aad;
    EXPECT_EQ(aad, 0) << s.name << ": every row must be W-cacheable";
    EXPECT_NO_THROW(cal::CompiledBundleResidual{s.prob}) << s.name;
  }
  const Shape s = swaps::shapes::fx_xccy();
  cal::Instrument lookback; bool found = false;
  for (const auto& in : s.prob.instruments) if (in.quote == cal::QuoteKind::XccyMtmBasis) { lookback = in; found = true; break; }
  ASSERT_TRUE(found);
  for (auto& c : lookback.mtm.coupons) c.obs.compounded = true;  // a compounded PRODUCT observation: still AAD
  EXPECT_TRUE(cal::instrument_is_noncacheable(lookback, s.prob.curves));
  lookback = s.prob.instruments[0];
  for (auto& in : s.prob.instruments) if (in.quote == cal::QuoteKind::XccyMtmBasis) { lookback = in; break; }
  lookback.mtm.reset_num = -1;  // an incomplete MtM leg cannot be compiled — refused, not silently priced
  EXPECT_TRUE(cal::instrument_is_noncacheable(lookback, s.prob.curves));
}

// 4. The funding term is PRICED, not dropped: on the lagged EURUSD product the exact quote differs from the
//    dropped-term quotient by the term's own size (so the old shortcut cannot creep back unnoticed).
TEST(CompiledMtm, FundingTermIsPricedNotDropped) {
  const Shape s = swaps::shapes::fx_xccy();
  const cal::CompiledBundleResidual cr(s.prob);
  const Eigen::VectorXd exact = cr.model_rates(s.x_true);
  cal::BundleProblem dropped = s.prob;
  for (auto& in : dropped.instruments) if (in.quote == cal::QuoteKind::XccyMtmBasis) in.mtm.coupons.clear();  // no funding leg
  double worst = 0.0; int n = 0;
  for (int i = 0; i < s.prob.n_residuals(); ++i) {
    if (s.prob.instruments[i].quote != cal::QuoteKind::XccyMtmBasis) continue;
    cal::BundleProblem one = dropped; one.instruments = {dropped.instruments[i]}; one.instruments[0].market = 0.0;
    cal::BundleProblem oneb = s.prob; oneb.instruments = {s.prob.instruments[i]}; oneb.instruments[0].market = 0.0;
    const double q_exact = oneb.residuals<double>(s.x_true)[0], q_dropped = one.residuals<double>(s.x_true)[0];
    worst = std::max(worst, std::abs(q_exact - q_dropped)); ++n;
    EXPECT_NEAR(exact[i], s.prob.instruments[i].market, 1e-12);
  }
  std::cout << "  [mtm] funding term on the lagged EURUSD basis: max |exact - dropped| = " << worst * 1e4 << " bp over " << n << " rows\n";
  EXPECT_GT(worst, 1e-7) << "the funding term must be visible (the old shortcut dropped it)";
}

// 5. Streaming: a 25 bp move streamed on the compiled MtM rows lands where a cold calibrate lands.
TEST(CompiledMtm, StreamedTickMatchesColdCalibrateAfterA25bpMove) {
  const Shape s = swaps::shapes::fx_xccy();
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  sess.stream_update(s.q_small);
  const Eigen::VectorXd& xs = sess.stream_update(s.q_big);
  ASSERT_TRUE(sess.last_converged());
  cal::BundleProblem pb = s.prob;
  for (int i = 0; i < pb.n_residuals(); ++i) pb.instruments[i].market = s.q_big[i];
  api::BundleSession cold(pb);
  cold.calibrate(s.x0);
  const double d = (xs - cold.x()).cwiseAbs().maxCoeff();
  std::cout << "  [mtm] streamed vs cold after 25bp: |dx|max=" << d << "\n";
  EXPECT_LT(d, 1e-7);
}
