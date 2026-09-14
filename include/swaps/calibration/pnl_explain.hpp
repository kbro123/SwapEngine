#pragma once
// pnl_explain.hpp — P&L EXPLAIN: the desk-standard attribution of a book's NPV change between two
// dates/markets into CARRY (time value / financing), ROLL-DOWN (the curve's own shape sliding under
// fixed calendar time), MARKET-MOVE (the analytic delta ladder dotted into the quote change) and a
// RESIDUAL (the unexplained second-order piece). Tier-2 analytics.
//
// Templated/QuantLib-FREE: it reuses ONLY existing primitives — build_bundle_curves + the CurveHandle
// pricing kernel (portfolio::MultiCurveBook::value → float_leg_pv / annuity / xccy_mtm_leg_pv) for the
// reprices, and the caller's analytic delta ladder dP/dq (BundleSession::price_portfolio_risk /
// risk_operator + bucketed_delta) for the market term. No bump-and-reprice, no QuantLib.
//
// =================================================================================================
// THE CONTRACT (definitions are exact; the SUM is the non-negotiable invariant)
// =================================================================================================
// Given a book, a calibrated curve state x0 at t0 (market q0), a NEW state x1 at t1 (market q1), and
// dt = t1 - t0, define the double-precision repricing operator
//
//     NPV(book, x, "at t1")  ≡  book.value<double> off the curves built from x, with EVERY cashflow's
//                               curve-time advanced by dt (see "valuing at t1" below).
//
// The four components and their reference reprices:
//
//   total  = NPV(book, x1, at t1) - NPV(book, x0, at t0)
//            the actual mark-to-market change of the position over the interval.
//
//   carry  = NPV_rebased(book, x0, at t1) - NPV(book, x0, at t0)
//            the PURE time-value / financing return of holding the book from t0 to t1 with the curve
//            UNCHANGED and each cashflow kept at its own calendar date. "Rebased" means the surviving
//            (not-yet-paid) cashflows are discounted on the SAME curve x0 but with the numeraire rolled
//            to t1 — DF'(t) = DF_x0(t) / DF_x0(dt) per discount curve — i.e. financed forward at that
//            curve's own short rate. Forward-rate DF RATIOS are invariant under the rebase, so `carry`
//            is exactly the financing accretion (≈ NPV·(e^{r·dt}-1) for a self-discounting book; ~0 for
//            an at-par book). This is theta.
//
//   roll   = NPV(book, x0, at t1) - NPV_rebased(book, x0, at t1)
//            the ROLL-DOWN: what the first reprice picks up OVER the pure-financing reprice because the
//            cashflows have SLID down the (unchanged) curve — the forwards you realize by advancing the
//            valuation date along today's curve shape. carry + roll therefore telescopes EXACTLY to the
//            full time effect NPV(book, x0, at t1) - NPV(book, x0, at t0) on the unchanged market.
//
//   market = ladder · (q1 - q0),  ladder_i = dP/dq_i   (the analytic delta ladder at (x0, q0))
//            the first-order market-move P&L. Per-instrument contributions ladder_i·(q1-q0)_i are
//            returned in `market_ladder`.
//
//   residual = total - carry - roll - market
//            everything left over — the SECOND-ORDER / cross term of the market move (and any effect of
//            the roll convention below). By CONSTRUCTION carry + roll + market + residual == total to
//            machine precision; that identity is the contract this module guarantees (asserted in the
//            test to 1e-9). Substituting the reprices above, residual telescopes to
//                residual = [NPV(book, x1, at t1) - NPV(book, x0, at t1)] - market
//            i.e. the exact horizon market move minus its linear (delta·dq) approximation — the genuine
//            unexplained piece.
//
// -------------------------------------------------------------------------------------------------
// "Valuing at t1" — the roll convention (documented approximation)
// -------------------------------------------------------------------------------------------------
// MultiCurveBook holds each coupon's times in CURVE time from the t0 reference date (FloatCoupon.pay /
// obs.sub_start / obs.sub_end / reset_time; FixedCoupon.pay). Advancing the valuation date by dt shifts
// every such time DOWN by dt. We therefore:
//   * DROP any coupon that has already paid (pay <= dt), and any position left with no floating cashflow
//     (an xccy position also needs a surviving mtm leg); a fully-paid book prices to 0.
//   * For the surviving coupons, subtract dt from every curve-time and FLOOR the forecast/reset times at
//     0 (a coupon straddling t1 keeps its pay date but its already-elapsed accrual is dropped rather than
//     evaluated at a negative curve-time — a negative time returns DF ≡ 1, curve_module.hpp integral(t <= 0) == 0, not DF > 1). Accrual
//     fractions (tau_pay/tau/tau_index) and `realized` are left AS-IS: we do NOT re-fix a partially
//     elapsed coupon. This is the one approximation, and it lives entirely inside `residual` for a market
//     move and cancels exactly for a pure roll (both t1-legs use the same rolled book at the same x). It
//     never touches the carry+roll+market+residual == total identity.
// The pure-financing `carry` leg instead keeps cashflows at their ORIGINAL dates (drop-only, no shift)
// and rebases the numeraire, so `carry` is slide-free and `roll` isolates the slide.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"  // BundleProblem, build_bundle_curves, CurveHandle
#include "swaps/calibration/diagnostics.hpp"     // CalibrationSession, seed_or_flat
#include "swaps/calibration/regularize.hpp"      // RegSpec
#include "swaps/portfolio/portfolio.hpp"         // MultiCurveBook
#include "swaps/pricing/cashflows.hpp"           // FloatCoupon / FixedCoupon

namespace swaps::calibration {

// The decomposition result. carry + roll + market + residual == total (to machine precision).
struct PnlExplain {
  double total = 0.0;     // NPV(x1, t1) - NPV(x0, t0)
  double carry = 0.0;     // pure time value / financing (theta), curve unchanged
  double roll = 0.0;      // roll-down: the curve's shape sliding under fixed calendar time
  double market = 0.0;    // ladder · (q1 - q0), the first-order market move
  double residual = 0.0;  // total - carry - roll - market (the second-order / unexplained piece)
  double npv_t0 = 0.0;    // book NPV at (x0, t0)
  double npv_t1 = 0.0;    // book NPV at (x1, t1)
  std::vector<double> market_ladder;  // per-instrument market P&L: ladder_i · (q1-q0)_i
};

namespace detail {

// A CurveHandle that rebases another handle's numeraire to the dt-horizon: DF'(t) = DF(t)/DF(dt),
// equivalently integral'(t) = integral(t) - integral(dt). Forwards (hence forward-rate DF ratios) are
// unchanged, so only the DISCOUNTING of surviving cashflows is financed forward — the `carry` leg.
struct RebasedHandle : CurveHandle<double> {
  const CurveHandle<double>* base;
  double idt;  // integral(dt) of the base curve
  RebasedHandle(const CurveHandle<double>* b, double dt) : base(b), idt(b->integral(dt)) {}
  double forward(double t) const override { return base->forward(t); }
  double integral(double t) const override { return base->integral(t) - idt; }
  double discount(double t) const override { using std::exp; return exp(-(base->integral(t) - idt)); }
  void set_forwards(const Eigen::Matrix<double, Eigen::Dynamic, 1>&) override {}
};

}  // namespace detail

// Advance the valuation date of `in` by dt. `shift` true => subtract dt from every cashflow curve-time
// (floored at 0) — the roll leg; `shift` false => keep times, only drop already-paid coupons — the
// carry (numeraire-rebase) leg. Positions with no surviving floating cashflow are dropped.
inline portfolio::MultiCurveBook roll_book(const portfolio::MultiCurveBook& in, double dt, bool shift) {
  using Book = portfolio::MultiCurveBook;
  Book out;
  auto proc_float = [&](const std::vector<pricing::FloatCoupon>& legs) {
    std::vector<pricing::FloatCoupon> keep;
    for (pricing::FloatCoupon c : legs) {
      if (c.pay <= dt) continue;  // already paid
      if (shift) {
        c.pay -= dt;
        for (auto& s : c.obs.sub_start) s = std::max(0.0, s - dt);
        for (auto& e : c.obs.sub_end) e = std::max(0.0, e - dt);
        if (c.reset_time >= 0.0) c.reset_time = std::max(0.0, c.reset_time - dt);
      }
      keep.push_back(std::move(c));
    }
    return keep;
  };
  auto proc_fixed = [&](const std::vector<pricing::FixedCoupon>& legs) {
    std::vector<pricing::FixedCoupon> keep;
    for (pricing::FixedCoupon c : legs) {
      if (c.pay <= dt) continue;
      if (shift) c.pay -= dt;
      keep.push_back(c);
    }
    return keep;
  };
  for (const auto& pos : in.positions) {
    Book::Position q = pos;
    q.float_coupons = proc_float(pos.float_coupons);
    q.fixed_coupons = proc_fixed(pos.fixed_coupons);
    if (pos.kind == Book::Kind::Xccy) {
      q.mtm_coupons = proc_float(pos.mtm_coupons);
      if (q.float_coupons.empty() || q.mtm_coupons.empty()) continue;  // xccy needs both legs alive
    } else if (q.float_coupons.empty()) {
      continue;
    }
    out.positions.push_back(std::move(q));
  }
  return out;
}

// Reprice a MultiCurveBook off bundle state x (double pass). rebase_dt > 0 rolls every discount curve's
// numeraire forward to that horizon (the carry leg); rebase_dt <= 0 prices as-is. Empty book => 0.
inline double book_npv(const BundleProblem& prob, const portfolio::MultiCurveBook& book,
                       const Eigen::VectorXd& x, double rebase_dt = 0.0) {
  if (book.positions.empty()) return 0.0;
  const auto C = build_bundle_curves<double>(
      prob.curves, [&](int c, int i) { return x[prob.offset(c) + i]; });
  if (rebase_dt <= 0.0) {
    const auto cof = [&C](int i) -> const CurveHandle<double>& { return *C[i]; };
    return book.value<double>(cof);
  }
  std::vector<detail::RebasedHandle> R;
  R.reserve(C.size());
  for (const auto& h : C) R.emplace_back(h.get(), rebase_dt);  // base pts into the stable C[i]
  const auto cof = [&R](int i) -> const CurveHandle<double>& { return R[i]; };
  return book.value<double>(cof);
}

// The decomposition. `prob` supplies the curve topology (curves + offsets) that x0/x1 index. `ladder` is
// the analytic delta ladder dP/dq at (x0, q0) — length = n instruments — and `dq` = q1 - q0 (same length).
// t0/t1 enter only through dt = t1 - t0.
inline PnlExplain pnl_explain(const BundleProblem& prob, const portfolio::MultiCurveBook& book,
                              const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, double dt,
                              const Eigen::VectorXd& ladder, const Eigen::VectorXd& dq) {
  const portfolio::MultiCurveBook rolled = roll_book(book, dt, /*shift=*/true);   // roll leg
  const portfolio::MultiCurveBook dropped = roll_book(book, dt, /*shift=*/false);  // carry (rebase) leg

  const double npv0 = book_npv(prob, book, x0, 0.0);        // (x0, t0)
  const double cr_t1 = book_npv(prob, rolled, x0, 0.0);     // (x0, t1): carry+roll on the unchanged market
  const double carry_t1 = book_npv(prob, dropped, x0, dt);  // (x0, t1): pure financing, numeraire rebased
  const double npv1 = book_npv(prob, rolled, x1, 0.0);      // (x1, t1): the new market at the horizon

  PnlExplain out;
  out.npv_t0 = npv0;
  out.npv_t1 = npv1;
  out.total = npv1 - npv0;
  out.carry = carry_t1 - npv0;
  out.roll = cr_t1 - carry_t1;
  out.market = (ladder.size() && dq.size()) ? ladder.dot(dq) : 0.0;
  out.residual = out.total - out.carry - out.roll - out.market;
  out.market_ladder.resize(dq.size());
  for (int i = 0; i < dq.size(); ++i)
    out.market_ladder[i] = (i < ladder.size() ? ladder[i] : 0.0) * dq[i];
  return out;
}


// ---- the `pnl` verb's whole computation (E7 stage 6.7) ------------------------------------------------------------
// bundle0 calibrates to q0 -> x0 and the analytic ladder dP/dq at (x0, q0); an optional bundle1 of the same shape
// calibrates to q1 -> x1 and dq = q1 - q0 (absent: x1 = x0, dq = 0, pure carry / roll). An explicit x0 / x1 SEEDS the
// calibration and also OVERRIDES the state the decomposition reprices at (the ladder stays at the calibrated x0).
struct PnlRequest {
  BundleProblem bundle0;
  std::optional<BundleProblem> bundle1;  // absent => pure carry / roll
  portfolio::MultiCurveBook book;
  double dt_years = 0.0;
  std::optional<Eigen::VectorXd> x0, x1;  // seed AND decomposition override; a wrong length throws
  RegSpec reg;
};

struct PnlReport {
  PnlExplain explain;
  Eigen::VectorXd dq;
  double dt_years = 0.0;
  int n = 0;  // positions
};

template <class S>
concept PnlSession = CalibrationSession<S> &&
    requires(const S cs, const portfolio::MultiCurveBook& b, const RegSpec& r) {
      { cs.price_portfolio_risk(b, r).ladder } -> std::convertible_to<Eigen::VectorXd>;
    };

template <PnlSession Session>
PnlReport pnl_report(PnlRequest r) {
  if (r.bundle0.n_curves() == 0) throw std::invalid_argument("pnl: bundle0 has no curves");
  Session s0(std::move(r.bundle0));
  const BundleProblem& P0 = s0.problem();
  s0.calibrate(seed_or_flat(P0, r.x0, "pnl"), r.reg);
  const Eigen::VectorXd ladder = s0.price_portfolio_risk(r.book, r.reg).ladder;  // dP/dq at (x0, q0)
  if (r.x1 && r.x1->size() != P0.n_knots())
    throw std::invalid_argument("pnl: x1 length does not match the bundle's knot count");

  PnlReport out;
  out.dq = Eigen::VectorXd::Zero(P0.n_residuals());
  Eigen::VectorXd x1 = s0.x();
  if (r.bundle1) {
    if (r.bundle1->n_knots() != P0.n_knots())
      throw std::invalid_argument("pnl: bundle1 must share bundle0's curve topology (knot count differs)");
    if (r.bundle1->n_residuals() != P0.n_residuals())
      throw std::invalid_argument("pnl: bundle1 must share bundle0's instruments (residual count differs)");
    Session s1(std::move(*r.bundle1));
    s1.calibrate(seed_or_flat(s1.problem(), r.x1, "pnl"), r.reg);
    x1 = s1.x();
    out.dq = s1.problem().market() - P0.market();
  }
  const Eigen::VectorXd x0 = r.x0 ? *r.x0 : s0.x();
  if (r.x1) x1 = *r.x1;
  out.explain = pnl_explain(P0, r.book, x0, x1, r.dt_years, ladder, out.dq);
  out.dt_years = r.dt_years;
  out.n = static_cast<int>(r.book.positions.size());
  return out;
}

}  // namespace swaps::calibration
