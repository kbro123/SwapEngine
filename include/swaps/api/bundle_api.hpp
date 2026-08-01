#pragma once
// Public API facade for building, calibrating, querying and streaming a whole curve BUNDLE.
//
// This is the stable seam a web/service layer talks to (CLAUDE.md is the engine; this is its product
// contract). It has TWO faces:
//   1. A JSON <-> engine-object-graph mapping (bundle_from_json / bundle_to_json and the per-object
//      (de)serializers) so a caller in any language can DEFINE and CONSTRUCT the underlying objects --
//      curves, legs, coupons, observations, instruments, the whole BundleProblem -- as data.
//   2. A stateful BundleSession that spins up an engine instance over one BundleProblem and exposes
//      calibrate / build-curves / sample / price-an-instrument / risk-operator / streaming.
//
// The API core is QuantLib-FREE: it consumes the plain-data generic instrument model (design §3). A
// later layer can add convention-driven builders (schedule -> generic coupons via ql/extract.hpp);
// that is where index/calendar knowledge belongs (CLAUDE.md §1), not here.
//
// Boost.JSON is vendored (third_party/boost); only <boost/json/fwd.hpp> leaks into this header, so a
// consumer that never touches JSON pays nothing. The Boost.JSON implementation is compiled once inside
// api/bundle_api.cpp.

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include <boost/json/fwd.hpp>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/portfolio/portfolio.hpp"  // MultiCurveBook — the batched reprice kernel
#include "swaps/pricing/fixings.hpp"

namespace swaps::api {

namespace cal = swaps::calibration;

// ---- plain result structs (no Boost types, so they cross the header boundary cheaply) -------------

// A curve sampled on a time grid: continuously-compounded zero, instantaneous forward, discount factor.
struct CurveSample {
  int currency = 0;
  std::vector<double> t;
  std::vector<double> discount;
  std::vector<double> zero;     // integral(t)/t (cont-comp), == forward(0) at t==0
  std::vector<double> forward;  // instantaneous forward
};

// Optional Tikhonov smoothness regulariser (include/swaps/calibration/regularize.hpp): penalise the
// curvature of the listed curves' knot forwards. lambda <= 0 or empty `curves` => off. Needed for
// basis-only forecast curves whose forward shape is a rank-deficient null (CLAUDE.md §7b, EUR trio).
// The `tension` flag switches the operator from the discrete second-difference penalty to the continuous
// TENSION ENERGY mu*x^T(K2+sigma^2 K1)x (regularize.hpp, research note §5): `lambda` is then the row
// weight (mu = lambda^2) and `sigma` the tension parameter -- sigma = 0 is pure bending energy INT(f'')^2,
// sigma > 0 adds the membrane term INT(f')^2 (taut, overshoot-damped). sigma is ignored when tension=false.
struct RegSpec {
  double lambda = 0.0;
  std::vector<int> curves;
  bool tension = false;  // false: second-difference curvature; true: continuous tension energy
  double sigma = 0.0;    // tension parameter (tension=true only); 0 => pure curvature penalty
  bool on() const { return lambda > 0.0 && !curves.empty(); }
};

// Result of a batched portfolio reprice (BundleSession::price_portfolio). `price_us` is the ENGINE-
// measured wall time of the pure pricing pass ONLY (a steady_clock pair around the double NPV valuation
// off the calibrated curves — no JSON/marshalling), the exact analogue of last_solve_us() for calibration.
struct PortfolioReprice {
  double npv = 0.0;      // total book NPV off the currently calibrated curves (discount-currency units)
  double pv01 = 0.0;     // d(NPV) for a +1bp PARALLEL shift of every fitted knot forward, one AAD pass
  double price_us = 0.0;  // pure engine pricing time of the double NPV pass, microseconds
  int n = 0;             // number of positions priced
};

// Result of expressing a portfolio's risk in the bundle's CALIBRATION INSTRUMENTS (BundleSession::
// price_portfolio_risk). `curve_grad` is the portfolio NPV gradient wrt the fitted STATE (knot forwards),
// dP/dx, from one forward-AAD pass. `ladder` is that same risk expressed per calibration instrument:
// dP/dq = curve_grad^T · M where M = risk_operator() = dx/dq, so ladder[i] = Σⱼ curve_grad[j]·M[j,i] is the
// portfolio's delta to a unit move in instrument i's model quote (a full analytic delta ladder, no bumping).
// This is the primitive a Python layer combines across two sessions to TRANSFORM a risk ladder from one
// bundle's instruments to another's: T = J_A · M_B, delta_B = delta_A · T (see jacobian() / risk_operator()).
// `risk_us` is the ENGINE-measured wall time of the AAD reprice + the M multiply only (a steady_clock pair,
// the exact analogue of price_us / last_solve_us), also cached in last_risk_us().
struct PortfolioRisk {
  double npv = 0.0;                  // book NPV off the calibrated curves (== price_portfolio().npv)
  Eigen::VectorXd curve_grad;        // dP/dx, length n_knots (AAD gradient wrt the fitted state)
  Eigen::VectorXd ladder;            // dP/dq = curve_grad^T · M, length n_residuals (native delta ladder)
  double risk_us = 0.0;              // engine time of the AAD reprice + M multiply, microseconds
  int n = 0;                         // number of positions
};

// ---- JSON <-> engine object graph (definitions in bundle_api.cpp) --------------------------------
// Every field is optional on parse and defaults to the struct default, so a minimal document is valid.
cal::BundleProblem bundle_from_json(const boost::json::value& v);
boost::json::value bundle_to_json(const cal::BundleProblem& p);
cal::Instrument instrument_from_json(const boost::json::value& v);
boost::json::value instrument_to_json(const cal::Instrument& ins);

// A book of positions to reprice. Reuses the SAME coupon JSON shapes the instrument (de)serializers
// parse (obs/pay/tau_pay/... for a FloatCoupon, pay/tau/scale for a FixedCoupon) so the web reuses its
// existing schedule builders. Schema (every field optional, defaults to the struct default):
//   { "positions": [
//       { "kind":"swap", "notional":<double>, "fixed_rate":<double>,
//         "fwd_curve":<int>, "disc_curve":<int>, "float_coupons":[<FloatCoupon>...],
//         "fixed_curve":<int>, "fixed_coupons":[<FixedCoupon>...] },
//       { "kind":"xccy", "notional":<double>, "fx_spot":<double>,
//         "fwd_curve":<int>, "disc_curve":<int>, "float_coupons":[<FloatCoupon>...],   // domestic leg
//         "mtm_fwd_curve":<int>, "mtm_disc_curve":<int>,
//         "mtm_reset_num":<int>, "mtm_reset_den":<int>, "mtm_coupons":[<FloatCoupon>...] } ] }
swaps::portfolio::MultiCurveBook book_from_json(const boost::json::value& v);

// A flat starting guess sized to the problem: outright curves at `level`, spread curves at 0.
Eigen::VectorXd flat_x0(const cal::BundleProblem& prob, double level = 0.02);

// ---- the session facade --------------------------------------------------------------------------
class BundleSession {
 public:
  explicit BundleSession(cal::BundleProblem prob);

  // Cold-calibrate from x0. Engine is chosen automatically: the compiled W-cache analytic path for an
  // all-linear bundle (ParRate/ParSpread/Rate), the AAD analytic path when an FX-forward / MtM-xccy
  // instrument is present (the compiled engine rejects those) or when a regulariser is requested.
  const cal::CalibrationResult& calibrate(const Eigen::VectorXd& x0, const RegSpec& reg = {});

  // Re-solve to a NEW market vector by overwriting the instruments' targets and warm-calibrating from
  // the current x. This is the streaming fallback for bundles the W-cache can't represent (custom
  // interpolation regions / non-linear schemes): still a full analytic AAD solve, sub-millisecond.
  const cal::CalibrationResult& recalibrate(const Eigen::VectorXd& new_market, const RegSpec& reg = {});

  const cal::CalibrationResult& result() const { return result_; }
  const Eigen::VectorXd& x() const { return x_; }
  const cal::BundleProblem& problem() const { return prob_; }
  bool has_fx() const { return has_fx_; }
  bool has_modular() const { return has_modular_; }
  // A non-linear region scheme (MonotoneCubic's value-dependent filter) has no constant W, so its bundle
  // can't ride the W-cache — calibration/streaming route through the AAD engine. LINEAR custom regions
  // (Flat/Linear/NaturalCubic/Hermite) are W-cacheable and stream at microseconds like the shipped curve.
  bool has_nonlinear() const { return has_nonlinear_; }
  // A bid/offer BAND makes the calibration a soft least-squares fit (banded instruments sit off-market
  // within their band). This still streams on the fast frozen-Newton path: the streamer drives the
  // BANDED residual (residuals_vs), so it solves the soft least-squares, not an exact reprice. So a band
  // does NOT force recalibrate(). This flag is informational (e.g. to label a soft-calibrated bundle).
  bool has_band() const { return has_band_; }
  // True when the bundle cannot use the frozen-Newton streaming path (start_streaming/update) at all and
  // must recalibrate() each tick. Only a NON-LINEAR region scheme (MonotoneCubic) qualifies now: it has no
  // constant W for ANY curve, so there is no compiled engine to freeze. FX/MtM no longer forces this — the
  // HYBRID engine streams the cacheable rows on the W-cache and the FX/MtM rows on a width-reduced AAD
  // block, refreshing the AAD Jacobian only on staleness. A band is a soft target and never forced it.
  bool needs_recalibrate() const { return has_nonlinear_; }

  // Query the built curves at the current x on a shared time grid.
  std::vector<CurveSample> sample(const std::vector<double>& times) const;

  // Price an ARBITRARY instrument off the current curves: its model quote (par rate / par spread /
  // future rate / FX forward) and the residual against its own stored `market`.
  double model_quote(const cal::Instrument& ins) const;
  double residual(const cal::Instrument& ins) const;

  // The calibration Jacobian J = dq/dx (n_residuals x n_knots): ROWS are calibration instruments, COLUMNS
  // are the fitted knot forwards, so J(i,j) = ∂(model quote of instrument i)/∂x_j. This is the SAME J that
  // risk_operator() builds M from (risk_operator() calls this, so the two can never desync). One forward-AAD
  // pass over the stacked residual (residual = q − market for the linear ParRate/ParSpread/Rate instruments,
  // so d(residual)/dx == dq/dx). A Python layer transforms a risk ladder from bundle A to bundle B via
  // T = J_A · M_B (n_res_A x n_res_B), delta_B = delta_A · T. `reg` is accepted for signature symmetry with
  // risk_operator() but does not affect J (a regulariser changes M through RᵀR, never the quote Jacobian).
  Eigen::MatrixXd jacobian(const RegSpec& reg = {}) const;

  // The analytic risk operator M = dx/dq = (JᵀJ + RᵀR)⁻¹ Jᵀ  (n_knots x n_residuals). Left-multiply a
  // portfolio's d(NPV)/dx (one AAD pass) by M for a full analytic delta ladder, no bumping (CLAUDE.md #4).
  Eigen::MatrixXd risk_operator(const RegSpec& reg = {}) const;

  // ---- batched portfolio reprice (the web "reprice N random swaps" feature) -----------------------
  // Reprice a full multi-curve + xccy book off the CURRENTLY CALIBRATED curves (the current x) and report
  // {npv, pv01, price_us, n}. The NPV is a pure double pass through the pricing kernel (float_leg_pv /
  // annuity / xccy_mtm_leg_pv), timed by a steady_clock around that pass ONLY — the engine stamps the
  // pricing time exactly as calibrate() stamps last_solve_us(), with no Python marshalling inside price_us.
  // PV01 is d(NPV) for a +1bp parallel shift of every fitted knot forward, from ONE forward-AAD pass (no
  // bump-and-reprice). Also cached in last_price_us(). The book references curves by their bundle index.
  PortfolioReprice price_portfolio(const swaps::portfolio::MultiCurveBook& book) const;
  // Convenience for a language binding: parse a book JSON document (schema on book_from_json) and reprice.
  PortfolioReprice price_portfolio_json(const std::string& book_json) const;

  // Express a book's risk in THIS bundle's calibration instruments: one forward-AAD pass gives the NPV and
  // its state gradient curve_grad = dP/dx, then ladder = curve_grad^T · M (M = risk_operator()) is the delta
  // per calibration quote — ladder[i] = Σⱼ curve_grad[j]·M[j,i]. The AAD reprice + the M multiply are ENGINE-
  // timed into risk_us (also last_risk_us()); the M FORMATION (the calibration Jacobian solve, book-
  // independent) is outside the clock, mirroring how price_portfolio times only the book-dependent pass.
  PortfolioRisk price_portfolio_risk(const swaps::portfolio::MultiCurveBook& book) const;
  // Convenience for a language binding: parse a book JSON document (same schema as book_from_json) and risk it.
  PortfolioRisk price_portfolio_risk_json(const std::string& book_json) const;

  // ---- streaming (any bundle with a constant W: hard, banded, portfolio, or mixed FX/MtM) ---------
  // Anchor a StreamingCalibrator at the current x; each stream_update(q) re-solves to the exact curve
  // for the new market q (frozen-Newton off the cached Jacobian, refreshed only on staleness). A mixed
  // FX/MtM bundle streams on the hybrid engine (its FX rows refresh their AAD Jacobian only on staleness).
  // Throws only for a bundle with no constant W at all (a MonotoneCubic region scheme) — recalibrate those.
  // step_tol > 0 overrides the frozen-Newton convergence tolerance (||dx||_inf); 0 keeps the exact default
  // (machine-precision reprice each tick). A looser tol stops in fewer corrector steps — a speed/accuracy
  // knob for a live viewer — while still refreshing the Jacobian on staleness (so it stays robust).
  void start_streaming(const RegSpec& reg = {}, double step_tol = 0.0);
  const Eigen::VectorXd& stream_update(const Eigen::VectorXd& new_market);
  bool streaming() const { return static_cast<bool>(stream_); }

  // ---- solve-time telemetry (measured by default; a steady_clock pair is ~100ns, <0.1% of a solve) --
  // Every calibrate/recalibrate/stream_update stamps the ENGINE-measured wall time of the solve itself
  // (no marshalling). Callers should report these instead of timing across a language boundary.
  double last_solve_us() const { return last_solve_us_; }        // most recent solve, any path (µs)
  double last_price_us() const { return last_price_us_; }        // most recent price_portfolio pricing pass (µs)
  double last_risk_us() const { return last_risk_us_; }          // most recent price_portfolio_risk pass (µs)
  // Running mean of stream_update solve times since the last start_streaming(); 0 before the first tick.
  double stream_avg_us() const { return stream_ticks_ ? stream_sum_us_ / stream_ticks_ : 0.0; }
  long stream_ticks() const { return stream_ticks_; }            // stream_update calls since start_streaming
  // Frozen-Newton introspection for the LAST stream_update (all 0 on the pure fast path): Gauss-Newton
  // steps taken, analytic-Jacobian refreshes triggered, and the market drift that provoked them.
  int last_newton_steps() const { return last_newton_steps_; }
  int last_refreshes() const { return last_refreshes_; }
  double last_drift() const { return last_drift_; }

  // ---- fixings as pricing context (E2) -----------------------------------------------------------
  // Any observation carrying a fixing_schedule is RESOLVED from this session's fixing table against the
  // evaluation date: past days -> `realized` (throwing/flagging MissingFixing if absent), future days ->
  // forecast sub-periods. Resolution runs on construction and on every set_evaluation_date/set_fixings.
  // It only rewrites `realized`/subs (constants) in prob_ -- it does NOT recompile: the next
  // calibrate()/stream picks the change up (a `realized`-only change is a residual-constant shift the
  // streamer absorbs; a past/future boundary move changes subs and refreshes W via streamer staleness).

  // Set the evaluation date (integer serial the caller defines); fixings strictly before it are fixed.
  void set_evaluation_date(int serial) { eval_date_ = serial; resolve_fixings(); }
  // Upsert fixings for one index and re-resolve every schedule-carrying observation in place. Returns the
  // number of observations still un-priceable (a required past fixing is missing) after the update.
  int set_fixings(const std::string& index, const std::vector<std::pair<int, double>>& rows) {
    fixings_.bulk_set(index, rows);  // notifies (unused here; the session drives resolution directly)
    return resolve_fixings();
  }
  int evaluation_date() const { return eval_date_; }
  int n_unresolved() const { return n_unresolved_; }
  const swaps::pricing::FixingTable& fixings() const { return fixings_; }

 private:
  // Collect pointers to every schedule-carrying observation in prob_ (futures obs + float-leg coupons,
  // recursing into portfolio components), and resolve them against {eval_date_, fixings_}. Returns the
  // count that could not resolve (missing a past fixing). Pointers are stable: prob_ is not resized.
  int resolve_fixings();

  cal::BundleProblem prob_;
  Eigen::VectorXd x_;
  cal::CalibrationResult result_;
  bool has_fx_ = false;
  bool has_modular_ = false;
  bool has_nonlinear_ = false;
  bool has_band_ = false;
  std::unique_ptr<cal::StreamingCalibrator<cal::BundleProblem>> stream_;
  swaps::pricing::FixingTable fixings_;
  int eval_date_ = 0;
  int n_unresolved_ = 0;

  // Solve-time telemetry (stamped by calibrate/recalibrate/stream_update; see the getters above).
  double last_solve_us_ = 0;
  // Pricing-time telemetry, stamped by the (const) price_portfolio; mutable so the query stays const.
  mutable double last_price_us_ = 0;
  // Risk-time telemetry, stamped by the (const) price_portfolio_risk; mutable so the query stays const.
  mutable double last_risk_us_ = 0;
  double stream_sum_us_ = 0;   // sum of stream_update solve times since start_streaming()
  long stream_ticks_ = 0;      // count of those ticks
  int last_newton_steps_ = 0;
  int last_refreshes_ = 0;
  double last_drift_ = 0;
};

// One-shot stateless JSON dispatcher for a web call. Request:
//   { "bundle": {...},                         (required) the BundleProblem object graph
//     "x0": [...],                             (optional) start; else a flat guess
//     "regularize": {"lambda":1.0,"curves":[3,4],   (optional) smoothness penalty on those curves;
//                     "tension":true,"sigma":0.0},   add tension:true for the continuous tension energy
//                                                     (sigma=0 curvature, sigma>0 taut) vs 2nd-difference
//     "sample_times": [...],                   (optional) grid to sample every curve on
//     "price": [ {instrument}, ... ],          (optional) instruments to price off the solved curves
//     "risk": true }                           (optional) include the dx/dq operator
// Response mirrors it: { "calibration": {...}, "x": [...], "curves": [...], "priced": [...],
//                        "risk_operator": [[...]] }. On error: { "error": "..." }.
std::string run_json(const std::string& request);

}  // namespace swaps::api
