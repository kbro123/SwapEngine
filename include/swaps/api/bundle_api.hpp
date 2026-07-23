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
struct RegSpec {
  double lambda = 0.0;
  std::vector<int> curves;
  bool on() const { return lambda > 0.0 && !curves.empty(); }
};

// ---- JSON <-> engine object graph (definitions in bundle_api.cpp) --------------------------------
// Every field is optional on parse and defaults to the struct default, so a minimal document is valid.
cal::BundleProblem bundle_from_json(const boost::json::value& v);
boost::json::value bundle_to_json(const cal::BundleProblem& p);
cal::Instrument instrument_from_json(const boost::json::value& v);
boost::json::value instrument_to_json(const cal::Instrument& ins);

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

  // Query the built curves at the current x on a shared time grid.
  std::vector<CurveSample> sample(const std::vector<double>& times) const;

  // Price an ARBITRARY instrument off the current curves: its model quote (par rate / par spread /
  // future rate / FX forward) and the residual against its own stored `market`.
  double model_quote(const cal::Instrument& ins) const;
  double residual(const cal::Instrument& ins) const;

  // The analytic risk operator M = dx/dq = (JᵀJ + RᵀR)⁻¹ Jᵀ  (n_knots x n_residuals). Left-multiply a
  // portfolio's d(NPV)/dx (one AAD pass) by M for a full analytic delta ladder, no bumping (CLAUDE.md #4).
  Eigen::MatrixXd risk_operator(const RegSpec& reg = {}) const;

  // ---- streaming (all-linear bundles only) -------------------------------------------------------
  // Anchor a StreamingCalibrator at the current x; each stream_update(q) re-solves to the exact curve
  // for the new market q (frozen-Newton off the cached Jacobian, refreshed only on staleness). Throws
  // if the bundle contains an FX/MtM instrument (not W-cacheable).
  void start_streaming(const RegSpec& reg = {});
  const Eigen::VectorXd& stream_update(const Eigen::VectorXd& new_market);
  bool streaming() const { return static_cast<bool>(stream_); }

 private:
  cal::BundleProblem prob_;
  Eigen::VectorXd x_;
  cal::CalibrationResult result_;
  bool has_fx_ = false;
  bool has_modular_ = false;
  bool has_nonlinear_ = false;
  std::unique_ptr<cal::StreamingCalibrator<cal::BundleProblem>> stream_;
};

// One-shot stateless JSON dispatcher for a web call. Request:
//   { "bundle": {...},                         (required) the BundleProblem object graph
//     "x0": [...],                             (optional) start; else a flat guess
//     "regularize": {"lambda":1.0,"curves":[3,4]},  (optional)
//     "sample_times": [...],                   (optional) grid to sample every curve on
//     "price": [ {instrument}, ... ],          (optional) instruments to price off the solved curves
//     "risk": true }                           (optional) include the dx/dq operator
// Response mirrors it: { "calibration": {...}, "x": [...], "curves": [...], "priced": [...],
//                        "risk_operator": [[...]] }. On error: { "error": "..." }.
std::string run_json(const std::string& request);

}  // namespace swaps::api
