#pragma once
// calibration/diagnostics.hpp — calibration HEALTH read off the calibration Jacobian (E7 stage 3.6): the
// `calib_report` verb's whole computation as calibration-layer library code, unit-testable on hand-built
// matrices and reachable by tools/mutate.py.
//
//   * quote_diagnostics     — per quote: model vs target, residual, and for a banded quote whether it sits in its
//                             band and the residual's slope there. BundleSession::quote_diagnostics() is its codec.
//   * jacobian_conditioning — the singular spectrum of J = dq/dx and sigma_max / sigma_min (>= 1; the largest
//                             double for an exact null).
//   * hat_diagonal          — per quote, clamp(J_i · M_i, 0, 1) with M = dx/dq: how well the quote pins its pillar.
//   * calibration_report    — calibrate, then all of the above, over any session type meeting CalibrationSession
//                             (the verb instantiates it with api::BundleSession; this header never includes api).
//
// identifiability divides the residual market scale D back out of M = risk_operator = J⁺·D, so it is the documented
// projector diag(J(JᵀJ+RᵀR)⁻¹Jᵀ): what a quote pins, not how its residual is scaled. FIXED 2026-09-13 -- a banded quote
// reported its decay and an FX forward P_ii/(q·T) (tests/risk_scale_repro_test.cpp).

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"

namespace swaps::calibration {

// ---- per-quote fit -------------------------------------------------------------------------------------------
struct QuoteDiagnostic {
  double model = 0.0, target = 0.0, residual = 0.0;
  bool soft = false;     // the quote carries a bid/offer band (band_upper > band_lower)
  bool in_band = false;  // soft, and the model sits inside the band
  double weight = 1.0;   // the residual's slope at the model: the decay inside a band, 1 outside or for a hard pin
  double lower = 0.0, upper = 0.0, decay = 1.0;  // the band (soft quotes only)
};

inline QuoteDiagnostic quote_diagnostic(const Instrument& ins, double model) {
  QuoteDiagnostic d;
  d.model = model;
  d.target = ins.market;
  d.residual = model - ins.market;
  d.soft = ins.band_upper > ins.band_lower;
  if (d.soft) {
    d.lower = ins.band_lower;
    d.upper = ins.band_upper;
    d.decay = ins.band_decay;
    d.in_band = ins.band_lower <= model && model <= ins.band_upper;
    d.weight = ins.penalty().slope_at(model, ins.market);
  }
  return d;
}

// Every quote's diagnostic at state x (curves built once).
inline std::vector<QuoteDiagnostic> quote_diagnostics(const BundleProblem& p, const Eigen::VectorXd& x) {
  const auto C = build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const CurveHandle<double>& { return *C[i]; };
  std::vector<QuoteDiagnostic> out;
  out.reserve(p.instruments.size());
  for (const Instrument& ins : p.instruments)
    out.push_back(quote_diagnostic(ins, instrument_model_quote<double>(ins, curve_of)));
  return out;
}

// ---- the Jacobian --------------------------------------------------------------------------------------------
struct JacobianConditioning {
  Eigen::VectorXd singular_values;  // descending; empty for an empty Jacobian
  double condition_number = 1.0;
};

inline JacobianConditioning jacobian_conditioning(const Eigen::MatrixXd& J) {
  JacobianConditioning c;
  if (J.rows() == 0 || J.cols() == 0) return c;
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);  // singular values only
  c.singular_values = svd.singularValues();
  if (c.singular_values.size() == 0) return c;
  const double smax = c.singular_values[0];
  const double smin = c.singular_values[c.singular_values.size() - 1];
  double cond = smin > 0.0 ? smax / smin : std::numeric_limits<double>::max();
  if (!std::isfinite(cond)) cond = std::numeric_limits<double>::max();
  c.condition_number = std::max(cond, 1.0);  // sigma_max >= sigma_min; guards rounding only
  return c;
}

// Per quote i, clamp(J_i · M_i, 0, 1): the diagonal of the model-resolution matrix J·M without forming it.
inline Eigen::VectorXd hat_diagonal(const Eigen::MatrixXd& J, const Eigen::MatrixXd& M) {
  if (M.rows() != J.cols() || M.cols() != J.rows())
    throw std::invalid_argument("hat_diagonal: M must be n_knots x n_residuals for an n_residuals x n_knots J");
  Eigen::VectorXd h(J.rows());
  for (Eigen::Index i = 0; i < J.rows(); ++i) h[i] = std::clamp(J.row(i).dot(M.col(i)), 0.0, 1.0);
  return h;
}

// Per quote, how well it pins its pillar: the hat diagonal of J against M with the residual market scale D divided back
// out (M = J⁺·D is BundleSession::risk_operator's contract). A zero scale -- a band with decay 0 -- carries no information.
inline Eigen::VectorXd identifiability(const Eigen::MatrixXd& J, const Eigen::MatrixXd& M, const Eigen::VectorXd& market_scale) {
  if (market_scale.size() != M.cols())
    throw std::invalid_argument("identifiability: the market scale needs one entry per residual");
  Eigen::MatrixXd P = M;
  for (Eigen::Index i = 0; i < P.cols(); ++i) {
    if (market_scale[i] != 0.0)
      P.col(i) /= market_scale[i];
    else
      P.col(i).setZero();
  }
  return hat_diagonal(J, P);
}

// ---- THE FRONT ANCHOR RULE (owner's rule, 2026-11-01; PRINCIPLES.md P15, ASSUMPTIONS K53) -----------------------
// Every curve's front is pinned by an instrument that reads THAT curve's forward over ONE window starting at the
// front: a Rate row forecast on it (a fixing, a deposit, a meeting-to-meeting or futures observation), a quotient row
// whose leg on the curve is a single coupon (a 1Y OIS, a 6M IRS to its first pillar), an FX forward reading its
// discount from the value date (one integral from 0), or a Portfolio with such a component (the CCP 1D zero-basis
// row). A multi-coupon swap is NOT an anchor, nor is any discount-only read: a par rate's discounting column vanishes
// at the front, which is how a spread curve with a knot per pillar ended up held by the smoothing tie-breaker alone.
// Region 0 flat-extrapolates its first knot, so an unanchored front silently carries a term pillar's value to day one.
struct FrontAnchor {
  int curve = -1;
  double first_knot = 0.0;  // the curve's first interpolation knot time (-1: no interp knots)
  int row = -1;             // the anchoring row (global residual index), -1 = UNANCHORED
  double start = 0.0, end = 0.0;  // the anchoring window on the curve
};
// The one window instrument `ins` reads of curve c's forward, if it is a single-window read (see above).
inline bool anchor_window(const Instrument& ins, int c, double& s, double& e) {
  const auto leg_window = [&](const FloatLeg& leg) {
    if (leg.forecast != c || leg.coupons.size() != 1) return false;
    const pricing::RateObservation& o = leg.coupons.front().obs;
    if (o.sub_start.empty()) return false;
    s = o.sub_start.front(); e = o.sub_end.back();
    return true;
  };
  switch (ins.quote) {
    case QuoteKind::Rate:
      if (ins.forecast != c || ins.obs.sub_start.empty()) return false;
      s = ins.obs.sub_start.front(); e = ins.obs.sub_end.back();
      return true;
    case QuoteKind::FxForward:
      if (ins.fx_num != c && ins.fx_den != c) return false;
      s = ins.fx_spot_time; e = ins.fx_time;
      return true;
    case QuoteKind::Portfolio:
      for (const auto& comp : ins.combination)
        if (anchor_window(comp.instrument, c, s, e)) return true;
      return false;
    case QuoteKind::ParRate: case QuoteKind::ParSpread: case QuoteKind::ZeroCouponRate: case QuoteKind::XccyMtmBasis: case QuoteKind::Npv:
      return leg_window(ins.fwd) || leg_window(ins.bench);
    default:
      return false;
  }
}
// Per curve, the first row that anchors its front, or row = -1. The window starts within `front_tol` years of the
// value date (a spot lag; a futures window that opened before the value date starts at a negative time) and ENDS AT
// the first knot: the front knot IS that instrument's maturity -- the knot sits at the window's end or within
// `lag_tol` after it (a knot at the instrument's PAYMENT time, a pay lag past the accrual end). A leading Flat region
// is the meeting-date front, whose first segment is one value however long: there the window only has to reach the
// knot (the SOFR reference market: FOMC knots under a 1M futures strip). A window that ends past a non-Flat first
// knot pins that knot together with the next one and leaves the front to extrapolation -- not an anchor.
inline std::vector<FrontAnchor> front_anchors(const BundleProblem& p, double front_tol = 0.03, double lag_tol = 0.03) {
  std::vector<FrontAnchor> out;
  for (int c = 0; c < p.n_curves(); ++c) {
    FrontAnchor a;
    a.curve = c;
    const auto& cs = p.curves[static_cast<std::size_t>(c)];
    a.first_knot = cs.n_interp_knots() > 0 ? cs.regions.front().knots.front() : -1.0;
    const bool flat_front = !cs.regions.empty() && cs.regions.front().scheme == curve::Scheme::Flat;
    for (int r = 0; r < p.n_residuals() && a.row < 0; ++r) {
      double s = 0.0, e = 0.0;
      if (!anchor_window(p.row(r), c, s, e) || s > front_tol) continue;
      const bool ends_at_knot = flat_front ? e + lag_tol >= a.first_knot : (a.first_knot >= e - 1e-6 && a.first_knot <= e + lag_tol);
      if (ends_at_knot) { a.row = r; a.start = s; a.end = e; }
    }
    out.push_back(a);
  }
  return out;
}

// The seed a report calibrates from: the caller's, length-checked, else the market-implied flat seed.
inline Eigen::VectorXd seed_or_flat(const BundleProblem& p, const std::optional<Eigen::VectorXd>& x0,
                                    const std::string& where) {
  if (!x0) return flat_x0(p);
  if (x0->size() != p.n_knots())
    throw std::invalid_argument(where + ": x0 length does not match the bundle's knot count");
  return *x0;
}

// ---- the report, over any calibrating session --------------------------------------------------------------------
template <class S>
concept CalibrationSession =
    std::constructible_from<S, BundleProblem> && requires(S s, const S cs, const Eigen::VectorXd& x, const RegSpec& r) {
      { s.calibrate(x) } -> std::convertible_to<const CalibrationResult&>;  // smoothing is region data: no argument
      { cs.problem() } -> std::convertible_to<const BundleProblem&>;
      { cs.x() } -> std::convertible_to<const Eigen::VectorXd&>;
      { cs.jacobian() } -> std::convertible_to<Eigen::MatrixXd>;
      { cs.risk_operator(r) } -> std::convertible_to<Eigen::MatrixXd>;  // r: the RISK-side damping
    };

struct CalibrationReportRequest {
  BundleProblem bundle;
  std::optional<Eigen::VectorXd> x0;  // absent => flat_x0
  RegSpec reg;                        // the RISK-side damping of the operator (never the calibration: that is region data)
};
struct QuoteReport {
  QuoteDiagnostic fit;
  double identifiability = 0.0;
};
struct CalibrationReport {
  CalibrationResult calibration;
  JacobianConditioning conditioning;
  std::vector<QuoteReport> quotes;  // instrument order
};

template <CalibrationSession Session>
CalibrationReport calibration_report(CalibrationReportRequest req) {
  Session sess(std::move(req.bundle));
  const Eigen::VectorXd x0 = seed_or_flat(sess.problem(), req.x0, "calib_report");
  CalibrationReport out;
  out.calibration = sess.calibrate(x0);
  const Eigen::MatrixXd J = sess.jacobian();              // n_quotes x n_knots
  const Eigen::MatrixXd M = sess.risk_operator(req.reg);  // n_knots x n_res
  out.conditioning = jacobian_conditioning(J);
  const std::vector<QuoteDiagnostic> fits = quote_diagnostics(sess.problem(), sess.x());
  const Eigen::VectorXd ident = identifiability(J, M, residual_market_scale(sess.problem().instruments));
  out.quotes.reserve(fits.size());
  for (std::size_t i = 0; i < fits.size(); ++i) out.quotes.push_back({fits[i], ident[static_cast<Eigen::Index>(i)]});
  return out;
}

}  // namespace swaps::calibration
