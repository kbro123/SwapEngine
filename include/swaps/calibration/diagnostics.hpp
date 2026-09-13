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
    d.weight = band_slope(model, ins.band_lower, ins.band_upper, ins.band_decay);
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
      { s.calibrate(x, r) } -> std::convertible_to<const CalibrationResult&>;
      { cs.problem() } -> std::convertible_to<const BundleProblem&>;
      { cs.x() } -> std::convertible_to<const Eigen::VectorXd&>;
      { cs.jacobian(r) } -> std::convertible_to<Eigen::MatrixXd>;
      { cs.risk_operator(r) } -> std::convertible_to<Eigen::MatrixXd>;
    };

struct CalibrationReportRequest {
  BundleProblem bundle;
  std::optional<Eigen::VectorXd> x0;  // absent => flat_x0
  RegSpec reg;                        // enters M through RᵀR, never J
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
  out.calibration = sess.calibrate(x0, req.reg);
  const Eigen::MatrixXd J = sess.jacobian(req.reg);       // n_res x n_knots
  const Eigen::MatrixXd M = sess.risk_operator(req.reg);  // n_knots x n_res
  out.conditioning = jacobian_conditioning(J);
  const std::vector<QuoteDiagnostic> fits = quote_diagnostics(sess.problem(), sess.x());
  const Eigen::VectorXd ident = identifiability(J, M, residual_market_scale(sess.problem().instruments));
  out.quotes.reserve(fits.size());
  for (std::size_t i = 0; i < fits.size(); ++i) out.quotes.push_back({fits[i], ident[static_cast<Eigen::Index>(i)]});
  return out;
}

}  // namespace swaps::calibration
