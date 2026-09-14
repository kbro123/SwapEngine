#pragma once
// calibration/bundle_state.hpp — reading and forking a calibrated bundle STATE x (E7 stage 5.1): the pieces the
// scenario / scenario_grid / var verbs and BundleSession::sample each wrote by hand, as calibration-layer library
// code over any state vector.
//
//   * CurveSample / sample_bundle_curves — every curve's discount, zero and forward on a time grid at x
//     (BundleSession::sample is this at the session's own x).
//   * shift_interp_forwards — the "fork over the market": a per-curve rate shift added to each curve's
//     interpolation forwards. A turn's jump (the state entries after a curve's interpolation knots) is left alone:
//     a shift moves the level of the forward curve, not a localized turn.
//   * book_value_at — a MultiCurveBook's value at x through the templated curve handles (the kernel
//     BundleSession::price_portfolio uses; CompiledMultiCurveBook is its compiled twin).
//
// The arithmetic is the verbs' own, one addition or evaluation per entry, so moving them here is bitwise
// (tests/scenario_golden_test.cpp).

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace swaps::calibration {

// A curve sampled on a time grid: continuously-compounded zero, instantaneous forward, discount factor.
struct CurveSample {
  int currency = 0;
  std::vector<double> t;
  std::vector<double> discount;
  std::vector<double> zero;     // integral(t)/t (cont-comp), == forward(0) at t==0
  std::vector<double> forward;  // instantaneous forward
};

inline std::vector<CurveSample> sample_bundle_curves(const BundleProblem& p, const Eigen::VectorXd& x,
                                                     const std::vector<double>& times) {
  if (x.size() != p.n_knots())
    throw std::invalid_argument("sample_bundle_curves: x length does not match the bundle's knot count");
  const auto C = build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  std::vector<CurveSample> out(p.n_curves());
  for (int c = 0; c < p.n_curves(); ++c) {
    out[c].currency = p.curves[c].currency;
    out[c].t = times;
    out[c].discount.reserve(times.size());
    out[c].zero.reserve(times.size());
    out[c].forward.reserve(times.size());
    for (double t : times) {
      out[c].discount.push_back(C[c]->discount(t));
      out[c].forward.push_back(C[c]->forward(t));
      out[c].zero.push_back(t > 1e-12 ? C[c]->integral(t) / t : C[c]->forward(0.0));
    }
  }
  return out;
}

// x with curve_delta[c] (rate space) added to every interpolation forward of curve c.
inline Eigen::VectorXd shift_interp_forwards(const BundleProblem& p, Eigen::VectorXd x,
                                             const std::vector<double>& curve_delta) {
  if (x.size() != p.n_knots())
    throw std::invalid_argument("shift_interp_forwards: x length does not match the bundle's knot count");
  if (curve_delta.size() != static_cast<std::size_t>(p.n_curves()))
    throw std::invalid_argument("shift_interp_forwards: need one shift per curve");
  for (int c = 0; c < p.n_curves(); ++c) {
    const double d = curve_delta[static_cast<std::size_t>(c)];
    if (d == 0.0) continue;
    x.segment(p.offset(c), p.curves[c].n_interp_knots()).array() += d;
  }
  return x;
}

inline double book_value_at(const portfolio::MultiCurveBook& book, const BundleProblem& p, const Eigen::VectorXd& x) {
  if (book.positions.empty()) return 0.0;
  if (x.size() != p.n_knots())
    throw std::invalid_argument("book_value_at: x length does not match the bundle's knot count");
  const auto C = build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const CurveHandle<double>& { return *C[i]; };
  return book.value<double>(curve_of);
}

}  // namespace swaps::calibration
