#pragma once
// Shared W-cache primitives used by the multi-curve compiled engine (pricing/compiled_book.hpp).
//
// The whole optimization thesis (CLAUDE.md §2): for a FIXED curve structure the log-discount is LINEAR
// in the knot forwards, integral(t) = w(t)·x, so DF(t) = exp(-w(t)·x). integral_weight_matrix builds the
// weight rows w(t) ONCE (one AAD pass -- the integral is linear so its gradient is the weight row,
// independent of x). The per-curve / per-instrument gather-and-reduce machinery lives in compiled_book.hpp;
// this header is just the weight-matrix builder plus a small vector helper both consumers share.

#include <Eigen/Core>

#include <type_traits>
#include <limits>
#include <string>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/curve_module.hpp"  // runtime ModularCurve -> generic (any-region) W-cache

namespace swaps::pricing {

// THE W-cache. W(i,:) such that integral(times[i]) = W(i,:)*x, for ANY region layout -- shipped
// (curve::flat_hermite / flat_bspline) or user-composed. Built via ONE AAD pass: a linear-in-values
// curve has integral(t) = w(t)*x, so the gradient IS the weight row, independent of x. Setup only, not
// the hot path.
//
// A flat front was never a requirement of the cache -- only linearity is, and that holds regardless of
// how the regions are composed. Linearity IS required though: ModularCurve reports it at runtime
// (MonotoneCubic's value-dependent Hyman filter is non-linear), so we check here and route a non-linear
// composition to the AAD engine rather than silently caching a wrong W.
//
// For a BSpline region the free values x are CONTROL POINTS (docs/bezier-and-moments.md Part A); use
// bspline_collocation below to present a risk ladder in the forward-at-knot basis.
inline Eigen::MatrixXd integral_weight_matrix(const std::vector<curve::CurveModule>& regions,
                                              const std::vector<double>& times) {
  auto c = curve::make_modular_curve<ad::Dual>(regions);
  // LINEARITY IS LOCAL (2026-09-10). A value-dependent region makes the integral non-linear only from where
  // that region starts; every earlier time still has a constant weight row, because regions are built front
  // to back and never feed backwards. So refuse the TIMES that reach a non-linear region, not the whole
  // curve. The router keeps its rows inside `curve_linear_horizon`; this is the backstop that turns a
  // routing mistake into a loud error rather than a silently wrong W.
  if (!c.is_linear_map()) {
    double horizon = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < regions.size(); ++i) {
      if (curve::scheme_is_linear(regions[i].scheme)) continue;
      if (i > 0 && !regions[i - 1].knots.empty()) horizon = regions[i - 1].knots.back();
      break;
    }
    for (double t : times)
      if (!(t <= horizon))
        throw std::invalid_argument(
            "integral_weight_matrix: time " + std::to_string(t) + " reaches a value-dependent region (the "
            "curve's linear horizon is " + std::to_string(horizon) + "); that row belongs on the AAD engine.");
  }
  const int m = c.n_knots();
  Eigen::MatrixXd W(static_cast<int>(times.size()), m);
  c.set_forwards(ad::seed(Eigen::VectorXd::Constant(m, 0.03)));
  for (std::size_t i = 0; i < times.size(); ++i) {
    const ad::Dual I = c.integral(times[i]);
    if (I.derivatives().size() == m)
      W.row(static_cast<int>(i)) = I.derivatives().transpose();
    else
      W.row(static_cast<int>(i)).setZero();
  }
  return W;
}

// FORWARD weight rows: psi(i,:) such that forward(times[i]) = psi(i,:)·x, for the same linear region layouts as
// integral_weight_matrix (the forward is linear in x whenever the integral is). One Dual pass, setup only. Used by
// the compiled MOMENT path (BundleFloatBatch): ∫f² over a window becomes the quadratic form xᵀ(Σ_k w_k psi_k psi_kᵀ)x
// with the Gauss nodes' rows precomputed at compile.
inline Eigen::MatrixXd forward_weight_matrix(const std::vector<curve::CurveModule>& regions,
                                             const std::vector<double>& times) {
  auto c = curve::make_modular_curve<ad::Dual>(regions);
  if (!c.is_linear_map())
    throw std::invalid_argument("forward_weight_matrix: requires linear interpolation regions (MonotoneCubic is value-dependent)");
  const int m = c.n_knots();
  Eigen::MatrixXd P(static_cast<int>(times.size()), m);
  c.set_forwards(ad::seed(Eigen::VectorXd::Constant(m, 0.03)));
  for (std::size_t i = 0; i < times.size(); ++i) {
    const ad::Dual f = c.forward(times[i]);
    if (f.derivatives().size() == m)
      P.row(static_cast<int>(i)) = f.derivatives().transpose();
    else
      P.row(static_cast<int>(i)).setZero();
  }
  return P;
}

// B-spline RISK TRANSFORM (docs/bezier-and-moments.md Part A). The B-spline free variables are CONTROL
// POINTS, which do not lie on the curve, so a raw risk ladder is control-point sensitivity. B maps the
// free vars x to the forward-at-knot values (front forwards are identity; back = de Boor of the control
// points): forward_at_knot = B x, with B fixed by the knot structure (extracted once via one AAD pass).
// So a control-point gradient g_P and the familiar forward-at-knot gradient g_f relate by g_f = B^{-T} g_P
// (i.e. present the ladder in either basis). Same shape as integral_weight_matrix -- a linear map of the
// curve extracted at setup.
inline Eigen::MatrixXd bspline_collocation(const std::vector<double>& meeting,
                                           const std::vector<double>& back) {
  const int m = static_cast<int>(meeting.size() + back.size());
  std::vector<double> knots = meeting;
  knots.insert(knots.end(), back.begin(), back.end());
  auto c = curve::make_modular_curve<ad::Dual>(curve::flat_bspline(meeting, back));
  c.set_forwards(ad::seed(Eigen::VectorXd::Constant(m, 0.03)));
  Eigen::MatrixXd B(m, m);
  for (int i = 0; i < m; ++i) {
    const ad::Dual f = c.forward(knots[i]);
    if (f.derivatives().size() == m)
      B.row(i) = f.derivatives().transpose();
    else
      B.row(i).setZero();
  }
  return B;
}

namespace detail {
inline Eigen::VectorXi to_vec(const std::vector<int>& v) {
  return Eigen::Map<const Eigen::VectorXi>(v.data(), static_cast<int>(v.size()));
}
inline Eigen::VectorXd to_vec(const std::vector<double>& v) {
  return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<int>(v.size()));
}
}  // namespace detail

}  // namespace swaps::pricing
