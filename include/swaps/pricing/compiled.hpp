#pragma once
// Shared W-cache primitives used by the multi-curve compiled engine (pricing/compiled_book.hpp).
//
// The whole optimization thesis (CLAUDE.md §2): for a FIXED curve structure the log-discount is LINEAR
// in the knot forwards, integral(t) = w(t)·x, so DF(t) = exp(-w(t)·x). integral_weight_matrix builds the
// weight rows w(t) ONCE (one AAD pass -- the integral is linear so its gradient is the weight row,
// independent of x). The per-curve / per-instrument gather-and-reduce machinery lives in compiled_book.hpp;
// this header is just the weight-matrix builder plus a small vector helper both consumers share.

#include <Eigen/Core>

#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/curve/calibration_curve.hpp"

namespace swaps::pricing {

// Which back-end interpolation the W-cache builds. Default Hermite = the shipped curve, unchanged.
enum class BackScheme { Hermite, BSpline };

// W(i,:) such that integral(times[i]) = W(i,:)*x. Built via one AAD pass (integral is linear, so the
// gradient is the weight row, independent of x). Setup only, not the hot path. `scheme` selects the
// back-end interpolation; for BSpline the free values x are B-spline CONTROL POINTS (Part A).
inline Eigen::MatrixXd integral_weight_matrix(const std::vector<double>& meeting,
                                              const std::vector<double>& back,
                                              const std::vector<double>& times,
                                              BackScheme scheme = BackScheme::Hermite) {
  const int m = static_cast<int>(meeting.size() + back.size());
  Eigen::MatrixXd W(static_cast<int>(times.size()), m);
  auto fill = [&](auto& c) {
    c.set_forwards(ad::seed(Eigen::VectorXd::Constant(m, 0.03)));
    for (std::size_t i = 0; i < times.size(); ++i) {
      const ad::Dual I = c.integral(times[i]);
      if (I.derivatives().size() == m)
        W.row(static_cast<int>(i)) = I.derivatives().transpose();
      else
        W.row(static_cast<int>(i)).setZero();
    }
  };
  if (scheme == BackScheme::BSpline) {
    auto c = curve::make_bspline_curve<ad::Dual>(meeting, back);
    fill(c);
  } else {
    auto c = curve::make_calibration_curve<ad::Dual>(meeting, back);
    fill(c);
  }
  return W;
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
  auto c = curve::make_bspline_curve<ad::Dual>(meeting, back);
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
