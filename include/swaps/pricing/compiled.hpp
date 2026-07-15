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

// W(i,:) such that integral(times[i]) = W(i,:)*x. Built via one AAD pass (integral is linear, so the
// gradient is the weight row, independent of x). Setup only, not the hot path.
inline Eigen::MatrixXd integral_weight_matrix(const std::vector<double>& meeting,
                                              const std::vector<double>& back,
                                              const std::vector<double>& times) {
  const int m = static_cast<int>(meeting.size() + back.size());
  auto c = curve::make_calibration_curve<ad::Dual>(meeting, back);
  c.set_forwards(ad::seed(Eigen::VectorXd::Constant(m, 0.03)));
  Eigen::MatrixXd W(static_cast<int>(times.size()), m);
  for (std::size_t i = 0; i < times.size(); ++i) {
    const ad::Dual I = c.integral(times[i]);
    if (I.derivatives().size() == m)
      W.row(static_cast<int>(i)) = I.derivatives().transpose();
    else
      W.row(static_cast<int>(i)).setZero();
  }
  return W;
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
