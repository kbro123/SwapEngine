#pragma once
// Forward-mode "vector-dual" AAD scalar (CLAUDE.md §1).
//
// A Dual carries a value and a length-M derivative vector. Seeding the M knot forwards with unit
// derivatives and evaluating the residual code ONCE yields every column of the M-wide Jacobian in a
// single differentiated pass — the same templated kernel that prices with `double`.

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

namespace swaps::ad {

using Dual = Eigen::AutoDiffScalar<Eigen::VectorXd>;

// Seed x (length M) as vector-duals: x[i] carries value x[i] and derivative e_i.
inline Eigen::Matrix<Dual, Eigen::Dynamic, 1> seed(const Eigen::VectorXd& x) {
  const int m = static_cast<int>(x.size());
  Eigen::Matrix<Dual, Eigen::Dynamic, 1> xd(m);
  for (int i = 0; i < m; ++i) xd[i] = Dual(x[i], m, i);
  return xd;
}

}  // namespace swaps::ad
