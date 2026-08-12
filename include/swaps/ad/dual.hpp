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

// ---- Pooled dual (R11): allocation-free forward AAD for narrow problems -------------------------
// `Dual`'s gradient is an `Eigen::VectorXd`, so EVERY intermediate operation heap-allocates a fresh
// gradient vector — the dominant cost of a width-M AAD Jacobian sweep. `DualPooled<MaxW>` keeps the
// gradient a DYNAMIC-length vector (so its runtime semantics are byte-for-byte `Dual`'s: an empty
// gradient still means "constant", so scalar constants and default-constructed temporaries carry a
// zero-SIZE gradient — never garbage, unlike a fixed-size `Matrix<double,MaxW,1>` derivative would)
// but stores it IN-OBJECT with a fixed maximum capacity MaxW, so no operation touches the heap for a
// problem of width ≤ MaxW. Arithmetic still runs over the ACTUAL runtime size, so results are the
// exact `Dual` doubles in the exact order — the oracle gate holds bit-for-bit. Fall back to `Dual`
// (heap) when the width exceeds MaxW.
template <int MaxW>
using DualPooled = Eigen::AutoDiffScalar<Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxW, 1>>;

// The pooled-dual width used by the AAD Jacobian fast path: covers a single classic curve and the
// typical width-reduced hybrid touch set; wider bundles fall back to the heap `Dual`. Sized to keep
// the in-object gradient buffer modest (MaxW doubles per scalar).
inline constexpr int kPooledMaxW = 48;

template <int MaxW>
inline Eigen::Matrix<DualPooled<MaxW>, Eigen::Dynamic, 1> seed_pooled(const Eigen::VectorXd& x) {
  const int m = static_cast<int>(x.size());
  Eigen::Matrix<DualPooled<MaxW>, Eigen::Dynamic, 1> xd(m);
  for (int i = 0; i < m; ++i) xd[i] = DualPooled<MaxW>(x[i], m, i);  // value x[i], gradient e_i (size m)
  return xd;
}

}  // namespace swaps::ad
