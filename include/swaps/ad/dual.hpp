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
// 64 since 2026-09-12, measured not guessed. A block WIDER than this falls back to the heap dual, where
// every arithmetic operation allocates its gradient vector -- a cliff, not a gradient: the desk_mixed ladder
// rung touches 55 knots and at 48 it cost 19,182 allocations and 4.49 ms per tick, while mixed_scheme at
// width 29 cost 26 allocations for the same kind of work. Raising this to 64 took desk_mixed to 180
// allocations and 2.16 ms -- 106x fewer, 2.1x faster -- and moved no other rung.
//
// THIS MOVES THE CLIFF, IT DOES NOT REMOVE IT. A bundle wider than 64 touched knots falls off again just as
// sharply. Removing it properly means dispatching on width to one of several MaxW instantiations, or a
// small-buffer-optimised gradient; until then ShapeLadder.StreamingTickIsAllocationFreeOnEveryCompiledShape
// asserts every rung stays pooled, so falling off is a test failure and not a silent 100x.
inline constexpr int kPooledMaxW = 64;

// DIRECTIONAL seed (E3-A4/D7, 2026-09-10): every knot carries the SAME single derivative slot with value 1, so
// one heap-free pass of a Scalar = DualPooled<1> function f yields f.derivatives()[0] = Σⱼ ∂f/∂xⱼ -- the
// all-ones directional derivative (a parallel-shift PV01) -- at the cost of one double pass, instead of a
// full-width gradient (208 heap-vector duals on the chain fixture: 71k allocations to compute ONE sum).
using DualDir = DualPooled<1>;
inline Eigen::Matrix<DualDir, Eigen::Dynamic, 1> seed_directional(const Eigen::VectorXd& x) {
  const int m = static_cast<int>(x.size());
  Eigen::Matrix<DualDir, Eigen::Dynamic, 1> xd(m);
  for (int i = 0; i < m; ++i) xd[i] = DualDir(x[i], 1, 0);  // value x[i], derivative [1]
  return xd;
}

template <int MaxW>
inline Eigen::Matrix<DualPooled<MaxW>, Eigen::Dynamic, 1> seed_pooled(const Eigen::VectorXd& x) {
  const int m = static_cast<int>(x.size());
  Eigen::Matrix<DualPooled<MaxW>, Eigen::Dynamic, 1> xd(m);
  for (int i = 0; i < m; ++i) xd[i] = DualPooled<MaxW>(x[i], m, i);  // value x[i], gradient e_i (size m)
  return xd;
}

}  // namespace swaps::ad
