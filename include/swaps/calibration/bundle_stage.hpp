#pragma once
// Staged bundle calibration (CLAUDE.md Stage 3): decompose the curve dependency graph and solve only
// what must be solved jointly. Curve c (an instrument's forecast leg) DEPENDS on the other curves that
// instrument references (benchmark / discount). Condense into strongly-connected components (Tarjan):
//   - a singleton SCC (a curve whose dependencies are already solved) -> a LOCAL LM over just its block,
//     with the earlier curves FROZEN as constants;
//   - a multi-curve SCC (a genuine cycle of mutual dependence)        -> a joint LM over just those.
// SCCs are solved in dependency-first order. On a triangular chain (SOFR->FF->PRIME->PRIME2) every SCC
// is a singleton, so it is four small local solves -- like-for-like with QuantLib's sequential bootstrap
// -- and no 28-knot global solve. A cycle forces exactly (and only) its members into a joint solve.
//
// Freezing a solved curve is free: the multi-curve kernel is templated per curve argument, and a frozen
// curve enters as a constant (Scalar(value), zero gradient), so the block's AAD Jacobian is the small
// per-block one automatically -- no kernel change.

#include <Eigen/Core>

#include <functional>
#include <memory>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/calibration_curve.hpp"

namespace swaps::calibration {

// SCCs of the curve dependency graph, in dependency-first (solve) order.
inline std::vector<std::vector<int>> bundle_dependency_order(const BundleProblem& p) {
  const int N = p.n_curves();
  std::vector<std::vector<int>> adj(N);
  auto add = [&](int c, int d) { if (d != c) adj[c].push_back(d); };  // c depends on d
  // A spread curve depends STRUCTURALLY on its base (base must be built first); if the base is also a
  // free, mutually-referenced curve, an instrument edge closes the cycle -> they land in one SCC.
  for (int c = 0; c < N; ++c)
    if (p.curves[c].base >= 0) add(c, p.curves[c].base);
  for (const auto& s : p.swaps) add(s.forecast, s.discount);
  for (const auto& b : p.bases) { add(b.forecast, b.benchmark); add(b.forecast, b.discount); }
  // A generic instrument pins its quoted leg's forecast curve, which therefore depends on every OTHER
  // curve its legs reference. A `Rate` instrument references only its own forecast curve -> no edge.
  for (const auto& ins : p.instruments) {
    if (ins.quote == QuoteKind::Rate) continue;
    const int c = ins.primary_curve();
    add(c, ins.fwd.discount);
    add(c, ins.fixed.discount);
    if (ins.quote == QuoteKind::ParSpread) { add(c, ins.bench.forecast); add(c, ins.bench.discount); }
  }

  std::vector<int> idx(N, -1), low(N, 0), stk;
  std::vector<char> onstk(N, 0);
  std::vector<std::vector<int>> sccs;  // Tarjan emits in reverse-topo = dependency-first order
  int counter = 0;
  std::function<void(int)> dfs = [&](int v) {
    idx[v] = low[v] = counter++;
    stk.push_back(v);
    onstk[v] = 1;
    for (int w : adj[v]) {
      if (idx[w] == -1) { dfs(w); low[v] = std::min(low[v], low[w]); }
      else if (onstk[w]) low[v] = std::min(low[v], idx[w]);
    }
    if (low[v] == idx[v]) {
      std::vector<int> comp;
      for (;;) {
        int w = stk.back();
        stk.pop_back();
        onstk[w] = 0;
        comp.push_back(w);
        if (w == v) break;
      }
      sccs.push_back(std::move(comp));
    }
  };
  for (int v = 0; v < N; ++v)
    if (idx[v] == -1) dfs(v);
  return sccs;
}

// A view of the bundle restricted to one SCC block: the block's curves are FREE (the parameter vector),
// every curve the block's instruments also reference is FROZEN at `solved`. Exposes the standard problem
// interface so `calibrate` / `aad_jacobian` drive it unchanged.
class BundleBlockProblem {
 public:
  BundleBlockProblem(const BundleProblem& p, const std::vector<int>& block, const Eigen::VectorXd& solved)
      : p_(&p), block_(block), solved_(solved), free_(p.n_curves(), 0) {
    for (int c : block_) { free_[c] = 1; nk_ += p.curves[c].n_knots(); }
    for (int i = 0; i < static_cast<int>(p.swaps.size()); ++i)
      if (free_[p.swaps[i].forecast]) swaps_.push_back(i);
    for (int i = 0; i < static_cast<int>(p.bases.size()); ++i)
      if (free_[p.bases[i].forecast]) bases_.push_back(i);
    for (int i = 0; i < static_cast<int>(p.avg_futs.size()); ++i)
      if (free_[p.avg_futs[i].forecast]) avg_futs_.push_back(i);
    for (int i = 0; i < static_cast<int>(p.comp_futs.size()); ++i)
      if (free_[p.comp_futs[i].forecast]) comp_futs_.push_back(i);
    for (int i = 0; i < static_cast<int>(p.instruments.size()); ++i)
      if (free_[p.instruments[i].primary_curve()]) gen_.push_back(i);
  }

  int n_knots() const { return nk_; }
  int n_residuals() const {
    return static_cast<int>(swaps_.size() + bases_.size() + avg_futs_.size() + comp_futs_.size() +
                            gen_.size());
  }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& xb) const {
    // A block-sized ZERO gradient (not an empty one): frozen curves must carry derivatives of the same
    // size as the free curves, or Eigen AutoDiffScalar adds mismatched-size derivative vectors -> NaN.
    // `xb[0] * 0.0` scales a seeded Dual to value 0 with a full-size zero gradient (and is just 0.0 for double).
    const Scalar zero_grad = xb.size() > 0 ? xb[0] * 0.0 : Scalar(0.0);
    // Build EVERY curve (build_bundle_curves needs each spread's base present): free curves from the
    // block vector, frozen curves as constants with a block-sized zero gradient.
    const auto C = build_bundle_curves<Scalar>(p_->curves, [&](int c, int i) -> Scalar {
      if (free_[c]) return xb[block_offset(c) + i];
      return zero_grad + solved_[p_->offset(c) + i];
    });
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    int row = 0;
    for (int i : avg_futs_) {
      const auto& a = p_->avg_futs[i];
      r[row++] = pricing::averaged_future_rate<Scalar>(a.sched, *C[a.forecast]) + (a.convexity - a.market_rate);
    }
    for (int i : comp_futs_) {
      const auto& cf = p_->comp_futs[i];
      r[row++] = pricing::compounded_future_rate<Scalar>(cf.sched, *C[cf.forecast]) + (cf.convexity - cf.market_rate);
    }
    for (int i : swaps_) {
      const auto& s = p_->swaps[i];
      r[row++] = pricing::ois_par_rate<Scalar>(s.sched, *C[s.forecast], *C[s.discount]) - Scalar(s.market_rate);
    }
    for (int i : bases_) {
      const auto& b = p_->bases[i];
      r[row++] = pricing::basis_par_spread<Scalar>(b.sched, *C[b.forecast], *C[b.benchmark], *C[b.discount]) -
                 Scalar(b.market_rate);
    }
    const auto curve_of = [&C](int i) -> const CurveHandle<Scalar>& { return *C[i]; };
    for (int i : gen_) r[row++] = instrument_residual<Scalar>(p_->instruments[i], curve_of);
    return r;
  }

  // Offset of curve c's block within the block parameter vector (block order).
  int block_offset(int c) const {
    int o = 0;
    for (int b : block_) {
      if (b == c) return o;
      o += p_->curves[b].n_knots();
    }
    return -1;
  }

 private:
  const BundleProblem* p_;
  std::vector<int> block_, swaps_, bases_, avg_futs_, comp_futs_, gen_;
  Eigen::VectorXd solved_;
  std::vector<char> free_;
  int nk_ = 0;
};

// Staged solve: dependency-decompose, then LM each SCC block in order with earlier blocks frozen.
inline CalibrationResult calibrate_staged(const BundleProblem& p, const Eigen::VectorXd& x0,
                                          bool use_aad = true) {
  const auto blocks = bundle_dependency_order(p);
  Eigen::VectorXd x = x0;
  int iters = 0;
  for (const auto& block : blocks) {
    BundleBlockProblem bp(p, block, x);
    Eigen::VectorXd xb(bp.n_knots());
    int o = 0;
    for (int c : block) {
      const int go = p.offset(c), nk = p.curves[c].n_knots();
      xb.segment(o, nk) = x.segment(go, nk);
      o += nk;
    }
    const CalibrationResult res = calibrate(bp, xb, use_aad);
    iters += res.iterations;
    o = 0;
    for (int c : block) {
      const int go = p.offset(c), nk = p.curves[c].n_knots();
      x.segment(go, nk) = res.x.segment(o, nk);
      o += nk;
    }
  }
  CalibrationResult out;
  out.x = x;
  out.iterations = iters;
  out.rms_residual = p.residuals<double>(x).norm() / std::sqrt(static_cast<double>(p.n_residuals()));
  out.stationarity = -1.0;  // not computed in staged mode (would need the full joint Jacobian)
  return out;
}

}  // namespace swaps::calibration
