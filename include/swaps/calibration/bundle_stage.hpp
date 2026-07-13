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
  for (const auto& s : p.swaps) add(s.forecast, s.discount);
  for (const auto& b : p.bases) { add(b.forecast, b.benchmark); add(b.forecast, b.discount); }

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
    std::vector<char> ref(p.n_curves(), 0);
    for (int c : block_) ref[c] = 1;
    for (int i : swaps_) { ref[p.swaps[i].forecast] = 1; ref[p.swaps[i].discount] = 1; }
    for (int i : bases_) {
      const auto& b = p.bases[i];
      ref[b.forecast] = ref[b.benchmark] = ref[b.discount] = 1;
    }
    for (int i : avg_futs_) ref[p.avg_futs[i].forecast] = 1;
    for (int i : comp_futs_) ref[p.comp_futs[i].forecast] = 1;
    for (int c = 0; c < p.n_curves(); ++c)
      if (ref[c]) referenced_.push_back(c);
  }

  int n_knots() const { return nk_; }
  int n_residuals() const {
    return static_cast<int>(swaps_.size() + bases_.size() + avg_futs_.size() + comp_futs_.size());
  }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& xb) const {
    // A block-sized ZERO gradient (not an empty one): frozen curves must carry derivatives of the same
    // size as the free curves, or Eigen AutoDiffScalar adds mismatched-size derivative vectors -> NaN.
    // `xb[0] * 0.0` scales a seeded Dual to value 0 with a full-size zero gradient (and is just 0.0 for double).
    const Scalar zero_grad = xb.size() > 0 ? xb[0] * 0.0 : Scalar(0.0);
    std::vector<std::unique_ptr<curve::CalibrationCurve<Scalar>>> C(p_->n_curves());
    for (int c : referenced_) {
      const auto& spec = p_->curves[c];
      const int nk = spec.n_knots();
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> xi(nk);
      if (free_[c]) {
        const int bo = block_offset(c);
        for (int i = 0; i < nk; ++i) xi[i] = xb[bo + i];       // free: from the block parameter vector
      } else {
        const int go = p_->offset(c);
        for (int i = 0; i < nk; ++i) xi[i] = zero_grad + solved_[go + i];  // frozen: constant, sized zero gradient
      }
      auto cc = std::make_unique<curve::CalibrationCurve<Scalar>>(
          curve::make_calibration_curve<Scalar>(spec.meeting, spec.back));
      cc->set_forwards(xi);
      C[c] = std::move(cc);
    }
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
  std::vector<int> block_, referenced_, swaps_, bases_, avg_futs_, comp_futs_;
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
