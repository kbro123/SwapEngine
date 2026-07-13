#pragma once
// BundleProblem -- multiple curves calibrated SIMULTANEOUSLY over one stacked parameter vector
// x = [x_0 ; x_1 ; ...] (CLAUDE.md Stage 3). Each instrument references curves by index for its
// forecast and discount roles; cross-curve instruments (basis swaps) depend on two curves and couple
// their blocks in the joint solve.
//
// The key reuse: it exposes the SAME interface as CalibrationProblem -- residuals<Scalar>(x),
// n_knots(), n_residuals() -- so `calibrate`, `aad_jacobian`, StreamingCalibrator and the risk ladder
// (all templated on the problem type) drive it UNCHANGED. The block-structured Jacobian falls straight
// out of AAD over the stacked residual: a curve-0-only instrument has an identically-zero derivative
// w.r.t. curve-1's knots, so its Jacobian block is zero automatically -- no hand-built sparsity.
//
// Curves are Flat+Hermite (make_calibration_curve) for now; per-curve heterogeneous composition via
// ModularCurve/CurveModule is a later refinement.

#include <Eigen/Core>

#include <vector>

#include "swaps/curve/calibration_curve.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace swaps::calibration {

class BundleProblem {
 public:
  struct CurveSpec {
    std::vector<double> meeting;  // front (flat) knot times
    std::vector<double> back;     // back (Hermite) knot times
    int n_knots() const { return static_cast<int>(meeting.size() + back.size()); }
  };
  // OIS: float forecasts `forecast`, everything discounts `discount` (forecast == discount = single-curve).
  struct Swap {
    int forecast, discount;
    pricing::OisSwap sched;  // QuantLib-extracted accrual/pay schedule
    double market_rate;
  };
  // Basis swap over an arbitrary benchmark: spread leg forecasts `forecast`, benchmark leg forecasts
  // `benchmark`, both legs discount `discount`. (benchmark == discount is the plain FF-SOFR case;
  // benchmark != discount lets a curve be quoted as a spread over ANOTHER forecast curve -- a chain.)
  struct Basis {
    int forecast, benchmark, discount;
    pricing::OisSwap sched;
    double market_rate;  // the par basis spread
  };
  // Futures forecast ONE curve (a rate, no discounting). Front-end instruments (SOFR 1M/3M, FF 1M).
  struct AvgFut {
    int forecast;
    pricing::AveragedFuture sched;
    double convexity, market_rate;  // market_rate = 1 - price/100
  };
  struct CompFut {
    int forecast;
    pricing::CompoundedFuture sched;
    double convexity, market_rate;
  };

  std::vector<CurveSpec> curves;
  std::vector<Swap> swaps;
  std::vector<Basis> bases;
  std::vector<AvgFut> avg_futs;
  std::vector<CompFut> comp_futs;

  int n_curves() const { return static_cast<int>(curves.size()); }
  int n_knots() const {
    int n = 0;
    for (const auto& c : curves) n += c.n_knots();
    return n;
  }
  int n_residuals() const {
    return static_cast<int>(swaps.size() + bases.size() + avg_futs.size() + comp_futs.size());
  }
  // Start index of curve k's block within the stacked x.
  int offset(int k) const {
    int o = 0;
    for (int i = 0; i < k; ++i) o += curves[i].n_knots();
    return o;
  }

  // Stacked residual: build each curve from its x-block, price every instrument off the curves it
  // references, r_i = model_i - market_i. Templated on Scalar so AAD flows through the whole bundle.
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    std::vector<curve::CalibrationCurve<Scalar>> C;
    C.reserve(curves.size());
    int off = 0;
    for (const auto& spec : curves) {
      auto c = curve::make_calibration_curve<Scalar>(spec.meeting, spec.back);
      const int n = spec.n_knots();
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> xi(n);
      for (int i = 0; i < n; ++i) xi[i] = x[off + i];
      c.set_forwards(xi);
      C.push_back(std::move(c));
      off += n;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    int row = 0;
    for (const auto& a : avg_futs)
      r[row++] = pricing::averaged_future_rate<Scalar>(a.sched, C[a.forecast]) + (a.convexity - a.market_rate);
    for (const auto& cf : comp_futs)
      r[row++] = pricing::compounded_future_rate<Scalar>(cf.sched, C[cf.forecast]) + (cf.convexity - cf.market_rate);
    for (const auto& s : swaps)
      r[row++] = pricing::ois_par_rate<Scalar>(s.sched, C[s.forecast], C[s.discount]) - Scalar(s.market_rate);
    for (const auto& b : bases)
      r[row++] =
          pricing::basis_par_spread<Scalar>(b.sched, C[b.forecast], C[b.benchmark], C[b.discount]) -
          Scalar(b.market_rate);
    return r;
  }
};

}  // namespace swaps::calibration
