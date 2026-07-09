#pragma once
// The calibration problem: free variables x = knot forwards, residual vector r(x) in RATE units.
//
// QuantLib-free and templated on Scalar (CLAUDE.md §1): the schedules are extracted from QuantLib
// once (swaps/ql/extract.hpp) and stored as plain data here; residuals(x) rebuilds the curve and
// reprices every instrument with the templated kernel. With Scalar = double this drives the LM
// solve; with Scalar = AutoDiffScalar (Phase 3) the SAME code yields the analytic Jacobian.
//
// Residual convention (CLAUDE.md §2): everything is in rate units so futures (quoted as prices) do
// not swamp swaps. A futures market quote of price P contributes target rate (1 - P/100); the model
// side is the reference rate plus its convexity adjustment.

#include <Eigen/Core>

#include <vector>

#include "swaps/curve/two_region_forward_curve.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace swaps::calibration {

struct CalibrationProblem {
  // Curve topology (year fractions on the curve day count).
  std::vector<double> meeting_times;
  std::vector<double> back_times;

  struct SwapInst {
    pricing::OisSwap sched;
    double market_rate;  // par quote
  };
  struct CompFutInst {
    pricing::CompoundedFuture sched;
    double convexity;
    double market_rate;  // 1 - price/100
  };
  struct AvgFutInst {
    pricing::AveragedFuture sched;
    double convexity;
    double market_rate;  // 1 - price/100
  };
  std::vector<SwapInst> swaps;
  std::vector<CompFutInst> comp_futs;
  std::vector<AvgFutInst> avg_futs;

  int n_knots() const { return static_cast<int>(meeting_times.size() + back_times.size()); }
  int n_residuals() const {
    return static_cast<int>(swaps.size() + comp_futs.size() + avg_futs.size());
  }

  // r(x) in rate units. Ordering: averaged futures, compounded futures, swaps (stable, so a
  // caller can map residual index back to instrument).
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    curve::TwoRegionForwardCurve<Scalar> c(meeting_times, back_times);
    c.set_forwards(x);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    int i = 0;
    for (const auto& a : avg_futs)
      r[i++] = pricing::averaged_future_rate<Scalar>(a.sched, c) + Scalar(a.convexity) -
               Scalar(a.market_rate);
    for (const auto& cf : comp_futs)
      r[i++] = pricing::compounded_future_rate<Scalar>(cf.sched, c) + Scalar(cf.convexity) -
               Scalar(cf.market_rate);
    for (const auto& s : swaps)
      r[i++] = pricing::ois_par_rate<Scalar>(s.sched, c) - Scalar(s.market_rate);
    return r;
  }
};

}  // namespace swaps::calibration
