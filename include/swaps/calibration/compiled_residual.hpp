#pragma once
// Vectorized calibration residual (Stage 2) -- the calibration analogue of CompiledPortfolio, on the
// SAME shared DF engine (swaps/pricing/compiled.hpp). Instead of the scalar per-cashflow
// curve.discount loop (~12us), one reprice is DF = exp(-Wx) then gathered/reduced DF-combinations
// then the cheap per-type transform:
//   1M/3M futures -> DF-ratio rates (+ convexity)
//   swaps         -> par rate = floatleg / annuity
// residual = model rates - market, in the residual order the LM/Jacobian code expects
// (averaged futures, compounded futures, swaps).

#include <Eigen/Core>

#include <vector>

#include "swaps/calibration/problem.hpp"
#include "swaps/pricing/compiled.hpp"

namespace swaps::calibration {

class CompiledResidual {
 public:
  explicit CompiledResidual(const CalibrationProblem& p)
      : n_avg_(static_cast<int>(p.avg_futs.size())),
        n_comp_(static_cast<int>(p.comp_futs.size())),
        n_swap_(static_cast<int>(p.swaps.size())) {
    pricing::TimeIndex ti;

    for (const auto& a : p.avg_futs) avg_.add(a.sched, a.convexity, ti);
    avg_.finalize();
    for (const auto& c : p.comp_futs) comp_.add(c.sched, c.convexity, ti);
    comp_.finalize();

    std::vector<const pricing::OisSwap*> swaps;
    for (const auto& s : p.swaps) swaps.push_back(&s.sched);
    float_.build(swaps, ti);
    fixed_.build(swaps, ti);

    df_ = pricing::CompiledDiscounts(p.meeting_times, p.back_times, ti.times());

    market_.resize(n_avg_ + n_comp_ + n_swap_);
    int i = 0;
    for (const auto& a : p.avg_futs) market_[i++] = a.market_rate;
    for (const auto& c : p.comp_futs) market_[i++] = c.market_rate;
    for (const auto& s : p.swaps) market_[i++] = s.market_rate;
  }

  int n_residuals() const { return n_avg_ + n_comp_ + n_swap_; }
  int n_times() const { return df_.n_times(); }

  // Model rates in residual order (averaged futures, compounded futures, swap par rates).
  Eigen::VectorXd model_rates(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd DF = df_(x);
    Eigen::VectorXd out(n_residuals());
    if (n_avg_) out.head(n_avg_) = avg_.rate(DF);
    if (n_comp_) out.segment(n_avg_, n_comp_) = comp_.rate(DF);
    if (n_swap_) {
      const Eigen::VectorXd fpv = float_.pv(DF);
      const Eigen::VectorXd ann = fixed_.annuity(DF);
      out.tail(n_swap_) = fpv.array() / ann.array();
    }
    return out;
  }

  Eigen::VectorXd residuals(const Eigen::VectorXd& x) const { return model_rates(x) - market_; }

 private:
  int n_avg_, n_comp_, n_swap_;
  pricing::CompiledDiscounts df_;
  pricing::CompiledAveragedFutures avg_;
  pricing::CompiledCompoundedFutures comp_;
  pricing::CompiledFloatLegs float_;
  pricing::CompiledFixedLegs fixed_;
  Eigen::VectorXd market_;
};

}  // namespace swaps::calibration
