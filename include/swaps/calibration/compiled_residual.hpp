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

  // Analytic calibration Jacobian J = dr/dx, WITHOUT AAD. It factors through the DFs:
  //   J = dr/dDF * dDF/dx = -(dr/dDF * diag(DF)) * W        (dDF/dx = -diag(DF)*W from DF=exp(-Wx))
  // dr/dDF is the per-instrument pricing sensitivity (analytic, sparse) scattered into G (n_res x
  // n_times); the W matmul then maps DF-space back to knot space. No VectorXd-allocating dual sweep.
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd DF = df_(x);
    const int N = n_residuals(), T = df_.n_times();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(N, T);  // dr/dDF

    for (int k = 0; k < static_cast<int>(avg_.subS.size()); ++k) {
      const int j = avg_.fut[k], sd = avg_.subS[k], se = avg_.subE[k];
      const double ip = avg_.inv_period[j];
      G(j, sd) += ip / DF[se];
      G(j, se) += -ip * DF[sd] / (DF[se] * DF[se]);
    }
    for (int j = 0; j < comp_.size(); ++j) {
      const int row = n_avg_ + j, s = comp_.s[j], e = comp_.e[j];
      const double it = comp_.inv_tau[j];
      G(row, s) += it / DF[e];
      G(row, e) += -it * DF[s] / (DF[e] * DF[e]);
    }
    if (n_swap_) {
      const Eigen::VectorXd fpv = float_.pv(DF), ann = fixed_.annuity(DF);
      Eigen::MatrixXd dfpv = Eigen::MatrixXd::Zero(n_swap_, T), dann = Eigen::MatrixXd::Zero(n_swap_, T);
      for (int i = 0; i < static_cast<int>(float_.pay.size()); ++i) {
        const int sw = float_.swap[i], pay = float_.pay[i], aS = float_.accS[i], aE = float_.accE[i];
        dfpv(sw, pay) += DF[aS] / DF[aE] - 1.0;
        dfpv(sw, aS) += DF[pay] / DF[aE];
        dfpv(sw, aE) += -DF[pay] * DF[aS] / (DF[aE] * DF[aE]);
      }
      for (int i = 0; i < static_cast<int>(fixed_.pay.size()); ++i)
        dann(fixed_.swap[i], fixed_.pay[i]) += fixed_.tau[i];
      for (int j = 0; j < n_swap_; ++j)
        G.row(n_avg_ + n_comp_ + j) =
            dfpv.row(j) / ann[j] - fpv[j] * dann.row(j) / (ann[j] * ann[j]);
    }
    return -((G * DF.asDiagonal()) * df_.W());
  }

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
