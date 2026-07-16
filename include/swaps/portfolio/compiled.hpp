#pragma once
// Vectorized ("compiled") portfolio repricing — the real-time / many-curves path (CLAUDE.md §2, §5).
//
// Now built on the SAME multi-curve engine as the calibration residual (pricing/compiled_book.hpp): a
// book is just instruments valued off DF = exp(-Wx). CompiledPortfolio registers each position's float
// and fixed legs on one self-discounting curve, then a reprice is:
//     DF = exp(-W x)                                             (one matvec + vectorized exp)
//     float_pv_p = R_float · (DF[pay]·(DF[accS]/DF[accE]-1))     (gathered coupons, sparse reduction)
//     annuity_p  = R_fixed · (tau·DF[pay])
//     NPV_p      = notional_p·(float_pv_p - fixed_rate_p·annuity_p)
// No per-swap / per-coupon scalar loop and no QuantLib object dispatch in the hot path. Sharing the
// BundleFloatBatch / BundleFixedLegs primitives means one compiled kernel for calibration AND analytics.

#include <Eigen/Core>

#include <vector>

#include "swaps/pricing/compiled_book.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace swaps::portfolio {

class CompiledPortfolio {
 public:
  CompiledPortfolio(const std::vector<double>& meeting_times, const std::vector<double>& back_times,
                    const Portfolio& pf) {
    cs_.init({pricing::CurveStructure{meeting_times, back_times, -1}});  // one self-discounting curve
    const int P = static_cast<int>(pf.positions.size());
    fixed_rate_.resize(P);
    notional_.resize(P);
    for (int p = 0; p < P; ++p) {
      const auto& pos = pf.positions[p];
      fixed_rate_[p] = pos.fixed_rate;
      notional_[p] = pos.notional;
      float_.add(cs_, 0, 0, pos.sched);  // forecast = discount = curve 0
      fixed_.add(cs_, 0, pos.sched);
    }
    cs_.finalize();
    float_.finalize();
    fixed_.finalize();
  }

  int n_swaps() const { return static_cast<int>(notional_.size()); }
  int n_times() const { return cs_.n_times(); }

  // Per-swap NPV for knot forwards x. Fully vectorized.
  Eigen::VectorXd npv(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd DF = cs_.df(x);
    return (notional_.array() *
            (float_.pv(DF).array() - fixed_rate_.array() * fixed_.annuity(DF).array()))
        .matrix();
  }

  double total_npv(const Eigen::VectorXd& x) const { return npv(x).sum(); }

 private:
  pricing::CompiledCurveSet cs_;
  pricing::BundleFloatBatch float_;
  pricing::BundleFixedLegs fixed_;
  Eigen::VectorXd fixed_rate_, notional_;
};

}  // namespace swaps::portfolio
