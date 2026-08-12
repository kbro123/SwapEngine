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

#include <algorithm>
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
      float_.add(cs_, 0, 0, pos.float_coupons);  // forecast = discount = curve 0
      fixed_.add(cs_, 0, pos.fixed_coupons);
    }
    cs_.finalize();
    float_.finalize();
    fixed_.finalize();
    nf_ = (notional_.array() * fixed_rate_.array()).matrix();  // notional⊙fixed_rate (annuity row-scale)
  }

  int n_swaps() const { return static_cast<int>(notional_.size()); }
  int n_times() const { return cs_.n_times(); }

  // Per-swap NPV for knot forwards x. Fully vectorized AND allocation-free per call (writes into
  // reusable scratch, returns a const ref) -- so a real-time book reprice never touches the allocator.
  const Eigen::VectorXd& npv(const Eigen::VectorXd& x) const {
    cs_.df_into(x, df_);  // DF into scratch; float_/fixed_ pv/annuity return refs into their own scratch
    const Eigen::VectorXd& pv = float_.pv(df_);
    const Eigen::VectorXd& ann = fixed_.annuity(df_);
    npv_ = (notional_.array() * (pv.array() - fixed_rate_.array() * ann.array())).matrix();
    return npv_;
  }

  double total_npv(const Eigen::VectorXd& x) const { return npv(x).sum(); }

  // BATCHED reprice over a grid of curve-states X (n_knots x n_states) -> per-swap NPVs (n_swaps x n_states)
  // — the MC-exposure kernel (hot-path design R3+R12). One W·X GEMM discounts every state at once; the cheap
  // sparse coupon/annuity reduce runs per state into the reused single-state scratch. Column j is bit-
  // identical to npv(X.col(j)) — the SAME kernel, so an arbitrary SIMULATED x just works (no calibration).
  const Eigen::MatrixXd& npv_grid(const Eigen::MatrixXd& X) const {
    if (float_.std_shape()) {
      // BATCHED path: the coupon gather INDICES are state-invariant, so we gather each coupon's rows ONCE
      // and let Eigen fuse the divide/subtract/multiply across states, accumulating each coupon straight
      // into its instrument's PV/annuity row (a fused segment-sum — no materialized coupon grid, no SpMM).
      // TILED over states (block width kStateTile) so the row-major DF tile and the n_inst × tile PV/annuity
      // grids stay cache-resident — a full-width intermediate would be hundreds of MB and turn the kernel
      // memory-bound. Per tile:  W·X GEMM → row-major DF tile → pv_grid/annuity_grid fused reduce →
      // NPV = notional ⊙ pv_grid − (notional⊙fixed_rate) ⊙ annuity_grid (two lazy DiagonalMatrix products).
      const Eigen::Index ns = X.cols();
      npvg_.resize(n_swaps(), ns);
      for (Eigen::Index j0 = 0; j0 < ns; j0 += kStateTile) {
        const Eigen::Index bw = std::min<Eigen::Index>(kStateTile, ns - j0);
        cs_.df_into(X.middleCols(j0, bw), dfg_row_);  // row-major DF tile (n_times × bw)
        const pricing::RowMatrixXd& pvg = float_.pv_grid(dfg_row_);
        const pricing::RowMatrixXd& anng = fixed_.annuity_grid(dfg_row_);
        npvg_.middleCols(j0, bw).noalias() =
            notional_.asDiagonal() * pvg - nf_.asDiagonal() * anng;
      }
      return npvg_;
    }
    // NON-standard shape (spread / weighted sub-periods / tau_pay≠tau_index): the batched reduce doesn't
    // apply, so fall back to the per-column single-state kernel over the same shared W·X GEMM.
    cs_.df_into(X, dfg_);  // one GEMM: DF grid = exp(-W·X)
    npvg_.resize(n_swaps(), X.cols());
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      df_ = dfg_.col(j);  // reuse the single-state DF scratch (contiguous for the gather loops)
      npvg_.col(j) = (notional_.array() *
                      (float_.pv(df_).array() - fixed_rate_.array() * fixed_.annuity(df_).array()))
                         .matrix();
    }
    return npvg_;
  }

 private:
  // State-tile width for the batched npv_grid: process this many curve-states at a time so the row-major
  // DF tile and the n_inst × tile PV/annuity grids stay cache-resident (a full-width DF/NPV grid would be
  // hundreds of MB and make the kernel memory-bound). 512 keeps a ~1000-coupon book's tiles in L2/L3.
  static constexpr Eigen::Index kStateTile = 512;

  pricing::CompiledCurveSet cs_;
  pricing::BundleFloatBatch float_;
  pricing::BundleFixedLegs fixed_;
  Eigen::VectorXd fixed_rate_, notional_, nf_;  // nf_ = notional⊙fixed_rate
  mutable Eigen::VectorXd df_, npv_;      // reusable per-reprice scratch
  mutable Eigen::MatrixXd dfg_, npvg_;    // reusable col-major DF (fallback) / NPV grids for npv_grid
  mutable pricing::RowMatrixXd dfg_row_;  // reusable row-major DF grid for the batched path
};

}  // namespace swaps::portfolio
