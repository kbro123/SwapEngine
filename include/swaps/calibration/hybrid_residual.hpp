#pragma once
// HybridBundleResidual -- the residual engine for a BundleProblem that MIXES W-cacheable instruments
// with a few non-cacheable ones (FX/MtM, or a Portfolio containing them). Instead of dropping the whole
// bundle to AAD because one instrument isn't W-cacheable, it splits the work:
//   * cacheable rows  -> a CompiledBundleResidual over that subset (the W-cache: DF = exp(-Wx), analytic
//     Jacobian) -- the microsecond majority;
//   * non-cacheable rows -> an AadBlock (width-reduced forward-AAD over just those instruments).
// then stitches both into the bundle's residual vector and block Jacobian in insertion order.
//
// When NOTHING is non-cacheable (the overwhelmingly common case) it is a transparent, zero-overhead
// delegate to a single CompiledBundleResidual -- the fast path is byte-for-byte unchanged. It only pays
// the split/stitch cost when an FX/MtM (or a portfolio holding one) is actually present.

#include <Eigen/Core>

#include <vector>

#include "swaps/calibration/aad_block.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/problem.hpp"

namespace swaps::calibration {

// A W-cache-incompatible LEAF anywhere in an instrument (FX/MtM, incl. nested inside a Portfolio).
inline bool instrument_is_noncacheable(const Instrument& ins) {
  if (ins.quote == QuoteKind::FxForward || ins.quote == QuoteKind::XccyMtmBasis) return true;
  if (ins.quote == QuoteKind::Portfolio)
    for (const auto& c : ins.combination)
      if (instrument_is_noncacheable(c.instrument)) return true;
  return false;
}

class HybridBundleResidual {
 public:
  explicit HybridBundleResidual(const BundleProblem& p)
      : n_res_(static_cast<int>(p.instruments.size())),
        cacheable_(build_cacheable(p)) {
    // Partition already happened in build_cacheable (it filled cache_rows_); collect the rest for AAD.
    std::vector<Instrument> nc;
    std::vector<int> nc_rows;
    for (int r = 0; r < n_res_; ++r)
      if (instrument_is_noncacheable(p.instruments[r])) { nc.push_back(p.instruments[r]); nc_rows.push_back(r); }
    nc_.init(p.curves, std::move(nc), std::move(nc_rows), p.n_knots());
  }

  int n_residuals() const { return n_res_; }
  int n_times() const { return cacheable_.n_times(); }

  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const {
    if (nc_.empty()) return cacheable_.model_rates(x);
    scatter(cacheable_.model_rates(x), out_);
    nc_.model_rates_into(x, out_);
    return out_;
  }

  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const {
    if (nc_.empty()) return cacheable_.residuals(x);
    scatter(cacheable_.residuals(x), res_);   // cacheable rows (incl. their bands)
    nc_.residuals_into(x, res_);               // non-cacheable rows: the true residual (e.g. FX log-basis)
    return res_;
  }

  // Streaming residual/Jacobian against a live market q. Only reached for a bundle WITHOUT a non-cacheable
  // block (those route to recalibrate), so it is a straight delegate.
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    return cacheable_.residuals_vs(x, q);
  }
  Eigen::MatrixXd jacobian_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    return cacheable_.jacobian_vs(x, q);
  }

  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {
    if (nc_.empty()) return cacheable_.jacobian(x);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n_res_, nknots(x));
    const Eigen::MatrixXd Jc = cacheable_.jacobian(x);   // n_cacheable x n_knots
    for (std::size_t j = 0; j < cache_rows_.size(); ++j) J.row(cache_rows_[j]) = Jc.row(static_cast<int>(j));
    nc_.jacobian_into(x, J);                             // scatters the non-cacheable rows (touched cols)
    return J;
  }

 private:
  static int nknots(const Eigen::VectorXd& x) { return static_cast<int>(x.size()); }

  // Build the CompiledBundleResidual over only the cacheable instruments (same curves), recording each
  // cacheable sub-row's GLOBAL row for the stitch. When nothing is non-cacheable, this is the whole
  // bundle and cache_rows_ is the identity, so every delegate above returns the compiled buffers as-is.
  CompiledBundleResidual build_cacheable(const BundleProblem& p) {
    BundleProblem c;
    c.curves = p.curves;
    for (int r = 0; r < static_cast<int>(p.instruments.size()); ++r)
      if (!instrument_is_noncacheable(p.instruments[r])) {
        c.instruments.push_back(p.instruments[r]);
        cache_rows_.push_back(r);
      }
    return CompiledBundleResidual(c);
  }

  void scatter(const Eigen::VectorXd& sub, Eigen::VectorXd& full) const {
    full.setZero(n_res_);
    for (std::size_t j = 0; j < cache_rows_.size(); ++j) full[cache_rows_[j]] = sub[static_cast<int>(j)];
  }

  int n_res_;
  std::vector<int> cache_rows_;  // cacheable sub-row -> global residual row (identity when nc empty)
  CompiledBundleResidual cacheable_;
  AadBlock nc_;
  mutable Eigen::VectorXd out_, res_;
};

}  // namespace swaps::calibration
