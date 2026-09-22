#pragma once
// A ROW ENGINE (architecture review item 2, 2026-09-22): evaluates a SUBSET of a bundle's residual rows against
// the stacked state x, writing into GLOBAL-row-indexed outputs. HybridBundleResidual holds a LIST of them and
// stitches; the tier of every row is simply which engine claimed it. Two implementations today:
//   * CompiledRows -- the W-cache (CompiledBundleResidual) over the rows the compiled batch can express;
//   * AadBlock     -- width-reduced forward-AAD over the rest (a compounded observation, an incomplete or
//                     seasoned MtM leg; under the horizon partition, rows reading a value-dependent region).
// A third engine registers by implementing this and claiming rows; the hybrid needs no new member for it.
//
// Contract: `rows()` are the global rows this engine owns; every *_into writes exactly those entries of `out`
// (residuals / model rates) or those rows of J (a Jacobian engine may write only its touched columns provided
// it zeroes its rows first), so the hybrid never pre-zeroes on the stitched path. Quote updates arrive by
// global row and are ignored for rows the engine does not own.
#include <Eigen/Core>

#include <vector>

namespace swaps::calibration {

struct RowEngine {
  virtual ~RowEngine() = default;
  virtual const std::vector<int>& rows() const = 0;
  virtual void set_quote(int global_row, double market, double lower, double upper, double decay) = 0;
  virtual void set_market(int global_row, double market) = 0;
  virtual void model_rates_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const = 0;
  virtual void residuals_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const = 0;
  virtual void residuals_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::VectorXd& out) const = 0;
  virtual void jacobian_into(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const = 0;
  // The streaming Jacobian against a live market q, and (when r is non-null) the residuals_vs values of the
  // engine's rows into *r -- consistent with J by construction (S2).
  virtual void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J,
                                Eigen::VectorXd* r) const = 0;
};

}  // namespace swaps::calibration
