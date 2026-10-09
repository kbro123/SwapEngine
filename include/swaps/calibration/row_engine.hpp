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
  // The same, and the rows' MODEL VALUES into *model when non-null (the streamer's walk carries model values, stage B).
  virtual void residuals_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::VectorXd& out,
                                 Eigen::VectorXd* model) const = 0;
  virtual void jacobian_into(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const = 0;
  // The streaming Jacobian against a live market q, and (when r is non-null) the residuals_vs values of the
  // engine's rows into *r -- consistent with J by construction (S2).
  virtual void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J,
                                Eigen::VectorXd* r) const = 0;
  // The residual rows' derivative ALONG a state direction, out[row] = (J·dir)[row], without forming J (the book's
  // parallel PV01; 2026-10-06). An engine computes it the cheapest way it has: the compiled one contracts its
  // partials with the DF tangent as it scatters them, the AAD block runs a width-one directional dual.
  virtual void directional_into(const Eigen::VectorXd& x, const Eigen::VectorXd& dir, Eigen::VectorXd& out) const = 0;
  // The FX spot of a row's resetting (MtM) leg -- a market datum like its quote (SC2 FX moves on a book of Npv rows).
  // A row without such a leg ignores it.
  virtual void set_mtm_fx_spot(int global_row, double fx_spot) = 0;
};

}  // namespace swaps::calibration
