#pragma once
// Multi-curve vectorized discount engine (Stage 3 W-cache). Generalizes pricing/compiled.hpp from ONE
// curve to a BUNDLE of curves calibrated over a stacked x = [x_0; x_1; ...], WITHOUT leaving the linear
// W-cache fast path.
//
// The key structural fact (CLAUDE.md §2): every curve's log-discount is LINEAR in the stacked x, even a
// spread curve (integral_spread just ADDS to integral_base). So if we CONCATENATE every curve's DF
// vector into one global DF and give each cashflow a global index, then
//     DF_all = exp(-W_all x)
// with a block-structured W_all whose row for (curve c, time t) is the log-DF weight of c at t (c's own
// knot weights, PLUS the base's weights if c is a spread curve). The single-curve gather/reduce
// primitives and the analytic Jacobian then carry over UNCHANGED -- the multi-curve-ness lives entirely
// in the global indices and W_all's block structure. One curve is the trivial special case.
//
// This header holds the curve-agnostic pricing primitives: the multi-curve DF engine (CompiledCurveSet)
// and role-aware leg / futures batches that gather from the global DF. Consumers (calibration residual,
// portfolio NPV) are thin final transforms on float_pv / annuity / futures-rate -- see
// calibration/compiled_bundle.hpp.

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <map>
#include <utility>
#include <vector>

#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/compiled.hpp"  // integral_weight_matrix, detail::to_vec

namespace swaps::pricing {

// A curve's knot structure + parameterization, mirroring BundleProblem::CurveSpec but with no
// calibration dependency (so this header stays in the pricing layer).
struct CurveStructure {
  std::vector<double> meeting, back;
  int base = -1;  // -1 = outright; else this curve = curves[base] + spread (spread knots)
  int n_knots() const { return static_cast<int>(meeting.size() + back.size()); }
};

// DF_all = exp(-W_all x) over every (curve, time) registered, concatenated into one global vector.
// reg(curve, t) returns a stable GLOBAL index in registration order; finalize() builds W_all by
// scattering each curve's log-DF weight rows (own knots + spread-base ancestry) into their global rows.
class CompiledCurveSet {
 public:
  void init(const std::vector<CurveStructure>& specs) {
    specs_ = specs;
    knot_offset_.assign(specs.size(), 0);
    int o = 0;
    for (std::size_t c = 0; c < specs.size(); ++c) {
      knot_offset_[c] = o;
      o += specs[c].n_knots();
    }
    n_knots_ = o;
  }

  int reg(int curve, double t) {
    const auto key = std::make_pair(curve, t);
    auto it = idx_.find(key);
    if (it != idx_.end()) return it->second;
    const int g = static_cast<int>(pts_.size());
    idx_.emplace(key, g);
    pts_.push_back(key);
    return g;
  }

  void finalize() {
    W_ = Eigen::MatrixXd::Zero(static_cast<int>(pts_.size()), n_knots_);
    // Group registered times by curve (one integral_weight_matrix call per curve), then scatter each
    // computed weight row back to its GLOBAL row index.
    std::vector<std::vector<int>> rows(specs_.size());
    std::vector<std::vector<double>> tms(specs_.size());
    for (int g = 0; g < static_cast<int>(pts_.size()); ++g) {
      rows[pts_[g].first].push_back(g);
      tms[pts_[g].first].push_back(pts_[g].second);
    }
    for (int c = 0; c < static_cast<int>(specs_.size()); ++c) {
      if (tms[c].empty()) continue;
      const Eigen::MatrixXd Wc = logdf_weight(c, tms[c]);
      for (int k = 0; k < static_cast<int>(rows[c].size()); ++k) W_.row(rows[c][k]) = Wc.row(k);
    }
  }

  Eigen::VectorXd df(const Eigen::VectorXd& x) const { return (-(W_ * x).array()).exp(); }
  const Eigen::MatrixXd& W() const { return W_; }
  int n_times() const { return static_cast<int>(pts_.size()); }
  int n_knots() const { return n_knots_; }

 private:
  // Log-DF weight rows of curve c at `times`: own knot weights in c's block, PLUS the base's log-DF
  // weights (recursively) if c is a spread curve. integral_spread(t) adds to integral_base(t), so the
  // total stays linear in the stacked x -- the whole reason a spread curve keeps the W-cache.
  Eigen::MatrixXd logdf_weight(int c, const std::vector<double>& times) const {
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(static_cast<int>(times.size()), n_knots_);
    W.middleCols(knot_offset_[c], specs_[c].n_knots()) =
        integral_weight_matrix(specs_[c].meeting, specs_[c].back, times);
    if (specs_[c].base >= 0) W += logdf_weight(specs_[c].base, times);
    return W;
  }

  std::vector<CurveStructure> specs_;
  std::vector<int> knot_offset_;
  int n_knots_ = 0;
  std::map<std::pair<int, double>, int> idx_;
  std::vector<std::pair<int, double>> pts_;  // global index -> (curve, time)
  Eigen::MatrixXd W_;
};

// Float legs of N instruments, each forecasting curve `fc` (accrual DF ratio) and discounting curve
// `dc` (pay DF). Coupon k: DF[pay]*(DF[accS]/DF[accE]-1). Indices are GLOBAL into the CompiledCurveSet.
struct BundleFloatLegs {
  Eigen::VectorXi pay, accS, accE, inst;  // per coupon (all instruments flattened); inst = owner
  Eigen::SparseMatrix<double> R;          // n_inst x n_coupons, 0/1 reduction to per-instrument
  int n_inst = 0;

  void add(CompiledCurveSet& cs, int fc, int dc, const OisSwap& s) {
    for (std::size_t i = 0; i < s.float_pay.size(); ++i) {
      p_.push_back(cs.reg(dc, s.float_pay[i]));
      s_.push_back(cs.reg(fc, s.float_acc_start[i]));
      e_.push_back(cs.reg(fc, s.float_acc_end[i]));
      row_.push_back(n_inst);
    }
    ++n_inst;
  }
  void finalize() {
    pay = detail::to_vec(p_);
    accS = detail::to_vec(s_);
    accE = detail::to_vec(e_);
    inst = detail::to_vec(row_);
    std::vector<Eigen::Triplet<double>> trip;
    for (int k = 0; k < static_cast<int>(row_.size()); ++k) trip.emplace_back(row_[k], k, 1.0);
    R.resize(n_inst, static_cast<int>(row_.size()));
    R.setFromTriplets(trip.begin(), trip.end());
  }
  Eigen::VectorXd pv(const Eigen::VectorXd& DF) const {
    // Materialize the per-coupon vector BEFORE the sparse reduction: Eigen's sparse*dense accesses its
    // dense operand repeatedly, so handing it an unevaluated gather+divide expression re-does that work
    // -- ~1.28x slower than reducing a contiguous VectorXd (measured).
    const Eigen::VectorXd coupon = DF(pay).array() * (DF(accS).array() / DF(accE).array() - 1.0);
    return R * coupon;
  }

 private:
  std::vector<int> p_, s_, e_, row_;
};

// Fixed-leg annuities of N instruments discounting curve `dc`: ann = sum(tau*DF[pay]).
struct BundleFixedLegs {
  Eigen::VectorXi pay, inst;
  Eigen::VectorXd tau;
  Eigen::SparseMatrix<double> R;
  int n_inst = 0;

  void add(CompiledCurveSet& cs, int dc, const OisSwap& s) {
    for (std::size_t i = 0; i < s.fixed_pay.size(); ++i) {
      p_.push_back(cs.reg(dc, s.fixed_pay[i]));
      t_.push_back(s.fixed_accrual[i]);
      row_.push_back(n_inst);
    }
    ++n_inst;
  }
  void finalize() {
    pay = detail::to_vec(p_);
    tau = detail::to_vec(t_);
    inst = detail::to_vec(row_);
    std::vector<Eigen::Triplet<double>> trip;
    for (int k = 0; k < static_cast<int>(row_.size()); ++k) trip.emplace_back(row_[k], k, 1.0);
    R.resize(n_inst, static_cast<int>(row_.size()));
    R.setFromTriplets(trip.begin(), trip.end());
  }
  Eigen::VectorXd annuity(const Eigen::VectorXd& DF) const {
    const Eigen::VectorXd disc = tau.array() * DF(pay).array();  // materialize before sparse reduction
    return R * disc;
  }

 private:
  std::vector<int> p_, row_;
  std::vector<double> t_;
};

// 3M compounded futures forecasting curve `fc`: rate = (DF[s]/DF[e]-1)*inv_tau + convexity.
struct BundleCompFutures {
  Eigen::VectorXi s, e;
  Eigen::VectorXd inv_tau, convexity;

  void add(CompiledCurveSet& cs, int fc, const CompoundedFuture& f, double conv) {
    s_.push_back(cs.reg(fc, f.start));
    e_.push_back(cs.reg(fc, f.end));
    it_.push_back(1.0 / f.accrual);
    cv_.push_back(conv);
  }
  void finalize() {
    s = detail::to_vec(s_);
    e = detail::to_vec(e_);
    inv_tau = detail::to_vec(it_);
    convexity = detail::to_vec(cv_);
  }
  int size() const { return static_cast<int>(inv_tau.size()); }
  Eigen::VectorXd rate(const Eigen::VectorXd& DF) const {
    return ((DF(s).array() / DF(e).array() - 1.0) * inv_tau.array() + convexity.array()).matrix();
  }

 private:
  std::vector<int> s_, e_;
  std::vector<double> it_, cv_;
};

// 1M averaged futures forecasting curve `fc`: rate = (realized + sum_days(DF[subS]/DF[subE]-1))*inv + conv.
struct BundleAvgFutures {
  Eigen::VectorXi subS, subE, fut;
  Eigen::SparseMatrix<double> R;
  Eigen::VectorXd realized, inv_period, convexity;

  void add(CompiledCurveSet& cs, int fc, const AveragedFuture& f, double conv) {
    for (std::size_t i = 0; i < f.sub_start.size(); ++i) {
      ss_.push_back(cs.reg(fc, f.sub_start[i]));
      se_.push_back(cs.reg(fc, f.sub_end[i]));
      row_.push_back(n_);
    }
    rz_.push_back(f.realized_sum);
    ip_.push_back(1.0 / f.period_yf);
    cv_.push_back(conv);
    ++n_;
  }
  void finalize() {
    subS = detail::to_vec(ss_);
    subE = detail::to_vec(se_);
    fut = detail::to_vec(row_);
    realized = detail::to_vec(rz_);
    inv_period = detail::to_vec(ip_);
    convexity = detail::to_vec(cv_);
    std::vector<Eigen::Triplet<double>> trip;
    for (int k = 0; k < static_cast<int>(row_.size()); ++k) trip.emplace_back(row_[k], k, 1.0);
    R.resize(n_, static_cast<int>(row_.size()));
    R.setFromTriplets(trip.begin(), trip.end());
  }
  int size() const { return n_; }
  Eigen::VectorXd rate(const Eigen::VectorXd& DF) const {
    const Eigen::VectorXd sum = R * (DF(subS).array() / DF(subE).array() - 1.0).matrix();
    return ((realized + sum).array() * inv_period.array() + convexity.array()).matrix();
  }

 private:
  int n_ = 0;
  std::vector<int> ss_, se_, row_;
  std::vector<double> rz_, ip_, cv_;
};

}  // namespace swaps::pricing
