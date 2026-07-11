#pragma once
// Shared vectorized discount engine (Stage 2). The calibration instrument set and a risk portfolio
// are the SAME thing: instruments valued off one DF vector. This header holds the common expensive
// core -- DF = exp(-Wx) over the union of all distinct cashflow times, plus the DF-linear-combination
// primitives (float legs, annuities, futures reductions). Both CompiledPortfolio (NPVs) and
// CompiledResidual (rates) are thin consumers that differ only in the final cheap transform.
//
// Everything here is plain double + Eigen dense/sparse: no per-cashflow curve.discount, no spline
// re-solve, no AAD, no scalar per-instrument loop. `W` is built ONCE (one AAD pass; the integral is
// linear so its gradient is the weight row) and reused across repricings of the same structure.

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <map>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/curve/two_region_forward_curve.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace swaps::pricing {

// W(i,:) such that integral(times[i]) = W(i,:)*x. Built via one AAD pass (integral is linear, so the
// gradient is the weight row, independent of x). Setup only, not the hot path.
inline Eigen::MatrixXd integral_weight_matrix(const std::vector<double>& meeting,
                                              const std::vector<double>& back,
                                              const std::vector<double>& times) {
  const int m = static_cast<int>(meeting.size() + back.size());
  curve::TwoRegionForwardCurve<ad::Dual> c(meeting, back);
  c.set_forwards(ad::seed(Eigen::VectorXd::Constant(m, 0.03)));
  Eigen::MatrixXd W(static_cast<int>(times.size()), m);
  for (std::size_t i = 0; i < times.size(); ++i) {
    const ad::Dual I = c.integral(times[i]);
    if (I.derivatives().size() == m)
      W.row(static_cast<int>(i)) = I.derivatives().transpose();
    else
      W.row(static_cast<int>(i)).setZero();
  }
  return W;
}

// Registers distinct cashflow times, giving each a stable row index into the shared DF vector.
class TimeIndex {
 public:
  int operator()(double t) {
    auto it = map_.find(t);
    if (it != map_.end()) return it->second;
    const int k = static_cast<int>(times_.size());
    map_.emplace(t, k);
    times_.push_back(t);
    return k;
  }
  const std::vector<double>& times() const { return times_; }
  int size() const { return static_cast<int>(times_.size()); }

 private:
  std::map<double, int> map_;
  std::vector<double> times_;
};

// DF = exp(-W x) over the registered times. The shared expensive core.
class CompiledDiscounts {
 public:
  CompiledDiscounts() = default;
  CompiledDiscounts(const std::vector<double>& meeting, const std::vector<double>& back,
                    const std::vector<double>& times)
      : W_(integral_weight_matrix(meeting, back, times)) {}

  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const { return (-(W_ * x).array()).exp(); }
  int n_times() const { return static_cast<int>(W_.rows()); }
  const Eigen::MatrixXd& W() const { return W_; }

 private:
  Eigen::MatrixXd W_;
};

namespace detail {
inline Eigen::VectorXi to_vec(const std::vector<int>& v) {
  return Eigen::Map<const Eigen::VectorXi>(v.data(), static_cast<int>(v.size()));
}
inline Eigen::VectorXd to_vec(const std::vector<double>& v) {
  return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<int>(v.size()));
}
}  // namespace detail

// Float legs of N swaps: pv_j = sum over j's coupons of DF[pay]*(DF[accS]/DF[accE]-1).
struct CompiledFloatLegs {
  Eigen::VectorXi pay, accS, accE;      // per coupon (all swaps flattened)
  Eigen::SparseMatrix<double> R;        // N x n_coupons, 0/1 reduction to per-swap

  void build(const std::vector<const OisSwap*>& swaps, TimeIndex& ti) {
    std::vector<int> p, s, e;
    std::vector<Eigen::Triplet<double>> trip;
    for (int j = 0; j < static_cast<int>(swaps.size()); ++j)
      for (std::size_t i = 0; i < swaps[j]->float_pay.size(); ++i) {
        const int k = static_cast<int>(p.size());
        p.push_back(ti(swaps[j]->float_pay[i]));
        s.push_back(ti(swaps[j]->float_acc_start[i]));
        e.push_back(ti(swaps[j]->float_acc_end[i]));
        trip.emplace_back(j, k, 1.0);
      }
    pay = detail::to_vec(p);
    accS = detail::to_vec(s);
    accE = detail::to_vec(e);
    R.resize(static_cast<int>(swaps.size()), static_cast<int>(p.size()));
    R.setFromTriplets(trip.begin(), trip.end());
  }
  Eigen::VectorXd pv(const Eigen::VectorXd& DF) const {
    return R * (DF(pay).array() * (DF(accS).array() / DF(accE).array() - 1.0)).matrix();
  }
};

// Fixed-leg annuities of N swaps: ann_j = sum over j's coupons of tau*DF[pay].
struct CompiledFixedLegs {
  Eigen::VectorXi pay;
  Eigen::VectorXd tau;
  Eigen::SparseMatrix<double> R;  // N x n_coupons

  void build(const std::vector<const OisSwap*>& swaps, TimeIndex& ti) {
    std::vector<int> p;
    std::vector<double> t;
    std::vector<Eigen::Triplet<double>> trip;
    for (int j = 0; j < static_cast<int>(swaps.size()); ++j)
      for (std::size_t i = 0; i < swaps[j]->fixed_pay.size(); ++i) {
        const int k = static_cast<int>(p.size());
        p.push_back(ti(swaps[j]->fixed_pay[i]));
        t.push_back(swaps[j]->fixed_accrual[i]);
        trip.emplace_back(j, k, 1.0);
      }
    pay = detail::to_vec(p);
    tau = detail::to_vec(t);
    R.resize(static_cast<int>(swaps.size()), static_cast<int>(p.size()));
    R.setFromTriplets(trip.begin(), trip.end());
  }
  Eigen::VectorXd annuity(const Eigen::VectorXd& DF) const {
    return R * (tau.array() * DF(pay).array()).matrix();
  }
};

// 3M compounded futures: rate_j = (DF[s]/DF[e]-1)*inv_tau + convexity.
struct CompiledCompoundedFutures {
  Eigen::VectorXi s, e;
  Eigen::VectorXd inv_tau, convexity;

  void add(const CompoundedFuture& f, double conv, TimeIndex& ti) {
    s_.push_back(ti(f.start));
    e_.push_back(ti(f.end));
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

// 1M arithmetic-average futures: rate_j = (realized_j + sum_days(DF[subS]/DF[subE]-1))*inv_period + conv.
struct CompiledAveragedFutures {
  Eigen::VectorXi subS, subE;      // per business day (all futures flattened)
  Eigen::SparseMatrix<double> R;   // N x n_subperiods
  Eigen::VectorXd realized, inv_period, convexity;

  void add(const AveragedFuture& f, double conv, TimeIndex& ti) {
    for (std::size_t i = 0; i < f.sub_start.size(); ++i) {
      ss_.push_back(ti(f.sub_start[i]));
      se_.push_back(ti(f.sub_end[i]));
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
