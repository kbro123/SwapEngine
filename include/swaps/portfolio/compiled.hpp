#pragma once
// Vectorized ("compiled") portfolio repricing — the real-time / many-curves path (CLAUDE.md §2, §5).
//
// For a FIXED curve structure the log-discount is linear in the knot forwards: integral(t) = w(t)·x.
// Stack every distinct portfolio cashflow time's weight row into W (n_times × n_knots) ONCE. Then a
// full portfolio reprice for a new x is:
//     DF = exp(-W x)                                  (one matvec + one vectorized exp)
//     float coupon_k = DF[pay]·(DF[accS]/DF[accE]-1)  (gathered, elementwise over ALL coupons)
//     fixed_k        = tau_k · DF[pay]                (gathered, elementwise)
//     leg values     = R_float · coupons , R_fixed · fixed   (sparse 0/1 reductions to per-swap)
//     NPV_p          = notional_p·(floatpv_p - fixedrate_p·annuity_p)
// No per-swap, per-coupon scalar loop and no QuantLib object dispatch in the hot path — that loop is
// exactly the baseline we are beating.

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <map>
#include <vector>

#include "swaps/pricing/compiled.hpp"  // shared integral_weight_matrix (uses the calibration curve)
#include "swaps/portfolio/portfolio.hpp"

namespace swaps::portfolio {

using pricing::integral_weight_matrix;  // W over the calibration curve -- one source, not two

class CompiledPortfolio {
 public:
  CompiledPortfolio(const std::vector<double>& meeting_times, const std::vector<double>& back_times,
                    const Portfolio& pf) {
    std::map<double, int> index;  // distinct cashflow time -> row in DF
    std::vector<double> times;
    auto idx = [&](double t) {
      auto it = index.find(t);
      if (it != index.end()) return it->second;
      const int k = static_cast<int>(times.size());
      index.emplace(t, k);
      times.push_back(t);
      return k;
    };

    const int P = static_cast<int>(pf.positions.size());
    fixed_rate_.resize(P);
    notional_.resize(P);
    std::vector<Eigen::Triplet<double>> tf, tx;
    std::vector<int> fpay, faccs, facce, xpay;
    std::vector<double> xtau;

    for (int p = 0; p < P; ++p) {
      const auto& pos = pf.positions[p];
      fixed_rate_[p] = pos.fixed_rate;
      notional_[p] = pos.notional;
      for (std::size_t i = 0; i < pos.sched.float_pay.size(); ++i) {
        const int k = static_cast<int>(fpay.size());
        fpay.push_back(idx(pos.sched.float_pay[i]));
        faccs.push_back(idx(pos.sched.float_acc_start[i]));
        facce.push_back(idx(pos.sched.float_acc_end[i]));
        tf.emplace_back(p, k, 1.0);
      }
      for (std::size_t i = 0; i < pos.sched.fixed_pay.size(); ++i) {
        const int k = static_cast<int>(xpay.size());
        xpay.push_back(idx(pos.sched.fixed_pay[i]));
        xtau.push_back(pos.sched.fixed_accrual[i]);
        tx.emplace_back(p, k, 1.0);
      }
    }

    W_ = integral_weight_matrix(meeting_times, back_times, times);
    f_pay_ = Eigen::Map<Eigen::VectorXi>(fpay.data(), fpay.size());
    f_accs_ = Eigen::Map<Eigen::VectorXi>(faccs.data(), faccs.size());
    f_acce_ = Eigen::Map<Eigen::VectorXi>(facce.data(), facce.size());
    x_pay_ = Eigen::Map<Eigen::VectorXi>(xpay.data(), xpay.size());
    x_tau_ = Eigen::Map<Eigen::VectorXd>(xtau.data(), xtau.size());
    Rfloat_.resize(P, static_cast<int>(fpay.size()));
    Rfloat_.setFromTriplets(tf.begin(), tf.end());
    Rfixed_.resize(P, static_cast<int>(xpay.size()));
    Rfixed_.setFromTriplets(tx.begin(), tx.end());
  }

  int n_swaps() const { return static_cast<int>(notional_.size()); }
  int n_times() const { return static_cast<int>(W_.rows()); }

  // Per-swap NPV for knot forwards x. Fully vectorized.
  Eigen::VectorXd npv(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd DF = (-(W_ * x).array()).exp();
    const Eigen::VectorXd coupon =
        DF(f_pay_).array() * (DF(f_accs_).array() / DF(f_acce_).array() - 1.0);
    const Eigen::VectorXd fixed = x_tau_.array() * DF(x_pay_).array();
    const Eigen::VectorXd float_pv = Rfloat_ * coupon;
    const Eigen::VectorXd annuity = Rfixed_ * fixed;
    return (notional_.array() * (float_pv.array() - fixed_rate_.array() * annuity.array())).matrix();
  }

  double total_npv(const Eigen::VectorXd& x) const { return npv(x).sum(); }

 private:
  Eigen::MatrixXd W_;
  Eigen::VectorXi f_pay_, f_accs_, f_acce_, x_pay_;
  Eigen::VectorXd x_tau_, fixed_rate_, notional_;
  Eigen::SparseMatrix<double> Rfloat_, Rfixed_;
};

}  // namespace swaps::portfolio
