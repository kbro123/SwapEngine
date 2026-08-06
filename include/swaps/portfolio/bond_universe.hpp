#pragma once
// Bulk bond analytics — a whole UNIVERSE of bonds repriced as ONE vectorized sweep, the bond analogue of
// the swap book (portfolio/compiled.hpp). Two kernels, matching the two pricing modes in pricing/bond.hpp:
//
//   BondUniverse      — YIELD space, curve-free. The universe's street data is stacked into padded Eigen
//                       matrices and price<->yield is a BATCHED Newton across every bond at once (a few
//                       iterations, one SIMD sweep per compounding period), instead of a per-bond solve
//                       loop. This is the "price/yield calcs over a large universe" path, penny-perfect vs
//                       QuantLib::BondFunctions per bond.
//
//   CompiledBondBook  — CURVE space. A bond is Σ amount·DF(pay), LINEAR in DF = exp(-Wx), so the universe
//                       rides the SAME W-cache (pricing/compiled_book.hpp CompiledCurveSet) as calibration
//                       and the swap book: build W once, then a reprice is one matvec + vectorized exp +
//                       sparse per-bond reductions. Gives PV / dirty / clean and per-bond z-spread with no
//                       per-bond QuantLib pricing.
//
// Both are allocation-light on the hot path (reusable scratch) and templated where AAD matters.

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "swaps/pricing/bond.hpp"
#include "swaps/pricing/compiled_book.hpp"

namespace swaps::portfolio {

// -------------------------------------------------------------------------------------------------
// YIELD-SPACE universe: batched price<->yield/duration/convexity, curve-free.
// -------------------------------------------------------------------------------------------------
// The street data of B bonds is padded to K = max cashflow count. Padding entries have amount 0 and
// exponent 0: base^0·0 = 0 and their derivative weights (exponent/f, exponent(exponent+1)) are 0, so a
// padded lane contributes nothing to price OR its derivatives — no explicit mask needed (§5 no-remainder).
class BondUniverse {
 public:
  void set(const std::vector<pricing::YieldBond>& bonds) {
    const int B = static_cast<int>(bonds.size());
    int K = 0;
    for (const auto& b : bonds) K = std::max(K, static_cast<int>(b.flows.size()));
    E_.setZero(B, K);
    A_.setZero(B, K);
    freq_.resize(B);
    accrued_.resize(B);
    for (int b = 0; b < B; ++b) {
      freq_[b] = bonds[b].freq;
      accrued_[b] = bonds[b].accrued;
      for (int k = 0; k < static_cast<int>(bonds[b].flows.size()); ++k) {
        E_(b, k) = bonds[b].flows[k].exponent;
        A_(b, k) = bonds[b].flows[k].amount;
      }
    }
  }

  int size() const { return static_cast<int>(freq_.size()); }

  // Per-bond DIRTY price at per-bond yields `y` (B-vector). One SIMD sweep per cashflow column.
  const Eigen::VectorXd& dirty_prices(const Eigen::VectorXd& y) const {
    value_pass(y, /*want_deriv=*/false);
    return dirty_;
  }
  Eigen::VectorXd clean_prices(const Eigen::VectorXd& y) const {
    return dirty_prices(y) - accrued_;
  }

  // Per-bond street yields from per-bond CLEAN prices — a BATCHED Newton on the dirty price. Every bond
  // steps together; converged bonds have ~0 residual so their step is ~0 (no masking). Returns yields.
  const Eigen::VectorXd& yields_from_clean(const Eigen::VectorXd& clean, double tol = 1e-13,
                                           int max_iter = 100) const {
    const int B = size();
    const Eigen::VectorXd target = clean + accrued_;  // solve on dirty
    y_.setConstant(B, 0.05);
    for (int it = 0; it < max_iter; ++it) {
      value_pass(y_, /*want_deriv=*/true);
      const Eigen::VectorXd resid = dirty_ - target;
      if (resid.cwiseAbs().maxCoeff() <= tol) break;
      y_.array() -= resid.array() / d1_.array();  // Newton, elementwise across the universe
    }
    return y_;
  }

  // Per-bond modified duration and convexity at yields `y` (QuantLib BondFunctions convention).
  Eigen::VectorXd modified_durations(const Eigen::VectorXd& y) const {
    value_pass(y, /*want_deriv=*/true);
    return (-d1_.array() / dirty_.array()).matrix();
  }
  Eigen::VectorXd convexities(const Eigen::VectorXd& y) const {
    value_pass(y, /*want_deriv=*/true);
    return (d2_.array() / dirty_.array()).matrix();
  }

 private:
  // Fills dirty_ (and d1_/d2_ when want_deriv) for per-bond yields `y`. base = 1 + y/f (per bond); each
  // cashflow column contributes CF·base^{−E}, computed as CF·exp(−E·ln base) so the whole column is one
  // vectorized transcendental sweep over the B lanes.
  void value_pass(const Eigen::VectorXd& y, bool want_deriv) const {
    const int B = static_cast<int>(y.size()), K = static_cast<int>(E_.cols());
    const Eigen::ArrayXd base = 1.0 + y.array() / freq_.array();
    const Eigen::ArrayXd lnbase = base.log();
    dirty_.setZero(B);
    if (want_deriv) { d1_.setZero(B); d2_.setZero(B); }
    for (int k = 0; k < K; ++k) {
      const Eigen::ArrayXd Ek = E_.col(k).array();
      const Eigen::ArrayXd p = A_.col(k).array() * (-Ek * lnbase).exp();  // CF·base^{−E}
      dirty_.array() += p;
      if (want_deriv) {
        d1_.array() += -(Ek / freq_.array()) * p / base;                      // CF·(−E/f)·base^{−E−1}
        d2_.array() += (Ek * (Ek + 1.0) / (freq_.array() * freq_.array())) * p / (base * base);
      }
    }
  }

  Eigen::MatrixXd E_, A_;             // B×K padded exponents / amounts
  Eigen::ArrayXd freq_;               // per-bond compounding freq (used in array math)
  Eigen::VectorXd accrued_;           // per-bond accrued (used in price vector arithmetic)
  mutable Eigen::VectorXd y_, dirty_, d1_, d2_;  // reusable scratch
};

// -------------------------------------------------------------------------------------------------
// CURVE-SPACE book: bonds priced off a discount curve via the W-cache.
// -------------------------------------------------------------------------------------------------
// Registers every bond's cashflow pay times AND settlement time on one self-discounting CurveStructure,
// builds W once, then a reprice is DF = exp(-Wx) followed by a sparse per-bond reduction. PV(today) is
// LINEAR in DF; the dirty price divides by DF(settle) per bond (a cheap elementwise step).
class CompiledBondBook {
 public:
  CompiledBondBook(const std::vector<double>& meeting_times, const std::vector<double>& back_times,
                   const std::vector<pricing::Bond>& bonds) {
    cs_.init({pricing::CurveStructure{meeting_times, back_times, -1}});  // one self-discounting curve
    const int B = static_cast<int>(bonds.size());
    settle_idx_.resize(B);
    accrued_.resize(B);
    bond_of_flow_.clear();
    amount_.clear();
    pay_idx_.clear();
    flow_dt_.clear();
    for (int b = 0; b < B; ++b) {
      settle_idx_[b] = cs_.reg(0, bonds[b].settle);
      accrued_[b] = bonds[b].accrued;
      for (const auto& f : bonds[b].flows) {
        pay_idx_.push_back(cs_.reg(0, f.pay));
        amount_.push_back(f.amount);
        bond_of_flow_.push_back(b);
        flow_dt_.push_back(f.pay - bonds[b].settle);  // z-spread discount horizon from settlement
      }
    }
    cs_.finalize();
    // Sparse bond×flow reduction R (0/1) and dense flow arrays for the gather.
    n_bonds_ = B;
    n_flows_ = static_cast<int>(amount_.size());
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(n_flows_);
    for (int j = 0; j < n_flows_; ++j) trip.emplace_back(bond_of_flow_[j], j, 1.0);
    R_.resize(n_bonds_, n_flows_);
    R_.setFromTriplets(trip.begin(), trip.end());
    amount_v_ = Eigen::Map<Eigen::VectorXd>(amount_.data(), n_flows_);
    pay_v_ = Eigen::Map<Eigen::VectorXi>(pay_idx_.data(), n_flows_);
    settle_v_ = Eigen::Map<Eigen::VectorXi>(settle_idx_.data(), n_bonds_);
    accrued_v_ = Eigen::Map<Eigen::VectorXd>(accrued_.data(), n_bonds_);
  }

  int n_bonds() const { return n_bonds_; }
  int n_times() const { return cs_.n_times(); }

  // Per-bond PV today = Σ amount·DF(pay). Const ref into reusable scratch (no per-reprice allocation).
  const Eigen::VectorXd& pv(const Eigen::VectorXd& x) const {
    cs_.df_into(x, df_);
    flow_.resize(n_flows_);
    const double* __restrict df = df_.data();
    const int* __restrict p = pay_v_.data();
    const double* __restrict a = amount_v_.data();
    for (int j = 0; j < n_flows_; ++j) flow_[j] = a[j] * df[p[j]];  // auto-vectorized gather
    pv_.noalias() = R_ * flow_;
    return pv_;
  }

  // Per-bond model DIRTY price = PV(today) / DF(settle).
  const Eigen::VectorXd& dirty_prices(const Eigen::VectorXd& x) const {
    pv(x);  // fills df_ and pv_
    dirty_.resize(n_bonds_);
    const double* __restrict df = df_.data();
    const int* __restrict s = settle_v_.data();
    for (int b = 0; b < n_bonds_; ++b) dirty_[b] = pv_[b] / df[s[b]];
    return dirty_;
  }
  Eigen::VectorXd clean_prices(const Eigen::VectorXd& x) const {
    return dirty_prices(x) - accrued_v_;
  }

  // Per-bond z-spread (continuous, on curve time) to a per-bond target DIRTY price. A scalar Newton per
  // bond off the current DFs — z-spread is inherently a per-bond quantity, so this is a B-loop, not the
  // vectorized hot path. Uses the already-computed df_ from a reprice at `x`.
  Eigen::VectorXd z_spreads(const Eigen::VectorXd& x, const Eigen::VectorXd& target_dirty,
                            double tol = 1e-14, int max_iter = 100) const {
    cs_.df_into(x, df_);
    Eigen::VectorXd z(n_bonds_);
    // Walk flows grouped by bond (flows are pushed bond-major in the ctor, so contiguous per bond).
    int j = 0;
    for (int b = 0; b < n_bonds_; ++b) {
      const double df_settle = df_[settle_v_[b]];
      const int j0 = j;
      while (j < n_flows_ && bond_of_flow_[j] == b) ++j;
      double s = 0.0;
      for (int it = 0; it < max_iter; ++it) {
        double price = 0.0, dprice = 0.0;
        for (int m = j0; m < j; ++m) {
          const double dt = flow_dt_[m];
          const double v = amount_v_[m] * (df_[pay_v_[m]] / df_settle) * std::exp(-s * dt);
          price += v;
          dprice += -dt * v;
        }
        const double resid = price - target_dirty[b];
        if (std::abs(resid) <= tol) break;
        s -= resid / dprice;
      }
      z[b] = s;
    }
    return z;
  }

 private:
  pricing::CompiledCurveSet cs_;
  int n_bonds_ = 0, n_flows_ = 0;
  std::vector<int> pay_idx_, settle_idx_, bond_of_flow_;
  std::vector<double> amount_, accrued_, flow_dt_;
  Eigen::VectorXd amount_v_, accrued_v_;
  Eigen::VectorXi pay_v_, settle_v_;
  Eigen::SparseMatrix<double> R_;
  mutable Eigen::VectorXd df_, flow_, pv_, dirty_;
};

}  // namespace swaps::portfolio
