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
// YIELD-SPACE universe: batched price<->yield/duration/convexity, curve-free — the price↔YTM sweep.
// -------------------------------------------------------------------------------------------------
// The "W-cache" for yield-to-maturity. YTM pricing has no shared curve — every bond carries its OWN yield
// y_b — so the classic DF = exp(-Wx) over one knot vector does NOT apply. But the SAME structure-only
// precompute + vectorized hot loop pattern does, and there is a bigger lever unique to YTM:
//
//   P(y) = Σ_i CF_i·(1+y/f)^{−E_i} = v^w · Σ_i CF_i·v^i     with  v = 1/(1+y/f),  E_i = w + i
//
// For a REGULAR bond the exponents are the arithmetic sequence E_i = w + i (w = fraction of the current
// coupon period remaining at settlement), so the powers are GEOMETRIC: the whole price is v^w times a
// POLYNOMIAL in v. That polynomial (and its first two derivatives, for yield-solve/duration/convexity) is
// evaluated by HORNER — pure FMAs, NO per-cashflow transcendental — with a single pow(v,w) per bond. So a
// per-Newton-iteration sweep is O(cashflows) fused-multiply-adds + O(bonds) pows, not O(cashflows) exps.
// The structure-only cache is the coefficient matrix A_ (amounts at integer powers), the offset w_ and f_
// — built once, independent of y (exactly the W-cache spirit: structure once, cheap reprice).
//
// The powers are geometric ONLY when every E_i = w + integer (no odd coupon breaking the grid). A builder
// bond is always regular; an externally-supplied irregular YieldBond falls back to the general exp path
// (CF_i·exp(−E_i·ln base) per cashflow), which is still correct, just without the FMA fast path. Padding
// lanes carry amount 0 and are self-annihilating in BOTH paths (Horner: a 0 leading coefficient; exp: 0·…)
// — no scalar remainder path (§5).
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
    w_.setZero(B);
    simple_.setZero(B);
    regular_ = true;
    any_simple_ = false;
    for (int b = 0; b < B; ++b) {
      freq_[b] = bonds[b].conv.freq;
      accrued_[b] = bonds[b].accrued;
      // Per-bond stub-discount mode as a 0/1 lane mask (pricing::YieldBond::simple_stub resolves the
      // final-period rule). Kept as a MASK, not a branch, so a mixed universe still runs one SIMD sweep
      // with no scalar remainder path (CLAUDE.md §5).
      if (bonds[b].simple_stub()) { simple_[b] = 1.0; any_simple_ = true; }
      const auto& fl = bonds[b].flows;
      const double w0 = fl.empty() ? 0.0 : fl[0].exponent;  // E_0 = w
      w_[b] = w0;
      for (int k = 0; k < static_cast<int>(fl.size()); ++k) {
        E_(b, k) = fl[k].exponent;
        A_(b, k) = fl[k].amount;
        // Regular iff E_k == w + k (integer-spaced from a common offset) for every flow of every bond.
        if (std::abs(fl[k].exponent - (w0 + double(k))) > 1e-9) regular_ = false;
      }
    }
  }

  int size() const { return static_cast<int>(freq_.size()); }
  bool is_regular() const { return regular_; }  // true => the Horner (FMA) fast path is in use

  // Per-bond accrued interest at settlement (per unit notional). FAST by construction: accrued is a pure
  // schedule quantity (coupon·day-fraction), independent of yield/price, so it is computed ONCE per bond
  // at build (build/bond.hpp) and cached — reading it back for the whole universe is O(1), no recompute.
  // Returns a const ref to the cached vector (no allocation). It is what converts clean<->dirty in BOTH
  // directions (yields_from_clean adds it; clean_prices subtracts it).
  const Eigen::VectorXd& accrued() const { return accrued_; }

  // REVERSE direction (yield -> price), fast. Per-bond DIRTY price at per-bond yields `y` (B-vector) via
  // the value-only Horner pass (no derivatives). Const ref into reusable scratch — allocation-free per
  // call, so re-marking a universe as yields move never touches the allocator.
  const Eigen::VectorXd& dirty_prices(const Eigen::VectorXd& y) const {
    value_pass(y, /*want_deriv=*/false);
    return dirty_;
  }
  // Per-bond CLEAN price = dirty - accrued. Also allocation-free (writes into `clean_` scratch): the
  // Horner value pass fills dirty_, then one vectorized subtract of the cached accrued.
  const Eigen::VectorXd& clean_prices(const Eigen::VectorXd& y) const {
    value_pass(y, /*want_deriv=*/false);
    clean_.noalias() = dirty_ - accrued_;
    return clean_;
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
  // Fills dirty_ (and, when want_deriv, d1_ = dP/dy and d2_ = d²P/dy²) at per-bond yields `y`.
  void value_pass(const Eigen::VectorXd& y, bool want_deriv) const {
    if (regular_) horner_pass(y, want_deriv);
    else exp_pass(y, want_deriv);
  }

  // FAST PATH. P = v^w·Q(v), Q(v) = Σ_k A_{·k}·v^k. Horner accumulates Q, Q', Q'' across the B lanes in
  // one FMA sweep per cashflow column (no transcendental); the single pow(v,w) per bond and the chain
  // rule v→y then give dirty/d1/d2. v = 1/(1+y/f), v' = −v²/f, v'' = 2v³/f².
  void horner_pass(const Eigen::VectorXd& y, bool want_deriv) const {
    const int K = static_cast<int>(A_.cols());
    const Eigen::ArrayXd v = 1.0 / (1.0 + y.array() / freq_.array());
    q_.setZero(v.size()); d_.setZero(v.size()); e_.setZero(v.size());
    for (int k = K - 1; k >= 0; --k) {
      // synthetic-differentiation Horner: q→Q, d→Q', e→Q''/2 (update e, then d, then q — order matters).
      if (want_deriv) {
        e_ = e_ * v + d_;
        d_ = d_ * v + q_;
      }
      q_ = q_ * v + A_.col(k).array();
    }
    const Eigen::ArrayXd vw = v.pow(w_.array());  // one transcendental per bond (fractional current period)
    const Eigen::ArrayXd w = w_.array();
    const Eigen::ArrayXd vp = -(v * v) / freq_.array();                             // dv/dy
    const Eigen::ArrayXd vpp = 2.0 * (v * v * v) / (freq_.array() * freq_.array());  // d²v/dy²

    // COMPOUND stub (the default, and the only path when no bond in the universe is simple): the
    // arithmetic below is untouched, so a compound-only universe is bit-for-bit what it always was.
    if (!any_simple_) {
      dirty_ = (vw * q_).matrix();
      if (!want_deriv) return;
      const Eigen::ArrayXd Qp = d_, Qpp = 2.0 * e_;
      // P_v = v^w(w·Q/v + Q');  P_vv = v^w(w(w−1)Q/v² + 2w·Q'/v + Q'').
      const Eigen::ArrayXd Pv = vw * (w * q_ / v + Qp);
      const Eigen::ArrayXd Pvv = vw * (w * (w - 1.0) * q_ / (v * v) + 2.0 * w * Qp / v + Qpp);
      d1_ = (Pv * vp).matrix();
      d2_ = (Pvv * vp * vp + Pv * vpp).matrix();
      return;
    }

    // MIXED universe. Both closing forms share the SAME Horner accumulators Q, Q', Q'' — only the leading
    // factor differs — so they are evaluated over all lanes and blended by the 0/1 mask. Branchless, one
    // sweep, no per-bond dispatch. `SIMPLE` divides by D = 1 + w·y/f instead of multiplying by v^w:
    //     P = Q/D,  P' = Q'v'/D − Q·D'/D²,  P'' = (Q''v'²+Q'v'')/D − 2Q'v'D'/D² + 2Q·D'²/D³   (D' = w/f)
    const Eigen::ArrayXd m = simple_.array(), mc = 1.0 - m;
    const Eigen::ArrayXd D = 1.0 + w * y.array() / freq_.array(), Dp = w / freq_.array();
    dirty_ = (mc * (vw * q_) + m * (q_ / D)).matrix();
    if (!want_deriv) return;
    const Eigen::ArrayXd Qp = d_, Qpp = 2.0 * e_;
    const Eigen::ArrayXd Pv = vw * (w * q_ / v + Qp);
    const Eigen::ArrayXd Pvv = vw * (w * (w - 1.0) * q_ / (v * v) + 2.0 * w * Qp / v + Qpp);
    const Eigen::ArrayXd c1 = Pv * vp, c2 = Pvv * vp * vp + Pv * vpp;            // compound d1/d2
    const Eigen::ArrayXd Qy = Qp * vp, Qyy = Qpp * vp * vp + Qp * vpp;           // dQ/dy, d²Q/dy²
    const Eigen::ArrayXd s1 = Qy / D - q_ * Dp / (D * D);                        // simple d1
    const Eigen::ArrayXd s2 = Qyy / D - 2.0 * Qy * Dp / (D * D) + 2.0 * q_ * Dp * Dp / (D * D * D);
    d1_ = (mc * c1 + m * s1).matrix();
    d2_ = (mc * c2 + m * s2).matrix();
  }

  // GENERAL PATH (irregular schedules). Per cashflow column: CF·base^{−E}, base = 1 + y/f, one vectorized
  // exp over the B lanes; derivatives via the exact per-cashflow y-partials.
  void exp_pass(const Eigen::VectorXd& y, bool want_deriv) const {
    const int K = static_cast<int>(E_.cols());
    const Eigen::ArrayXd base = 1.0 + y.array() / freq_.array();
    const Eigen::ArrayXd lnbase = base.log();
    // A SIMPLE-stub lane shifts its exponents by w (so the stub is not compounded) and divides by
    // D = 1 + w·y/f at the end; a COMPOUND lane shifts by 0 and divides by 1. One expression, both modes.
    const Eigen::ArrayXd shift = simple_.array() * w_.array();
    const Eigen::ArrayXd D = 1.0 + shift * y.array() / freq_.array();
    const Eigen::ArrayXd Dp = shift / freq_.array();
    Eigen::ArrayXd N = Eigen::ArrayXd::Zero(y.size()), N1 = N, N2 = N;
    for (int k = 0; k < K; ++k) {
      const Eigen::ArrayXd Ek = E_.col(k).array() - shift;
      const Eigen::ArrayXd p = A_.col(k).array() * (-Ek * lnbase).exp();  // CF·base^{−(E−shift)}
      N += p;
      if (want_deriv) {
        N1 += -(Ek / freq_.array()) * p / base;                      // CF·(−E/f)·base^{−E−1}
        N2 += (Ek * (Ek + 1.0) / (freq_.array() * freq_.array())) * p / (base * base);
      }
    }
    dirty_ = (N / D).matrix();
    if (!want_deriv) return;
    d1_ = (N1 / D - N * Dp / (D * D)).matrix();
    d2_ = (N2 / D - 2.0 * N1 * Dp / (D * D) + 2.0 * N * Dp * Dp / (D * D * D)).matrix();
  }

  Eigen::MatrixXd E_, A_;             // B×K padded exponents / amounts (A_ = the coupon-polynomial coeffs)
  Eigen::ArrayXd freq_;               // per-bond compounding freq (used in array math)
  Eigen::VectorXd accrued_;           // per-bond accrued (used in price vector arithmetic)
  Eigen::ArrayXd w_;                  // per-bond current-period fraction w = E_0 (the v^w offset)
  Eigen::ArrayXd simple_;             // per-bond 0/1: 1 => SIMPLE stub discounting (App B / final period)
  bool any_simple_ = false;           // false => the untouched compound-only closing form is used
  bool regular_ = true;               // all E_i = w + integer => Horner fast path applies
  mutable Eigen::VectorXd y_, dirty_, clean_, d1_, d2_;  // reusable scratch
  mutable Eigen::ArrayXd q_, d_, e_;                     // Horner accumulators (Q, Q', Q''/2)
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
