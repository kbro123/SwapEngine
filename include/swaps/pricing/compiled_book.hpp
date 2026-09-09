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
// and role-aware leg batches that gather from the global DF. Consumers (calibration residual, portfolio
// NPV) are thin final transforms on pv / annuity / rate -- see calibration/compiled_bundle.hpp.
//
// There is exactly ONE float primitive (BundleFloatBatch, design §4): the compiled form of the generic
// FloatCoupon/RateObservation model. Legs and futures, compounded and averaged, spread and no spread,
// any day-count basis are all the same batch with different DATA.

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cassert>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/compiled.hpp"  // integral_weight_matrix, detail::to_vec
#include "swaps/pricing/curve_spec.hpp"  // CurveStructure (shared with the calibration layer)

namespace swaps::pricing {

// REDUCE LAYOUT (E4.D, measured 2026-09-09 on every rung of the shape ladder, tools/check_perf.py shape_*):
//   1  SEGMENT (default): coupons of an instrument (and sub-periods of a coupon) are CONTIGUOUS in registration
//      order, so each coupon's gather-product accumulates straight into its instrument's slot — no coupon
//      vector, no sparse matrix. Stream tick 0.50-0.66x of the sparse form on every compiled shape, 0.42x on the
//      desk-scale chain (177 -> 74 us); refresh 0.78-0.90x; Jacobian 0.84-0.96x; never slower.
//   0  SPARSE (reference): materialise the coupon vector, then R_cpn * coupon (Eigen SpMV). Kept selectable
//      (-DSWAPS_REDUCE_LAYOUT=0) as the A/B baseline; not used by any build.
//   A PADDED coupon-major layout (slots x instruments with a DF==1 sentinel, instruments as SIMD lanes) was
//   measured too and was neutral to 1.2x SLOWER on the ticks (padding waste, worse locality); deleted.
#ifndef SWAPS_REDUCE_LAYOUT
#define SWAPS_REDUCE_LAYOUT 1
#endif
static_assert(SWAPS_REDUCE_LAYOUT == 0 || SWAPS_REDUCE_LAYOUT == 1, "SWAPS_REDUCE_LAYOUT: 0 (sparse reference) or 1 (segment)");

// Row-major dense matrix: storing the batched DF grid row-major makes DFg.row(t) — one registered time
// across ALL curve-states — a CONTIGUOUS span, which is what lets the batched leg reduce gather a coupon's
// pay/start/end rows with unit-stride (SIMD-friendly) loads. Only the batched MC-exposure grid path uses
// it; calibration and single-state pricing stay on Eigen's default column-major VectorXd/MatrixXd.
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// CurveStructure (the per-curve topology) is shared with the calibration layer -- see curve_spec.hpp.

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
  // Allocation-free DF: writes exp(-W_all x) into the caller's `out` scratch (bit-identical to df(x)).
  void df_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const {
    out.noalias() = W_ * x;
    out = (-out.array()).exp();
  }
  // BATCHED DF (hot-path design R12): a MATRIX of curve-states X (n_knots x n_states) -> DF grid
  // (n_times x n_states) = exp(-W_all·X), one GEMM + one vectorized exp. This is the exposure/MC lever:
  // repricing 10k paths x 100 nodes shares ONE W·X matmul instead of N_states separate matvecs. Column j is
  // bit-identical to df_into(X.col(j), .). Alloc-free after the first sizing of `out`.
  void df_into(const Eigen::MatrixXd& X, Eigen::MatrixXd& out) const {
    out.noalias() = W_ * X;
    out = (-out.array()).exp();
  }
  // ROW-MAJOR batched DF grid (design R12 coupon-batch): the SAME exp(-W_all·X) as the column-major
  // overload above (column j bit-identical to df_into(X.col(j),·)), but stored ROW-MAJOR so that each
  // DFg.row(t) is contiguous. The per-coupon row gathers in BundleFloatBatch::pv_grid /
  // BundleFixedLegs::annuity_grid then run unit-stride across states instead of striding a column-major
  // grid by n_times. Calibration NEVER calls this — its df_into(VectorXd/MatrixXd) overloads are UNCHANGED.
  // Alloc-free after `out` is first sized. X is an Eigen::Ref so a column-BLOCK of a larger state grid
  // (X.middleCols(j0, bw)) can be discounted without copying — the batched exposure loop tiles the states
  // so the DF/coupon working set stays cache-resident.
  void df_into(const Eigen::Ref<const Eigen::MatrixXd>& X, RowMatrixXd& out) const {
    out.noalias() = W_ * X;
    out = (-out.array()).exp();
  }
  const Eigen::MatrixXd& W() const { return W_; }
  // FORWARD weight rows of curve c at `times` (forward_c(t) = row·x): own regions + base ancestry + a 0/1
  // indicator column per turn (a turn adds δ to the forward inside its window). Setup only (the moment path's
  // quadratic forms are built from these at registration); callable once init() has run.
  Eigen::MatrixXd forward_rows(int c, const std::vector<double>& times) const {
    const int ni = specs_[c].n_interp_knots();
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(static_cast<int>(times.size()), n_knots_);
    P.middleCols(knot_offset_[c], ni) = forward_weight_matrix(specs_[c].modules(), times);
    for (int j = 0; j < static_cast<int>(specs_[c].turns.size()); ++j) {
      const int col = knot_offset_[c] + ni + j;
      const auto& w = specs_[c].turns[j];
      for (int i = 0; i < static_cast<int>(times.size()); ++i) P(i, col) = (times[i] >= w.start && times[i] < w.end) ? 1.0 : 0.0;
    }
    if (specs_[c].base >= 0) P += forward_rows(specs_[c].base, times);
    return P;
  }
  int n_times() const { return static_cast<int>(pts_.size()); }
  int n_knots() const { return n_knots_; }

 private:
  // Log-DF weight rows of curve c at `times`: own knot weights in c's block, PLUS the base's log-DF
  // weights (recursively) if c is a spread curve. integral_spread(t) adds to integral_base(t), so the
  // total stays linear in the stacked x -- the whole reason a spread curve keeps the W-cache.
  //
  // A curve's state block is [ interpolation knots | one δ per turn ]. The interp knots take the
  // region W-cache; each turn δ takes a closed-form `overlap` column (docs/turns-calibration.md §2):
  // the log-DF integral gains Σⱼ δⱼ·overlap(tᵢ, turnⱼ), which is LINEAR in the δⱼ and needs no AAD. A
  // spread/dependent curve observes the base's turns FOR FREE via the same base recursion below.
  Eigen::MatrixXd logdf_weight(int c, const std::vector<double>& times) const {
    const int ni = specs_[c].n_interp_knots();
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(static_cast<int>(times.size()), n_knots_);
    W.middleCols(knot_offset_[c], ni) =
        integral_weight_matrix(specs_[c].modules(), times);  // one W-cache, any region layout
    for (int j = 0; j < static_cast<int>(specs_[c].turns.size()); ++j) {
      const int col = knot_offset_[c] + ni + j;  // δⱼ sits after the interp knots in c's block
      for (int i = 0; i < static_cast<int>(times.size()); ++i)
        W(i, col) = turn_overlap(times[i], specs_[c].turns[j]);
    }
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

// THE float batch (design §4) -- the compiled form of the generic FloatCoupon/RateObservation model.
// N instruments, each a list of coupons, each coupon an observation with ANY number of weighted
// sub-periods. Two sparse reductions:
//
//   sub   = DF[s_k]/DF[e_k] - 1                      (per sub-period, forecast curve)
//   num   = R_sub * sub                              (sub-periods -> per-coupon numerator; w_k are
//                                                     R_sub's VALUES, so no separate w-multiply)
//   pv    = R_cpn * ( DF[pay] * (num + konst) * k )  (coupons -> per-instrument leg PV)
//
// in the k-form of pricing/cashflows.hpp float_coupon_pv (konst = realized + spread*tau_index,
// k = tau_pay/tau_index), which is what makes the one-sub-period/no-spread/tau_pay==tau_index shape
// reduce BIT-EXACTLY to the pre-generalization DF[pay]*(DF[accS]/DF[accE]-1).
//
// A FUTURE is the same batch stopping earlier: rate = (num + realized)*inv_tau + convexity. So the
// old BundleFloatLegs, BundleCompFutures and BundleAvgFutures are all THIS type -- "1M averaged SOFR
// future" vs "3M compounded future" vs "IBOR coupon" is data (how many sub-periods, what weights),
// not a type (design §1).
//
// All curve indices are GLOBAL into the CompiledCurveSet.
struct BundleFloatBatch {
  // Per sub-period (every coupon of every instrument, flattened).
  Eigen::VectorXi subS, subE, sub_cpn;  // sub_cpn = owning coupon
  Eigen::VectorXd sub_w;                // w_k (also carried as R_sub's values)
  Eigen::SparseMatrix<double> R_sub;    // n_coupons x n_subs, values = w_k
  // Per coupon.
  Eigen::VectorXi pay, inst;                       // inst = owning instrument; pay < 0 for futures
  Eigen::VectorXd konst, k, realized, inv_tau, convexity;
  Eigen::SparseMatrix<double> R_cpn;               // n_inst x n_coupons, 0/1
  int n_inst = 0;
  // True iff R_sub is EXACTLY the identity (one unit-weight sub-period per coupon, in order) -- the
  // standard compounded-OIS / IBOR / single-period-future shape. Then R_sub*v == v and we skip the
  // reduction entirely, so generalizing costs the hot path nothing.
  bool sub_is_identity = false;
  // True iff EVERY coupon has konst == 0 and k == 1 -- no spread, nothing realized, and
  // tau_pay == tau_index: the standard OIS / portfolio shape. The "+ konst" and "* k" passes are then
  // pure overhead over the whole coupon vector (~1.28x on a 1000-swap book, measured), so pv() fuses
  // to the minimal pre-generalization expression. Genericity must cost the hot path nothing (design §4).
  bool cpn_is_plain = false;

  // ---- MOMENT coupons (RateObservation::fixing_step > 0; docs/bezier-and-moments.md Part B) -------------------
  // An arithmetic-average window [a,b] priced from curve MOMENTS instead of ~250 daily sub-periods:
  //     num = ln(DF[a]·INV[b]) + ½·step·∫f² (+ ⅙·step3·∫f³),   ∫f² = xᵀ Q x with Q = Σ_k w_k psi_k psi_kᵀ
  // Q (over the coupon's knot SUPPORT) and, when step3 > 0, the Gauss nodes' forward rows are precomputed at
  // registration, so a tick costs one log + |S|² multiply-adds per coupon (|S| ≈ 6-16) instead of a daily gather
  // loop — the reason a Fed funds curve can be as cheap as a SOFR curve (shape ladder: 534 → ~5 µs tick).
  // The linear term is exact; the moment correction is the documented ~5e-9 approximation of the daily sum
  // (the exact daily path stays available with fixing_step == 0). Same Gauss rule as the templated
  // curve_forward_sq_integral (32 panels × 2 nodes) so both paths agree to rounding.
  struct MomentCoupon {
    int cpn = -1, sub = -1;              // coupon index; its single sub-period (the [a,b] bracket)
    double step = 0.0, step3 = 0.0;      // ½·step and ⅙·step3 are applied in set_state
    std::vector<int> support;            // global knot indices with nonzero forward weight over [a,b]
    Eigen::MatrixXd Q;                   // |S|×|S|: Σ_k w_k psi_k psi_kᵀ (restricted to the support)
    Eigen::MatrixXd Psi;                 // (step3 > 0 only) nodes × |S| forward rows
    Eigen::VectorXd wq;                  // (step3 > 0 only) Gauss weights per node
    // per-state scratch (set_state): the correction value and its gradient over the support
    double mom = 0.0;
    Eigen::VectorXd xs, dmom, fnode;
  };
  bool has_moment() const { return !moments_.empty(); }
  const std::vector<MomentCoupon>& moments() const { return moments_; }
  // Evaluate every moment coupon's correction (and its x-gradient) at the stacked knot state x. MUST precede
  // num/pv/rate/d_* on a batch with moment coupons (they throw otherwise). Allocation-free after first use.
  void set_state(const Eigen::VectorXd& x) const {
    if (state_set_ && state_x_.size() == x.size() && (state_x_.array() == x.array()).all()) return;  // memo on x
    state_x_ = x;
    for (auto& mc : moments_) {
      const int n = static_cast<int>(mc.support.size());
      mc.xs.resize(n);
      for (int j = 0; j < n; ++j) mc.xs[j] = x[mc.support[j]];
      mc.dmom.resize(n);
      mc.dmom.noalias() = mc.Q * mc.xs;                       // ∇(½ xᵀQx) = Qx (Q symmetric)
      double m = 0.5 * mc.step * mc.xs.dot(mc.dmom);           // ½·step·xᵀQx
      mc.dmom *= mc.step;
      if (mc.step3 > 0.0) {
        mc.fnode.resize(mc.Psi.rows());
        mc.fnode.noalias() = mc.Psi * mc.xs;                   // f at the Gauss nodes
        double cube = 0.0;
        for (int k = 0; k < mc.fnode.size(); ++k) {
          const double f2 = mc.fnode[k] * mc.fnode[k];
          cube += mc.wq[k] * f2 * mc.fnode[k];
          mc.fnode[k] = 0.5 * mc.step3 * mc.wq[k] * f2;       // reuse as the gradient weights g_k
        }
        m += (1.0 / 6.0) * mc.step3 * cube;
        mc.dmom.noalias() += mc.Psi.transpose() * mc.fnode;   // d/dx ⅙·step3·Σ w f³ = Σ_k g_k psi_k (one GEMV)
      }
      mc.mom = m;
    }
    state_set_ = true;
  }

  int n_coupons() const { return n_cpn_; }
  int size() const { return n_inst; }
  static constexpr int reduce_layout() { return SWAPS_REDUCE_LAYOUT; }

  // --- generic registration -------------------------------------------------------------------
  // One instrument = one float leg: coupons forecast `fc`, discount `dc`.
  void add(CompiledCurveSet& cs, int fc, int dc, const std::vector<FloatCoupon>& leg) {
    for (const auto& c : leg) {
      push_obs(cs, fc, c.obs);
      // FX `scale` (default 1) folds into the per-coupon k (× 1.0 is exact). A scale != 1 makes k != 1,
      // so the coupon drops off the cpn_is_plain fused fast path automatically -- exactly right: only a
      // foreign (converted) leg pays that cost, and the analytic Jacobian formula (pv = DF·A·k) is
      // unchanged because scale rides inside k as a constant.
      push_coupon(cs.reg(dc, c.pay), c.obs.realized + c.spread * c.obs.tau_index,
                  c.tau_pay / c.obs.tau_index * c.scale, c.obs.realized, 1.0 / c.obs.tau_index, 0.0);
    }
    ++n_inst;
  }
  // One instrument = one future on `obs` forecasting `fc`. No discounting: the terminal transform is
  // rate(), not pv(). `convexity` is an INPUT NUMBER (design §3) -- the model lives in tests.
  void add_future(CompiledCurveSet& cs, int fc, const RateObservation& o, double conv) {
    push_obs(cs, fc, o);
    push_coupon(-1, 0.0, 0.0, o.realized, 1.0 / o.tau_index, conv);
    ++n_inst;
  }

  void finalize() {
    subS = detail::to_vec(ss_);
    subE = detail::to_vec(se_);
    sub_cpn = detail::to_vec(sc_);
    sub_w = detail::to_vec(sw_);
    pay = detail::to_vec(p_);
    inst = detail::to_vec(row_);
    konst = detail::to_vec(konst_);
    k = detail::to_vec(k_);
    realized = detail::to_vec(rz_);
    inv_tau = detail::to_vec(it_);
    convexity = detail::to_vec(cv_);
    R_sub = build(sc_, sw_, n_cpn_);
    R_cpn = build(row_, std::vector<double>(row_.size(), 1.0), n_inst);
    // Segment offsets: coupons of one instrument and sub-periods of one coupon are contiguous by construction
    // (add() pushes a whole leg; push_obs pushes a whole observation) -- asserted, since the fused reduces rely on it.
    cpn_begin_.assign(n_inst + 1, 0);
    for (int c = 0; c < n_cpn_; ++c) { assert(c == 0 || row_[c] >= row_[c - 1]); ++cpn_begin_[row_[c] + 1]; }
    for (int i = 0; i < n_inst; ++i) cpn_begin_[i + 1] += cpn_begin_[i];
    sub_begin_.assign(n_cpn_ + 1, 0);
    for (std::size_t j = 0; j < sc_.size(); ++j) { assert(j == 0 || sc_[j] >= sc_[j - 1]); ++sub_begin_[sc_[j] + 1]; }
    for (int c = 0; c < n_cpn_; ++c) sub_begin_[c + 1] += sub_begin_[c];
    sub_is_identity = (static_cast<int>(sc_.size()) == n_cpn_);
    for (std::size_t j = 0; sub_is_identity && j < sc_.size(); ++j)
      sub_is_identity = (sc_[j] == static_cast<int>(j) && sw_[j] == 1.0);
    cpn_is_plain = true;
    for (std::size_t j = 0; cpn_is_plain && j < konst_.size(); ++j)
      cpn_is_plain = (konst_[j] == 0.0 && k_[j] == 1.0);
  }

  // --- pricing --------------------------------------------------------------------------------
  // SHARED RECIPROCALS (2026-09-09): every per-coupon ratio DF[s]/DF[e] is computed as DF[s]·INV[e] with
  // INV = 1/DF precomputed ONCE per DF evaluation (n_times reciprocals, ~250 on a desk bundle) instead of
  // one divide per sub-period (~7,000). The divide was the dominant per-coupon cost (13-15 cycle latency,
  // 4-cycle throughput); a multiply is 0.5/cycle. a·(1/b) differs from a/b by ≤ 1 ULP, so "bit-identical to
  // the templated kernel" becomes ~1e-16 RELATIVE parity (the T3 tests are at 1e-12). The hot callers
  // (CompiledBundleResidual, CompiledMultiCurveBook) pass their shared INV; the DF-only overloads below
  // compute it into per-batch scratch for everyone else (still n_times divides, never per coupon).
  //
  // NOTE on vectorisation: these pointer loops are NOT auto-vectorised by clang at -O3 -march=x86-64-v3
  // ("cannot identify array bounds": data-dependent gather indices) — verified with -Rpass-analysis on
  // 2026-09-09; the earlier "auto-vectorized gather" comments were wrong. With the reciprocal they are
  // load/multiply bound, which is close to the scalar floor; explicit SIMD is an E4.D decision.
  const Eigen::VectorXd& inverse_of(const Eigen::VectorXd& DF) const {
    inv_scratch_ = DF.cwiseInverse();  // alloc-free after first sizing
    return inv_scratch_;
  }

  // Per-coupon numerator num = sum_k w_k (DF[s_k]·INV[e_k] - 1). Returns a const ref into per-batch
  // scratch (no per-tick allocation); the reference is valid until the next call on THIS batch.
  const Eigen::VectorXd& num(const Eigen::VectorXd& DF) const { return num(DF, inverse_of(DF)); }
  const Eigen::VectorXd& num(const Eigen::VectorXd& DF, const Eigen::VectorXd& INV) const {
    const int n = static_cast<int>(subS.size());
    sub_.resize(n);
    const double* __restrict df = DF.data();
    const double* __restrict iv = INV.data();
    const int* __restrict ss = subS.data();
    const int* __restrict se = subE.data();
    double* __restrict out = sub_.data();
    for (int i = 0; i < n; ++i) out[i] = df[ss[i]] * iv[se[i]] - 1.0;
    if (!moments_.empty()) {  // moment brackets: ln(DF[a]·INV[b]) + ½·step·∫f² (+ ⅙·step3·∫f³), from set_state
      if (!state_set_) throw std::logic_error("BundleFloatBatch: set_state(x) must precede num/pv/rate on a batch with moment coupons");
      for (const auto& mc : moments_) out[mc.sub] = std::log(df[ss[mc.sub]] * iv[se[mc.sub]]) + mc.mom;
    }
    if (sub_is_identity) return sub_;  // R_sub == I: the reduction is a bitwise no-op
    if (SWAPS_REDUCE_LAYOUT == 0) { num_res_.noalias() = R_sub * sub_; return num_res_; }
    num_res_.resize(n_cpn_);  // segment-sum of the weighted sub-periods of each coupon (sub_w = R_sub's values)
    const double* __restrict sw = sub_w.data();
    const int* __restrict sb = sub_begin_.data();
    double* __restrict nr = num_res_.data();
    for (int c = 0; c < n_cpn_; ++c) { double acc = 0.0; for (int j = sb[c]; j < sb[c + 1]; ++j) acc += sw[j] * out[j]; nr[c] = acc; }
    return num_res_;
  }

  // Per-instrument float-leg PV. Returns a const ref into per-batch scratch (valid until the next call
  // on THIS batch) -- gen_pos_ and gen_neg_ are distinct objects, so model_rates can hold both at once.
  const Eigen::VectorXd& pv(const Eigen::VectorXd& DF) const { return pv(DF, inverse_of(DF)); }
  const Eigen::VectorXd& pv(const Eigen::VectorXd& DF, const Eigen::VectorXd& INV) const {
    // PERF RULE: what reaches R_cpn must be a materialized VectorXd -- handing Eigen's sparse*dense an
    // unevaluated gather/divide re-does that work per access (~1.28x slower, measured). On the identity
    // path we fuse the sub-period gather straight into the coupon vector, so the standard shape costs
    // exactly one materialized pass, as it did before this generalization.
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "pv() needs pay dates: this is a futures batch");
    Eigen::VectorXd& coupon = coupon_;  // reuse the per-batch scratch (no per-tick allocation)
    if (SWAPS_REDUCE_LAYOUT >= 1) {  // SEGMENT: each coupon's gather-product accumulates into its instrument's slot
      pv_res_.setZero(n_inst);
      const double* __restrict df = DF.data();
      const double* __restrict iv = INV.data();
      const int* __restrict p = pay.data();
      const double* __restrict kk = konst.data();
      const double* __restrict kv = k.data();
      const int* __restrict cb = cpn_begin_.data();
      double* __restrict out = pv_res_.data();
      if (!moments_.empty() || !sub_is_identity) {
        const Eigen::VectorXd& nm = num(DF, INV);
        const double* __restrict nn = nm.data();
        for (int i = 0; i < n_inst; ++i) { double acc = 0.0; for (int c = cb[i]; c < cb[i + 1]; ++c) acc += df[p[c]] * (nn[c] + kk[c]) * kv[c]; out[i] = acc; }
      } else {
        const int* __restrict ss = subS.data();
        const int* __restrict se = subE.data();
        if (cpn_is_plain)
          for (int i = 0; i < n_inst; ++i) { double acc = 0.0; for (int c = cb[i]; c < cb[i + 1]; ++c) acc += df[p[c]] * (df[ss[c]] * iv[se[c]] - 1.0); out[i] = acc; }
        else
          for (int i = 0; i < n_inst; ++i) { double acc = 0.0; for (int c = cb[i]; c < cb[i + 1]; ++c) acc += df[p[c]] * (df[ss[c]] * iv[se[c]] - 1.0 + kk[c]) * kv[c]; out[i] = acc; }
      }
      return pv_res_;
    }
    if (!moments_.empty() || !sub_is_identity) {  // moment coupons or a weighted/multi-sub-period batch: go through num()
      const Eigen::VectorXd& nm = num(DF, INV);  // materialised per-coupon numerator (sub_ or num_res_)
      coupon.resize(n_cpn_);
      const double* __restrict df = DF.data();
      const int* __restrict p = pay.data();
      const double* __restrict nn = nm.data();
      const double* __restrict kk = konst.data();
      const double* __restrict kv = k.data();
      double* __restrict out = coupon.data();
      for (int i = 0; i < n_cpn_; ++i) out[i] = df[p[i]] * (nn[i] + kk[i]) * kv[i];  // hand gather: no IndexedView temporaries
    }
    else if (sub_is_identity && cpn_is_plain) {  // standard shape: identical work to pre-generalization
      // Hand-written FUSED gather instead of Eigen's IndexedView (which materializes DF(pay)/DF(subS)/
      // DF(subE) into temporaries — allocations on the tick). Not auto-vectorised (see inverse_of).
      coupon.resize(n_cpn_);
      const double* __restrict df = DF.data();
      const double* __restrict iv = INV.data();
      const int* __restrict p = pay.data();
      const int* __restrict ss = subS.data();
      const int* __restrict se = subE.data();
      double* __restrict out = coupon.data();
      for (int i = 0; i < n_cpn_; ++i) out[i] = df[p[i]] * (df[ss[i]] * iv[se[i]] - 1.0);
    }
    else if (sub_is_identity) {
      coupon.resize(n_cpn_);
      const double* __restrict df = DF.data();
      const double* __restrict iv = INV.data();
      const int* __restrict p = pay.data();
      const int* __restrict ss = subS.data();
      const int* __restrict se = subE.data();
      const double* __restrict kk = konst.data();
      const double* __restrict kv = k.data();
      double* __restrict out = coupon.data();
      for (int i = 0; i < n_cpn_; ++i)
        out[i] = df[p[i]] * (df[ss[i]] * iv[se[i]] - 1.0 + kk[i]) * kv[i];
    }
    pv_res_.noalias() = R_cpn * coupon;
    return pv_res_;
  }

  // Per-instrument PV from a PRECOMPUTED per-coupon numerator `num_cpn` (== num(DF)). Lets the
  // Jacobian's value pass share the single sub-period gather with its derivative pass (d_pv_from_num)
  // instead of each re-gathering. BIT-IDENTICAL to pv(DF): on the plain path (konst == 0, k == 1) it
  // is DF[pay]*num_cpn, exactly the fused pv coupon; otherwise the same DF[pay]*(num+konst)*k form.
  const Eigen::VectorXd& pv_from_num(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF) const {
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "pv_from_num() needs pay dates");
    if (n_cpn_ == 0) {
      pv_res_.setZero(n_inst);
      return pv_res_;
    }
    Eigen::VectorXd& coupon = coupon_;
    coupon.resize(n_cpn_);
    {
      const double* __restrict df = DF.data();
      const int* __restrict p = pay.data();
      const double* __restrict nn = num_cpn.data();
      const double* __restrict kk = konst.data();
      const double* __restrict kv = k.data();
      double* __restrict out = coupon.data();
      if (cpn_is_plain)
        for (int i = 0; i < n_cpn_; ++i) out[i] = df[p[i]] * nn[i];
      else
        for (int i = 0; i < n_cpn_; ++i) out[i] = df[p[i]] * (nn[i] + kk[i]) * kv[i];
    }
    // (IndexedView form retired 2026-09-09: it materialised index temporaries on every Jacobian — E3-A2/A3.)
    if (SWAPS_REDUCE_LAYOUT >= 1) {
      pv_res_.resize(n_inst);
      const int* __restrict cb = cpn_begin_.data(); const double* __restrict cc = coupon.data(); double* __restrict out = pv_res_.data();
      for (int i = 0; i < n_inst; ++i) { double acc = 0.0; for (int c = cb[i]; c < cb[i + 1]; ++c) acc += cc[c]; out[i] = acc; }
      return pv_res_;
    }
    pv_res_.noalias() = R_cpn * coupon;
    return pv_res_;
  }

  // --- BATCHED pricing over a curve-state GRID (MC-exposure hot path, design R12 coupon-batch) --------
  // True iff this batch is the STANDARD shape (one unit-weight sub-period per coupon, no spread / nothing
  // realized / tau_pay==tau_index) — the only shape the batched grid reduce below handles. On that shape
  // sub-period i IS coupon i, so subS/subE index by coupon directly. Non-standard batches fall back to the
  // per-column pv() loop (the gather INDICES are still state-invariant, but konst/k/weights re-enter).
  bool std_shape() const { return sub_is_identity && cpn_is_plain; }

  // Per-instrument leg PV for a WHOLE grid of curve-states at once. DFg is the ROW-MAJOR DF grid
  // (n_times x n_states) from CompiledCurveSet::df_into. The coupon gather INDICES (pay/subS/subE/inst)
  // are state-invariant — only the DF VALUES change across states — so per coupon we gather its three rows
  // ONCE and let Eigen fuse divide/subtract/multiply across ALL states in a vectorized row op, then
  // ACCUMULATE it straight into its owning instrument's PV row (coupons of one instrument are a contiguous
  // block, so R_cpn is just a segment-sum):
  //     pv_grid.row(inst[i]) += DFg.row(pay[i]) ⊙ (DFg.row(subS[i]) ⊘ DFg.row(subE[i]) − 1)
  // Fusing the reduction into the gather avoids materializing the big n_cpn × n_states coupon grid and the
  // SpMM over it — pv_grid_ is only n_inst × n_states (stays cache-hot). Column j is == pv(DFg.col(j)) to
  // rounding. Const ref into per-batch scratch (valid until the next call on THIS batch); alloc-free after
  // warmup. STANDARD shape only — assert-guarded.
  const RowMatrixXd& pv_grid(const RowMatrixXd& DFg) const {
    assert(std_shape() && "pv_grid(): non-standard shape — caller must use the per-column pv() fallback");
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "pv_grid() needs pay dates: this is a futures batch");
    // The per-coupon denominator DF[subE] takes only n_times DISTINCT values, but there are n_cpn ≫ n_times
    // coupons — so invert the DF grid ONCE per tile (n_times reciprocals per state) and turn each coupon's
    // expensive divide into a cheap multiply (n_cpn mults per state). Bit-identical to DF[s]/DF[e] only up
    // to reciprocal rounding — a/b vs a·(1/b) differ by ≤1 ULP, far inside the col-0 rel-1e-9 check.
    inv_dfg_ = DFg.array().inverse();
    pv_grid_.setZero(n_inst, DFg.cols());
    const int* __restrict p = pay.data();
    const int* __restrict ss = subS.data();
    const int* __restrict se = subE.data();
    const int* __restrict in = inst.data();
    for (int i = 0; i < n_cpn_; ++i)
      pv_grid_.row(in[i]).array() +=
          DFg.row(p[i]).array() * (DFg.row(ss[i]).array() * inv_dfg_.row(se[i]).array() - 1.0);
    return pv_grid_;
  }

  // Per-instrument (per-future) rate = (num + realized)*inv_tau + convexity. Const ref into scratch.
  const Eigen::VectorXd& rate(const Eigen::VectorXd& DF) const { return rate(DF, inverse_of(DF)); }
  const Eigen::VectorXd& rate(const Eigen::VectorXd& DF, const Eigen::VectorXd& INV) const {
    if (sub_is_identity && moments_.empty()) {
      const int n = static_cast<int>(subS.size());
      rate_res_.resize(n);
      const double* __restrict df = DF.data();
      const double* __restrict iv = INV.data();
      const int* __restrict ss = subS.data();
      const int* __restrict se = subE.data();
      const double* __restrict rz = realized.data();
      const double* __restrict it = inv_tau.data();
      const double* __restrict cv = convexity.data();
      double* __restrict out = rate_res_.data();
      for (int i = 0; i < n; ++i) out[i] = (df[ss[i]] * iv[se[i]] - 1.0 + rz[i]) * it[i] + cv[i];
      return rate_res_;
    }
    rate_res_ = ((num(DF, INV).array() + realized.array()) * inv_tau.array() + convexity.array()).matrix();
    return rate_res_;
  }

  // --- analytic sensitivities (the ANALYTIC Jacobian's dr/dDF; no AAD in the hot loop) ----------
  // d(pv)/dDF, accumulated (with `sign`) into rows [row0, row0+n_inst) of `d`. For the generic coupon
  // pv = DF[pay]·A·k with A = Σ_k w_k(DF[s_k]/DF[e_k] − 1) + konst (the k-form):
  //     d pv / d DF[pay] = A·k
  //     d pv / d DF[s_k] = + DF[pay]·k·w_k / DF[e_k]
  //     d pv / d DF[e_k] = − DF[pay]·k·w_k·DF[s_k] / DF[e_k]²
  // tau_index vanishes into k, so at k=1, w=1, konst=0 (one sub-period, no spread,
  // tau_pay == tau_index) these are EXACTLY the pre-generalization partials.
  //
  // TWO passes, not one: d pv/d DF[pay] = A·k depends on the coupon's WHOLE numerator, i.e. on OTHER
  // DF entries, so A must be materialized before anything can be scattered to the pay column.
  //
  // `+=` (never `=`) is load-bearing -- two structural aliases are live and MUST accumulate:
  // consecutive averaged sub-periods share a registered time (e_k == s_{k+1}), and pay == e_k when
  // the forecast and discount curves coincide with no payment lag.
  // Takes the SAME PRECOMPUTED per-coupon numerator `num_cpn` (== num(DF)) that pv_from_num used, so
  // the sub-period gather is done ONCE per Jacobian call, not once for the value and again for the
  // derivative. A == num_cpn + konst is the k-form numerator; the pay-column term uses it and the
  // s/e-column terms use DF directly. BIT-IDENTICAL to the previous d_pv (which recomputed num(DF)).
  template <class Mat>  // any dense matrix (row-major preferred: the scatters are row-local)
  void d_pv_from_num(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF, Mat& d,
                     int row0, double sign) const {
    d_pv_from_num(num_cpn, DF, inverse_of(DF), d, row0, sign);
  }
  template <class Mat>
  void d_pv_from_num(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF, const Eigen::VectorXd& INV,
                     Mat& d, int row0, double sign) const {
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "d_pv_from_num() needs pay dates");
    for (int i = 0; i < n_cpn_; ++i)
      d(row0 + inst[i], pay[i]) += sign * (num_cpn[i] + konst[i]) * k[i];
    for (int j = 0; j < static_cast<int>(subS.size()); ++j) {
      const int i = sub_cpn[j], s = subS[j], e = subE[j];
      const double f = sign * DF[pay[i]] * k[i] * sub_w[j];
      const double ie = INV[e];
      d(row0 + inst[i], s) += f * ie;
      d(row0 + inst[i], e) += -f * DF[s] * ie * ie;
    }
    // moment brackets: d ln(DF[a]/DF[b]) / dDF = +INV[a] on a, −INV[b] on b (replacing the ratio partials above)
    for (const auto& mc : moments_) {
      const int i = mc.cpn, s = subS[mc.sub], e = subE[mc.sub];
      const double f = sign * DF[pay[i]] * k[i] * sub_w[mc.sub];
      const double ie = INV[e];
      d(row0 + inst[i], s) -= f * ie;                 // undo the ratio partials
      d(row0 + inst[i], e) -= -f * DF[s] * ie * ie;
      d(row0 + inst[i], s) += f * INV[s];             // log-bracket partials
      d(row0 + inst[i], e) += -f * ie;
    }
  }
  // DIRECT x-space derivative of the moment corrections (the part that is NOT a function of DF): for each moment
  // coupon c, ∂pv_inst/∂x_j += DF[pay_c]·k_c·w·∂mom_c/∂x_j on its support. Delivered through `add(inst, j, value)` so
  // the caller can scatter into its own Jacobian with the row's quotient / band factors. Precondition: set_state(x).
  template <class Add>
  void moment_direct_pv(const Eigen::VectorXd& DF, double sign, Add&& add) const {
    for (const auto& mc : moments_) {
      const int i = mc.cpn;
      const double f = sign * DF[pay[i]] * k[i] * sub_w[mc.sub];
      for (int j = 0; j < static_cast<int>(mc.support.size()); ++j) add(inst[i], mc.support[j], f * mc.dmom[j]);
    }
  }
  // Same for a futures batch: ∂rate_inst/∂x_j += inv_tau·w·∂mom_c/∂x_j.
  template <class Add>
  void moment_direct_rate(Add&& add) const {
    for (const auto& mc : moments_) {
      const int i = mc.cpn;
      const double f = inv_tau[i] * sub_w[mc.sub];
      for (int j = 0; j < static_cast<int>(mc.support.size()); ++j) add(inst[i], mc.support[j], f * mc.dmom[j]);
    }
  }
  // d(rate)/dDF for a futures batch (realized and convexity are constants -> zero derivative):
  //     d rate / d DF[s_k] = + w_k·inv_tau / DF[e_k]
  //     d rate / d DF[e_k] = − w_k·inv_tau·DF[s_k] / DF[e_k]²
  template <class Mat>
  void d_rate(const Eigen::VectorXd& DF, Mat& d, int row0) const { d_rate(DF, inverse_of(DF), d, row0); }
  template <class Mat>
  void d_rate(const Eigen::VectorXd& DF, const Eigen::VectorXd& INV, Mat& d, int row0) const {
    for (int j = 0; j < static_cast<int>(subS.size()); ++j) {
      const int i = sub_cpn[j], s = subS[j], e = subE[j], r = row0 + inst[i];
      const double f = inv_tau[i] * sub_w[j];
      const double ie = INV[e];
      d(r, s) += f * ie;
      d(r, e) += -f * DF[s] * ie * ie;
    }
    for (const auto& mc : moments_) {  // log-bracket partials for moment windows (see d_pv_from_num)
      const int i = mc.cpn, s = subS[mc.sub], e = subE[mc.sub], r = row0 + inst[i];
      const double f = inv_tau[i] * sub_w[mc.sub];
      const double ie = INV[e];
      d(r, s) += -f * ie + f * INV[s];
      d(r, e) += f * DF[s] * ie * ie - f * ie;
    }
  }

 private:
  void push_sub(CompiledCurveSet& cs, int fc, double s_t, double e_t, double w) {
    ss_.push_back(cs.reg(fc, s_t));
    se_.push_back(cs.reg(fc, e_t));
    sc_.push_back(n_cpn_);  // the coupon about to be pushed
    sw_.push_back(w);
  }
  void push_obs(CompiledCurveSet& cs, int fc, const RateObservation& o) {
    // GUARD: the compiled batch computes the ARITHMETIC sum Σ w_k(DF/DF−1); it cannot represent the
    // COMPOUNDED product (RFR lookback/lockout). Those are pricing coupons, never calibration
    // instruments, so they price through the templated float_coupon_pv instead -- reject them here
    // rather than silently summing what should be multiplied.
    if (o.compounded)
      throw std::invalid_argument(
          "CompiledBook: compounded (RFR lookback/lockout) observation cannot use the arithmetic "
          "W-cache batch; price it through the templated kernel");
    if (!o.fixing_schedule.empty() && !o.resolved)
      throw std::runtime_error(
          "CompiledBook: a fixings-resolvable observation was compiled before resolution against a fixing "
          "table (its realized part would silently be zero) -- attach fixings / set the evaluation date first");
    const bool weighted = !o.weight.empty();
    if (o.fixing_step > 0.0) {  // MOMENT path: one bracket + a precomputed quadratic form over the window
      if (o.sub_start.size() != 1 || (weighted && o.weight.size() != 1))
        throw std::invalid_argument("CompiledBook: a moment-path observation has exactly one window (one optional weight)");
      MomentCoupon mc;
      mc.cpn = n_cpn_;
      mc.sub = static_cast<int>(ss_.size());
      mc.step = o.fixing_step; mc.step3 = o.fixing_step3;
      push_sub(cs, fc, o.sub_start[0], o.sub_end[0], weighted ? o.weight[0] : 1.0);  // the day-count ratio rides R_sub
      // The SAME composite 2-point Gauss rule as pricing::curve_forward_sq_integral (subdiv = 32).
      const double a = o.sub_start[0], b = o.sub_end[0];
      constexpr int subdiv = 32;
      const double gx = 0.5773502691896257, H = (b - a) / subdiv;
      std::vector<double> nodes; std::vector<double> wts;
      for (int s = 0; s < subdiv; ++s) {
        const double mid = a + (s + 0.5) * H, h = 0.5 * H;
        for (int sg = -1; sg <= 1; sg += 2) { nodes.push_back(mid + sg * gx * h); wts.push_back(h); }
      }
      const Eigen::MatrixXd P = cs.forward_rows(fc, nodes);  // nodes × n_knots
      for (int j = 0; j < P.cols(); ++j)
        if ((P.col(j).array() != 0.0).any()) mc.support.push_back(j);
      const int S = static_cast<int>(mc.support.size());
      Eigen::MatrixXd Ps(static_cast<int>(nodes.size()), S);
      for (int j = 0; j < S; ++j) Ps.col(j) = P.col(mc.support[j]);
      mc.Q = Eigen::MatrixXd::Zero(S, S);
      for (int kq = 0; kq < Ps.rows(); ++kq) mc.Q.noalias() += wts[kq] * (Ps.row(kq).transpose() * Ps.row(kq));
      if (mc.step3 > 0.0) {  // cubic term: 8 panels x 2 nodes (curve_forward_cube_integral's rule), same support
        constexpr int subdiv3 = 8;
        const double H3 = (b - a) / subdiv3;
        std::vector<double> n3, w3;
        for (int s = 0; s < subdiv3; ++s) {
          const double mid = a + (s + 0.5) * H3, h = 0.5 * H3;
          for (int sg = -1; sg <= 1; sg += 2) { n3.push_back(mid + sg * gx * h); w3.push_back(h); }
        }
        const Eigen::MatrixXd P3 = cs.forward_rows(fc, n3);
        mc.Psi.resize(static_cast<int>(n3.size()), S);
        for (int j = 0; j < S; ++j) mc.Psi.col(j) = P3.col(mc.support[j]);
        mc.wq = Eigen::Map<const Eigen::VectorXd>(w3.data(), static_cast<int>(w3.size()));
      }
      moments_.push_back(std::move(mc));
      return;
    }
    for (std::size_t j = 0; j < o.sub_start.size(); ++j)
      push_sub(cs, fc, o.sub_start[j], o.sub_end[j], weighted ? o.weight[j] : 1.0);
  }
  void push_coupon(int pay_idx, double konst, double kk, double rz, double inv_tau, double conv) {
    p_.push_back(pay_idx);
    konst_.push_back(konst);
    k_.push_back(kk);
    rz_.push_back(rz);
    it_.push_back(inv_tau);
    cv_.push_back(conv);
    row_.push_back(n_inst);
    ++n_cpn_;
  }
  static Eigen::SparseMatrix<double> build(const std::vector<int>& row, const std::vector<double>& val,
                                           int n_rows) {
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(row.size());
    for (int j = 0; j < static_cast<int>(row.size()); ++j) trip.emplace_back(row[j], j, val[j]);
    Eigen::SparseMatrix<double> M(n_rows, static_cast<int>(row.size()));
    M.setFromTriplets(trip.begin(), trip.end());
    return M;
  }

  int n_cpn_ = 0;
  std::vector<int> ss_, se_, sc_, p_, row_;
  std::vector<int> cpn_begin_, sub_begin_;                // segment offsets (coupons of an instrument; subs of a coupon)
  std::vector<double> sw_, konst_, k_, rz_, it_, cv_;
  // Per-batch reusable scratch (sized on first use) so pv/num/rate never allocate in the hot loop.
  // Each returns a const ref into these; because gen_pos_/gen_neg_ are distinct batch objects, two
  // results (one per batch) are simultaneously live in model_rates without aliasing.
  mutable Eigen::VectorXd coupon_, sub_, num_res_, pv_res_, rate_res_, inv_scratch_;
  mutable std::vector<MomentCoupon> moments_;  // per-coupon precomputed quadratic forms (set_state fills mom/dmom)
  mutable bool state_set_ = false;
  mutable Eigen::VectorXd state_x_;  // set_state memo key
  // Batched-grid scratch (pv_grid): pv_grid_ is n_inst x n_states (coupons accumulated straight into their
  // instrument row); inv_dfg_ is the per-tile reciprocal DF grid (n_times x n_states). Sized on first use.
  mutable RowMatrixXd pv_grid_, inv_dfg_;
};

// Fixed-leg annuities of N instruments discounting curve `dc`: ann = sum(tau*DF[pay]).
struct BundleFixedLegs {
  Eigen::VectorXi pay, inst;
  Eigen::VectorXd tau;
  Eigen::SparseMatrix<double> R;
  int n_inst = 0;

  // One instrument = one fixed leg of generic coupons, discounting `dc`. FX `scale` (default 1) folds
  // into tau (× 1.0 exact), so a foreign annuity is converted with no separate field and d_annuity's
  // tau_i partial stays correct.
  void add(CompiledCurveSet& cs, int dc, const std::vector<FixedCoupon>& leg) {
    for (const auto& c : leg) {
      p_.push_back(cs.reg(dc, c.pay));
      t_.push_back(c.tau * c.scale);
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
    cpn_begin_.assign(n_inst + 1, 0);
    for (std::size_t c = 0; c < row_.size(); ++c) { assert(c == 0 || row_[c] >= row_[c - 1]); ++cpn_begin_[row_[c] + 1]; }
    for (int i = 0; i < n_inst; ++i) cpn_begin_[i + 1] += cpn_begin_[i];
  }
  const Eigen::VectorXd& annuity(const Eigen::VectorXd& DF) const {
    const int n = static_cast<int>(pay.size());
    disc_.resize(n);  // per-batch reusable scratch (no per-tick allocation); auto-vectorized gather
    const double* __restrict df = DF.data();
    const int* __restrict p = pay.data();
    const double* __restrict t = tau.data();
    double* __restrict d = disc_.data();
    if (SWAPS_REDUCE_LAYOUT >= 1) {
      ann_res_.setZero(n_inst);
      const int* __restrict cb = cpn_begin_.data(); double* __restrict out = ann_res_.data();
      for (int i = 0; i < n_inst; ++i) { double acc = 0.0; for (int c = cb[i]; c < cb[i + 1]; ++c) acc += t[c] * df[p[c]]; out[i] = acc; }
      return ann_res_;
    }
    for (int i = 0; i < n; ++i) d[i] = t[i] * df[p[i]];
    ann_res_.noalias() = R * disc_;
    return ann_res_;
  }
  // BATCHED annuity over a curve-state GRID (MC-exposure hot path, design R12 coupon-batch). DFg is the
  // ROW-MAJOR DF grid (n_times x n_states). The pay INDICES are state-invariant; only DF values change, so
  // gather each coupon's pay row ONCE, scale by tau across all states, and ACCUMULATE straight into its
  // instrument's annuity row:  ann_grid.row(inst[i]) += tau[i]·DFg.row(pay[i]). Column j equals
  // annuity(DFg.col(j)) to GEMM rounding. Const ref into per-batch scratch; alloc-free after warmup.
  const RowMatrixXd& annuity_grid(const RowMatrixXd& DFg) const {
    const int n = static_cast<int>(pay.size());
    ann_grid_.setZero(n_inst, DFg.cols());  // accumulate coupons straight into their instrument row (R is
    const int* __restrict p = pay.data();   // a segment-sum: one instrument's coupons are contiguous), so
    const int* __restrict in = inst.data(); // no n_cpn × n_states disc grid and no SpMM — ann_grid_ stays
    const double* __restrict t = tau.data();// cache-hot at n_inst × n_states.
    for (int i = 0; i < n; ++i) ann_grid_.row(in[i]).array() += t[i] * DFg.row(p[i]).array();
    return ann_grid_;
  }
  // d(annuity)/dDF[pay_i] = tau_i, accumulated into rows [row0, row0+n_inst).
  template <class Mat>
  void d_annuity(Mat& d, int row0) const {
    for (int i = 0; i < static_cast<int>(pay.size()); ++i) d(row0 + inst[i], pay[i]) += tau[i];
  }

 private:
  std::vector<int> p_, row_;
  std::vector<double> t_;
  std::vector<int> cpn_begin_;              // segment offsets (coupons of an instrument)
  mutable Eigen::VectorXd disc_, ann_res_;  // per-batch reusable scratch for annuity()
  mutable RowMatrixXd ann_grid_;  // reusable scratch for annuity_grid()
};

}  // namespace swaps::pricing
