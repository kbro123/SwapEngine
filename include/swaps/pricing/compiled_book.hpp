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
#include <set>
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
    eval_x_.assign(specs.size(), Eigen::VectorXd());
  }

  // EXPERIMENT (exp/piecewise-linear-w): PIECEWISE-LINEAR mode. A curve with a value-dependent region takes
  // its W at a STATE (integral_weight_matrix_at) instead of refusing times past its linear horizon; the W is
  // then exact throughout that state's branch-pattern cell. The owner re-points the state and re-runs
  // finalize() when the pattern changes. Call before finalize(). Fully linear curves are unaffected.
  void enable_pwl() { pwl_ = true; }
  bool pwl() const { return pwl_; }
  bool value_dependent(int c) const {
    for (const auto& r : specs_[c].regions)
      if (!curve::scheme_is_linear(r.scheme)) return true;
    return false;
  }
  // The state a value-dependent curve's W is taken at (its interp-knot segment of the stacked x).
  void set_eval_state(const Eigen::VectorXd& x) {
    for (int c = 0; c < static_cast<int>(specs_.size()); ++c)
      if (value_dependent(c)) eval_x_[c] = x.segment(knot_offset_[c], specs_[c].n_interp_knots());
  }
  int knot_offset(int c) const { return knot_offset_[c]; }
  const std::vector<CurveStructure>& specs() const { return specs_; }

  // ---- ANALYTIC RE-TAKE (EXPERIMENT, exp/piecewise-linear-w) ------------------------------------------
  // For a curve c with exactly one value-dependent (MonotoneCubic) region, every registered time t obeys
  //     integral_c(t) = L_t·x_c + B_t·m        (m = the region's N FILTERED tangents)
  // with L_t, B_t structure-only, and m = Φ_P·G·x_c (Φ_P decoded from the branch pattern). So when nodes
  // j1..jk change formula, W's c-block changes by exactly ΔW = B[:, j]·ΔM[j, :] -- rank k, only on rows
  // whose curve's ANCESTRY includes c (a spread curve's row carries its base's row at the same time) and
  // whose B row is nonzero (t past the region start). Applied in place to W_ and the ancestry blocks.
  struct PwlRow { int g, d, blk, blk_row, blk_col; };
  struct PwlCurve {
    int c = -1, off = 0, ni = 0, N = 0;
    std::vector<PwlRow> rows;
    Eigen::MatrixXd B;  // rows x N
    Eigen::MatrixXd L;  // rows x ni: d integral / d x_c holding the filtered tangents fixed
    Eigen::MatrixXd D;  // rows x ni: the last applied delta (scratch, sized once; the owner mirrors it into W^T)
  };
  // Structure-only; call after finalize(). Returns the pwl index, or -1 if c is not analytic-capable.
  int pwl_prepare(int c) {
    if (!pwl_ || !value_dependent(c)) return -1;
    const auto mods = specs_[c].modules();
    const int ni = specs_[c].n_interp_knots();
    auto crv = curve::make_modular_curve<double>(mods);
    if (crv.n_value_dependent_regions() != 1) return -1;
    crv.set_forwards(Eigen::VectorXd::Constant(ni, 0.03));
    const int N = crv.pwl_nodes();
    PwlCurve P;
    P.c = c; P.off = knot_offset_[c]; P.ni = ni; P.N = N;
    // One AAD pass over [x_c (ni) | m (N)] with the filtered tangents replaced by independent duals.
    using D = ad::Dual;
    const int w = ni + N;
    std::vector<D> xd(ni), md(N);
    for (int k = 0; k < ni; ++k) xd[k] = D(0.03, w, k);
    for (int j = 0; j < N; ++j) md[j] = D(0.0, w, ni + j);
    auto cd = curve::make_modular_curve<D>(mods);
    cd.set_tangent_override(md.data());
    cd.set_forwards(xd);
    std::vector<Eigen::RowVectorXd> Brows, Lrows;
    for (int g = 0; g < static_cast<int>(pts_.size()); ++g) {
      const int d = pts_[g].first;
      bool anc = false;
      for (int a = d; a >= 0 && !anc; a = specs_[a].base) anc = (a == c);
      if (!anc) continue;
      const D I = cd.integral(pts_[g].second);
      if (I.derivatives().size() != w) continue;
      const Eigen::RowVectorXd b = I.derivatives().tail(N).transpose();
      if (b.cwiseAbs().maxCoeff() == 0.0) continue;  // before the region: pattern-independent row
      PwlRow r{g, d, -1, -1, -1};
      for (int k = 0; k < static_cast<int>(blk_rows_[d].size()); ++k)
        if (blk_rows_[d][k] == g) { r.blk_row = k; break; }
      for (int b2 = 0; b2 < static_cast<int>(blk_[d].size()); ++b2)
        if (blk_[d][b2].off <= P.off && P.off + ni <= blk_[d][b2].off + blk_[d][b2].nk) {
          r.blk = b2;
          r.blk_col = P.off - blk_[d][b2].off;
          break;
        }
      if (r.blk < 0 || r.blk_row < 0) return -1;  // layout not understood: stay on the AAD re-take
      P.rows.push_back(r);
      Brows.push_back(b);
      Lrows.push_back(I.derivatives().head(ni).transpose());
    }
    P.B.resize(static_cast<int>(Brows.size()), N);
    P.L.resize(static_cast<int>(Brows.size()), ni);
    for (int r = 0; r < static_cast<int>(Brows.size()); ++r) { P.B.row(r) = Brows[r]; P.L.row(r) = Lrows[r]; }
    P.D.setZero(static_cast<int>(Brows.size()), ni);
    pwlc_.push_back(std::move(P));
    return static_cast<int>(pwlc_.size()) - 1;
  }
  // FULL reset of c's affected rows to the cell of formula rows M (N x ni): W = L + B·M, exactly the recorded
  // pattern's map. Used instead of an AAD re-take through the filter, which at a DEGENERATE state (a flat curve:
  // S == m_raw == 0) follows the `correction != m` VALUE test and silently takes the neighbouring cell's
  // derivative -- a W that disagrees with the recorded pattern (found: a flat cold seed broke every later update).
  void pwl_set(int pc, const Eigen::Ref<const Eigen::MatrixXd>& M) {
    PwlCurve& P = pwlc_[pc];
    for (int r = 0; r < static_cast<int>(P.rows.size()); ++r)
      P.D.row(r) = P.L.row(r) + P.B.row(r) * M - W_.row(P.rows[r].g).segment(P.off, P.ni);
    apply_delta(P);
  }
  // ΔW = B[:, nodes]·dM (dM: k x ni, row q the change of node nodes[q]'s formula·G). Allocation-free.
  void pwl_rank_update(int pc, const std::vector<int>& nodes, const Eigen::Ref<const Eigen::MatrixXd>& dM) {
    PwlCurve& P = pwlc_[pc];
    P.D.setZero();
    for (int q = 0; q < static_cast<int>(nodes.size()); ++q) P.D.noalias() += P.B.col(nodes[q]) * dM.row(q);
    apply_delta(P);
  }
  void apply_delta(PwlCurve& P) {
    for (int r = 0; r < static_cast<int>(P.rows.size()); ++r) {
      const PwlRow& pr = P.rows[r];
      W_.row(pr.g).segment(P.off, P.ni) += P.D.row(r);
      blk_[pr.d][pr.blk].W.row(pr.blk_row).segment(pr.blk_col, P.ni) += P.D.row(r);
    }
  }
  const PwlCurve& pwl_curve(int pc) const { return pwlc_[pc]; }

  // ---- THE PIECEWISE-LINEAR TRACKER (moved here from the calibration router, 2026-09-22) -------------------
  // One per value-dependent curve. Phase 2 of the tier: the region's filter inputs z are LINEAR in x, so G =
  // d(prefilter)/dx is taken ONCE (one AAD pass) and every sync reads the branch pattern off z = G·x with the
  // region's own recorder (pattern_from_prefilter) -- no curve rebuild. When a node's pattern changes, its
  // formula row (the region's node_formula, over z) times G is the new row of M = Φ_P·G, and W moves by the
  // rank-k update ΔW = B·ΔM (pwl_rank_update). Nothing here names a scheme: the region describes its cells.
  struct PwlTracker {
    int c = -1, off = 0, ni = 0, npb = 0;
    curve::ModularCurve<double> crv;
    std::vector<unsigned char> cur, cached;
    Eigen::MatrixXd G;   // d(filter inputs)/dx: structure-only
    Eigen::VectorXd z;   // G·x scratch
    int pc = -1, N = 0;  // analytic re-take structure (-1: this curve re-takes W by AAD)
    std::vector<double> phi;
    Eigen::MatrixXd M, dM;  // current formula rows Φ_P·G (N x ni); the change scratch
    Eigen::RowVectorXd row;
    std::vector<int> nodes;
    bool dirty = false;
  };
  // Build the trackers (after finalize, on a pwl-enabled set).
  void pwl_init() {
    trk_.clear();
    for (int k = 0; k < static_cast<int>(specs_.size()); ++k) {
      if (!value_dependent(k)) continue;
      const int ni = specs_[k].n_interp_knots();
      PwlTracker t;
      t.c = k; t.off = knot_offset_[k]; t.ni = ni;
      t.crv = curve::make_modular_curve<double>(specs_[k].modules());
      // G = d(prefilter)/dx, ONE AAD pass: the filter inputs precede the filter, so this is exact at any x.
      auto cd = curve::make_modular_curve<ad::Dual>(specs_[k].modules());
      cd.set_forwards(ad::seed(Eigen::VectorXd::Constant(ni, 0.03)));
      std::vector<ad::Dual> z;
      cd.prefilter_into(z);
      t.G.resize(static_cast<int>(z.size()), ni);
      for (int i = 0; i < static_cast<int>(z.size()); ++i)
        t.G.row(i) = z[static_cast<std::size_t>(i)].derivatives().size() == ni
                         ? Eigen::RowVectorXd(z[static_cast<std::size_t>(i)].derivatives().transpose())
                         : Eigen::RowVectorXd::Zero(ni);
      t.z.resize(t.G.rows());
      t.crv.set_forwards(Eigen::VectorXd::Constant(ni, 0.03));  // fixes each region's node spacing once
      t.pc = pwl_prepare(k);
      if (t.pc >= 0) {
        t.N = t.crv.pwl_nodes();
        t.npb = t.crv.pwl_pattern_bytes();
        t.M.setZero(t.N, ni);
        t.dM.resize(t.N, ni);
        t.phi.assign(static_cast<std::size_t>(t.crv.pwl_prefilter_size()), 0.0);
        t.row.resize(ni);
        t.nodes.reserve(static_cast<std::size_t>(t.N));
      }
      trk_.push_back(std::move(t));
    }
  }
  // Keep W in step with x's branch pattern. Per call: z = G·x and the region's recorder on doubles. On a
  // change: the ANALYTIC rank-k update (nodes whose formula changed) when every changed curve supports it,
  // else a full re-take. The owner supplies the three effects on ITS derived data (it mirrors W's deltas):
  //   rank_update(pc, nodes, dM)   set(pc, M)   full_retake()  -- the last also re-seats every analytic curve.
  // Returns true iff W changed. The first sync is always a full re-take.
  template <class RankUpdate, class Set, class FullRetake>
  bool pwl_sync(const Eigen::VectorXd& x, RankUpdate&& rank_update, Set&& set, FullRetake&& full_retake) {
    bool changed = false, analytic = pwl_synced_;
    for (auto& t : trk_) {
      t.z.noalias() = t.G * x.segment(t.off, t.ni);  // the one-pass linear map to the filter inputs
      t.crv.pattern_from_prefilter(t.z, t.cur);
      t.dirty = (t.cur != t.cached);
      if (t.dirty) {
        changed = true;
        if (t.pc < 0 || t.cur.size() != t.cached.size()) analytic = false;
      }
    }
    if (pwl_synced_ && !changed) return false;
    if (analytic) {
      for (auto& t : trk_) {
        if (!t.dirty) continue;
        t.nodes.clear();
        for (int j = 0; j < t.N; ++j) {
          bool same = true;
          for (int b = 0; b < t.npb && same; ++b) same = t.cur[static_cast<std::size_t>(j * t.npb + b)] == t.cached[static_cast<std::size_t>(j * t.npb + b)];
          if (same) continue;
          formula_row(t, t.cur, j);
          t.dM.row(static_cast<int>(t.nodes.size())) = t.row - t.M.row(j);
          t.M.row(j) = t.row;
          t.nodes.push_back(j);
        }
        rank_update(t.pc, t.nodes, t.dM.topRows(static_cast<int>(t.nodes.size())));
        t.cached = t.cur;  // same size: no allocation
      }
      ++pwl_analytic_;
    } else {
      // Full reset. An AAD re-take only for curves the analytic path does not cover; every analytic-capable
      // curve is then SET to its recorded pattern's W (L + B·M), which stays exact at degenerate states.
      bool need_aad = false;
      for (const auto& t : trk_) need_aad = need_aad || t.pc < 0;
      if (need_aad) full_retake();
      for (auto& t : trk_) {
        t.cached = t.cur;
        if (t.pc < 0) continue;
        for (int j = 0; j < t.N; ++j) { formula_row(t, t.cached, j); t.M.row(j) = t.row; }
        set(t.pc, t.M);
      }
      pwl_synced_ = true;
    }
    ++pwl_rebuilds_;
    if (pwl_stats_) {  // opt-in diagnostics: a std::set insert allocates, so it is OFF unless asked for
      std::vector<unsigned char> key;
      for (const auto& t : trk_) key.insert(key.end(), t.cached.begin(), t.cached.end());
      pwl_seen_.insert(std::move(key));
    }
    return true;
  }
  int pwl_rebuilds() const { return pwl_rebuilds_; }
  int pwl_analytic() const { return pwl_analytic_; }
  int pwl_distinct() const { return static_cast<int>(pwl_seen_.size()); }
  void enable_pwl_stats() { pwl_stats_ = true; }

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
    // BLOCK-SPARSE form of W for the per-tick DF stage (E4.D "per-curve W ancestry", 2026-09-10): a curve's
    // log-DF row is nonzero only on its OWN knot block and its base chain's, so the dense n_times x n_knots
    // GEMV multiplied zeros for every other curve (the desk: 857 times x 67 knots, ~2.8x the nonzeros --
    // the DF stage was 73 % of a Newton step there). Per curve: its registered rows, and one dense
    // (rows x n_knots_a) block per ancestry curve a; df_into sums the blocks and scatters. W_ stays dense
    // for the Jacobian's transposed product, rowsum_ and the batched (matrix) overloads.
    blk_rows_.assign(specs_.size(), {});
    blk_.assign(specs_.size(), {});
    blk_contig_.assign(specs_.size(), -1);
    int max_rows = 0;
    for (int c = 0; c < static_cast<int>(specs_.size()); ++c) {
      if (rows[c].empty()) continue;
      blk_rows_[c] = rows[c];
      max_rows = std::max(max_rows, static_cast<int>(rows[c].size()));
      // Rows registered in one unbroken ascending run (every single-curve bundle) write straight into `out`.
      bool contig = true;
      for (std::size_t k = 1; k < rows[c].size() && contig; ++k) contig = rows[c][k] == rows[c][k - 1] + 1;
      if (contig) blk_contig_[c] = rows[c].front();
      // The ancestry's knot blocks, MERGED where they are adjacent in the state (a spread chain c -> c-1 -> ... -> 0
      // is one contiguous prefix): one GEMV per maximal column range, so a deep chain costs one call, not one per
      // ancestor (36 tiny GEMVs on the 8-curve chain cost more in call overhead than the zeros they skipped).
      std::vector<std::pair<int, int>> ranges;  // [off, off + nk) per ancestry curve
      for (int a = c; a >= 0; a = specs_[a].base)
        if (specs_[a].n_knots() > 0) ranges.emplace_back(knot_offset_[a], knot_offset_[a] + specs_[a].n_knots());
      std::sort(ranges.begin(), ranges.end());
      std::vector<std::pair<int, int>> merged;
      for (const auto& r : ranges) {
        if (!merged.empty() && merged.back().second == r.first) merged.back().second = r.second;
        else merged.push_back(r);
      }
      for (const auto& r : merged) {
        const int off = r.first, nk = r.second - r.first;
        Eigen::MatrixXd B(static_cast<int>(rows[c].size()), nk);
        for (int k = 0; k < static_cast<int>(rows[c].size()); ++k) B.row(k) = W_.row(rows[c][k]).segment(off, nk);
        blk_[c].push_back({off, nk, std::move(B)});
      }
    }
    blk_tmp_.resize(max_rows);
  }

  Eigen::VectorXd df(const Eigen::VectorXd& x) const {
    Eigen::VectorXd out;
    df_into(x, out);
    return out;
  }
  // Allocation-free DF: writes exp(-W_all x) into the caller's `out` scratch (df(x) calls this, so the two are
  // identical). Block-sparse: each curve's rows sum their ancestry blocks only (see finalize); the batched
  // matrix overloads below still use the dense W, so a column of those agrees with this to rounding, not bit.
  void df_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const {
    out.resize(static_cast<int>(pts_.size()));
    for (std::size_t c = 0; c < blk_.size(); ++c) {
      const auto& rows = blk_rows_[c];
      if (rows.empty()) continue;
      const int m = static_cast<int>(rows.size());
      auto acc = blk_contig_[c] >= 0 ? out.segment(blk_contig_[c], m) : blk_tmp_.head(m);
      bool first = true;
      for (const auto& b : blk_[c]) {
        if (first) { acc.noalias() = b.W * x.segment(b.off, b.nk); first = false; }
        else acc.noalias() += b.W * x.segment(b.off, b.nk);
      }
      if (first) acc.setZero();
      if (blk_contig_[c] < 0) {
        double* __restrict o = out.data();
        const double* __restrict a = acc.data();
        const int* __restrict r = rows.data();
        for (int k = 0; k < m; ++k) o[r[k]] = a[k];
      }
    }
    out = (-out.array()).exp();
  }
  // BATCHED DF (hot-path design R12): a MATRIX of curve-states X (n_knots x n_states) -> DF grid
  // (n_times x n_states) = exp(-W_all·X), one GEMM + one vectorized exp. This is the exposure/MC lever:
  // repricing 10k paths x 100 nodes shares ONE W·X matmul instead of N_states separate matvecs. Column j
  // equals df_into(X.col(j), .) to rounding (dense GEMM vs the block-sparse vector path). Alloc-free after
  // the first sizing of `out`.
  void df_into(const Eigen::MatrixXd& X, Eigen::MatrixXd& out) const {
    out.noalias() = W_ * X;
    out = (-out.array()).exp();
  }
  // ROW-MAJOR batched DF grid (design R12 coupon-batch): the SAME exp(-W_all·X) as the column-major
  // overload above (column j equal to df_into(X.col(j),·) to rounding), but stored ROW-MAJOR so that each
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
  // The breakpoints of curve c's forward (its regions' pieces -- de Boor breakpoints for a B-spline -- its
  // turn edges, and its base chain's), sorted unique. Setup only: the moment path's quadratic forms are
  // built on knot-aligned Gauss panels between these (cashflows.hpp moment_gauss_nodes), so the compiled
  // and templated kernels integrate the SAME nodes.
  std::vector<double> pieces(int c) const {
    std::vector<double> out;
    for (int k = c; k >= 0; k = specs_[k].base) {
      auto crv = curve::make_modular_curve<double>(specs_[k].modules());
      crv.set_forwards(Eigen::VectorXd::Zero(specs_[k].n_interp_knots()));
      const std::vector<double> p = crv.pieces();
      out.insert(out.end(), p.begin(), p.end());
      for (const auto& w : specs_[k].turns) { out.push_back(w.start); out.push_back(w.end); }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

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
    if (pwl_ && value_dependent(c)) {  // EXPERIMENT: W of the state's branch-pattern cell
      const Eigen::VectorXd xs = eval_x_[c].size() == ni ? eval_x_[c] : Eigen::VectorXd::Constant(ni, 0.03);
      W.middleCols(knot_offset_[c], ni) = integral_weight_matrix_at(specs_[c].modules(), times, xs);
    } else {
      W.middleCols(knot_offset_[c], ni) =
          integral_weight_matrix(specs_[c].modules(), times);  // one W-cache, any region layout
    }
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
  bool pwl_ = false;                     // EXPERIMENT: value-dependent curves take W at eval_x_
  std::vector<Eigen::VectorXd> eval_x_;  // EXPERIMENT: per-curve W evaluation state (empty = default seed)
  std::vector<PwlCurve> pwlc_;           // EXPERIMENT: analytic re-take structure per value-dependent curve
  std::vector<PwlTracker> trk_;          // the tier's trackers (pwl_init)
  bool pwl_synced_ = false, pwl_stats_ = false;
  int pwl_rebuilds_ = 0, pwl_analytic_ = 0;  // syncs that changed W; of which analytic rank-k updates
  std::set<std::vector<unsigned char>> pwl_seen_;  // opt-in diagnostics: distinct cells visited
  // M's row j for pattern `pat`: the region's own formula for node j over its prefilter inputs, times G.
  static void formula_row(PwlTracker& t, const std::vector<unsigned char>& pat, int j) {
    t.crv.pwl_node_formula(j, pat.data() + j * t.npb, t.phi.data());
    t.row.noalias() = Eigen::Map<const Eigen::RowVectorXd>(t.phi.data(), static_cast<Eigen::Index>(t.phi.size())) * t.G;
  }
  std::map<std::pair<int, double>, int> idx_;
  std::vector<std::pair<int, double>> pts_;  // global index -> (curve, time)
  Eigen::MatrixXd W_;
  // Block-sparse W for df_into (finalize): per curve its global rows and one dense block per ancestry curve.
  struct AncestryBlock { int off, nk; Eigen::MatrixXd W; };
  std::vector<std::vector<int>> blk_rows_;
  std::vector<std::vector<AncestryBlock>> blk_;
  std::vector<int> blk_contig_;  // first global row when a curve's rows are one ascending run (no scatter), else -1
  mutable Eigen::VectorXd blk_tmp_;
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
//
// WHERE A LEG'S VALUE GOES (2026-09-22): the derivative scatters and the moment direct terms take a LegMap policy that
// says, per leg (batch instrument), which output row it lands on and with what sign. The two policies:
//   LegOffset{row0, sign} -- leg i -> row0 + i, one sign for the batch (the book twin, the tests);
//   LegTable{row, sign}   -- leg i -> row[i], sign[i]: several signed legs of ONE quotient row in ONE batch, which is
//                            how CompiledBundleResidual folds +bench/-fwd/+mtm into a single float batch (no padded
//                            empty legs, no per-kind batches).
// Both are indexed by COUPON on the hot path (one load per coupon, exactly the old `row0 + inst[c]`): the batch
// hands LegOffset its per-coupon leg array, and a LegTable is the caller's per-COUPON row/sign arrays
// (BundleFloatBatch::coupon_leg() maps a coupon to its leg so a caller can build them once).
struct LegOffset {
  int row0; double sign; const int* inst = nullptr;
  int row(int c) const { return row0 + inst[c]; }
  double sgn(int) const { return sign; }
};
struct LegTable {
  const int* rows; const double* signs;  // per COUPON
  int row(int c) const { return rows[c]; }
  double sgn(int c) const { return signs[c]; }
};
struct BundleFloatBatch {
  // Per sub-period (every coupon of every instrument, flattened).
  Eigen::VectorXi subS, subE, sub_cpn;  // sub_cpn = owning coupon
  Eigen::VectorXd sub_w;                // w_k (also carried as R_sub's values)
  Eigen::SparseMatrix<double> R_sub;    // n_coupons x n_subs, values = w_k
  // Per coupon.
  Eigen::VectorXi pay, inst;                       // inst = owning instrument; pay < 0 for futures
  // MtM (FX-reset-notional) coupons (2026-09-09): value = R·(A + DF[dE] − DF[dS]) with R = DF[rN]·INV[rD] the
  // notional reset ratio at the reset time (numerator / denominator discount curves) and A the ordinary coupon
  // DF[pay]·(num + konst)·k; dS/dE are the accrual start/end on the DISCOUNT curve (the notional exchanges).
  // -1 on every entry for a constant-notional coupon. The exact templated pricing::xccy_mtm_leg_pv form — a
  // product of registered DFs, hence W-cacheable with hand-written partials (no "negligible term" shortcut).
  Eigen::VectorXi rN, rD, dS, dE;
  Eigen::VectorXd konst, k, realized, inv_tau, convexity;
  Eigen::SparseMatrix<double> R_cpn;               // n_inst x n_coupons, 0/1
  int n_inst = 0;
  // True iff EVERY observation is standard() (one implicit-unit-weight sub-period per coupon, in
  // order): R_sub is then EXACTLY the identity, R_sub*v == v, and the reduction is skipped entirely.
  bool sub_is_identity = false;
  // True iff EVERY coupon is standard() (no spread, nothing realized, tau_pay == tau_index, no scale):
  // the "+ konst" and "* k" passes are then pure overhead over the whole coupon vector (~1.28x on a 1000-swap
  // book, measured), so pv() fuses to the minimal pre-generalization expression. Both flags are aggregates of
  // the per-coupon predicates from cashflows.hpp (see finalize); the per-leg forms drive the value pass.
  bool cpn_is_plain = false;

  // ---- MOMENT coupons (RateObservation::fixing_step > 0; docs/bezier-and-moments.md Part B) -------------------
  // An arithmetic-average window [a,b] priced from curve MOMENTS instead of ~250 daily sub-periods:
  //     num = ln(DF[a]·INV[b]) + ½·step·∫f² (+ ⅙·step3·∫f³),   ∫f² = xᵀ Q x with Q = Σ_k w_k psi_k psi_kᵀ
  // Q (over the coupon's knot SUPPORT) and, when step3 > 0, the Gauss nodes' forward rows are precomputed at
  // registration, so a tick costs one log + |S|² multiply-adds per coupon (|S| ≈ 6-16) instead of a daily gather
  // loop — the reason a Fed funds curve can be as cheap as a SOFR curve (shape ladder: 534 → ~5 µs tick).
  // The linear term is exact; the moment correction is the documented ~5e-9 approximation of the daily sum
  // (the exact daily path stays available with fixing_step == 0). Same Gauss rule as the templated
  // curve_forward_sq_integral (moment_gauss_nodes, knot-aligned) so both paths agree to rounding.
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
  bool has_mtm() const { return has_mtm_; }
  bool mtm_leg(int leg) const { return mtm_leg_[static_cast<std::size_t>(leg)] != 0; }
  static constexpr int reduce_layout() { return SWAPS_REDUCE_LAYOUT; }

  // --- generic registration -------------------------------------------------------------------
  // One instrument = one float leg: coupons forecast `fc`, discount `dc`.
  void add(CompiledCurveSet& cs, int fc, int dc, const std::vector<FloatCoupon>& leg) {
    for (const auto& c : leg) {
      push_obs(cs, fc, c.obs);
      // FX `scale` (default 1) folds into the per-coupon k (× 1.0 is exact). A scale != 1 is a non-standard
      // coupon (FloatCoupon::standard), so it leaves the fused fast path -- exactly right: only a foreign
      // (converted) leg pays that cost, and the analytic Jacobian formula (pv = DF·A·k) is unchanged because
      // scale rides inside k as a constant.
      push_coupon(cs.reg(dc, c.pay), c.obs.realized + c.spread * c.obs.tau_index,
                  c.tau_pay / c.obs.tau_index * c.scale, c.obs.realized, 1.0 / c.obs.tau_index, 0.0,
                  c.obs.standard(), c.standard());
    }
    mtm_leg_.push_back(0);
    ++n_inst;
  }
  // One instrument = one MtM (FX-resettable-notional) funding leg: coupons forecast `fc`, discount `dc`, notional
  // fx·DF[num](reset)/DF[den](reset) with reset = coupon.reset_time (>= 0) else the period start. fx_spot is NOT
  // folded in (the XccyMtmBasis quote divides it out: mtm/(fx·ann)); the caller scales if it wants the raw PV.
  void add_mtm(CompiledCurveSet& cs, int fc, int dc, int num, int den, const std::vector<FloatCoupon>& leg) {
    mtm_leg_.push_back(0);  // set below once a coupon proves the leg MtM (an empty leg stays plain)
    for (const auto& c : leg) {
      // A SEASONED coupon (fixed FX reset, settled initial exchange, past reset) is a constant times a
      // reduced flow set, not a product of registered DFs: it prices on the templated kernel. The hybrid
      // router (Instrument::noncacheable) sends such instruments to the AAD block, so this only fires
      // when a caller compiles one directly.
      if (!c.accrual_set && (c.obs.sub_start.empty() || c.obs.sub_end.empty()))
        throw std::runtime_error(
            "CompiledBook: a fully-fixed MtM coupon has no accrual period or observation window to place its "
            "notional exchanges on (set accrual_start/accrual_end)");  // identical to pricing::xccy_mtm_leg_pv
      if (c.seasoned_mtm())
        throw std::invalid_argument(
            "CompiledBook: a seasoned MtM coupon (fixed reset_fx, settled or past-reset accrual) is not W-cacheable; "
            "price it through the templated kernel (the hybrid engine routes it there)");
      const double s = c.accrual_set ? c.accrual_start : c.obs.sub_start.front();
      const double e = c.accrual_set ? c.accrual_end : c.obs.sub_end.back();
      const double reset = (c.reset_time >= 0.0) ? c.reset_time : s;
      push_obs(cs, fc, c.obs);
      push_coupon(cs.reg(dc, c.pay), c.obs.realized + c.spread * c.obs.tau_index,
                  c.tau_pay / c.obs.tau_index * c.scale, c.obs.realized, 1.0 / c.obs.tau_index, 0.0,
                  c.obs.standard(), c.standard());
      rn_.back() = cs.reg(num, reset); rd_.back() = cs.reg(den, reset);
      ds_.back() = cs.reg(dc, s);      de_.back() = cs.reg(dc, e);
      has_mtm_ = true;
    }
    if (!leg.empty()) mtm_leg_.back() = 1;
    ++n_inst;
  }
  // One instrument = one future on `obs` forecasting `fc`. No discounting: the terminal transform is
  // rate(), not pv(). `convexity` is an INPUT NUMBER (design §3) -- the model lives in tests.
  void add_future(CompiledCurveSet& cs, int fc, const RateObservation& o, double conv) {
    push_obs(cs, fc, o);
    push_coupon(-1, 0.0, 0.0, o.realized, 1.0 / o.tau_index, conv, o.standard(), o.standard());
    mtm_leg_.push_back(0);
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
    rN = detail::to_vec(rn_); rD = detail::to_vec(rd_); dS = detail::to_vec(ds_); dE = detail::to_vec(de_);
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
    // The fast-path flags are AGGREGATES of what each coupon SAYS of itself (RateObservation::standard /
    // FloatCoupon::standard, asked once at registration -- the definition the templated kernel reads too),
    // never a rediscovery from the packed arrays. Batch-wide (the book's grid path, the reference layout) and PER LEG
    // (2026-09-22: the value pass picks the coupon formula per leg, so one weighted, moment or MtM leg in a
    // batch no longer takes every other leg off the fused standard-shape gather -- the residual folds a whole
    // bundle's legs into one batch now). sub0_[c] is coupon c's first sub-period: on a standard observation it
    // IS the coupon's only one, so the fused gather reads ss[sub0[c]] where the batch-wide path read ss[c].
    sub_is_identity = true;
    cpn_is_plain = true;
    for (int c = 0; c < n_cpn_; ++c) {
      sub_is_identity = sub_is_identity && obs_std_[static_cast<std::size_t>(c)];
      cpn_is_plain = cpn_is_plain && cpn_std_[static_cast<std::size_t>(c)];
    }
    sub0_.assign(sub_begin_.begin(), sub_begin_.end() - 1);
    // The fused per-leg gather reads a coupon's ONLY sub-period's DF indices directly (cs0/ce0): one load, exactly
    // as the batch-wide fast path read ss[c] -- not ss[sub0[c]], a dependent double indirection that measurably
    // lengthened the gather chain (+18 % on a plain single-curve residual, A/B 2026-09-22).
    cs0_.resize(n_cpn_); ce0_.resize(n_cpn_);
    for (int c = 0; c < n_cpn_; ++c) { cs0_[c] = subS[sub0_[static_cast<std::size_t>(c)]]; ce0_[c] = subE[sub0_[static_cast<std::size_t>(c)]]; }
    leg_general_.assign(static_cast<std::size_t>(n_inst), 0);
    leg_plain_.assign(static_cast<std::size_t>(n_inst), 1);
    for (int c = 0; c < n_cpn_; ++c) {
      const std::size_t i = static_cast<std::size_t>(row_[c]);
      if (!obs_std_[static_cast<std::size_t>(c)]) leg_general_[i] = 1;
      if (!cpn_std_[static_cast<std::size_t>(c)]) leg_plain_[i] = 0;
    }
    any_general_leg_ = false;
    for (unsigned char g : leg_general_) any_general_leg_ = any_general_leg_ || g;
    // ONE kind byte per leg for the value pass's dispatch: 0 = standard (fused), 1 = standard observations with
    // spread / k / scale, 2 = needs the materialised numerator, 3 = MtM.
    leg_kind_.assign(static_cast<std::size_t>(n_inst), 0);
    for (int i = 0; i < n_inst; ++i) {
      const std::size_t u = static_cast<std::size_t>(i);
      leg_kind_[u] = mtm_leg_[u] ? 3 : leg_general_[u] ? 2 : leg_plain_[u] ? 0 : 1;
    }
    mom_of_.assign(static_cast<std::size_t>(n_cpn_), -1);
    for (int q = 0; q < static_cast<int>(moments_.size()); ++q) mom_of_[static_cast<std::size_t>(moments_[static_cast<std::size_t>(q)].cpn)] = q;
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
  // coupon vector -> per-instrument PV under the configured reduce layout (segment-sum or the sparse reference).
  const Eigen::VectorXd& reduce_coupons(const Eigen::VectorXd& coupon) const {
    if (SWAPS_REDUCE_LAYOUT >= 1) {
      pv_res_.resize(n_inst);
      const int* __restrict cb = cpn_begin_.data(); const double* __restrict cc = coupon.data(); double* __restrict out = pv_res_.data();
      for (int i = 0; i < n_inst; ++i) { double acc = 0.0; for (int c = cb[i]; c < cb[i + 1]; ++c) acc += cc[c]; out[i] = acc; }
      return pv_res_;
    }
    pv_res_.noalias() = R_cpn * coupon;
    return pv_res_;
  }
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
  // Every leg's PV ACCUMULATED with a sign into a caller slot: out[row[leg]] += sign[leg]·pv(leg), legs in order
  // (the caller zeroes `out`). This is how a quotient's numerator Σ sign·pv(leg) is formed in the batch's own
  // per-leg loop, with no second pass over the legs; (0 + pv_a) + (−pv_b) == pv_a − pv_b bit for bit. SEGMENT
  // layout only (the reference layout has no per-leg loop); pv() is this with row = identity, sign = +1.
  void pv_into(const Eigen::VectorXd& DF, const Eigen::VectorXd& INV, Eigen::VectorXd& out, const int* row, const double* sign) const {
    pv_impl(DF, INV, out.data(), row, sign);
  }
  const Eigen::VectorXd& pv(const Eigen::VectorXd& DF, const Eigen::VectorXd& INV) const {
    if (SWAPS_REDUCE_LAYOUT >= 1) {
      pv_res_.resize(n_inst);
      pv_impl(DF, INV, pv_res_.data(), nullptr, nullptr);
      return pv_res_;
    }
    return pv_reference(DF, INV);
  }
  const Eigen::VectorXd& pv_reference(const Eigen::VectorXd& DF, const Eigen::VectorXd& INV) const {
    // PERF RULE: what reaches R_cpn must be a materialized VectorXd -- handing Eigen's sparse*dense an
    // unevaluated gather/divide re-does that work per access (~1.28x slower, measured). On the identity
    // path we fuse the sub-period gather straight into the coupon vector, so the standard shape costs
    // exactly one materialized pass, as it did before this generalization.
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "pv() needs pay dates: this is a futures batch");
    Eigen::VectorXd& coupon = coupon_;  // reuse the per-batch scratch (no per-tick allocation)
    if (has_mtm_) {  // (reference layout) MtM coupons: R·(A + DF[dE] − DF[dS]); constant-notional coupons in the same batch: A
      const Eigen::VectorXd& nm = num(DF, INV);
      coupon.resize(n_cpn_);
      const double* __restrict df = DF.data();
      const double* __restrict iv = INV.data();
      const int* __restrict p = pay.data();
      const double* __restrict nn = nm.data();
      const double* __restrict kk = konst.data();
      const double* __restrict kv = k.data();
      const int* __restrict rn = rN.data(); const int* __restrict rdn = rD.data();
      const int* __restrict dsn = dS.data(); const int* __restrict den = dE.data();
      double* __restrict out = coupon.data();
      for (int i = 0; i < n_cpn_; ++i) {
        double v = df[p[i]] * (nn[i] + kk[i]) * kv[i];
        if (rn[i] >= 0) v = (v + df[den[i]] - df[dsn[i]]) * (df[rn[i]] * iv[rdn[i]]);
        out[i] = v;
      }
      return reduce_coupons(coupon);
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

  // SEGMENT: each coupon's gather-product accumulates into its leg's slot (row = nullptr: slot i, written; else
  // out[row[i]] += sign[i]·pv, accumulated).
  void pv_impl(const Eigen::VectorXd& DF, const Eigen::VectorXd& INV, double* __restrict out, const int* __restrict row, const double* __restrict sign) const {
    {
      // The formula is chosen PER LEG (2026-09-22): an MtM leg R·(A + DF[dE] − DF[dS]) reads the materialised
      // numerator; a plain leg keeps the fused standard-shape gather it always had -- so one MtM leg in the batch
      // costs the par-swap legs beside it nothing (the residual folds every leg of a bundle into ONE batch now).
      // No batch-wide numerator pass: a leg that needs the materialised numerator (weighted / multi-sub-period /
      // moment coupons, or an MtM leg) computes it INLINE per coupon -- the same segment-sum num() would form,
      // in the same order -- so the plain legs beside it never pay for a pass they do not read.
      if (!moments_.empty() && !state_set_) throw std::logic_error("BundleFloatBatch: set_state(x) must precede num/pv/rate on a batch with moment coupons");
      const double* __restrict df = DF.data();
      const double* __restrict iv = INV.data();
      const int* __restrict p = pay.data();
      const double* __restrict kk = konst.data();
      const double* __restrict kv = k.data();
      const int* __restrict cb = cpn_begin_.data();
      const int* __restrict ss = subS.data();
      const int* __restrict se = subE.data();
      const int* __restrict s0 = sub0_.data();
      const int* __restrict cs0 = cs0_.data();
      const int* __restrict ce0 = ce0_.data();
      const int* __restrict sb = sub_begin_.data();
      const double* __restrict sw = sub_w.data();
      const int* __restrict mo = mom_of_.data();
      const int* __restrict rn = rN.data(); const int* __restrict rdn = rD.data();
      const int* __restrict dsn = dS.data(); const int* __restrict den = dE.data();
      const unsigned char* __restrict lk = leg_kind_.data();
      // coupon c's numerator, exactly as num() forms it: sum_j w_j (DF[s_j]*INV[e_j] - 1), a moment bracket's
      // w * (ln(DF[a]*INV[b]) + mom)
      const unsigned char* __restrict os = obs_std_.data();
      const auto numer = [&](int c) {
        if (os[c]) return df[cs0[c]] * iv[ce0[c]] - 1.0;  // standard: the one gather (== 1.0 * that, bit for bit)
        if (mo[c] >= 0) { const int j = s0[c]; return sw[j] * (std::log(df[ss[j]] * iv[se[j]]) + moments_[static_cast<std::size_t>(mo[c])].mom); }
        double n = 0.0;
        for (int j = sb[c]; j < sb[c + 1]; ++j) n += sw[j] * (df[ss[j]] * iv[se[j]] - 1.0);
        return n;
      };
      for (int i = 0; i < n_inst; ++i) {
        double acc = 0.0;
        if (lk[i] == 3) {
          for (int c = cb[i]; c < cb[i + 1]; ++c) {
            const double v = df[p[c]] * (numer(c) + kk[c]) * kv[c];
            acc += (v + df[den[c]] - df[dsn[c]]) * (df[rn[c]] * iv[rdn[c]]);
          }
        } else if (lk[i] == 2) {  // weighted / multi-sub-period / moment coupons: the inline numerator
          for (int c = cb[i]; c < cb[i + 1]; ++c) acc += df[p[c]] * (numer(c) + kk[c]) * kv[c];
        } else if (lk[i] == 0) {  // the standard shape: the fused gather, exactly the pre-generalization expression
          for (int c = cb[i]; c < cb[i + 1]; ++c) acc += df[p[c]] * (df[cs0[c]] * iv[ce0[c]] - 1.0);
        } else {
          for (int c = cb[i]; c < cb[i + 1]; ++c) acc += df[p[c]] * (df[cs0[c]] * iv[ce0[c]] - 1.0 + kk[c]) * kv[c];
        }
        if (row) out[row[i]] += sign[i] * acc;
        else out[i] = acc;
      }
    }
  }
  // Per-instrument PV from a PRECOMPUTED per-coupon numerator `num_cpn` (== num(DF)). Lets the
  // Jacobian's value pass share the single sub-period gather with its derivative pass (d_pv_from_num)
  // instead of each re-gathering. BIT-IDENTICAL to pv(DF): on the plain path (konst == 0, k == 1) it
  // is DF[pay]*num_cpn, exactly the fused pv coupon; otherwise the same DF[pay]*(num+konst)*k form.
  const Eigen::VectorXd& pv_from_num(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF) const {
    pv_res_.resize(n_inst);
    pv_from_num_impl(num_cpn, DF, pv_res_.data(), nullptr, nullptr);
    return pv_res_;
  }
  // The accumulate form of pv_from_num (see pv_into): out[row[leg]] += sign[leg]·pv(leg).
  void pv_from_num_into(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF, Eigen::VectorXd& out, const int* row, const double* sign) const {
    pv_from_num_impl(num_cpn, DF, out.data(), row, sign);
  }
  void pv_from_num_impl(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF, double* __restrict res, const int* __restrict row, const double* __restrict sign) const {
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "pv_from_num() needs pay dates");
    if (n_cpn_ == 0) {
      if (!row) for (int i = 0; i < n_inst; ++i) res[i] = 0.0;
      return;
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
      if (has_mtm_) {  // per LEG: the MtM legs take the reset-ratio form, the plain legs beside them their usual one
        const double* __restrict iv = inverse_of(DF).data();
        const int* __restrict rn = rN.data(); const int* __restrict rdn = rD.data();
        const int* __restrict dsn = dS.data(); const int* __restrict den = dE.data();
        const int* __restrict cb = cpn_begin_.data();
        const unsigned char* __restrict ml = mtm_leg_.data();
        const unsigned char* __restrict lp = leg_plain_.data();
        for (int i = 0; i < n_inst; ++i) {
          if (ml[i])
            for (int c = cb[i]; c < cb[i + 1]; ++c) {
              const double v = df[p[c]] * (nn[c] + kk[c]) * kv[c];
              out[c] = (v + df[den[c]] - df[dsn[c]]) * (df[rn[c]] * iv[rdn[c]]);
            }
          else if (lp[i])
            for (int c = cb[i]; c < cb[i + 1]; ++c) out[c] = df[p[c]] * nn[c];
          else
            for (int c = cb[i]; c < cb[i + 1]; ++c) out[c] = df[p[c]] * (nn[c] + kk[c]) * kv[c];
        }
      } else if (cpn_is_plain)
        for (int i = 0; i < n_cpn_; ++i) out[i] = df[p[i]] * nn[i];
      else {
        const int* __restrict cb = cpn_begin_.data();
        const unsigned char* __restrict lp = leg_plain_.data();
        for (int i = 0; i < n_inst; ++i) {
          if (lp[i]) for (int c = cb[i]; c < cb[i + 1]; ++c) out[c] = df[p[c]] * nn[c];
          else for (int c = cb[i]; c < cb[i + 1]; ++c) out[c] = df[p[c]] * (nn[c] + kk[c]) * kv[c];
        }
      }
    }
    // (IndexedView form retired 2026-09-09: it materialised index temporaries on every Jacobian — E3-A2/A3.)
    if (SWAPS_REDUCE_LAYOUT >= 1) {
      const int* __restrict cb = cpn_begin_.data(); const double* __restrict cc = coupon.data();
      for (int i = 0; i < n_inst; ++i) {
        double acc = 0.0;
        for (int c = cb[i]; c < cb[i + 1]; ++c) acc += cc[c];
        if (row) res[row[i]] += sign[i] * acc;
        else res[i] = acc;
      }
      return;
    }
    Eigen::Map<Eigen::VectorXd>(res, n_inst).noalias() = R_cpn * coupon;  // reference layout: identity map only
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
    d_pv_from_num(num_cpn, DF, inverse_of(DF), d, LegOffset{row0, sign, inst.data()});
  }
  template <class Mat>
  void d_pv_from_num(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF, const Eigen::VectorXd& INV,
                     Mat& d, int row0, double sign) const {
    d_pv_from_num(num_cpn, DF, INV, d, LegOffset{row0, sign, inst.data()});
  }
  // The coupon -> leg map, so a caller can build per-coupon row/sign tables once (LegTable).
  const Eigen::VectorXi& coupon_leg() const { return inst; }
  // The general form: coupon c's partials land on row m.row(c) with sign m.sgn(c) (LegOffset / LegTable).
  // Runs PER LEG: an MtM leg takes the reset-ratio partials, every other leg the plain ones -- one MtM leg in
  // the batch costs the plain legs nothing (the batch-wide has_mtm_ path used to multiply every partial by R = 1).
  template <class Mat, class LegMap>
  void d_pv_from_num(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF, const Eigen::VectorXd& INV,
                     Mat& d, const LegMap& m) const {
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "d_pv_from_num() needs pay dates");
    if (has_mtm_) {
      for (int leg = 0; leg < n_inst; ++leg) {
        const int c0 = cpn_begin_[leg], c1 = cpn_begin_[leg + 1];
        if (c0 == c1) continue;
        const int j0 = sub_begin_[c0], j1 = sub_begin_[c1];
        if (mtm_leg_[static_cast<std::size_t>(leg)]) {
          d_pv_mtm_leg(num_cpn, DF, INV, d, m, c0, c1, j0, j1);
        } else {
          for (int i = c0; i < c1; ++i)
            d(m.row(i), pay[i]) += m.sgn(i) * (num_cpn[i] + konst[i]) * k[i];
          for (int j = j0; j < j1; ++j) {
            const int i = sub_cpn[j], s = subS[j], e = subE[j];
            const double f = m.sgn(i) * DF[pay[i]] * k[i] * sub_w[j];
            const double ie = INV[e];
            d(m.row(i), s) += f * ie;
            d(m.row(i), e) += -f * DF[s] * ie * ie;
          }
        }
      }
      for (const auto& mc : moments_) {  // moment brackets (either leg kind): the log-bracket swap, scaled by R on an MtM leg
        const int i = mc.cpn, s = subS[mc.sub], e = subE[mc.sub];
        const double R = rN[i] >= 0 ? DF[rN[i]] * INV[rD[i]] : 1.0;
        const double f = m.sgn(i) * R * DF[pay[i]] * k[i] * sub_w[mc.sub];
        const double ie = INV[e];
        d(m.row(i), s) += -f * ie + f * INV[s];
        d(m.row(i), e) += f * DF[s] * ie * ie - f * ie;
      }
      return;
    }
    for (int i = 0; i < n_cpn_; ++i)
      d(m.row(i), pay[i]) += m.sgn(i) * (num_cpn[i] + konst[i]) * k[i];
    for (int j = 0; j < static_cast<int>(subS.size()); ++j) {
      const int i = sub_cpn[j], s = subS[j], e = subE[j];
      const double f = m.sgn(i) * DF[pay[i]] * k[i] * sub_w[j];
      const double ie = INV[e];
      d(m.row(i), s) += f * ie;
      d(m.row(i), e) += -f * DF[s] * ie * ie;
    }
    // moment brackets: d ln(DF[a]/DF[b]) / dDF = +INV[a] on a, −INV[b] on b (replacing the ratio partials above)
    for (const auto& mc : moments_) {
      const int i = mc.cpn, s = subS[mc.sub], e = subE[mc.sub];
      const double f = m.sgn(i) * DF[pay[i]] * k[i] * sub_w[mc.sub];
      const double ie = INV[e];
      d(m.row(i), s) -= f * ie;                 // undo the ratio partials
      d(m.row(i), e) -= -f * DF[s] * ie * ie;
      d(m.row(i), s) += f * INV[s];             // log-bracket partials
      d(m.row(i), e) += -f * ie;
    }
  }
  // The MtM leg's partials over coupons [c0, c1) and their sub-periods [j0, j1): value = R·(A + DF[dE] − DF[dS]),
  // R = DF[rN]·INV[rD], A = DF[pay]·(num+konst)·k.
  //   ∂/∂DF[pay] = R·(num+konst)·k;  ∂/∂DF[s],[e] = R·(the ordinary A partials);  ∂/∂DF[dE] = +R;  ∂/∂DF[dS] = −R;
  //   ∂/∂DF[rN] = value·INV[rN];      ∂/∂DF[rD] = −value·INV[rD].     (aliased indices accumulate via +=)
  template <class Mat, class LegMap>
  void d_pv_mtm_leg(const Eigen::VectorXd& num_cpn, const Eigen::VectorXd& DF, const Eigen::VectorXd& INV, Mat& d,
                    const LegMap& m, int c0, int c1, int j0, int j1) const {
    for (int i = c0; i < c1; ++i) {
      const double R = DF[rN[i]] * INV[rD[i]], sign = m.sgn(i);
      const double A = DF[pay[i]] * (num_cpn[i] + konst[i]) * k[i];
      d(m.row(i), pay[i]) += sign * R * (num_cpn[i] + konst[i]) * k[i];
      const double value = R * (A + DF[dE[i]] - DF[dS[i]]);
      d(m.row(i), dE[i]) += sign * R;
      d(m.row(i), dS[i]) += -sign * R;
      d(m.row(i), rN[i]) += sign * value * INV[rN[i]];
      d(m.row(i), rD[i]) += -sign * value * INV[rD[i]];
    }
    for (int j = j0; j < j1; ++j) {
      const int i = sub_cpn[j], s = subS[j], e = subE[j];
      const double f = m.sgn(i) * (DF[rN[i]] * INV[rD[i]]) * DF[pay[i]] * k[i] * sub_w[j];
      const double ie = INV[e];
      d(m.row(i), s) += f * ie;
      d(m.row(i), e) += -f * DF[s] * ie * ie;
    }
  }
  // DIRECT x-space derivative of the moment corrections (the part that is NOT a function of DF): for each moment
  // coupon c, ∂pv_inst/∂x_j += DF[pay_c]·k_c·w·∂mom_c/∂x_j on its support. Delivered through `add(inst, j, value)` so
  // the caller can scatter into its own Jacobian with the row's quotient / band factors. Precondition: set_state(x).
  template <class Add>
  void moment_direct_pv(const Eigen::VectorXd& DF, double sign, Add&& add) const {
    moment_direct_pv(DF, LegOffset{0, sign, inst.data()}, std::forward<Add>(add));
  }
  template <class LegMap, class Add>
  void moment_direct_pv(const Eigen::VectorXd& DF, const LegMap& m, Add&& add) const {
    for (const auto& mc : moments_) {
      const int i = mc.cpn;
      const double R = (has_mtm_ && rN[i] >= 0) ? DF[rN[i]] / DF[rD[i]] : 1.0;
      const double f = m.sgn(i) * R * DF[pay[i]] * k[i] * sub_w[mc.sub];
      for (int j = 0; j < static_cast<int>(mc.support.size()); ++j) add(m.row(i), mc.support[j], f * mc.dmom[j]);
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
      // KNOT-ALIGNED 4-pt Gauss per piece panel -- the SAME node set as pricing::curve_forward_sq_integral
      // (moment_gauss_nodes), exact for the square of a cubic piece (E3-R4 / G2, 2026-09-10; the old
      // unaligned 32×2 rule lost 1.7e-7 of the rate on the Flat-front Fed-funds shape).
      const double a = o.sub_start[0], b = o.sub_end[0];
      const std::vector<double> pieces = cs.pieces(fc);
      std::vector<double> nodes, wts;
      moment_gauss_nodes(pieces, a, b, 4, 1, nodes, wts);
      const Eigen::MatrixXd P = cs.forward_rows(fc, nodes);  // nodes × n_knots
      for (int j = 0; j < P.cols(); ++j)
        if ((P.col(j).array() != 0.0).any()) mc.support.push_back(j);
      const int S = static_cast<int>(mc.support.size());
      Eigen::MatrixXd Ps(static_cast<int>(nodes.size()), S);
      for (int j = 0; j < S; ++j) Ps.col(j) = P.col(mc.support[j]);
      mc.Q = Eigen::MatrixXd::Zero(S, S);
      for (int kq = 0; kq < Ps.rows(); ++kq) mc.Q.noalias() += wts[kq] * (Ps.row(kq).transpose() * Ps.row(kq));
      if (mc.step3 > 0.0) {  // cubic term: knot-aligned 5-pt Gauss (curve_forward_cube_integral's rule), same support
        std::vector<double> n3, w3;
        moment_gauss_nodes(pieces, a, b, 5, 1, n3, w3);
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
  void push_coupon(int pay_idx, double konst, double kk, double rz, double inv_tau, double conv, bool obs_std, bool cpn_std) {
    obs_std_.push_back(obs_std ? 1 : 0);
    cpn_std_.push_back(cpn_std ? 1 : 0);
    p_.push_back(pay_idx);
    konst_.push_back(konst);
    k_.push_back(kk);
    rz_.push_back(rz);
    it_.push_back(inv_tau);
    cv_.push_back(conv);
    row_.push_back(n_inst);
    rn_.push_back(-1); rd_.push_back(-1); ds_.push_back(-1); de_.push_back(-1);
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
  std::vector<int> rn_, rd_, ds_, de_;                     // MtM reset / notional-exchange indices (-1 = none)
  bool has_mtm_ = false;
  std::vector<unsigned char> mtm_leg_;                     // per leg (batch instrument): 1 iff it is an MtM leg
  std::vector<unsigned char> obs_std_, cpn_std_;           // per coupon: obs.standard() / coupon.standard(), asked at registration
  std::vector<unsigned char> leg_general_, leg_plain_;     // per leg: any non-standard observation / every coupon standard
  std::vector<unsigned char> leg_kind_;                    // per leg: 0 fused / 1 fused with konst,k / 2 materialised numerator / 3 MtM
  std::vector<int> sub0_;                                  // per coupon: its first sub-period (== its only one on an identity leg)
  Eigen::VectorXi cs0_, ce0_;                              // per coupon: that sub-period's start/end DF indices (the fused gather)
  std::vector<int> mom_of_;                                // per coupon: its moment-coupon index, or -1
  bool any_general_leg_ = false;
  std::vector<int> cpn_begin_, sub_begin_;                // segment offsets (coupons of an instrument; subs of a coupon)
  std::vector<double> sw_, konst_, k_, rz_, it_, cv_;
  // Per-batch reusable scratch (sized on first use) so pv/num/rate never allocate in the hot loop.
  // Each returns a const ref into these; because gen_pos_/gen_neg_ are distinct batch objects, two
  // results (one per batch) are simultaneously live in model_rates without aliasing.
  mutable Eigen::VectorXd coupon_, sub_, num_res_, pv_res_, rate_res_, inv_scratch_, rmul_;
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
