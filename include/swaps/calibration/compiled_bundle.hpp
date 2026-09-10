#pragma once
// Compiled (W-cache) residual for the multi-curve BundleProblem -- the bundle analogue of
// CompiledResidual, built on the multi-curve pricing/compiled_book.hpp engine. Produces model rates /
// residuals and the ANALYTIC block Jacobian (no AAD in the hot loop), so the warm & streaming
// re-calibrators drive the bundle on the microsecond fast path instead of an AAD sweep per refresh.
//
// Same factorization as the single-curve CompiledResidual: DF_all = exp(-W_all x) once, then the cheap
// per-type transform (futures DF-ratios, swap par rate = float_pv/annuity, basis = (pv_bench-pv_fwd)/
// annuity), and J = -(dr/dDF diag(DF)) W_all with dr/dDF the analytic per-instrument sensitivity
// scattered into the global DF columns. Residual order matches BundleProblem::residuals: the generic
// instruments in insertion order (batched internally by quote kind, then scattered to each row).

#include <Eigen/Core>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/pricing/compiled_book.hpp"

namespace swaps::calibration {

// A single-curve CalibrationProblem IS a 1-curve bundle: one self-discounting outright curve, every
// instrument forecasting and discounting curve 0. This lets the single-curve CompiledResidual be a
// thin delegate to CompiledBundleResidual -- one compiled engine, not two.
inline BundleProblem single_curve_bundle(const CalibrationProblem& p) {
  BundleProblem b;
  b.curves.push_back({.regions = curve::flat_hermite(p.meeting_times, p.back_times)});  // base = -1
  for (Instrument ins : p.instruments) {  // by value: force every role onto the single curve, so the
    ins.fwd.forecast = ins.fwd.discount = 0;  // compiled path cannot disagree with price_residuals(),
    ins.bench.forecast = ins.bench.discount = 0;  // which resolves every role to the one curve.
    ins.fixed.discount = 0;
    ins.forecast = 0;
    b.instruments.push_back(std::move(ins));
  }
  return b;
}

class CompiledBundleResidual {
 public:
  explicit CompiledBundleResidual(const BundleProblem& p)
      : n_gen_(static_cast<int>(p.instruments.size())), market_(p.market()) {
    cs_.init(p.curves);  // p.curves ARE pricing::CurveStructure now (BundleCurveSpec is an alias), no copy
    fx_row_.assign(n_gen_, 0);
    band_lo_.assign(n_gen_, 0.0);
    band_up_.assign(n_gen_, 0.0);
    band_dc_.assign(n_gen_, 1.0);
    band_.reserve(n_gen_);  // rebuild_bands() never allocates after this

    register_generic(p);

    cs_.finalize();  // builds W_all now that every (curve, time) is registered
    gen_pos_.finalize();
    gen_neg_.finalize();
    gen_fixed_.finalize();
    gen_mtm_.finalize();
    has_moment_ = gen_pos_.has_moment() || gen_neg_.has_moment() || gen_rate_.has_moment() || gen_mtm_.has_moment();
    gen_rate_.finalize();

    // Jacobian scratch buffers, sized ONCE here and reused (setZero) every call -- no per-iteration
    // allocation of the N×T / nq×T / nr×T dense matrices.
    const int T = cs_.n_times();
    G_.resize(n_gen_, T);
    dnum_.resize(static_cast<int>(q_rows_.size()), T);
    dann_.resize(static_cast<int>(q_rows_.size()), T);
    dr_.resize(static_cast<int>(r_rows_.size()), T);

    // ---- the support-blocked Jacobian's STRUCTURE (built once; see jacobian_vs) --------------------
    // G is structurally BLOCK-SPARSE: row r is nonzero only at the DF times instrument r's legs actually
    // registered (its pay dates + sub-period boundaries + FX pillar times). Those positions are known
    // exactly from the batch index arrays, so record them per residual row as a CSR support list. The
    // final product then sums ONLY over each row's support instead of a dense n_res × T × n_knots GEMM.
    {
      std::vector<std::vector<int>> sup(n_gen_);
      const auto add_float = [&](const pricing::BundleFloatBatch& b, const std::vector<Scatter>& rows) {
        for (int i = 0; i < b.n_coupons(); ++i)
          if (b.pay[i] >= 0) sup[rows[b.inst[i]].row].push_back(b.pay[i]);  // futures carry pay = -1
        for (int j = 0; j < static_cast<int>(b.subS.size()); ++j) {
          const int r = rows[b.inst[b.sub_cpn[j]]].row;
          sup[r].push_back(b.subS[j]);
          sup[r].push_back(b.subE[j]);
        }
      };
      add_float(gen_pos_, q_rows_);
      add_float(gen_neg_, q_rows_);
      add_float(gen_rate_, r_rows_);
      add_float(gen_mtm_, q_rows_);
      for (int i = 0; i < gen_mtm_.n_coupons(); ++i)  // the MtM coupon's reset ratio and notional exchanges
        if (gen_mtm_.rN[i] >= 0) {
          const int r = q_rows_[gen_mtm_.inst[i]].row;
          sup[r].push_back(gen_mtm_.rN[i]); sup[r].push_back(gen_mtm_.rD[i]);
          sup[r].push_back(gen_mtm_.dS[i]); sup[r].push_back(gen_mtm_.dE[i]);
        }
      for (int i = 0; i < static_cast<int>(gen_fixed_.pay.size()); ++i)
        sup[q_rows_[gen_fixed_.inst[i]].row].push_back(gen_fixed_.pay[i]);
      for (const auto& f : fx_rows_) {
        sup[f.row].push_back(f.idx_num);
        sup[f.row].push_back(f.idx_den);
      }
      for (int r = 0; r < n_gen_; ++r) {
        std::sort(sup[r].begin(), sup[r].end());
        sup[r].erase(std::unique(sup[r].begin(), sup[r].end()), sup[r].end());
      }
      // TIME-MAJOR CSR (t -> the residual rows whose support contains t): the product iterates times in
      // the OUTER loop so one W row stays L1-hot across all (~n_res/n_curves) rows that share it -- the
      // axpy stream then only writes Jt, instead of re-reading a different W row per term.
      tsup_ptr_.assign(T + 1, 0);
      for (int r = 0; r < n_gen_; ++r)
        for (int t : sup[r]) ++tsup_ptr_[t + 1];
      for (int t = 0; t < T; ++t) tsup_ptr_[t + 1] += tsup_ptr_[t];
      tsup_row_.resize(tsup_ptr_[T]);
      {
        std::vector<int> cur(tsup_ptr_.begin(), tsup_ptr_.end() - 1);
        for (int r = 0; r < n_gen_; ++r)
          for (int t : sup[r]) tsup_row_[cur[t]++] = r;
      }
      // W transposed once: Wt.col(t) == W.row(t), CONTIGUOUS (col-major). And each W row is itself
      // sparse -- time t on curve c touches only c's ancestry knots -- so record its nonzero column SPAN
      // [wlo, whi) once and axpy only that segment (a spread-chain's early curves touch a fraction of
      // the state, so this cuts both the flops and the Jt write traffic).
      Wt_ = cs_.W().transpose();
      const int nk = static_cast<int>(Wt_.rows());
      wlo_.assign(T, 0);
      whi_.assign(T, 0);
      for (int t = 0; t < T; ++t) {
        int lo = 0, hi = nk;
        while (lo < nk && Wt_(lo, t) == 0.0) ++lo;
        while (hi > lo && Wt_(hi - 1, t) == 0.0) --hi;
        wlo_[t] = lo;
        whi_[t] = hi;
      }
    }
  }

  int n_residuals() const { return n_gen_; }
  int n_times() const { return cs_.n_times(); }

  // Overwrite the quote RHS -- targets and soft-quote bands -- WITHOUT touching the compiled structure.
  // The W-cache, batches and scatter maps depend only on topology (legs, times, curve roles); the market
  // vector and the band list are plain per-row data read at residual time. This is what makes a session's
  // rebind/recalibrate a true warm path: mutate the RHS here, re-solve on the SAME engine, no recompile.
  // `p` must be the same problem shape this engine was compiled from (row count enforced; topology is the
  // caller's contract, guarded upstream by the structure fingerprint).
  void set_quotes(const BundleProblem& p) {
    if (p.n_residuals() != n_gen_)
      throw std::invalid_argument("CompiledBundleResidual::set_quotes: instrument count differs");
    for (int row = 0; row < n_gen_; ++row) {
      const Instrument& ins = p.instruments[row];
      set_quote(row, ins.market, ins.band_lower, ins.band_upper, ins.band_decay);
    }
    // The DF memo (df_x_) keys on x alone -- DF = exp(-Wx) is quote-independent -- so it stays valid.
  }
  // SCALAR quote updates (the object model, 2026-09-10): the compiled structure is immutable, the quote RHS
  // is per-row data. No Instrument is copied. An FX-forward row never carries a band (see the ctor).
  void set_quote(int row, double market, double lower, double upper, double decay) {
    market_[row] = market;
    if (!fx_row_[row]) {
      band_lo_[row] = lower;
      band_up_[row] = upper;
      band_dc_[row] = decay;
      bands_dirty_ = true;
    }
  }
  void set_market(int row, double market) { market_[row] = market; }
  double market_of(int row) const { return market_[row]; }
  // DF = exp(-W_all x) (memoized on x). Exposed for profiling / downstream analytics.
  const Eigen::VectorXd& discount_factors(const Eigen::VectorXd& x) const { return df_at(x); }

  // Model rates in BundleProblem's residual order: the generic instruments in insertion order, batched
  // by quote kind internally then scattered back to each instrument's own row.
  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd& DF = df_at(x);
    const Eigen::VectorXd& INV = inv_;  // valid whenever df_at(x) is (same memo)
    if (has_moment_) { gen_pos_.set_state(x); gen_neg_.set_state(x); gen_rate_.set_state(x); gen_mtm_.set_state(x); }
    out_.setZero(n_residuals());  // ACCUMULATE: a portfolio row sums its components' weighted quotes; a
                                  // plain row has one entry with weight 1 (0 + 1·q == q, bit-identical).
    if (!q_rows_.empty()) {
      const Eigen::VectorXd& ann = gen_fixed_.annuity(DF);  // refs into DISTINCT batch objects,
      const Eigen::VectorXd& pp = gen_pos_.pv(DF, INV);     // so all three are simultaneously live
      const Eigen::VectorXd& pn = gen_neg_.pv(DF, INV);
      if (gen_mtm_.has_mtm()) {  // + the MtM funding leg (already divided by fx_spot: R is the bare reset ratio)
        const Eigen::VectorXd& pm = gen_mtm_.pv(DF, INV);
        for (std::size_t j = 0; j < q_rows_.size(); ++j) {
          const int i = static_cast<int>(j);
          out_[q_rows_[j].row] += q_rows_[j].weight * (pp[i] - pn[i] + pm[i]) / ann[i];
        }
      } else {
        for (std::size_t j = 0; j < q_rows_.size(); ++j) {
          const int i = static_cast<int>(j);
          out_[q_rows_[j].row] += q_rows_[j].weight * (pp[i] - pn[i]) / ann[i];
        }
      }
    }
    // Zero-coupon rows: the ParRate quotient just accumulated is transformed in place (standalone rows only,
    // so out_[row] IS the quotient). Before the FX/turn rows (disjoint) and before any band (residuals_vs).
    for (const auto& z : zc_rows_) out_[z.row] = zero_coupon_transform_d(out_[z.row], z.tau).first;
    if (!r_rows_.empty()) {
      const Eigen::VectorXd& v = gen_rate_.rate(DF, INV);
      for (std::size_t j = 0; j < r_rows_.size(); ++j)
        out_[r_rows_[j].row] += r_rows_[j].weight * v[static_cast<int>(j)];
    }
    // TURN rows (docs/turns-calibration.md): the model quote is the raw jump δ = x[state index]. This is
    // a STATE-PIN, not a function of DF -- read it straight from x. Accumulate like every other row (a
    // standalone turn is one entry, weight 1, on its own row).
    for (const auto& t : turn_rows_) out_[t.row] += t.weight * x[t.state_index];
    // FX forward OUTRIGHT F = fx_spot · DF_num(T) / DF_den(T) (the model quote; its LOG-basis residual is
    // applied in residuals_vs). Just a DF ratio -- the two DFs were registered into W like any other.
    for (const auto& f : fx_rows_)
      out_[f.row] = f.fx_spot * DF[f.idx_num] / DF[f.idx_den];
    return out_;
  }

  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const { return residuals_vs(x, market_); }

  // Residual against an ARBITRARY market q -- the live streaming feed, not the stored anchor market. With
  // no bands this is model_rates(x) - q; with a band it is the SOFT residual w(q_model)·(q_model - q).
  // This is what lets the frozen-Newton streamer solve the banded LEAST-SQUARES on the fast path (drive
  // this residual, not model_rates-q): at the soft minimum dx = J⁺·r -> 0 even though r != 0.
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    const Eigen::VectorXd& mr = model_rates(x);  // model_rates fills out_; res_ (a distinct member) holds r
    res_ = mr - q;
    for (const auto& b : bands())  // banded rows: the Huber band residual (problem.hpp band_residual)
      res_[b.row] = band_residual_d(mr[b.row], q[b.row], b.lower, b.upper, b.decay).first;
    // FX rows: the residual is the implied-basis discrepancy (ln F_model − ln q)/T in RATE units, NOT the
    // raw outright difference F − q. (FX rows are never banded, so this cleanly overwrites mr − q.)
    for (const auto& f : fx_rows_) {
      using std::log;
      res_[f.row] = (log(mr[f.row]) - log(q[f.row])) / f.fx_time;
    }
    return res_;
  }

  // Analytic block Jacobian J = dr/dx = -(dr/dDF diag(DF)) W_all. dr/dDF (G) is the per-instrument
  // pricing sensitivity scattered into the global DF columns; the W_all matmul maps DF-space back to
  // the stacked knot space. Block structure (an instrument touches only its role curves' knots) falls
  // out automatically because those are the only nonzero W_all columns for its DF entries.
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const { return jacobian_vs(x, market_); }

  // Jacobian of residuals_vs(x, q): the band chain-rule term (q_model - q) uses the SAME market q as the
  // residual, so a frozen-Newton streamer's M is consistent with the residual it drives.
  Eigen::MatrixXd jacobian_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    const Eigen::VectorXd& DF = df_at(x);
    const Eigen::VectorXd& INV = inv_;
    if (has_moment_) { gen_pos_.set_state(x); gen_neg_.set_state(x); gen_rate_.set_state(x); gen_mtm_.set_state(x); }
    // Capture the model quotes for banded rows BEFORE the batch scratch below is overwritten.
    const std::vector<Band>& bl = bands();
    if (!bl.empty()) {
      const Eigen::VectorXd& mr = model_rates(x);
      qb_.resize(static_cast<int>(bl.size()));
      for (std::size_t k = 0; k < bl.size(); ++k) qb_[static_cast<int>(k)] = mr[bl[k].row];
    }
    G_.setZero();  // reuse the scratch buffer (sized once in the ctor)
    pricing::RowMatrixXd& G = G_;

    // Quotient rows: (pv_pos - pv_neg)/annuity, the ONE transform behind both ParRate (an empty `neg`
    // leg => pv_neg == 0) and ParSpread. Each batch row j lands on the instrument's own residual row.
    if (!q_rows_.empty()) {
      const int nq = static_cast<int>(q_rows_.size());
      // Compute each batch's per-coupon numerator (the sub-period gather + reduce) ONCE, then feed it
      // to BOTH the value pass (pv_from_num) and the derivative pass (d_pv_from_num) -- the gather no
      // longer runs a second time for the derivative. Const refs: each accessor returns a ref into ITS
      // OWN batch object's scratch (gen_pos_/gen_neg_/gen_fixed_ are distinct), so all stay live -- no
      // per-call vector copies. Only `num` (a genuine difference) lands in a reused member scratch.
      const Eigen::VectorXd& num_pos = gen_pos_.num(DF, INV);
      const Eigen::VectorXd& num_neg = gen_neg_.num(DF, INV);
      num_.noalias() = gen_pos_.pv_from_num(num_pos, DF) - gen_neg_.pv_from_num(num_neg, DF);
      if (gen_mtm_.has_mtm()) { num_mtm_ = gen_mtm_.num(DF, INV); num_ += gen_mtm_.pv_from_num(num_mtm_, DF); }
      const Eigen::VectorXd& num = num_;
      const Eigen::VectorXd& ann = gen_fixed_.annuity(DF);
      dnum_.setZero();
      dann_.setZero();
      pricing::RowMatrixXd& dnum = dnum_;
      pricing::RowMatrixXd& dann = dann_;
      gen_pos_.d_pv_from_num(num_pos, DF, INV, dnum, 0, 1.0);
      gen_neg_.d_pv_from_num(num_neg, DF, INV, dnum, 0, -1.0);
      if (gen_mtm_.has_mtm()) gen_mtm_.d_pv_from_num(num_mtm_, DF, INV, dnum, 0, 1.0);
      gen_fixed_.d_annuity(dann, 0);
      for (int j = 0; j < nq; ++j)  // d(num/ann) = dnum/ann - num·dann/ann²; accumulate (portfolio rows)
        G.row(q_rows_[j].row) +=
            q_rows_[j].weight * (dnum.row(j) / ann[j] - num[j] * dann.row(j) / (ann[j] * ann[j]));
      // Zero-coupon chain rule: dr/dDF = (dr/dq)·dq/dDF with q = num/ann the row's own quotient (a zc row
      // is standalone, weight 1). Applied BEFORE the band scale below, exactly as residuals_vs orders them.
      for (const auto& z : zc_rows_) {
        const double sc = zero_coupon_transform_d(num[z.batch] / ann[z.batch], z.tau).second;
        G.row(z.row) *= sc;
        if (has_moment_) row_scale_[z.row] *= sc;
      }
      if (has_moment_) { ann_keep_ = ann; num_keep_ = num; }  // the quotient factors for the direct terms below
    }
    // `Rate` rows ARE the futures batch's rate rows (convexity is a constant -> zero row).
    if (!r_rows_.empty()) {
      const int nr = static_cast<int>(r_rows_.size());
      dr_.setZero();
      pricing::RowMatrixXd& dr = dr_;
      gen_rate_.d_rate(DF, INV, dr, 0);
      for (int j = 0; j < nr; ++j) G.row(r_rows_[j].row) += r_rows_[j].weight * dr.row(j);
    }
    // FX rows: r = (ln F − ln q)/T with F = fx_spot·DF_num/DF_den. Only TWO dr/dDF entries per row:
    //   dr/dDF_num = +1/(DF_num·T),  dr/dDF_den = −1/(DF_den·T).
    // The -(G·diag(DF))·W matmul below then yields the constant (W_den − W_num)/T row (ln F is affine in
    // x). FX rows are disjoint from the batch/band rows, so G starts at zero here.
    for (const auto& f : fx_rows_) {
      G(f.row, f.idx_num) += 1.0 / (DF[f.idx_num] * f.fx_time);
      G(f.row, f.idx_den) += -1.0 / (DF[f.idx_den] * f.fx_time);
    }
    // Band chain rule: dr/dx = (dr/dq)·dq/dx with dr/dq = decay inside the band, 1 outside (the Huber
    // residual, problem.hpp). Scale each banded row's dr/dDF (G) by that slope before the W matmul (the
    // matmul is linear, so scaling commutes). Turn rows have a zero G row (no DF dependence), so scaling
    // them here is a no-op -- their band factor is applied to the DIRECT ∂δ/∂x entry below instead.
    if (has_moment_) row_scale_.setOnes(n_residuals());
    for (std::size_t k = 0; k < bl.size(); ++k) {
      const Band& b = bl[k];
      const int i = static_cast<int>(k);
      const double sc = band_residual_d(qb_[i], q[b.row], b.lower, b.upper, b.decay).second;
      G.row(b.row) *= sc;
      if (has_moment_) row_scale_[b.row] *= sc;
    }
    // SUPPORT-BLOCKED product replacing the dense -(G·diag(DF))·W GEMM: J.row(r) = -Σ_{t ∈ sup(r)}
    // G(r,t)·DF[t]·W.row(t). G's nonzeros per row are exactly the row's registered times (recorded once
    // in the ctor), so this sums Σ|sup| × span terms instead of n_res × T × n_knots -- the audit's U1
    // (the dense GEMM was ~38x the residual cost at desk scale). Three structural exploits, all
    // ctor-precomputed: TIME-MAJOR order (one W row stays L1-hot across every residual row sharing it),
    // TRANSPOSED accumulation (Jt.col(r) -= c·Wt.col(t): contiguous axpys; one transpose at the end),
    // and W-row SPANS (time t touches only its curve's ancestry knots [wlo, whi)). Turn rows have empty
    // support -> zero rows, exactly as the GEMM gave.
    Jt_.setZero(Wt_.rows(), n_gen_);
    const int T = static_cast<int>(Wt_.cols());
    for (int t = 0; t < T; ++t) {
      const int lo = wlo_[t], len = whi_[t] - lo;
      if (len <= 0) continue;
      const auto w = Wt_.col(t).segment(lo, len);
      const double dft = DF[t];
      for (int s = tsup_ptr_[t]; s < tsup_ptr_[t + 1]; ++s) {
        const int r = tsup_row_[s];
        const double c = G(r, t) * dft;
        if (c != 0.0) Jt_.col(r).segment(lo, len).noalias() -= c * w;
      }
    }
    Eigen::MatrixXd J = Jt_.transpose();
    // MOMENT coupons: the ½·step·xᵀQx correction is a function of x, not of DF, so its derivative enters J
    // DIRECTLY (after the W product), through the same quotient / band / zc factors as the row's DF terms:
    //   quotient rows: ∂r/∂x_j += weight·scale·(∂num/∂x_j)/ann,  rate rows: ∂r/∂x_j += weight·scale·∂rate/∂x_j.
    if (has_moment_) {
      gen_pos_.moment_direct_pv(DF, 1.0, [&](int bi, int j, double v) {
        const auto& s = q_rows_[bi]; J(s.row, j) += s.weight * row_scale_[s.row] * v / ann_keep_[bi]; });
      gen_neg_.moment_direct_pv(DF, -1.0, [&](int bi, int j, double v) {
        const auto& s = q_rows_[bi]; J(s.row, j) += s.weight * row_scale_[s.row] * v / ann_keep_[bi]; });
      gen_mtm_.moment_direct_pv(DF, 1.0, [&](int bi, int j, double v) {
        const auto& s = q_rows_[bi]; J(s.row, j) += s.weight * row_scale_[s.row] * v / ann_keep_[bi]; });
      gen_rate_.moment_direct_rate([&](int bi, int j, double v) {
        const auto& s = r_rows_[bi]; J(s.row, j) += s.weight * row_scale_[s.row] * v; });
    }
    // TURN rows: r = (banded) (δ − market) with δ = weight·x[state index] -- LINEAR in x, and independent
    // of every DF, so its Jacobian is a single DIRECT entry ∂r/∂x[state index], not part of the W matmul.
    // The band slope dr/dq (decay inside, 1 outside) multiplies that entry (matches residuals_vs).
    for (const auto& t : turn_rows_) {
      double factor = t.weight;
      for (const auto& b : bl)
        if (b.row == t.row) {
          const double qm = t.weight * x[t.state_index];  // the model quote for this row (== mr[t.row])
          factor *= band_residual_d(qm, q[t.row], b.lower, b.upper, b.decay).second;
          break;
        }
      J(t.row, t.state_index) += factor;
    }
    return J;
  }

 private:
  // DF = exp(-W_all x), memoized on x. model_rates(x) and jacobian(x) are called at the SAME x within
  // an LM step (the accepted point), so they share ONE W*x + exp instead of recomputing it. The gate is
  // exact equality on x (short-circuit on size), so the returned DF is bit-identical to cs_.df(x).
  const Eigen::VectorXd& df_at(const Eigen::VectorXd& x) const {
    if (x.size() != df_x_.size() || (x.array() != df_x_.array()).any()) {
      cs_.df_into(x, df_);  // allocation-free recompute into the df_ scratch
      inv_ = df_.cwiseInverse();  // the shared reciprocals: n_times divides ONCE, none per coupon
      df_x_ = x;
    }
    return df_;
  }

  // Register the generic instruments, preserving their insertion order in the RESIDUAL rows while
  // batching them by quote kind (a batch must be homogeneous, and d_pv/d_rate scatter into a
  // CONTIGUOUS row block). q_rows_/r_rows_ map a batch position back to its residual row, so a mixed
  // list of ParRate/ParSpread/Rate instruments still fills exactly the rows BundleProblem documents.
  //
  // ParRate and ParSpread share ONE pair of float batches: both are (pv_pos - pv_neg)/annuity, with
  // ParRate contributing an EMPTY `neg` leg (a leg with no coupons has no entries in R_cpn, so its
  // pv row is exactly 0.0 and its d_pv contributes nothing).
  void register_generic(const BundleProblem& p) {
    for (int row = 0; row < static_cast<int>(p.instruments.size()); ++row) {
      const Instrument& ins = p.instruments[row];
      // A bid/offer band re-weights this row's residual (r = w(q)·(q-market)); the weight is a per-row
      // scalar post-transform, so the row stays on the W-cache path. Captured here, applied below.
      // NOT for an FX forward: its residual is the log-basis transform, which the templated
      // instrument_residual never bands -- banding it here (residual AND Jacobian row) made the compiled
      // and AAD paths disagree on a banded FX pin.
      fx_row_[row] = (ins.quote == QuoteKind::FxForward) ? 1 : 0;
      band_lo_[row] = ins.band_lower;
      band_up_[row] = ins.band_upper;
      band_dc_[row] = ins.band_decay;
      if (ins.band_upper > ins.band_lower && ins.quote != QuoteKind::FxForward)
        band_.push_back({row, ins.band_lower, ins.band_upper, ins.band_decay});
      register_at(ins, row, 1.0, p.curves);
    }
  }

  // Register one instrument's legs into the batches, targeting residual `row` with `weight`. A Portfolio
  // recurses -- each component registers onto the SAME row with the product of weights -- so a butterfly
  // of par swaps becomes three weighted batch entries summed into one row, fully on the W-cache path.
  // Only a genuinely non-W-cacheable LEAF forces the AAD engine: an FX forward or MtM leg nested in a
  // Portfolio, an incomplete or SEASONED MtM leg, a compounded observation (hybrid_residual.hpp
  // instrument_is_noncacheable). A standalone FX forward and a par MtM leg compile (since 2026-09-09).
  void register_at(const Instrument& ins, int row, double weight, const std::vector<BundleCurveSpec>& curves) {
    static const std::vector<pricing::FloatCoupon> no_leg;
    if (ins.quote == QuoteKind::Portfolio) {
      for (const auto& comp : ins.combination) register_at(comp.instrument, row, weight * comp.weight, curves);
      return;
    }
    if (ins.quote == QuoteKind::FxForward) {
      // FX forward F = fx_spot·DF_num(T)/DF_den(T). ln F is AFFINE in x (ln DF = -W·x), so the residual
      // (ln F − ln q)/T rides the W-cache: register the two DFs and remember them; dr/dDF is two entries.
      // Only STANDALONE FX (weight 1) -- a Σ of FX log-residuals inside a Portfolio isn't this transform.
      if (weight != 1.0)
        throw std::invalid_argument("CompiledBundleResidual: FX-forward inside a Portfolio is not W-cacheable; use the AAD engine");
      fx_rows_.push_back({row, cs_.reg(ins.fx_num, ins.fx_time), cs_.reg(ins.fx_den, ins.fx_time),
                          ins.fx_spot, ins.fx_time});
      return;
    }
    if (ins.quote == QuoteKind::XccyMtmBasis) {
      // EXACT (2026-09-09; until then the FX-reset funding term had to be numerically negligible and was
      // dropped, else the row went to AAD): the MtM basis quote (pv_self − pv_fx)/ann + mtm/(fx_spot·ann) is the
      // quotient of the pos/neg batches PLUS the MtM batch, whose coupons are the product of registered DFs
      // R·(A + DF[e] − DF[s]) with R = DF[num](reset)/DF[den](reset) — W-cacheable with hand-written partials
      // (BundleFloatBatch::add_mtm / d_pv_from_num). Linear in the row weight, so it composes inside a Portfolio.
      if (ins.mtm.forecast < 0 || ins.mtm.discount < 0 || ins.mtm.reset_num < 0 || ins.mtm.reset_den < 0)
        throw std::invalid_argument("CompiledBundleResidual: an XccyMtmBasis row needs a complete MtM leg (forecast/discount/reset_num/reset_den)");
      gen_pos_.add(cs_, ins.fwd.forecast, ins.fwd.discount, ins.fwd.coupons);         // + pv_self
      gen_neg_.add(cs_, ins.bench.forecast, ins.bench.discount, ins.bench.coupons);   // − pv_fx
      gen_fixed_.add(cs_, ins.fixed.discount, ins.fixed.coupons);                     // annuity
      gen_mtm_.add_mtm(cs_, ins.mtm.forecast, ins.mtm.discount, ins.mtm.reset_num, ins.mtm.reset_den, ins.mtm.coupons);
      q_rows_.push_back({row, weight});
      return;
    }
    if (ins.quote == QuoteKind::Rate) {
      gen_rate_.add_future(cs_, ins.forecast, ins.obs, ins.convexity);
      r_rows_.push_back({row, weight});
      return;
    }
    if (ins.quote == QuoteKind::TurnJump) {
      // State-pin on turn δ. δ lives at the END of its curve's state block: global offset of the curve +
      // its interpolation-knot count + the turn's index. No DF is touched -- model_rates/jacobian read x.
      int off = 0;
      for (int k = 0; k < ins.turn_curve; ++k) off += curves[k].n_knots();
      const int state_index = off + curves[ins.turn_curve].n_interp_knots() + ins.turn_index;
      if (ins.turn_index < 0 || ins.turn_index >= static_cast<int>(curves[ins.turn_curve].turns.size()))
        throw std::invalid_argument("CompiledBundleResidual: TurnJump turn_index out of range");
      turn_rows_.push_back({row, state_index, weight});
      return;
    }
    if (ins.quote == QuoteKind::ZeroCouponRate) {
      // The ParRate quotient of the same legs with a per-row nonlinear post-transform (problem.hpp
      // zero_coupon_transform). Only STANDALONE (weight 1): a Σ of transformed quotes is not one quotient.
      if (weight != 1.0)
        throw std::invalid_argument("CompiledBundleResidual: ZeroCouponRate inside a Portfolio is not W-cacheable; use the AAD engine");
      zc_rows_.push_back({row, static_cast<int>(q_rows_.size()), zero_coupon_tau(ins)});
    }
    const bool spread = (ins.quote == QuoteKind::ParSpread);
    const FloatLeg& pos = spread ? ins.bench : ins.fwd;  // ParSpread: +bench; ParRate: +fwd
    gen_pos_.add(cs_, pos.forecast, pos.discount, pos.coupons);
    if (spread)
      gen_neg_.add(cs_, ins.fwd.forecast, ins.fwd.discount, ins.fwd.coupons);  // ParSpread: -fwd
    else
      gen_neg_.add(cs_, 0, 0, no_leg);  // ParRate: nothing subtracted
    gen_fixed_.add(cs_, ins.fixed.discount, ins.fixed.coupons);
    gen_mtm_.add(cs_, 0, 0, no_leg);  // keeps the MtM batch index-aligned with the quotient rows (empty here)
    q_rows_.push_back({row, weight});
  }

  int n_gen_;
  pricing::CompiledCurveSet cs_;
  // The ONE generic block (design §3): ParRate/ParSpread share the pos/neg float pair + fixed annuity;
  // Rate futures use the rate batch.
  pricing::BundleFloatBatch gen_pos_, gen_neg_, gen_rate_, gen_mtm_;  // gen_mtm_: MtM funding legs, index-aligned with q_rows_
  pricing::BundleFixedLegs gen_fixed_;
  // Batch position -> (residual row, weight). A plain instrument is one batch entry with weight 1 on its
  // own row; a Portfolio's components are several batch entries that ACCUMULATE (weighted) onto the ONE
  // portfolio row -- which is exactly why a portfolio of W-cacheable components stays W-cacheable.
  struct Scatter { int row; double weight; };
  std::vector<Scatter> q_rows_, r_rows_;
  // Zero-coupon rows: residual row, its quotient batch index (into q_rows_/num/ann) and the single accrual τ.
  struct ZcRow { int row, batch; double tau; };
  std::vector<ZcRow> zc_rows_;
  // FX-forward rows: F = fx_spot·DF[idx_num]/DF[idx_den] at time fx_time; residual (ln F − ln q)/fx_time.
  // Affine in x (ln DF = −Wx), so it rides the W-cache with a constant Jacobian row -- no AAD needed.
  struct Fx { int row, idx_num, idx_den; double fx_spot, fx_time; };
  std::vector<Fx> fx_rows_;
  // Turn state-pin rows: r = (banded) δ − market, δ = weight·x[state_index]. Linear in x, no DF -- the
  // Jacobian is a direct unit entry (see jacobian_vs). Empty for a bundle with no turn instruments.
  struct TurnRow { int row, state_index; double weight; };
  std::vector<TurnRow> turn_rows_;
  // Bid/offer bands: a residual row whose value + Jacobian get the w(q) post-transform (see residuals /
  // jacobian). Empty for a plain bundle, so the fast path is untouched when no instrument is banded.
  struct Band { int row; double lower, upper, decay; };
  // The band table is DERIVED from the per-row arrays below (rebuilt lazily after a scalar set_quote; the
  // vector was reserved to n_gen_ in the ctor, so a rebuild never allocates).
  mutable std::vector<Band> band_;
  mutable bool bands_dirty_ = false;
  std::vector<char> fx_row_;                       // an FX-forward row: never banded
  std::vector<double> band_lo_, band_up_, band_dc_;  // per-row band (upper <= lower: none)
  const std::vector<Band>& bands() const {
    if (bands_dirty_) {
      band_.clear();
      for (int row = 0; row < n_gen_; ++row)
        if (!fx_row_[row] && band_up_[row] > band_lo_[row]) band_.push_back({row, band_lo_[row], band_up_[row], band_dc_[row]});
      bands_dirty_ = false;
    }
    return band_;
  }
  Eigen::VectorXd market_;
  // Mutable per-call scratch (② reused Jacobian buffers, ③ DF memo) -- state that only CACHES pure
  // functions of x, so const-ness of residuals()/jacobian() is preserved semantically.
  mutable Eigen::VectorXd df_, df_x_, inv_;
  bool has_moment_ = false;                    // any batch carries moment-path coupons (set_state + direct terms)
  mutable Eigen::VectorXd row_scale_, ann_keep_, num_keep_;  // jacobian_vs scratch for the moment direct terms
  mutable Eigen::VectorXd out_, res_;  // model_rates / residuals result scratch (const-ref returns)
  mutable Eigen::VectorXd qb_, num_, num_mtm_;   // jacobian_vs scratch: banded model quotes, quotient numerator, MtM numerator
  // ROW-MAJOR: every fill site (the d_* scatters, the per-row G assembly, the band row
  // scaling) and the product's G(r,t) reads are row-local, so row-major makes them contiguous
  // (a col-major .row() expression is strided by n_res -- it dominated the fill cost).
  mutable pricing::RowMatrixXd G_, dnum_, dann_, dr_;
  // Support-blocked Jacobian structure (ctor-built, constant): time-major CSR (time -> residual rows
  // using it), each W row's nonzero column span, the transposed W (contiguous per-time columns), and
  // the transposed-accumulation scratch.
  std::vector<int> tsup_ptr_, tsup_row_;
  std::vector<int> wlo_, whi_;
  Eigen::MatrixXd Wt_;
  mutable Eigen::MatrixXd Jt_;
};

}  // namespace swaps::calibration
