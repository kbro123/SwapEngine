#pragma once
// Compiled (W-cache) residual for the multi-curve BundleProblem -- the bundle analogue of
// CompiledResidual, built on the multi-curve pricing/compiled_book.hpp engine. Produces model rates /
// residuals and the ANALYTIC block Jacobian (no AAD in the hot loop), so the warm & streaming
// re-calibrators drive the bundle on the microsecond fast path instead of an AAD sweep per refresh.
//
// Same factorization as the single-curve CompiledResidual: DF_all = exp(-W_all x) once, then the cheap
// per-row transform, and J = -(dr/dDF diag(DF)) W_all with dr/dDF the analytic per-instrument sensitivity
// scattered into the global DF columns. Residual order matches BundleProblem::residuals: the generic
// instruments in insertion order (batched internally by shape, then scattered to each row).
//
// THE ROW MODEL (architecture review item 2, 2026-09-22). Every residual row is
//
//     r_row = rho_row( Σ_terms weight · xf(value) , q_row )
//
// ONE list of TERMS, each `weight × xf(value)` accumulated onto its row, where the VALUE comes from one of
// four SOURCES (Quotient = a float-numerator / annuity pair in the batches; Rate = a futures row of the
// rate batch; State = a raw state entry x[i], the turn jump; FxRatio = fx_spot · DF/DF), `xf` is a scalar
// per-TERM transform (none, or the zero-coupon compounding r = (1+τq)^(1/τ) − 1), and rho_row is the
// per-ROW residual map (plain q − market; the Huber bid/offer band; the FX log-basis (ln F − ln q)/T).
// A Portfolio is then nothing special: its components are terms with the product of weights onto the one
// row -- so a portfolio of zero-coupon rates or of FX forwards rides the W-cache exactly as the templated
// instrument_model_quote defines it (Σ of transformed component quotes, plain residual). Until 2026-09-22
// those two were refused here (five row families with per-family loops; the FX row ASSIGNED and the
// zero-coupon transform was applied after accumulation, so neither could sit inside a Σ).
// The Jacobian is the same chain: each term scatters weight·xf'·∂value/∂DF into G, rho' scales the row,
// the support-blocked −(G·diag(DF))·W product maps to knot space, and State terms add their direct entry.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
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
  // `pwl` (EXPERIMENT, exp/piecewise-linear-w): accept value-dependent (piecewise-linear) curves at any time;
  // their W is taken at a state and the OWNER must call rebuild_W(x) whenever x's branch pattern changes.
  explicit CompiledBundleResidual(const BundleProblem& p, bool pwl = false)
      : n_gen_(static_cast<int>(p.instruments.size())), market_(p.market()) {
    cs_.init(p.curves);  // p.curves ARE pricing::CurveStructure now (BundleCurveSpec is an alias), no copy
    if (pwl) cs_.enable_pwl();
    row_log_.assign(n_gen_, 0.0);
    band_lo_.assign(n_gen_, 0.0);
    band_up_.assign(n_gen_, 0.0);
    band_dc_.assign(n_gen_, 1.0);
    maps_.reserve(n_gen_);  // row_maps() never allocates after this

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
    dnum_.resize(static_cast<int>(qterm_.size()), T);
    dann_.resize(static_cast<int>(qterm_.size()), T);
    dr_.resize(static_cast<int>(rterm_.size()), T);
    tf_.resize(static_cast<int>(terms_.size()));
    row_scale_.resize(n_gen_);

    // ---- the support-blocked Jacobian's STRUCTURE (built once; see jacobian_vs) --------------------
    // G is structurally BLOCK-SPARSE: row r is nonzero only at the DF times instrument r's legs actually
    // registered (its pay dates + sub-period boundaries + FX pillar times). Those positions are known
    // exactly from the batch index arrays, so record them per residual row as a CSR support list. The
    // final product then sums ONLY over each row's support instead of a dense n_res × T × n_knots GEMM.
    {
      std::vector<std::vector<int>> sup(n_gen_);
      const auto add_float = [&](const pricing::BundleFloatBatch& b, const std::vector<int>& term_of) {
        for (int i = 0; i < b.n_coupons(); ++i)
          if (b.pay[i] >= 0) sup[terms_[term_of[b.inst[i]]].row].push_back(b.pay[i]);  // futures carry pay = -1
        for (int j = 0; j < static_cast<int>(b.subS.size()); ++j) {
          const int r = terms_[term_of[b.inst[b.sub_cpn[j]]]].row;
          sup[r].push_back(b.subS[j]);
          sup[r].push_back(b.subE[j]);
        }
      };
      add_float(gen_pos_, qterm_);
      add_float(gen_neg_, qterm_);
      add_float(gen_rate_, rterm_);
      add_float(gen_mtm_, qterm_);
      for (int i = 0; i < gen_mtm_.n_coupons(); ++i)  // the MtM coupon's reset ratio and notional exchanges
        if (gen_mtm_.rN[i] >= 0) {
          const int r = terms_[qterm_[gen_mtm_.inst[i]]].row;
          sup[r].push_back(gen_mtm_.rN[i]); sup[r].push_back(gen_mtm_.rD[i]);
          sup[r].push_back(gen_mtm_.dS[i]); sup[r].push_back(gen_mtm_.dE[i]);
        }
      for (int i = 0; i < static_cast<int>(gen_fixed_.pay.size()); ++i)
        sup[terms_[qterm_[gen_fixed_.inst[i]]].row].push_back(gen_fixed_.pay[i]);
      for (const Term& t : terms_)
        if (t.src == Src::FxRatio) {
          const FxRatio& f = fx_[t.idx];
          sup[t.row].push_back(f.idx_num);
          sup[t.row].push_back(f.idx_den);
          if (f.idx_snum >= 0) { sup[t.row].push_back(f.idx_snum); sup[t.row].push_back(f.idx_sden); }
        }
      for (int r = 0; r < n_gen_; ++r) {
        std::sort(sup[r].begin(), sup[r].end());
        sup[r].erase(std::unique(sup[r].begin(), sup[r].end()), sup[r].end());
      }
      // ROW-MAJOR CSR (r -> its support times): what the row-map scaling walks (it scales G on the support only).
      rsup_ptr_.assign(n_gen_ + 1, 0);
      for (int r = 0; r < n_gen_; ++r) rsup_ptr_[r + 1] = rsup_ptr_[r] + static_cast<int>(sup[r].size());
      rsup_t_.resize(rsup_ptr_[n_gen_]);
      for (int r = 0; r < n_gen_; ++r) std::copy(sup[r].begin(), sup[r].end(), rsup_t_.begin() + rsup_ptr_[r]);
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
      build_wt();
    }
  }

  // EXPERIMENT (exp/piecewise-linear-w): re-take W at x (its branch-pattern cell) and everything derived
  // from it -- the transposed copy, the per-time nonzero spans, the DF memo. The batches hold DF INDICES
  // only, so nothing else depends on W's values. Only valid on a pwl-constructed engine; costs one AAD pass
  // per value-dependent curve + the block rebuild, and runs only when the pattern actually changes.
  void rebuild_W(const Eigen::VectorXd& x) {
    cs_.set_eval_state(x);
    cs_.finalize();
    build_wt();
    df_stale_ = true;  // the memo keyed on x alone is stale: the same x now maps through a new W
  }
  // EXPERIMENT (analytic re-take): prepare the rank-k structure for curve c (-1 = unsupported), and apply
  // one update -- W, its ancestry blocks (CompiledCurveSet), W^T and the per-time spans. Allocation-free.
  int pwl_prepare(int c) { return cs_.pwl_prepare(c); }
  void pwl_rank_update(int pc, const std::vector<int>& nodes, const Eigen::Ref<const Eigen::MatrixXd>& dM) {
    cs_.pwl_rank_update(pc, nodes, dM);
    mirror_delta(pc);
  }
  void pwl_set(int pc, const Eigen::Ref<const Eigen::MatrixXd>& M) {
    cs_.pwl_set(pc, M);
    mirror_delta(pc);
  }
  void mirror_delta(int pc) {  // the last CompiledCurveSet delta into W^T and the spans
    const auto& P = cs_.pwl_curve(pc);
    for (int r = 0; r < static_cast<int>(P.rows.size()); ++r) {
      const int g = P.rows[r].g;
      Wt_.col(g).segment(P.off, P.ni) += P.D.row(r).transpose();
      wlo_[g] = std::min(wlo_[g], P.off);  // a zero entry may have become nonzero: widen (a superset is exact)
      whi_[g] = std::max(whi_[g], P.off + P.ni);
    }
    df_stale_ = true;
  }
  const pricing::CompiledCurveSet& curve_set() const { return cs_; }

  int n_residuals() const { return n_gen_; }
  int n_times() const { return cs_.n_times(); }
  // The number of TERMS (row contributions) the engine compiled: n_residuals for a plain bundle, more when
  // a Portfolio row has several components. Observability for the row model, not a hot-path quantity.
  int n_terms() const { return static_cast<int>(terms_.size()); }

  // Overwrite the quote RHS -- targets and soft-quote bands -- WITHOUT touching the compiled structure.
  // The W-cache, batches and scatter maps depend only on topology (legs, times, curve roles); the market
  // vector and the row maps are plain per-row data read at residual time. This is what makes a session's
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
  // is per-row data. No Instrument is copied. A log-residual (standalone FX-forward) row never carries a
  // band (the templated instrument_residual applies the log map first; see register_generic).
  void set_quote(int row, double market, double lower, double upper, double decay) {
    market_[row] = market;
    if (row_log_[row] == 0.0) {
      band_lo_[row] = lower;
      band_up_[row] = upper;
      band_dc_[row] = decay;
      maps_dirty_ = true;
    }
  }
  void set_market(int row, double market) { market_[row] = market; }
  double market_of(int row) const { return market_[row]; }
  // DF = exp(-W_all x) (memoized on x). Exposed for profiling / downstream analytics.
  const Eigen::VectorXd& discount_factors(const Eigen::VectorXd& x) const { return df_at(x); }

  // Model rates in BundleProblem's residual order: every term's weight·xf(value) accumulated onto its
  // row. A plain row is one term with weight 1 (0 + 1·q == q, bit-identical); a Portfolio row sums its
  // components' weighted (transformed) quotes in component order, exactly as instrument_model_quote does.
  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd& DF = df_at(x);
    const Eigen::VectorXd& INV = inv_;  // valid whenever df_at(x) is (same memo)
    if (has_moment_) { gen_pos_.set_state(x); gen_neg_.set_state(x); gen_rate_.set_state(x); gen_mtm_.set_state(x); }
    out_.setZero(n_residuals());
    // The batch passes: each accessor returns a ref into ITS OWN batch's scratch (distinct objects), so
    // all stay live across the term loop below. Empty batches are skipped (their refs are never read).
    const bool mtm = gen_mtm_.has_mtm();
    const Eigen::VectorXd* ann = nullptr; const Eigen::VectorXd* pp = nullptr; const Eigen::VectorXd* pn = nullptr;
    const Eigen::VectorXd* pm = nullptr; const Eigen::VectorXd* rt = nullptr;
    if (!qterm_.empty()) {
      ann = &gen_fixed_.annuity(DF);
      pp = &gen_pos_.pv(DF, INV);
      pn = &gen_neg_.pv(DF, INV);
      if (mtm) pm = &gen_mtm_.pv(DF, INV);  // the MtM funding leg (already divided by fx_spot: R is the bare reset ratio)
    }
    if (!rterm_.empty()) rt = &gen_rate_.rate(DF, INV);
    for (const Term& t : terms_) {
      switch (t.src) {
        case Src::Quotient: {
          double num = (*pp)[t.idx] - (*pn)[t.idx];
          if (mtm) num += (*pm)[t.idx];
          // (weight·num)/ann: the association the pre-row-model quotient loop used, bit for bit.
          if (t.xf == Xf::ZeroCoupon) out_[t.row] += t.weight * zero_coupon_transform_d(num / (*ann)[t.idx], t.tau).first;
          else out_[t.row] += t.weight * num / (*ann)[t.idx];
          break;
        }
        case Src::Rate: out_[t.row] += t.weight * (*rt)[t.idx]; break;
        case Src::State: out_[t.row] += t.weight * x[t.idx]; break;  // a STATE-PIN (the turn jump δ): straight from x, no DF
        case Src::FxRatio: out_[t.row] += t.weight * fx_value(fx_[t.idx], DF); break;
      }
    }
    return out_;
  }

  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const { return residuals_vs(x, market_); }

  // Residual against an ARBITRARY market q -- the live streaming feed, not the stored anchor market. A plain
  // row is model_rates(x) - q; a mapped row applies its residual map rho_row (the Huber band residual
  // w(q_model)·(q_model − q), or the FX log-basis (ln F − ln q)/T in RATE units). This is what lets the
  // frozen-Newton streamer solve the banded LEAST-SQUARES on the fast path (drive this residual, not
  // model_rates-q): at the soft minimum dx = J⁺·r -> 0 even though r != 0.
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    residuals_from(model_rates(x), q, res_);  // model_rates fills out_; res_ (a distinct member) holds r
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
    Eigen::MatrixXd J;
    jacobian_vs_into(x, q, J);
    return J;
  }
  // The same Jacobian written into a caller-owned J, resized only when its shape differs: the streamer keeps J in a member, so a
  // refresh allocates no Jacobian (C6, 2026-09-15). Same operations in the same order as jacobian_vs: bit-identical.
  // The same, and the residuals at (x, q) into *r (S2, 2026-09-15): what a streamer refresh reads its band sides from.
  // The residuals come from the ROW VALUES THE JACOBIAN PASS ITSELF ACCUMULATED (mrj_: the batches' numerators
  // and annuities, the rate batch, x, the FX ratios -- the same quantities the slopes rho' were taken at), so r
  // and J are consistent by construction and the refresh runs ONE set of batch passes, not a Jacobian's plus a
  // residual's. Equals residuals_vs(x, q) to rounding (pv vs pv_from_num assemble the same coupons).
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J, Eigen::VectorXd* r) const {
    if (!r) { jacobian_vs_into(x, q, J); return; }
    jacobian_core(x, q, J, /*values=*/true);
    residuals_from(mrj_, q, *r);
  }
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J) const {
    jacobian_core(x, q, J, /*values=*/false);
  }

 private:
  // r = rho_row(mr, q): a plain row is mr − q, a mapped row its residual map (band / log).
  void residuals_from(const Eigen::VectorXd& mr, const Eigen::VectorXd& q, Eigen::VectorXd& r) const {
    r = mr - q;
    for (const RowMap& m : row_maps()) r[m.row] = row_residual_d(m, mr[m.row], q[m.row]).first;
  }

  // The Jacobian pass. `values`: also accumulate every row's model value into mrj_ (the 4-arg form's residual);
  // the mapped rows' values are accumulated regardless, since their slope rho'(q_model) needs them.
  void jacobian_core(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J, bool values) const {
    const Eigen::VectorXd& DF = df_at(x);
    const Eigen::VectorXd& INV = inv_;
    if (has_moment_) { gen_pos_.set_state(x); gen_neg_.set_state(x); gen_rate_.set_state(x); gen_mtm_.set_state(x); }
    const std::vector<RowMap>& maps = row_maps();
    values = values || !maps.empty();
    G_.setZero();  // reuse the scratch buffer (sized once in the ctor)
    pricing::RowMatrixXd& G = G_;
    row_scale_.setOnes();  // the per-row residual-map slope rho' (1 for a plain row)

    // The batch derivative passes. Quotient terms need num/ann and their partials; Rate terms the rate
    // batch's. Each accessor returns a ref into ITS OWN batch object's scratch, so all stay live -- no
    // per-call vector copies. Only `num` (a genuine difference) lands in a reused member scratch.
    const Eigen::VectorXd* ann = nullptr;
    if (!qterm_.empty()) {
      // Compute each batch's per-coupon numerator (the sub-period gather + reduce) ONCE, then feed it
      // to BOTH the value pass (pv_from_num) and the derivative pass (d_pv_from_num) -- the gather no
      // longer runs a second time for the derivative.
      const Eigen::VectorXd& num_pos = gen_pos_.num(DF, INV);
      const Eigen::VectorXd& num_neg = gen_neg_.num(DF, INV);
      num_.noalias() = gen_pos_.pv_from_num(num_pos, DF) - gen_neg_.pv_from_num(num_neg, DF);
      if (gen_mtm_.has_mtm()) { num_mtm_ = gen_mtm_.num(DF, INV); num_ += gen_mtm_.pv_from_num(num_mtm_, DF); }
      ann = &gen_fixed_.annuity(DF);
      dnum_.setZero();
      dann_.setZero();
      gen_pos_.d_pv_from_num(num_pos, DF, INV, dnum_, 0, 1.0);
      gen_neg_.d_pv_from_num(num_neg, DF, INV, dnum_, 0, -1.0);
      if (gen_mtm_.has_mtm()) gen_mtm_.d_pv_from_num(num_mtm_, DF, INV, dnum_, 0, 1.0);
      gen_fixed_.d_annuity(dann_, 0);
      if (has_moment_) { ann_keep_ = *ann; num_keep_ = num_; }  // the quotient factors for the direct terms below
    }
    const Eigen::VectorXd* rt = nullptr;
    if (!rterm_.empty()) {
      dr_.setZero();
      gen_rate_.d_rate(DF, INV, dr_, 0);
      if (values) rt = &gen_rate_.rate(DF, INV);  // the rate batch's values (its own scratch; d_rate touched none of it)
    }
    if (values) mrj_.setZero(n_gen_);
    // ONE term loop: each term scatters weight·xf'(value)·∂value/∂DF onto its row of G (a Portfolio's
    // components ACCUMULATE), and -- when values are wanted -- weight·xf(value) onto its row of mrj_ with the
    // same association as model_rates (weight·num/den). tf_ keeps the chain factor weight·xf' for the moment
    // direct terms.
    {
      const Eigen::VectorXd& num = num_;
      const pricing::RowMatrixXd& dnum = dnum_;
      const pricing::RowMatrixXd& dann = dann_;
      const pricing::RowMatrixXd& dr = dr_;
      for (std::size_t k = 0; k < terms_.size(); ++k) {
        const Term& t = terms_[k];
        double f = t.weight;
        switch (t.src) {
          case Src::Quotient: {
            const int i = t.idx;
            // Zero-coupon chain rule: xf' at the term's OWN quotient num/ann, folded into the chain factor.
            if (t.xf == Xf::ZeroCoupon) f *= zero_coupon_transform_d(num[i] / (*ann)[i], t.tau).second;
            // d(num/ann) = dnum/ann - num·dann/ann²
            G.row(t.row) += f * (dnum.row(i) / (*ann)[i] - num[i] * dann.row(i) / ((*ann)[i] * (*ann)[i]));
            if (values) {
              if (t.xf == Xf::ZeroCoupon) mrj_[t.row] += t.weight * zero_coupon_transform_d(num[i] / (*ann)[i], t.tau).first;
              else mrj_[t.row] += t.weight * num[i] / (*ann)[i];
            }
            break;
          }
          case Src::Rate:  // `Rate` rows ARE the futures batch's rate rows (convexity is a constant -> zero row)
            G.row(t.row) += f * dr.row(t.idx);
            if (values) mrj_[t.row] += t.weight * (*rt)[t.idx];
            break;
          case Src::State:  // no DF dependence: the direct ∂/∂x entry is added after the W product
            if (values) mrj_[t.row] += t.weight * x[t.idx];
            break;
          case Src::FxRatio: {
            // F = fx_spot·DF_num/DF_den (· DF_sden/DF_snum): ∂F/∂DF_i = ±F/DF_i on its (two or four) DFs.
            // On a standalone row the Log map's slope 1/(F·T) then makes the row the constant (W_den − W_num)/T.
            const FxRatio& fx = fx_[t.idx];
            const double v = fx_value(fx, DF);
            const double F = f * v;
            G(t.row, fx.idx_num) += F * INV[fx.idx_num];
            G(t.row, fx.idx_den) += -F * INV[fx.idx_den];
            if (fx.idx_snum >= 0) {  // O-X3: F gains DF_den(t_s)/DF_num(t_s)
              G(t.row, fx.idx_sden) += F * INV[fx.idx_sden];
              G(t.row, fx.idx_snum) += -F * INV[fx.idx_snum];
            }
            if (values) mrj_[t.row] += t.weight * v;
            break;
          }
        }
        tf_[static_cast<int>(k)] = f;
      }
    }
    // Row map chain rule: dr/dx = rho'(q_model)·dq/dx -- decay inside a Huber band, 1 outside; 1/(F·T) on a
    // log row -- at the row value this pass accumulated. Scale each mapped row's dr/dDF (G) by that slope
    // before the W matmul (the matmul is linear, so scaling commutes), and keep it in row_scale_ for the
    // DIRECT entries (moment and State terms).
    for (const RowMap& m : maps) {
      const double sc = row_residual_d(m, mrj_[m.row], q[m.row]).second;
      for (int s = rsup_ptr_[m.row]; s < rsup_ptr_[m.row + 1]; ++s) G(m.row, rsup_t_[s]) *= sc;  // its support only: the product reads nothing else
      row_scale_[m.row] *= sc;
    }
    // SUPPORT-BLOCKED product replacing the dense -(G·diag(DF))·W GEMM: J.row(r) = -Σ_{t ∈ sup(r)}
    // G(r,t)·DF[t]·W.row(t). G's nonzeros per row are exactly the row's registered times (recorded once
    // in the ctor), so this sums Σ|sup| × span terms instead of n_res × T × n_knots -- the audit's U1
    // (the dense GEMM was ~38x the residual cost at desk scale). Three structural exploits, all
    // ctor-precomputed: TIME-MAJOR order (one W row stays L1-hot across every residual row sharing it),
    // TRANSPOSED accumulation (Jt.col(r) -= c·Wt.col(t): contiguous axpys; one transpose at the end),
    // and W-row SPANS (time t touches only its curve's ancestry knots [wlo, whi)). State-only rows have
    // empty support -> zero rows, exactly as the GEMM gave.
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
    J.resize(Jt_.cols(), Jt_.rows());
    J = Jt_.transpose();
    // MOMENT coupons: the ½·step·xᵀQx correction is a function of x, not of DF, so its derivative enters J
    // DIRECTLY (after the W product), through the same term chain factor and row slope as the row's DF terms:
    //   quotient terms: ∂r/∂x_j += tf·rho'·(∂num/∂x_j)/ann,  rate terms: ∂r/∂x_j += tf·rho'·∂rate/∂x_j.
    if (has_moment_) {
      const auto add_q = [&](int bi, int j, double v) {
        const int k = qterm_[bi];
        J(terms_[k].row, j) += tf_[k] * row_scale_[terms_[k].row] * v / ann_keep_[bi];
      };
      gen_pos_.moment_direct_pv(DF, 1.0, add_q);
      gen_neg_.moment_direct_pv(DF, -1.0, add_q);
      gen_mtm_.moment_direct_pv(DF, 1.0, add_q);
      gen_rate_.moment_direct_rate([&](int bi, int j, double v) {
        const int k = rterm_[bi];
        J(terms_[k].row, j) += tf_[k] * row_scale_[terms_[k].row] * v;
      });
    }
    // STATE terms: value = x[idx] -- LINEAR in x and independent of every DF, so the Jacobian is a single
    // DIRECT entry weight·rho' at the state index, not part of the W matmul (matches residuals_vs).
    for (const Term& t : terms_)
      if (t.src == Src::State) J(t.row, t.idx) += t.weight * row_scale_[t.row];
  }

  // ---- the row model ------------------------------------------------------------------------------
  enum class Src : unsigned char { Quotient, Rate, State, FxRatio };  // where a term's value comes from
  enum class Xf : unsigned char { None, ZeroCoupon };                 // the per-term scalar transform
  // One contribution weight·xf(value) to residual row `row`. `idx` indexes the source: the quotient batch
  // position (Quotient), the rate batch position (Rate), the global state entry (State), or fx_ (FxRatio).
  struct Term { int row; double weight; Src src; Xf xf; int idx; double tau; };
  // F = fx_spot·DF[idx_num]/DF[idx_den] (· DF[idx_sden]/DF[idx_snum] when a spot time is set, O-X3).
  struct FxRatio { int idx_num, idx_den; double fx_spot; int idx_snum = -1, idx_sden = -1; };
  static double fx_value(const FxRatio& f, const Eigen::VectorXd& DF) {
    double v = f.fx_spot * DF[f.idx_num] / DF[f.idx_den];
    if (f.idx_snum >= 0) v *= DF[f.idx_sden] / DF[f.idx_snum];  // O-X3: roll back from the spot date
    return v;
  }
  // The per-ROW residual map rho(q_model, q): a Huber bid/offer band {lower, upper, decay} (problem.hpp
  // band_residual) or the FX log-basis (ln q_model − ln q)/T (RATE units, CLAUDE.md §2: a 1bp basis error
  // maps to ~1bp regardless of tenor). A plain row has no entry. Returns {r, dr/dq_model}.
  struct RowMap { int row; bool log; double a, b, c; };  // log: a = T; band: a = lower, b = upper, c = decay
  static std::pair<double, double> row_residual_d(const RowMap& m, double mr, double q) {
    if (m.log) {
      using std::log;
      return {(log(mr) - log(q)) / m.a, 1.0 / (mr * m.a)};
    }
    return band_residual_d(mr, q, m.a, m.b, m.c);
  }

  // DF = exp(-W_all x), memoized on x. model_rates(x) and jacobian(x) are called at the SAME x within
  // an LM step (the accepted point), so they share ONE W*x + exp instead of recomputing it. The gate is
  // exact equality on x (short-circuit on size), so the returned DF is identical to cs_.df(x) (the same path).
  // W transposed once: Wt.col(t) == W.row(t), CONTIGUOUS (col-major). And each W row is itself
  // sparse -- time t on curve c touches only c's ancestry knots -- so record its nonzero column SPAN
  // [wlo, whi) once and axpy only that segment (a spread-chain's early curves touch a fraction of
  // the state, so this cuts both the flops and the Jt write traffic).
  void build_wt() {
    const int T = cs_.n_times();
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

  const Eigen::VectorXd& df_at(const Eigen::VectorXd& x) const {
    if (df_stale_ || x.size() != df_x_.size() || (x.array() != df_x_.array()).any()) {
      df_stale_ = false;
      cs_.df_into(x, df_);  // allocation-free recompute into the df_ scratch
      inv_ = df_.cwiseInverse();  // the shared reciprocals: n_times divides ONCE, none per coupon
      df_x_ = x;
    }
    return df_;
  }

  // Register the generic instruments, preserving their insertion order in the RESIDUAL rows while
  // batching their legs by shape (a batch must be homogeneous, and d_pv/d_rate scatter into a
  // CONTIGUOUS row block). Each instrument becomes one or more TERMS onto its row; qterm_/rterm_ map a
  // batch position back to its term, so a mixed list of quote kinds still fills exactly the rows
  // BundleProblem documents.
  //
  // ParRate and ParSpread share ONE pair of float batches: both are (pv_pos - pv_neg)/annuity, with
  // ParRate contributing an EMPTY `neg` leg (a leg with no coupons has no entries in R_cpn, so its
  // pv row is exactly 0.0 and its d_pv contributes nothing).
  void register_generic(const BundleProblem& p) {
    for (int row = 0; row < static_cast<int>(p.instruments.size()); ++row) {
      const Instrument& ins = p.instruments[row];
      // The row's residual map is decided by the TOP-LEVEL quote kind, exactly as instrument_residual does:
      // a standalone FX forward is the log-basis row (never banded -- the templated residual applies the log
      // map first); everything else takes the bid/offer band when one is set (r = w(q)·(q − market), a per-
      // row scalar post-transform, so the row stays on the W-cache path).
      row_log_[row] = (ins.quote == QuoteKind::FxForward) ? ins.fx_time : 0.0;
      band_lo_[row] = ins.band_lower;
      band_up_[row] = ins.band_upper;
      band_dc_[row] = ins.band_decay;
      register_at(ins, row, 1.0, p.curves);
    }
    maps_dirty_ = true;
  }

  // Register one instrument as terms onto residual `row` with `weight`. A Portfolio recurses -- each
  // component registers onto the SAME row with the product of weights -- so a butterfly of par swaps
  // becomes three weighted quotient terms summed into one row, fully on the W-cache path; likewise a
  // portfolio of FX forwards (a Σ of outrights, plain residual) or of zero-coupon rates (a Σ of transformed
  // quotients). Only a genuinely non-W-cacheable LEAF forces the AAD engine: an incomplete or SEASONED MtM
  // leg, or a compounded observation (hybrid_residual.hpp instrument_is_noncacheable).
  void register_at(const Instrument& ins, int row, double weight, const std::vector<BundleCurveSpec>& curves) {
    static const std::vector<pricing::FloatCoupon> no_leg;
    if (ins.quote == QuoteKind::Portfolio) {
      for (const auto& comp : ins.combination) register_at(comp.instrument, row, weight * comp.weight, curves);
      return;
    }
    if (ins.quote == QuoteKind::FxForward) {
      // F = fx_spot·DF_num(T)/DF_den(T): register the DFs (two, or four with a spot time) like any other.
      FxRatio f{cs_.reg(ins.fx_num, ins.fx_time), cs_.reg(ins.fx_den, ins.fx_time), ins.fx_spot};
      if (ins.fx_spot_time != 0.0) {  // O-X3: two more registered DFs, only when a spot time is set (W layout unchanged at 0)
        f.idx_snum = cs_.reg(ins.fx_num, ins.fx_spot_time);
        f.idx_sden = cs_.reg(ins.fx_den, ins.fx_spot_time);
      }
      fx_.push_back(f);
      terms_.push_back({row, weight, Src::FxRatio, Xf::None, static_cast<int>(fx_.size()) - 1, 0.0});
      return;
    }
    if (ins.quote == QuoteKind::Rate) {
      gen_rate_.add_future(cs_, ins.forecast, ins.obs, ins.convexity);
      rterm_.push_back(static_cast<int>(terms_.size()));
      terms_.push_back({row, weight, Src::Rate, Xf::None, static_cast<int>(rterm_.size()) - 1, 0.0});
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
      terms_.push_back({row, weight, Src::State, Xf::None, state_index, 0.0});
      return;
    }
    // Every remaining kind is a QUOTIENT term: (pv_pos − pv_neg [+ pv_mtm]) / annuity, with the batches
    // index-aligned at this term's quotient position (an unused leg is registered EMPTY so the alignment
    // holds; an empty leg's pv is exactly 0.0 and its d_pv contributes nothing).
    Xf xf = Xf::None;
    double tau = 0.0;
    if (ins.quote == QuoteKind::ZeroCouponRate) {  // the ParRate quotient of the same legs, then r = (1+τq)^(1/τ) − 1
      xf = Xf::ZeroCoupon;
      tau = zero_coupon_tau(ins);
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
    } else {
      const bool spread = (ins.quote == QuoteKind::ParSpread);
      const FloatLeg& pos = spread ? ins.bench : ins.fwd;  // ParSpread: +bench; ParRate: +fwd
      gen_pos_.add(cs_, pos.forecast, pos.discount, pos.coupons);
      if (spread)
        gen_neg_.add(cs_, ins.fwd.forecast, ins.fwd.discount, ins.fwd.coupons);  // ParSpread: -fwd
      else
        gen_neg_.add(cs_, 0, 0, no_leg);  // ParRate: nothing subtracted
      gen_fixed_.add(cs_, ins.fixed.discount, ins.fixed.coupons);
      gen_mtm_.add(cs_, 0, 0, no_leg);  // keeps the MtM batch index-aligned with the quotient terms (empty here)
    }
    qterm_.push_back(static_cast<int>(terms_.size()));
    terms_.push_back({row, weight, Src::Quotient, xf, static_cast<int>(qterm_.size()) - 1, tau});
  }

  int n_gen_;
  pricing::CompiledCurveSet cs_;
  // The ONE generic block (design §3): ParRate/ParSpread/ZeroCoupon/MtM share the pos/neg float pair + the
  // fixed annuity (+ the MtM funding batch); Rate futures use the rate batch.
  pricing::BundleFloatBatch gen_pos_, gen_neg_, gen_rate_, gen_mtm_;  // gen_mtm_: MtM funding legs, index-aligned with the quotient terms
  pricing::BundleFixedLegs gen_fixed_;
  // THE row model: every term, in registration (component) order, and the batch-position -> term maps
  // (quotient batch position i is term qterm_[i]; rate batch position j is term rterm_[j]).
  std::vector<Term> terms_;
  std::vector<int> qterm_, rterm_;
  std::vector<FxRatio> fx_;  // FxRatio terms' DF indices, by Term::idx
  // The per-row residual map data: a log row (T > 0; a standalone FX forward, never banded) or a band
  // (upper > lower). The RowMap list is DERIVED from these (rebuilt lazily after a scalar set_quote; the
  // vector was reserved to n_gen_ in the ctor, so a rebuild never allocates). Empty for a plain bundle, so
  // the fast path is untouched when no row is mapped.
  std::vector<double> row_log_;                      // T of a log row, 0.0 = not a log row
  std::vector<double> band_lo_, band_up_, band_dc_;  // per-row band (upper <= lower: none)
  mutable std::vector<RowMap> maps_;
  mutable bool maps_dirty_ = false;
  const std::vector<RowMap>& row_maps() const {
    if (maps_dirty_) {
      maps_.clear();
      for (int row = 0; row < n_gen_; ++row) {
        if (row_log_[row] > 0.0) maps_.push_back({row, true, row_log_[row], 0.0, 0.0});
        else if (band_up_[row] > band_lo_[row]) maps_.push_back({row, false, band_lo_[row], band_up_[row], band_dc_[row]});
      }
      maps_dirty_ = false;
    }
    return maps_;
  }
  Eigen::VectorXd market_;
  // Mutable per-call scratch (② reused Jacobian buffers, ③ DF memo) -- state that only CACHES pure
  // functions of x, so const-ness of residuals()/jacobian() is preserved semantically.
  mutable Eigen::VectorXd df_, df_x_, inv_;
  mutable bool df_stale_ = false;  // EXPERIMENT: W changed under an unchanged x (pwl re-take)
  bool has_moment_ = false;                    // any batch carries moment-path coupons (set_state + direct terms)
  mutable Eigen::VectorXd row_scale_, tf_, ann_keep_, num_keep_;  // jacobian_vs scratch: row slopes, term chain factors, the moment direct terms' quotient factors
  mutable Eigen::VectorXd out_, res_;  // model_rates / residuals result scratch (const-ref returns)
  mutable Eigen::VectorXd mrj_, num_, num_mtm_;  // jacobian_vs scratch: the pass's own row values, quotient numerator, MtM numerator
  // ROW-MAJOR: every fill site (the d_* scatters, the per-row G assembly, the row-map scaling) and the
  // product's G(r,t) reads are row-local, so row-major makes them contiguous (a col-major .row()
  // expression is strided by n_res -- it dominated the fill cost).
  mutable pricing::RowMatrixXd G_, dnum_, dann_, dr_;
  // Support-blocked Jacobian structure (ctor-built, constant): time-major CSR (time -> residual rows
  // using it), each W row's nonzero column span, the transposed W (contiguous per-time columns), and
  // the transposed-accumulation scratch.
  std::vector<int> tsup_ptr_, tsup_row_;
  std::vector<int> rsup_ptr_, rsup_t_;  // the same support, row-major (the row-map scaling)
  std::vector<int> wlo_, whi_;
  Eigen::MatrixXd Wt_;
  mutable Eigen::MatrixXd Jt_;
};

}  // namespace swaps::calibration
