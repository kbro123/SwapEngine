#pragma once
// AadBlock -- prices the NON-W-cacheable instruments of a bundle (FX/MtM, or a Portfolio that contains
// one) via the templated / forward-AAD path, so a HYBRID engine can keep the cacheable majority on the
// W-cache instead of dropping the whole bundle to AAD.
//
// This is the higher-cost path, so it is optimised for the fact that only the QUOTES change tick to tick
// -- the instruments, curves and knot structure are fixed at construction:
//   * WIDTH REDUCTION (the dominant saving). The Dual is Eigen::AutoDiffScalar<VectorXd> -- a DYNAMIC
//     gradient whose width, and every per-op allocation, scales with the number of seeded knots. So we
//     seed AAD only over the knots these instruments actually TOUCH (their curves + spread ancestors).
//     A couple of FX trades touching two curves differentiate w.r.t. ~20 knots, not the bundle's 100 --
//     the Dual pass and its allocations shrink proportionally, and the block Jacobian is scattered back
//     to those knots' columns.
//   * BUFFER REUSE. The seed vector's gradient basis is CONSTANT (it is the identity over the touched
//     knots), so it is built ONCE; each call only overwrites the seed VALUES. The sub-problem, the
//     touched-column map and the output targets are all sized once. (The per-operation gradient
//     allocations inside Eigen's AutoDiffScalar remain -- eliminating those needs a pooled Dual type,
//     a separate future change.)

#include <Eigen/Core>

#include <algorithm>
#include <set>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"

namespace swaps::calibration {

// ---- vectorised discount-factor cache (hot-path optimisation) ------------------------------------
// The block's coupon times are FIXED at init, so integral(t_i) is a constant linear map over the knot
// forwards: integral(t_i) = M.row(i) . x. So the whole per-step discount layer = one GEMV (M x) + one
// vectorised exp, then an O(log) slot lookup -- instead of the per-call region search + scalar exp that
// dominated the FX/MtM tick. This wraps a real curve: discount(t) returns the precomputed DF for a known
// time and FALLS BACK to the underlying curve for anything else, so it can never be wrong, only faster.
struct CachedDisc : CurveHandle<double> {
  const CurveHandle<double>* real = nullptr;
  const std::vector<double>* times = nullptr;   // sorted distinct query times (fallback search)
  const Eigen::VectorXd* df = nullptr;          // exp(-(M x + b)), refreshed once per residual eval
  // The pricing queries discounts in a FIXED order every eval, so we replay that order with a cursor --
  // direct index, no search. seq_slot[k] is the df index of the k-th discount() call; seq_t[k] its time
  // (guards against any order divergence -> fall back to a search, then an exact curve reprice).
  const std::vector<int>* seq_slot = nullptr;
  const std::vector<double>* seq_t = nullptr;
  mutable std::size_t cursor = 0;
  double discount(double t) const override {
    if (cursor < seq_t->size() && (*seq_t)[cursor] == t)    // fast path: the expected next DF, O(1)
      return (*df)[(*seq_slot)[cursor++]];
    const auto it = std::lower_bound(times->begin(), times->end(), t);  // order diverged: search
    if (it != times->end() && *it == t) return (*df)[static_cast<int>(it - times->begin())];
    return real->discount(t);                   // not a cached time: exact fallback
  }
  // The FX/MtM kernel prices purely off discount(); integral/forward just delegate for safety.
  double integral(double t) const override { return real->integral(t); }
  double forward(double t) const override { return real->forward(t); }
  void set_forwards(const Eigen::Matrix<double, Eigen::Dynamic, 1>&) override {}  // real curve owns knots
};

// Records every discount(t) a curve is asked for during a discovery replay of the block's pricing, so the
// cache set is EXACTLY what the kernel queries (query times are x-independent -- the schedule is fixed).
struct RecordingCurve : CurveHandle<double> {
  const CurveHandle<double>* real = nullptr;
  std::vector<double>* log = nullptr;
  double discount(double t) const override { log->push_back(t); return real->discount(t); }
  double integral(double t) const override { return real->integral(t); }
  double forward(double t) const override { return real->forward(t); }
  void set_forwards(const Eigen::Matrix<double, Eigen::Dynamic, 1>&) override {}
};

class AadBlock {
 public:
  AadBlock() = default;
  bool empty() const { return rows_.empty(); }
  int size() const { return static_cast<int>(rows_.size()); }
  const std::vector<int>& rows() const { return rows_; }

  // Build from the bundle's curves, the non-cacheable instruments, their GLOBAL residual rows, and the
  // global knot count. Determines the touched knots and pre-sizes the reusable width-reduced seed.
  void init(const std::vector<BundleCurveSpec>& curves, std::vector<Instrument> instruments,
            std::vector<int> rows, int n_knots) {
    n_knots_ = n_knots;
    rows_ = std::move(rows);
    sub_.curves = curves;
    sub_.instruments = std::move(instruments);
    if (rows_.empty()) return;

    // Touched knots = every knot of every curve any sub-instrument references, closed under spread
    // ancestry (a spread curve's DFs also depend on its base's knots).
    std::set<int> curves_used;
    for (const auto& ins : sub_.instruments) collect_curves(ins, curves_used);
    std::set<int> closed;
    for (int c : curves_used) {
      for (int a = c; a >= 0; a = sub_.curves[a].base) closed.insert(a);
    }
    std::set<int> knots;
    for (int c : closed) {
      const int off = sub_.offset(c), nk = sub_.curves[c].n_knots();
      for (int i = 0; i < nk; ++i) knots.insert(off + i);
    }
    touched_.assign(knots.begin(), knots.end());  // sorted

    // Build the width-reduced seed ONCE: gradient e_j on touched_[j], zero elsewhere. Only values change.
    const int w = static_cast<int>(touched_.size());
    xd_.resize(n_knots_);
    for (int k = 0; k < n_knots_; ++k) xd_[k].derivatives() = Eigen::VectorXd::Zero(w);
    for (int j = 0; j < w; ++j) xd_[touched_[j]].derivatives()[j] = 1.0;

    // Build the reusable curve objects ONCE (the structure is fixed tick to tick). Every residual/Jacobian
    // pass overwrites their forwards IN PLACE instead of reconstructing the handles/curves each call.
    dcurves_.build(sub_.curves);
    ducurves_.build(sub_.curves);

    build_df_cache();
  }

  // Model quotes of the block's instruments (doubles), written to out[global_row]. (Used only to fill the
  // uniform model_rates vector; a bundle with a non-cacheable instrument does not stream, so these rows
  // are never the driver of a frozen-Newton reprice.)
  void model_rates_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const {
    if (rows_.empty()) return;
    dcurves_.update([&](int c, int i) { return x[sub_.offset(c) + i]; });
    const auto curve_of = [this](int i) -> const CurveHandle<double>& { return dcurves_[i]; };
    for (int j = 0; j < size(); ++j)
      out[rows_[j]] = instrument_model_quote<double>(sub_.instruments[j], curve_of);
  }

  // True residuals (doubles) of the block's instruments against their stored markets, into out[global_row].
  void residuals_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const {
    if (rows_.empty()) return;
    refresh_curves(x);
    const auto curve_of = [this](int i) -> const CurveHandle<double>& { return *resolve_[i]; };
    for (int j = 0; j < size(); ++j)
      out[rows_[j]] = instrument_residual<double>(sub_.instruments[j], curve_of);
  }

  // Jacobian rows d(residual)/dx via WIDTH-REDUCED AAD, into J.row(global_row) of an (n_res x n_knots) J.
  // Only the touched columns are nonzero; the rest stay whatever the caller pre-zeroed.
  void jacobian_into(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const {
    if (rows_.empty()) return;
    for (int k = 0; k < n_knots_; ++k) xd_[k].value() = x[k];   // reuse the seed: values only
    ducurves_.update([&](int c, int i) { return xd_[sub_.offset(c) + i]; });
    const auto curve_of = [this](int i) -> const CurveHandle<ad::Dual>& { return ducurves_[i]; };
    const int w = static_cast<int>(touched_.size());
    for (int j = 0; j < size(); ++j) {
      const ad::Dual rj = instrument_residual<ad::Dual>(sub_.instruments[j], curve_of);
      J.row(rows_[j]).setZero();
      const auto& g = rj.derivatives();
      if (g.size() == w)
        for (int t = 0; t < w; ++t) J(rows_[j], touched_[t]) = g[t];
    }
  }

  // --- Streaming (frozen-Newton) forms: residual/Jacobian against a LIVE market q instead of the stored
  // mids, so a mixed FX/MtM bundle streams on the SAME hybrid engine (cacheable rows W-cache, these rows
  // AAD) rather than recalibrating each tick. residuals_vs_into runs every tick (cheap doubles);
  // jacobian_vs_into runs only on a Jacobian REFRESH (staleness), so the AAD sweep is off the hot path.

  // True residuals against the live market q, into out[global_row]: instrument_residual with q as the
  // target (FX gets ln F_model − ln q[row]; a banded row gets w(q_model)·(q_model − q[row])).
  void residuals_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::VectorXd& out) const {
    if (rows_.empty()) return;
    refresh_curves(x);
    const auto curve_of = [this](int i) -> const CurveHandle<double>& { return *resolve_[i]; };
    for (int j = 0; j < size(); ++j)
      out[rows_[j]] = instrument_residual<double>(sub_.instruments[j], curve_of, q[rows_[j]]);
  }

  // Jacobian of residuals_vs_into via WIDTH-REDUCED AAD, into the touched columns of J.row(global_row).
  // Consistent with residuals_vs_into by construction: AAD differentiates the SAME instrument_residual
  // (so the band chain-rule term (q_model − q) and the FX 1/F_model factor fall out automatically).
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J) const {
    if (rows_.empty()) return;
    for (int k = 0; k < n_knots_; ++k) xd_[k].value() = x[k];   // reuse the seed: values only
    ducurves_.update([&](int c, int i) { return xd_[sub_.offset(c) + i]; });
    const auto curve_of = [this](int i) -> const CurveHandle<ad::Dual>& { return ducurves_[i]; };
    const int w = static_cast<int>(touched_.size());
    for (int j = 0; j < size(); ++j) {
      const ad::Dual rj = instrument_residual<ad::Dual>(sub_.instruments[j], curve_of, q[rows_[j]]);
      J.row(rows_[j]).setZero();
      const auto& g = rj.derivatives();
      if (g.size() == w)
        for (int t = 0; t < w; ++t) J(rows_[j], touched_[t]) = g[t];
    }
  }

 private:
  // Curve roles an instrument reads (recursing through Portfolio components).
  static void collect_curves(const Instrument& ins, std::set<int>& s) {
    auto leg = [&](const FloatLeg& l) {
      s.insert(l.forecast); s.insert(l.discount);
      if (l.reset_num >= 0) s.insert(l.reset_num);
      if (l.reset_den >= 0) s.insert(l.reset_den);
    };
    switch (ins.quote) {
      case QuoteKind::Rate: s.insert(ins.forecast); break;
      case QuoteKind::FxForward:
        if (ins.fx_num >= 0) s.insert(ins.fx_num);
        if (ins.fx_den >= 0) s.insert(ins.fx_den);
        break;
      case QuoteKind::Portfolio:
        for (const auto& c : ins.combination) collect_curves(c.instrument, s);
        break;
      default:  // ParRate / ParSpread / XccyMtmBasis
        leg(ins.fwd); leg(ins.bench); leg(ins.mtm); s.insert(ins.fixed.discount);
        break;
    }
  }

  // Build the per-curve discount cache: discover the query times, then the constant affine map to their
  // integrals (integral(t) = M.row . x + b), so a residual eval computes DF = exp(-(M x + b)) vectorised.
  void build_df_cache() {
    const int NC = static_cast<int>(sub_.curves.size());
    disc_times_.assign(NC, {}); disc_M_.assign(NC, {}); disc_b_.assign(NC, {}); disc_df_.assign(NC, {});
    seq_t_.assign(NC, {}); seq_slot_.assign(NC, {});
    cached_.assign(NC, CachedDisc{}); resolve_.assign(NC, nullptr);
    cached_ids_.clear();
    for (int c = 0; c < NC; ++c) resolve_[c] = &dcurves_[c];  // default: the real reusable curve
    if (rows_.empty()) return;

    // 1) Discovery replay: record every discount(t) per curve IN ORDER. The query TIMES are x-independent
    //    (the schedule is fixed), so any x works -- price the block once at x = 0 with recording curves.
    dcurves_.update([](int, int) { return 0.0; });
    std::vector<std::vector<double>> seq(NC);   // ordered discount-query times, as the kernel asks them
    std::vector<RecordingCurve> rec(NC);
    for (int c = 0; c < NC; ++c) { rec[c].real = &dcurves_[c]; rec[c].log = &seq[c]; }
    const auto rec_of = [&](int i) -> const CurveHandle<double>& { return rec[i]; };
    for (const auto& ins : sub_.instruments) (void)instrument_residual<double>(ins, rec_of);

    // 2) Per queried curve: the DISTINCT sorted times (df layout + fallback search) and the AFFINE map to
    //    integral(t) via AAD (integral is affine in x for any streamable/constant-W scheme, so derivatives
    //    = M exactly at any seed and value at x=0 = b). Then the ordered play-list of slots (seq_slot).
    std::vector<ad::Dual> xd(n_knots_);
    for (int k = 0; k < n_knots_; ++k) { xd[k].value() = 0.0; xd[k].derivatives() = Eigen::VectorXd::Unit(n_knots_, k); }
    const auto Cu = build_bundle_curves<ad::Dual>(sub_.curves, [&](int c, int i) { return xd[sub_.offset(c) + i]; });
    for (int c = 0; c < NC; ++c) {
      if (seq[c].empty()) continue;
      std::vector<double> uniq = seq[c];
      std::sort(uniq.begin(), uniq.end());
      uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
      const int nt = static_cast<int>(uniq.size());
      disc_times_[c] = uniq;
      disc_M_[c].resize(nt, n_knots_);
      disc_b_[c].resize(nt);
      for (int i = 0; i < nt; ++i) {
        const ad::Dual I = Cu[c]->integral(uniq[i]);
        disc_b_[c][i] = I.value();
        if (I.derivatives().size() == n_knots_) disc_M_[c].row(i) = I.derivatives().transpose();
        else disc_M_[c].row(i).setZero();
      }
      disc_df_[c].resize(nt);
      // ordered play-list: slot of each recorded query, so eval is df[seq_slot[cursor++]] -- no search.
      seq_t_[c] = seq[c];
      seq_slot_[c].resize(seq[c].size());
      for (std::size_t k = 0; k < seq[c].size(); ++k)
        seq_slot_[c][k] = static_cast<int>(std::lower_bound(uniq.begin(), uniq.end(), seq[c][k]) - uniq.begin());
      cached_[c].real = &dcurves_[c]; cached_[c].times = &disc_times_[c]; cached_[c].df = &disc_df_[c];
      cached_[c].seq_t = &seq_t_[c]; cached_[c].seq_slot = &seq_slot_[c];
      resolve_[c] = &cached_[c];
      cached_ids_.push_back(c);
    }
  }

  // Per residual eval: update the reusable curves (fallback + integral/forward), recompute each cached
  // curve's DFs as one GEMV + one vectorised exp, and rewind its cursor to replay the fixed query order.
  void refresh_curves(const Eigen::VectorXd& x) const {
    dcurves_.update([&](int c, int i) { return x[sub_.offset(c) + i]; });
    for (int c : cached_ids_) { disc_df_[c] = (-(disc_M_[c] * x + disc_b_[c])).array().exp(); cached_[c].cursor = 0; }
  }

  BundleProblem sub_;          // the non-cacheable instruments over the SAME curves (built once)
  std::vector<int> rows_;      // block index -> global residual row
  std::vector<int> touched_;   // global knot indices these instruments differentiate w.r.t. (sorted)
  int n_knots_ = 0;
  mutable Eigen::Matrix<ad::Dual, Eigen::Dynamic, 1> xd_;  // reused width-reduced seed (values updated)
  mutable BundleCurveSet<double> dcurves_;     // reusable double curves: built once, forwards updated in place
  mutable BundleCurveSet<ad::Dual> ducurves_;  // reusable Dual curves for the width-reduced Jacobian

  // ---- vectorised discount cache (per curve; only for curves the block prices off) ----
  std::vector<std::vector<double>> disc_times_;  // [curve] sorted distinct query times
  std::vector<Eigen::MatrixXd> disc_M_;          // [curve] (n_times x n_knots): integral(t) = M.row.x + b
  std::vector<Eigen::VectorXd> disc_b_;          // [curve] constant term of the affine integral map
  mutable std::vector<Eigen::VectorXd> disc_df_; // [curve] exp(-(M x + b)), refreshed per eval
  std::vector<std::vector<double>> seq_t_;        // [curve] ordered query times (cursor replay guard)
  std::vector<std::vector<int>> seq_slot_;        // [curve] ordered df slots -> discount is df[seq_slot[k]]
  std::vector<CachedDisc> cached_;               // [curve] cache handle (stable address -> pointers hold)
  std::vector<const CurveHandle<double>*> resolve_;  // [curve] cached_ where cached, else the real curve
  std::vector<int> cached_ids_;                  // curve ids that have a cache (drives the refresh loop)
};

}  // namespace swaps::calibration
