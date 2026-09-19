#pragma once
// HybridBundleResidual -- the residual engine for a BundleProblem that MIXES W-cacheable instruments
// with a few non-cacheable ones (FX/MtM, or a Portfolio containing them). Instead of dropping the whole
// bundle to AAD because one instrument isn't W-cacheable, it splits the work:
//   * cacheable rows  -> a CompiledBundleResidual over that subset (the W-cache: DF = exp(-Wx), analytic
//     Jacobian) -- the microsecond majority;
//   * non-cacheable rows -> an AadBlock (width-reduced forward-AAD over just those instruments).
// then stitches both into the bundle's residual vector and block Jacobian in insertion order.
//
// When NOTHING is non-cacheable (the overwhelmingly common case) it is a transparent, zero-overhead
// delegate to a single CompiledBundleResidual -- the fast path is byte-for-byte unchanged. It only pays
// the split/stitch cost when an FX/MtM (or a portfolio holding one) is actually present.

#include <Eigen/Core>

#include <cstdlib>
#include <optional>
#include <set>
#include <limits>
#include <vector>

#include "swaps/calibration/aad_block.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/problem.hpp"

namespace swaps::calibration {

// True iff any observation this instrument prices is COMPOUNDED (RFR lookback/lockout). The compiled batch
// computes the arithmetic Σ and cannot represent the product (compiled_book push_obs would throw), so such
// an instrument must ride the AAD block. Recursive over Portfolio components.
inline bool has_compounded_obs(const Instrument& ins) {
  const auto leg = [](const FloatLeg& l) {
    for (const auto& c : l.coupons)
      if (c.obs.compounded) return true;
    return false;
  };
  if (ins.quote == QuoteKind::Rate && ins.obs.compounded) return true;
  if (leg(ins.fwd) || leg(ins.bench) || leg(ins.mtm)) return true;
  for (const auto& c : ins.combination)
    if (has_compounded_obs(c.instrument)) return true;
  return false;
}

// (curves_are_noncacheable — "do these curves have a constant W?" — moved to pricing/curve_handle.hpp in
// E6.4 and is re-exported by bundle_problem.hpp; it is a curve-set property, not a residual-engine one.)

// A curve handle that forwards to a real curve and records the LATEST time anything asks of it. Running the
// GENERIC pricer once against these is how the router learns which times a row touches on which curve: the
// answer comes from the pricer itself, so it cannot drift from what the row really reads (2026-09-10).
struct MaxTimeProbe : pricing::CurveHandle<double> {
  const pricing::CurveHandle<double>* real = nullptr;
  double* seen = nullptr;
  void note(double t) const { if (t > *seen) *seen = t; }
  double forward(double t) const override { note(t); return real->forward(t); }
  double integral(double t) const override { note(t); return real->integral(t); }
  double discount(double t) const override { note(t); return real->discount(t); }
  double turn_jump(int j) const override { return real->turn_jump(j); }
  void pieces_into(std::vector<double>& out) const override { real->pieces_into(out); }
  void set_forwards(const Eigen::VectorXd&) override {}
};

// Does every time this instrument touches sit at or below its curve's linear horizon? Only asked when some
// curve HAS a finite horizon, i.e. when a value-dependent region exists somewhere; otherwise every row is
// within reach by construction and the probe is skipped entirely (so an all-linear bundle pays nothing).
// An instrument that cannot even be priced at a dummy state is treated as non-cacheable, not as an error.
inline bool instrument_within_horizons(const Instrument& ins, const std::vector<BundleCurveSpec>& curves,
                                       const std::vector<double>& horizons) {
  const int NC = static_cast<int>(curves.size());
  std::vector<double> seen(static_cast<std::size_t>(NC), -std::numeric_limits<double>::infinity());
  const auto real = build_bundle_curves<double>(curves, [](int, int) { return 0.03; });
  std::vector<MaxTimeProbe> probe(static_cast<std::size_t>(NC));
  for (int c = 0; c < NC; ++c) {
    probe[static_cast<std::size_t>(c)].real = real[static_cast<std::size_t>(c)].get();
    probe[static_cast<std::size_t>(c)].seen = &seen[static_cast<std::size_t>(c)];
  }
  const auto of = [&probe](int i) -> const pricing::CurveHandle<double>& {
    return probe[static_cast<std::size_t>(i)];
  };
  try {
    (void)instrument_model_quote<double>(ins, of);
  } catch (const std::exception&) {
    return false;  // unpriceable at a dummy state: let the general path deal with it
  }
  for (int c = 0; c < NC; ++c)
    if (seen[static_cast<std::size_t>(c)] > horizons[static_cast<std::size_t>(c)]) return false;
  return true;
}

// True iff this instrument must go to the AAD block rather than the W-cache. Both cross-currency quotes are
// now W-cacheable in their standard form: a STANDALONE FX forward (affine (ln F − ln q)/T residual) and a
// MtM-xccy basis with a PAR funding leg (its FX-reset-notional term is identically zero, so it collapses to
// the ParSpread quotient). Only a non-par MtM funding leg -- a genuine curve-dependent notional -- still
// needs AAD. FX/MtM INSIDE a Portfolio are excluded too (the compiled transforms don't compose in a Σ).
// A compounded (RFR lookback/lockout) observation anywhere also forces AAD -- the batch's arithmetic Σ
// cannot represent the product. This asks only "can the BATCH express this row's shape?"; whether the row's
// times reach a value-dependent part of a curve is the separate, per-curve horizon question above.
inline bool instrument_is_noncacheable(const Instrument& ins, const std::vector<BundleCurveSpec>& curves) {
  if (has_compounded_obs(ins)) return true;
  // A MtM basis is cacheable only if its FX-reset funding term is NUMERICALLY negligible on the real
  // (until 2026-09-09 a numeric "funding term negligible" test decided this; the batch now prices the leg exactly)
  // An MtM xccy basis row is W-cacheable EXACTLY since 2026-09-09 (BundleFloatBatch::add_mtm prices the resetting
  // notional as a product of registered DFs); only an incomplete MtM leg (no reset roles) stays on AAD.
  if (ins.quote == QuoteKind::XccyMtmBasis) {
    if (ins.mtm.forecast < 0 || ins.mtm.discount < 0 || ins.mtm.reset_num < 0 || ins.mtm.reset_den < 0) return true;
    for (const auto& c : ins.mtm.coupons)
      if (pricing::mtm_coupon_is_seasoned(c)) return true;  // E3-S2/G4: a seasoned coupon prices on the templated kernel
    return false;
  }
  if (ins.quote == QuoteKind::Portfolio)
    for (const auto& c : ins.combination)
      if (c.instrument.quote == QuoteKind::FxForward ||
          c.instrument.quote == QuoteKind::ZeroCouponRate ||
          instrument_is_noncacheable(c.instrument, curves))
        return true;
  return false;
}

// EXPERIMENT (exp/piecewise-linear-w): the default for the piecewise-linear W tier. OFF unless the process sets
// SWAPS_EXP_PWL=1, so the shipped engine and every gate are byte-unchanged; the env switch lets the WHOLE
// correctness suite run on the new tier without editing a test.
inline bool exp_pwl_default() {
  static const bool on = [] { const char* e = std::getenv("SWAPS_EXP_PWL"); return e && e[0] == '1'; }();
  return on;
}

class HybridBundleResidual {
 public:
  explicit HybridBundleResidual(const BundleProblem& p) : HybridBundleResidual(p, exp_pwl_default()) {}
  // `pwl` (EXPERIMENT): rows that read past a value-dependent region's start stay on the COMPILED engine,
  // whose W is re-taken at x whenever x's branch pattern changes (sync) -- instead of going to the AAD block.
  HybridBundleResidual(const BundleProblem& p, bool pwl) : n_res_(static_cast<int>(p.instruments.size())) {
    validate_problem(p, "HybridBundleResidual");  // E1/E2/B12: refuse a malformed bundle before compiling it
    // ONE partition pass. Each instrument's cacheability is decided once (the MtM guard inside
    // instrument_is_noncacheable prices real cashflows, so it is not free -- do not re-ask per consumer).
    // Per-curve LINEAR HORIZONS replace the old whole-bundle veto (2026-09-10). A value-dependent region
    // used to send EVERY instrument in the bundle to the AAD block, including par swaps on curves that never
    // touched it. Now a row is only pushed off the W-cache if it actually reads past a horizon. When every
    // curve is fully linear -- which is every shipped bundle -- `mixed` is false and no probe runs at all,
    // so this costs nothing except where it buys something.
    const std::vector<double> horizons = pricing::curve_linear_horizons(p.curves);
    bool mixed = false;
    for (double h : horizons)
      if (h < std::numeric_limits<double>::infinity()) mixed = true;
    if (pwl && mixed && try_pwl(p)) return;
    BundleProblem c;
    c.curves = p.curves;
    std::vector<Instrument> nc;
    std::vector<int> nc_rows;
    cache_pos_.assign(n_res_, -1);
    for (int r = 0; r < n_res_; ++r) {
      const bool off_cache = instrument_is_noncacheable(p.instruments[r], p.curves) ||
                             (mixed && !instrument_within_horizons(p.instruments[r], p.curves, horizons));
      if (off_cache) {
        nc.push_back(p.instruments[r]);
        nc_rows.push_back(r);
      } else {
        cache_pos_[r] = static_cast<int>(cache_rows_.size());
        c.instruments.push_back(p.instruments[r]);
        cache_rows_.push_back(r);
      }
    }
    // Engaged for every fully-linear bundle (even an all-AAD instrument mix, which keeps n_times() honest),
    // and for a mixed bundle whenever some row stayed on the W-cache.
    if (!mixed || !c.instruments.empty()) cacheable_.emplace(c);
    nc_.init(p.curves, std::move(nc), std::move(nc_rows), p.n_knots());
  }

  int n_residuals() const { return n_res_; }
  // EXPERIMENT observability: is the piecewise-linear tier engaged, and how many W re-takes has it done.
  bool pwl_active() const { return pwl_; }
  int pwl_rebuilds() const { return pwl_rebuilds_; }
  int pwl_distinct() const { return static_cast<int>(pwl_seen_.size()); }  // distinct patterns (SWAPS_EXP_PWL_STATS=1)
  int pwl_analytic() const { return pwl_analytic_; }
  int n_times() const { return cacheable_ ? cacheable_->n_times() : 0; }
  // The ROW PARTITION, observable (item 5, 2026-09-10): which engine took a given global row. >=0 is the
  // row's index in the compiled W-cache sub-problem, -1 means it went to the AAD block. Exposed so a test
  // can pin WHERE a row was routed, not merely that the two routes agree -- a router that quietly sent
  // everything to the slow path would otherwise pass every parity test it has.
  int compiled_row(int row) const { return cache_pos_[row]; }
  int n_compiled_rows() const { return static_cast<int>(cache_rows_.size()); }
  // Is the AAD half on the heap-free pooled dual? False means its touched width exceeded ad::kPooledMaxW and
  // every dual operation now allocates -- a ~100x per-tick cost cliff that nothing else reports. Exposed so
  // the hot-path tests can fail on it rather than merely measure it (2026-09-12).
  bool aad_pooled() const { return nc_.empty() || nc_.pooled(); }
  int aad_width() const { return nc_.touched_width(); }

  // Overwrite the quote RHS (targets + bands) on BOTH halves without touching either's compiled/discovered
  // structure -- the engine-side of a warm rebind. `p` must have this engine's row count and topology
  // (guarded upstream by the structure fingerprint).
  void set_quotes(const BundleProblem& p) {
    if (p.n_residuals() != n_res_)
      throw std::invalid_argument("HybridBundleResidual::set_quotes: instrument count differs");
    for (int r = 0; r < n_res_; ++r) {
      const Instrument& ins = p.instruments[r];
      set_quote(r, ins.market, ins.band_lower, ins.band_upper, ins.band_decay);
    }
  }
  // SCALAR quote updates by GLOBAL row (the object model, 2026-09-10): no Instrument is copied -- until now
  // set_quotes deep-copied every cacheable Instrument to read four doubles (14,641 of a rebind's 15.7k
  // allocations on the census fixture, E3-D5).
  void set_quote(int row, double market, double lower, double upper, double decay) {
    const int j = cache_pos_[row];
    if (j >= 0) cacheable_->set_quote(j, market, lower, upper, decay);
    else nc_.set_quote(row, market, lower, upper, decay);
  }
  void set_market(const Eigen::VectorXd& q) {
    if (q.size() != n_res_) throw std::invalid_argument("HybridBundleResidual::set_market: market length differs");
    for (int r = 0; r < n_res_; ++r) {
      const int j = cache_pos_[r];
      if (j >= 0) cacheable_->set_market(j, q[r]);
      else nc_.set_market(r, q[r]);
    }
  }

  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const {
    sync(x);
    if (nc_.empty() && cacheable_) return cacheable_->model_rates(x);
    if (cacheable_) scatter(cacheable_->model_rates(x), out_);
    else out_.setZero(n_res_);
    nc_.model_rates_into(x, out_);
    return out_;
  }

  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const {
    sync(x);
    if (nc_.empty() && cacheable_) return cacheable_->residuals(x);
    if (cacheable_) scatter(cacheable_->residuals(x), res_);  // cacheable rows (incl. their bands)
    else res_.setZero(n_res_);
    nc_.residuals_into(x, res_);  // non-cacheable rows: the true residual (e.g. FX log-basis)
    return res_;
  }

  // Streaming residual/Jacobian against a live market q (global residual order). Drives frozen-Newton for
  // a mixed FX/MtM bundle too: cacheable rows on the W-cache `_vs` path, non-cacheable rows on the AAD
  // block's `_vs` forms. When nothing is non-cacheable it is a straight delegate (q is already global).
  // The cacheable engine indexes q by ITS sub-rows, so the global q is first gathered onto them.
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    sync(x);
    if (nc_.empty() && cacheable_) return cacheable_->residuals_vs(x, q);
    if (cacheable_) {
      gather_cache(q, qsub_);
      scatter(cacheable_->residuals_vs(x, qsub_), res_);  // cacheable rows (incl. their bands)
    } else {
      res_.setZero(n_res_);
    }
    nc_.residuals_vs_into(x, q, res_);  // non-cacheable rows against the live global q
    return res_;
  }
  Eigen::MatrixXd jacobian_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    Eigen::MatrixXd J;
    jacobian_vs_into(x, q, J);
    return J;
  }
  // The same Jacobian, and the residuals_vs(x, q) values into *r (S2, 2026-09-15): the AAD rows' from the sweep's dual values, the cacheable
  // rows' from their own residuals_vs -- so a streamer refresh reads its band sides without re-evaluating every model quote (on desk_mixed
  // that re-evaluation rebuilt every AAD curve: 192 us and 23 allocations per refresh).
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J, Eigen::VectorXd* r) const {
    if (!r) { jacobian_vs_into(x, q, J); return; }
    sync(x);
    if (nc_.empty() && cacheable_) { cacheable_->jacobian_vs_into(x, q, J, r); return; }
    r->resize(n_res_);
    J.resize(n_res_, nknots(x));
    if (cacheable_) {
      gather_cache(q, qsub_);
      cacheable_->jacobian_vs_into(x, qsub_, Jc_);
      for (std::size_t j = 0; j < cache_rows_.size(); ++j) J.row(cache_rows_[j]) = Jc_.row(static_cast<int>(j));
      const Eigen::VectorXd& rc = cacheable_->residuals_vs(x, qsub_);
      for (std::size_t j = 0; j < cache_rows_.size(); ++j) (*r)[cache_rows_[j]] = rc[static_cast<int>(j)];
    }
    nc_.jacobian_vs_into(x, q, J, r);
  }
  // Into a caller-owned J (C6, 2026-09-15): the cacheable half writes into a member scratch Jc_, so a warm call allocates no matrix.
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J) const {
    sync(x);
    if (nc_.empty() && cacheable_) {
      cacheable_->jacobian_vs_into(x, q, J);
      return;
    }
    // No zeroing: every row is overwritten -- a compiled row whole (below), an AAD row zeroed then written on its touched columns
    // (AadBlock::jacobian_impl). The row partition is total (constructor), so a garbage-filled J comes back exact (JacobianIntoParity).
    J.resize(n_res_, nknots(x));
    if (cacheable_) {
      gather_cache(q, qsub_);
      cacheable_->jacobian_vs_into(x, qsub_, Jc_);
      for (std::size_t j = 0; j < cache_rows_.size(); ++j) J.row(cache_rows_[j]) = Jc_.row(static_cast<int>(j));
    }
    nc_.jacobian_vs_into(x, q, J);  // scatters the non-cacheable rows (touched cols)
  }

  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {
    sync(x);
    if (nc_.empty() && cacheable_) return cacheable_->jacobian(x);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n_res_, nknots(x));
    if (cacheable_) {
      const Eigen::MatrixXd Jc = cacheable_->jacobian(x);  // n_cacheable x n_knots
      for (std::size_t j = 0; j < cache_rows_.size(); ++j) J.row(cache_rows_[j]) = Jc.row(static_cast<int>(j));
    }
    nc_.jacobian_into(x, J);  // scatters the non-cacheable rows (touched cols)
    return J;
  }

 private:
  static int nknots(const Eigen::VectorXd& x) { return static_cast<int>(x.size()); }

  struct PatternTracker {
    int off = 0, ni = 0;
    curve::ModularCurve<double> crv;
    std::vector<unsigned char> cur, cached;
    Eigen::MatrixXd G;   // d(filter inputs)/dx: structure-only (phase 2)
    Eigen::VectorXd z;   // G·x scratch
    // analytic re-take
    int pc = -1, N = 0;
    std::vector<double> h, phi;
    Eigen::MatrixXd M, dM;  // current formula rows Φ_P·G (N x ni); the change scratch
    Eigen::RowVectorXd row;
    std::vector<int> nodes;
    bool dirty = false;
  };

  // EXPERIMENT: engage the piecewise-linear tier. Only the SHAPE question routes a row to AAD (can the batch
  // express it at all?); the horizon question disappears. Refused (-> the shipped routing) if nothing would
  // stay compiled or the compiled engine rejects the bundle (e.g. moment-path coupons on a value-dependent
  // curve: forward_weight_matrix has no piecewise form yet).
  bool try_pwl(const BundleProblem& p) {
    BundleProblem c;
    c.curves = p.curves;
    std::vector<Instrument> nc;
    std::vector<int> nc_rows, rows, pos(n_res_, -1);
    for (int r = 0; r < n_res_; ++r) {
      if (instrument_is_noncacheable(p.instruments[r], p.curves)) {
        nc.push_back(p.instruments[r]);
        nc_rows.push_back(r);
      } else {
        pos[r] = static_cast<int>(rows.size());
        c.instruments.push_back(p.instruments[r]);
        rows.push_back(r);
      }
    }
    if (c.instruments.empty()) return false;
    try {
      cacheable_.emplace(c, /*pwl=*/true);
    } catch (const std::exception&) {
      cacheable_.reset();
      return false;
    }
    cache_rows_ = std::move(rows);
    cache_pos_ = std::move(pos);
    nc_.init(p.curves, std::move(nc), std::move(nc_rows), p.n_knots());
    auto& eng = *cacheable_;
    const auto& cs = eng.curve_set();
    for (int k = 0; k < static_cast<int>(p.curves.size()); ++k)
      if (cs.value_dependent(k)) {
        const int ni = p.curves[k].n_interp_knots();
        PatternTracker t{cs.knot_offset(k), ni, curve::make_modular_curve<double>(p.curves[k].modules())};
        // G = d(prefilter)/dx, ONE AAD pass: the filter inputs precede the filter, so this is exact at any x.
        auto cd = curve::make_modular_curve<ad::Dual>(p.curves[k].modules());
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
        // Analytic re-take structure (-1: this curve re-takes W by AAD).
        t.pc = eng.pwl_prepare(k);
        if (t.pc >= 0) {
          t.N = t.crv.pwl_nodes();
          t.h = *t.crv.pwl_spacing();
          t.M.setZero(t.N, ni);
          t.dM.resize(t.N, ni);
          t.phi.assign(static_cast<std::size_t>(2 * t.N - 1), 0.0);
          t.row.resize(ni);
          t.nodes.reserve(static_cast<std::size_t>(t.N));
        }
        trk_.push_back(std::move(t));
      }
    pwl_ = true;
    return true;
  }

  // M = Φ_P·G for the tracker's CACHED pattern, rows `j` (node j's filtered-tangent formula over x_c).
  static void formula_row(PatternTracker& t, const std::vector<unsigned char>& pat, int j) {
    curve::MonotoneCubic<double>::tangent_formula(pat[2 * j], pat[2 * j + 1], j, t.h, t.N, t.phi.data());
    t.row.noalias() = Eigen::Map<const Eigen::RowVectorXd>(t.phi.data(), 2 * t.N - 1) * t.G;
  }

  // EXPERIMENT: keep the compiled W in step with x's branch pattern. Per call: z = G·x and the filter's
  // branch logic on doubles (phase 2). On a change: the ANALYTIC rank-k update (nodes whose formula changed)
  // when every changed curve supports it, else the full AAD re-take. The first sync is always a full re-take.
  void sync(const Eigen::VectorXd& x) const {
    if (!pwl_) return;
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
    if (pwl_synced_ && !changed) return;
    auto& eng = const_cast<CompiledBundleResidual&>(*cacheable_);  // const API caches a pure fn of x
    if (analytic) {
      for (auto& t : trk_) {
        if (!t.dirty) continue;
        t.nodes.clear();
        for (int j = 0; j < t.N; ++j)
          if (t.cur[2 * j] != t.cached[2 * j] || t.cur[2 * j + 1] != t.cached[2 * j + 1]) {
            formula_row(t, t.cur, j);
            t.dM.row(static_cast<int>(t.nodes.size())) = t.row - t.M.row(j);
            t.M.row(j) = t.row;
            t.nodes.push_back(j);
          }
        eng.pwl_rank_update(t.pc, t.nodes, t.dM.topRows(static_cast<int>(t.nodes.size())));
        t.cached = t.cur;  // same size: no allocation
      }
      ++pwl_analytic_;
    } else {
      // Full reset. An AAD re-take only for curves the analytic path does not cover; every analytic-capable
      // curve is then SET to its recorded pattern's W (L + B·M), which stays exact at degenerate states.
      bool need_aad = false;
      for (const auto& t : trk_) need_aad = need_aad || t.pc < 0;
      if (need_aad) eng.rebuild_W(x);
      for (auto& t : trk_) {
        t.cached = t.cur;
        if (t.pc < 0) continue;
        for (int j = 0; j < t.N; ++j) { formula_row(t, t.cached, j); t.M.row(j) = t.row; }
        eng.pwl_set(t.pc, t.M);
      }
      pwl_synced_ = true;
    }
    ++pwl_rebuilds_;
    if (pwl_stats()) {  // opt-in diagnostics: a std::set insert allocates, so it is OFF on the measured path
      std::vector<unsigned char> key;
      for (const auto& t : trk_) key.insert(key.end(), t.cached.begin(), t.cached.end());
      pwl_seen_.insert(std::move(key));
    }
  }
  static bool pwl_stats() {
    static const bool on = [] { const char* e = std::getenv("SWAPS_EXP_PWL_STATS"); return e && e[0] == '1'; }();
    return on;
  }
  bool pwl_ = false;
  mutable bool pwl_synced_ = false;
  mutable int pwl_rebuilds_ = 0;
  mutable int pwl_analytic_ = 0;  // of which: rank-k analytic updates (the rest are full AAD re-takes)
  mutable std::set<std::vector<unsigned char>> pwl_seen_;  // EXPERIMENT diagnostics: distinct cells visited
  mutable std::vector<PatternTracker> trk_;

  void scatter(const Eigen::VectorXd& sub, Eigen::VectorXd& full) const {
    full.setZero(n_res_);
    for (std::size_t j = 0; j < cache_rows_.size(); ++j) full[cache_rows_[j]] = sub[static_cast<int>(j)];
  }

  // Gather a GLOBAL streaming market q onto the cacheable engine's sub-rows (qsub[j] = q[cache_rows_[j]]),
  // so cacheable_.residuals_vs / jacobian_vs see the live feed in their own sub-row order.
  void gather_cache(const Eigen::VectorXd& q, Eigen::VectorXd& qsub) const {
    qsub.resize(static_cast<int>(cache_rows_.size()));
    for (std::size_t j = 0; j < cache_rows_.size(); ++j) qsub[static_cast<int>(j)] = q[cache_rows_[j]];
  }

  int n_res_;
  std::vector<int> cache_rows_;  // cacheable sub-row -> global residual row (identity when nc empty)
  std::vector<int> cache_pos_;   // global residual row -> cacheable sub-row, or -1 (AAD block)
  // The compiled half. Disengaged ONLY when the curves themselves rule out a constant W (non-linear
  // interpolation scheme) -- then every row rides the AAD block and the hybrid is pure width-reduced AAD.
  std::optional<CompiledBundleResidual> cacheable_;
  AadBlock nc_;
  mutable Eigen::VectorXd out_, res_, qsub_;
  mutable Eigen::MatrixXd Jc_;  // jacobian_vs_into's cacheable-half scratch (single-thread, like res_)
};

}  // namespace swaps::calibration
