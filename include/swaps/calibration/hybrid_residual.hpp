#pragma once
// HybridBundleResidual -- the residual engine for a BundleProblem, a LIST OF ROW ENGINES (row_engine.hpp,
// 2026-09-22). Every row is claimed by exactly one engine:
//   * CompiledRows -- the W-cache (CompiledBundleResidual: DF = exp(-Wx), analytic Jacobian) over every row
//     the compiled batch can express -- the microsecond majority, and on every shipped bundle ALL rows;
//   * AadBlock     -- width-reduced forward-AAD over the rest (a compounded observation, an incomplete or
//     seasoned MtM leg; under the horizon partition, rows reading a value-dependent region).
// and the hybrid stitches their outputs into the bundle's residual vector and block Jacobian in insertion
// order. When ONE compiled engine owns every row (the overwhelmingly common case) the hybrid is a transparent,
// zero-overhead delegate to it -- the fast path is byte-for-byte the compiled engine's. A third engine
// registers by implementing RowEngine and claiming rows; nothing here needs a new member for it.

#include <Eigen/Core>

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "swaps/calibration/aad_block.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/row_engine.hpp"

namespace swaps::calibration {

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

// Which instruments must go to the AAD block rather than the W-cache is the INSTRUMENT'S own answer
// (Instrument::noncacheable: a compounded observation anywhere, or an incomplete / seasoned MtM funding leg;
// everything else compiles, nested in a Portfolio or not, since the 2026-09-22 row model). The router asks it
// once per row. Whether a row's times reach a value-dependent part of a curve is the separate, per-curve
// horizon question above.

// The piecewise-linear W tier is ON by default (adopted 2026-09-20): a MonotoneCubic curve is piecewise-linear
// in its knots, so rows reading the value-dependent region ride the compiled W-cache -- W exact inside a branch
// cell, re-taken analytically (rank-k) when x crosses one -- instead of dropping to the AAD block: desk_mixed
// residual 133 -> 4.6 us, Jacobian 747 -> 181 us, streamed tick 3.7-4.8x, parity ~1e-16 vs the pre-adoption
// router. `HybridBundleResidual(p, false)` is the opt-OUT: it restores the horizon partition (value-dependent
// rows to AAD) for a bisect or an A/B, and is what tests/router_partition_test.cpp pins the old routing with.
// The tier tracks PIECEWISE-linear value dependence only: a region that is not piecewise-linear in its knots
// (curve::scheme_is_piecewise_linear false -- MonotoneConvex) is never offered to it; its rows take the horizon
// partition. (Until 2026-09-22 the tier's trackers and the scheme's tangent formula lived HERE and an
// environment variable chose the routing; the trackers are the curve set's now (pricing::CompiledCurveSet::
// pwl_sync, off the region's own cell description), and routing is a constructor argument.)

// The compiled W-cache as a ROW ENGINE over a subset of the bundle's rows: a CompiledBundleResidual built over
// the sub-problem of those rows, with the gather of the live market onto its sub-rows and the scatter of its
// outputs back onto the global rows. `engine()` is exposed for the whole-bundle direct delegate (no stitch).
class CompiledRows final : public RowEngine {
 public:
  CompiledRows(const BundleProblem& sub, std::vector<int> rows, int n_res, bool pwl)
      : eng_(sub, pwl), rows_(std::move(rows)), pos_(static_cast<std::size_t>(n_res), -1) {
    for (std::size_t j = 0; j < rows_.size(); ++j) pos_[static_cast<std::size_t>(rows_[j])] = static_cast<int>(j);
  }
  const std::vector<int>& rows() const override { return rows_; }
  int local(int global_row) const { return pos_[static_cast<std::size_t>(global_row)]; }  // -1: not mine
  const CompiledBundleResidual& engine() const { return eng_; }
  CompiledBundleResidual& engine() { return eng_; }

  void set_quote(int row, double market, double lower, double upper, double decay) override {
    const int j = local(row);
    if (j >= 0) eng_.set_quote(j, market, lower, upper, decay);
  }
  void set_market(int row, double market) override {
    const int j = local(row);
    if (j >= 0) eng_.set_market(j, market);
  }
  void model_rates_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const override { scatter(eng_.model_rates(x), out); }
  void residuals_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const override { scatter(eng_.residuals(x), out); }
  void residuals_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::VectorXd& out) const override {
    gather(q);
    scatter(eng_.residuals_vs(x, qsub_), out);
  }
  void jacobian_into(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const override {
    const Eigen::MatrixXd Jc = eng_.jacobian(x);
    for (std::size_t j = 0; j < rows_.size(); ++j) J.row(rows_[j]) = Jc.row(static_cast<int>(j));
  }
  // Into the caller's J (C6): the sub-Jacobian lands in a member scratch, so a warm call allocates no matrix.
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J, Eigen::VectorXd* r) const override {
    gather(q);
    eng_.jacobian_vs_into(x, qsub_, Jc_, r ? &rc_ : nullptr);
    for (std::size_t j = 0; j < rows_.size(); ++j) J.row(rows_[j]) = Jc_.row(static_cast<int>(j));
    if (r)
      for (std::size_t j = 0; j < rows_.size(); ++j) (*r)[rows_[j]] = rc_[static_cast<int>(j)];
  }

 private:
  void gather(const Eigen::VectorXd& q) const {
    qsub_.resize(static_cast<int>(rows_.size()));
    for (std::size_t j = 0; j < rows_.size(); ++j) qsub_[static_cast<int>(j)] = q[rows_[j]];
  }
  void scatter(const Eigen::VectorXd& sub, Eigen::VectorXd& out) const {
    for (std::size_t j = 0; j < rows_.size(); ++j) out[rows_[j]] = sub[static_cast<int>(j)];
  }
  CompiledBundleResidual eng_;
  std::vector<int> rows_;  // sub-row -> global row
  std::vector<int> pos_;   // global row -> sub-row, or -1
  mutable Eigen::VectorXd qsub_, rc_;
  mutable Eigen::MatrixXd Jc_;
};

class HybridBundleResidual {
 public:
  // `pwl` (default ON): rows that read past a value-dependent region's start stay on the COMPILED engine, whose
  // W is re-taken at x whenever x's branch pattern changes -- instead of going to the AAD block.
  explicit HybridBundleResidual(const BundleProblem& p, bool pwl = true) : n_res_(static_cast<int>(p.instruments.size())) {
    validate_problem(p, "HybridBundleResidual");  // E1/E2/B12: refuse a malformed bundle before compiling it
    // ONE partition pass. Each instrument's cacheability is decided once (Instrument::noncacheable walks real
    // cashflows, so it is not free -- do not re-ask per consumer). Per-curve LINEAR HORIZONS replace the old
    // whole-bundle veto (2026-09-10): a value-dependent region used to send EVERY instrument in the bundle to
    // the AAD block, including par swaps on curves that never touched it. Now a row is only pushed off the
    // W-cache if it actually reads past a horizon -- and under the tier (default) not even then: only the
    // SHAPE question routes a row to AAD. When every curve is fully linear -- every shipped bundle -- `mixed`
    // is false and no probe runs at all.
    const std::vector<double> horizons = pricing::curve_linear_horizons(p.curves);
    bool mixed = false;
    for (double h : horizons)
      if (h < std::numeric_limits<double>::infinity()) mixed = true;
    // The tier tracks PIECEWISE-linear value dependence only (one constant W per branch cell). A region that is
    // not piecewise-linear (MonotoneConvex) is never offered to it: its rows take the horizon partition below.
    if (pwl && mixed && pricing::curves_piecewise_linear(p.curves) && build(p, horizons, /*use_horizons=*/false, /*pwl=*/true)) return;
    // The horizon partition (the tier off, or refused: nothing would stay compiled, or the compiled engine
    // rejects the bundle, e.g. moment-path coupons on a value-dependent curve).
    build(p, horizons, /*use_horizons=*/mixed, /*pwl=*/false);
  }

  int n_residuals() const { return n_res_; }
  // The piecewise-linear tier's observability (forwarded from the compiled engine): engaged, how many syncs
  // changed W, of which analytic, distinct cells visited (opt-in: enable_pwl_stats).
  bool pwl_active() const { return compiled_ && compiled_->engine().pwl_active(); }
  int pwl_rebuilds() const { return compiled_ ? compiled_->engine().pwl_rebuilds() : 0; }
  int pwl_distinct() const { return compiled_ ? compiled_->engine().pwl_distinct() : 0; }
  int pwl_analytic() const { return compiled_ ? compiled_->engine().pwl_analytic() : 0; }
  void enable_pwl_stats() { if (compiled_) compiled_->engine().enable_pwl_stats(); }
  int n_times() const { return compiled_ ? compiled_->engine().n_times() : 0; }
  // The ROW PARTITION, observable (item 5, 2026-09-10): which engine took a given global row. >=0 is the
  // row's index in the compiled W-cache sub-problem, -1 means it went to the AAD block. Exposed so a test
  // can pin WHERE a row was routed, not merely that the two routes agree -- a router that quietly sent
  // everything to the slow path would otherwise pass every parity test it has.
  int compiled_row(int row) const { return compiled_ ? compiled_->local(row) : -1; }
  int n_compiled_rows() const { return compiled_ ? static_cast<int>(compiled_->rows().size()) : 0; }
  // Is the AAD half on the heap-free pooled dual? False means its touched width exceeded ad::kPooledMaxW and
  // every dual operation now allocates -- a ~100x per-tick cost cliff that nothing else reports. Exposed so
  // the hot-path tests can fail on it rather than merely measure it (2026-09-12).
  bool aad_pooled() const { return !aad_ || aad_->pooled(); }
  int aad_width() const { return aad_ ? aad_->touched_width() : 0; }
  // The engines, for a consumer that wants to walk the list (which rows each owns).
  const std::vector<std::unique_ptr<RowEngine>>& engines() const { return engines_; }

  // Overwrite the quote RHS (targets + bands) on every engine without touching any compiled/discovered
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
  // SCALAR quote updates by GLOBAL row (the object model, 2026-09-10): no Instrument is copied; the row's
  // owning engine takes it.
  void set_quote(int row, double market, double lower, double upper, double decay) {
    engines_[static_cast<std::size_t>(owner_[static_cast<std::size_t>(row)])]->set_quote(row, market, lower, upper, decay);
  }
  void set_market(const Eigen::VectorXd& q) {
    if (q.size() != n_res_) throw std::invalid_argument("HybridBundleResidual::set_market: market length differs");
    for (int r = 0; r < n_res_; ++r) engines_[static_cast<std::size_t>(owner_[static_cast<std::size_t>(r)])]->set_market(r, q[r]);
  }

  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const {
    if (whole_) return compiled_->engine().model_rates(x);
    out_.resize(n_res_);
    for (const auto& e : engines_) e->model_rates_into(x, out_);
    return out_;
  }
  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const {
    if (whole_) return compiled_->engine().residuals(x);
    res_.resize(n_res_);
    for (const auto& e : engines_) e->residuals_into(x, res_);
    return res_;
  }
  // Streaming residual/Jacobian against a live market q (global residual order). Drives frozen-Newton for a
  // mixed bundle too: each engine's `_vs` form on its rows. When one compiled engine owns every row it is a
  // straight delegate (q is already global).
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    if (whole_) return compiled_->engine().residuals_vs(x, q);
    res_.resize(n_res_);
    for (const auto& e : engines_) e->residuals_vs_into(x, q, res_);
    return res_;
  }
  Eigen::MatrixXd jacobian_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    Eigen::MatrixXd J;
    jacobian_vs_into(x, q, J);
    return J;
  }
  // The same Jacobian, and the residuals_vs(x, q) values into *r (S2, 2026-09-15): each engine's from its own
  // pass -- so a streamer refresh reads its band sides without re-evaluating every model quote.
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J, Eigen::VectorXd* r) const {
    if (whole_) { compiled_->engine().jacobian_vs_into(x, q, J, r); return; }
    if (r) r->resize(n_res_);
    // No zeroing: every row is written -- a compiled row whole, an AAD row zeroed then written on its touched
    // columns. The row partition is total (constructor), so a garbage-filled J comes back exact (JacobianIntoParity).
    J.resize(n_res_, static_cast<int>(x.size()));
    for (const auto& e : engines_) e->jacobian_vs_into(x, q, J, r);
  }
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J) const {
    jacobian_vs_into(x, q, J, nullptr);
  }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {
    if (whole_) return compiled_->engine().jacobian(x);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n_res_, static_cast<int>(x.size()));
    for (const auto& e : engines_) e->jacobian_into(x, J);
    return J;
  }

 private:
  // Partition the rows and build the engine list. `use_horizons`: also push rows that read past a curve's
  // linear horizon to the AAD block (the horizon partition); `pwl`: build the compiled engine on the tier.
  // Returns false (and builds nothing) when the tier was asked for and cannot be engaged.
  bool build(const BundleProblem& p, const std::vector<double>& horizons, bool use_horizons, bool pwl) {
    BundleProblem c;
    c.curves = p.curves;
    std::vector<Instrument> nc;
    std::vector<int> nc_rows, rows;
    for (int r = 0; r < n_res_; ++r) {
      const bool off_cache = p.instruments[r].noncacheable() ||
                             (use_horizons && !instrument_within_horizons(p.instruments[r], p.curves, horizons));
      if (off_cache) {
        nc.push_back(p.instruments[r]);
        nc_rows.push_back(r);
      } else {
        c.instruments.push_back(p.instruments[r]);
        rows.push_back(r);
      }
    }
    if (pwl && c.instruments.empty()) return false;
    engines_.clear();
    compiled_ = nullptr;
    aad_ = nullptr;
    // The compiled engine is built for every fully-linear bundle (even an all-AAD instrument mix, which keeps
    // n_times() honest) and for a mixed bundle whenever some row stayed on the W-cache.
    if (!use_horizons || !c.instruments.empty()) {
      std::unique_ptr<CompiledRows> cr;
      try {
        cr = std::make_unique<CompiledRows>(c, rows, n_res_, pwl);
      } catch (const std::exception&) {
        if (pwl) return false;  // the tier is refused: the caller falls back to the horizon partition
        throw;
      }
      compiled_ = cr.get();
      engines_.push_back(std::move(cr));
    }
    if (!nc.empty()) {
      auto blk = std::make_unique<AadBlock>();
      blk->init(p.curves, std::move(nc), std::move(nc_rows), p.n_knots());
      aad_ = blk.get();
      engines_.push_back(std::move(blk));
    }
    owner_.assign(static_cast<std::size_t>(n_res_), -1);
    for (std::size_t k = 0; k < engines_.size(); ++k)
      for (int r : engines_[k]->rows()) owner_[static_cast<std::size_t>(r)] = static_cast<int>(k);
    whole_ = compiled_ && static_cast<int>(compiled_->rows().size()) == n_res_;
    return true;
  }

  int n_res_;
  std::vector<std::unique_ptr<RowEngine>> engines_;  // in tier order: the compiled W-cache, then the AAD block
  std::vector<int> owner_;                            // global row -> index into engines_
  CompiledRows* compiled_ = nullptr;                  // the compiled engine, if any (typed access for the observability)
  AadBlock* aad_ = nullptr;                           // the AAD block, if any
  bool whole_ = false;                                // ONE compiled engine owns every row: direct delegate, no stitch
  mutable Eigen::VectorXd out_, res_;
};

}  // namespace swaps::calibration
