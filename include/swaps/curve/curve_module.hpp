#pragma once
// ModularCurve<Scalar> -- THE forward curve. Build it from CurveModule pieces, each a set of knot times
// + an interpolation scheme over those knots (Flat / Linear / NaturalCubic / Hermite / MonotoneCubic /
// BSpline from regions.hpp), stitched left-to-right with a C0 (level) Boundary handoff.
//
// There is exactly ONE curve type and ONE factory (make_modular_curve). A "curve flavour" is DATA -- a
// list of modules -- not a type: the shipped flat-front/Hermite-back calibration curve, a B-spline back
// end and a monotone-cubic back end are just three different module lists (see the layout helpers
// below). That keeps every curve, whether shipped or user-composed in the web composer, on one code
// path: one set of region math, one linearity check, one W-cache, one risk path.
//
// Region types are erased, so composing a curve costs a virtual call per region per evaluation. That is
// setup-only cost: the microsecond calibration hot path runs on the cached W matrix (CompiledBundleResidual),
// never on a curve object. Measured against the compile-time alternative it was not slower --
// risk_full_jacobian 662us vs 809us -- because AAD gradient allocation dominates the dispatch entirely.
// Templated on Scalar so AAD flows through (e.g. to build W or a risk gradient).

#include <cmath>
#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "swaps/ad/dual.hpp"
#include "swaps/curve/regions.hpp"

namespace swaps::curve {

// Type-erased region: the runtime face of a compile-time region policy (Flat<S>, Hermite<S>, ...).
template <class S>
struct RegionIface {
  virtual ~RegionIface() = default;
  virtual int n_values() const = 0;
  virtual double t_end() const = 0;
  virtual bool linear() const = 0;
  virtual void build(const S* x, int off, const Boundary<S>& in) = 0;
  virtual S forward(double t) const = 0;
  virtual S integral(double t) const = 0;
  virtual S forward_d1(double t) const = 0;           // exact df/dt (0 outside / at a Flat step)
  virtual S forward_d2(double t) const = 0;           // exact d²f/dt²
  virtual std::vector<double> pieces() const = 0;     // breakpoints between which f is ONE analytic piece
  virtual Boundary<S> out() const = 0;
  // EXPERIMENT (exp/piecewise-linear-w): append this region's value-dependent branch trace (empty for every
  // LINEAR scheme -- it has no decisions). Equal traces over the whole curve => the same linear map of x.
  virtual void pattern_into(std::vector<unsigned char>&) const {}
  // Phase 2: the filter inputs (linear in x) and the pattern computed from them. Empty / no-op when linear.
  virtual void prefilter_into(std::vector<S>&) const {}
  virtual int prefilter_size() const { return 0; }
  virtual void pattern_from_prefilter(const double*, std::vector<unsigned char>&) const {}
  // Analytic re-take (EXPERIMENT): tangent override + the node structure the decoder needs. No-ops if linear.
  virtual void set_tangent_override(const S*) {}
  virtual int n_nodes() const { return 0; }
  // The piecewise-linear capability (2026-09-22): the region's own cell description. 0 / no-op when linear.
  virtual int node_pattern_bytes() const { return 0; }
  virtual void node_formula(int, const unsigned char*, double*) const {}
  // THE TANGENT (2026-10-31): the gradient of this region's integral at t, and of its outgoing forward, with respect
  // to its OWN values and the INCOMING join forward, at the state of the last build (the incoming integral enters
  // with coefficient 1 everywhere and no region reads the incoming slope, so the chain needs nothing else). A linear
  // region's is constant (taken once by one dual pass over the region alone: LinearTangentTwin); a value-dependent
  // region supplies its own closed form (MonotoneConvex). `has_tangent` false (MonotoneCubic) keeps the curve off
  // the tangent tier. `own` has n_values() entries.
  virtual bool has_tangent() const { return false; }
  virtual void integral_tangent(double, double*, double&) const { throw std::logic_error("region: no tangent"); }
  virtual void out_value_tangent(double*, double&) const { throw std::logic_error("region: no tangent"); }
};

// The constant tangent of a LINEAR region policy: one dual pass over the region alone, seeded on its own values and
// the incoming forward, taken once (the gradient of a linear map does not depend on the state) at the first request
// after the region knows whether it leads or follows. Setup only, and the same single AAD pass integral_weight_matrix
// makes for the whole curve -- here per region, so the chain can compose it with a value-dependent neighbour.
template <class PolicyDual>
struct LinearTangentTwin {
  std::unique_ptr<PolicyDual> twin;
  int n = 0;
  // The rows are CONSTANT, so each time's row is taken once and served from here afterwards: a dual evaluation
  // materialises a heap gradient, and the compiled engine's tangent sync asks the same registered times every
  // state -- the cache keeps the per-state path allocation-free after warm-up (the T4 invariant).
  struct Row { std::vector<double> own; double d_inv = 0.0; };
  mutable std::map<double, Row> rows;
  mutable std::unique_ptr<Row> out_row;
  template <class Make>
  void ensure(Make make, int n_values, double in_time, bool has_pred) {
    if (twin) return;
    n = n_values;
    twin = make();
    std::vector<ad::Dual> xd(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) xd[static_cast<std::size_t>(k)] = ad::Dual(0.03 + 0.002 * k, n + 1, k);
    // WIDTH-SIZED zeros for the boundary scalars the seed does not differentiate (the engine's idiom): an EMPTY
    // derivative vector is coherent with a sized one only when Eigen sees a plain vector on the other side; a
    // region's first-segment integral sums two such empties before adding a sized term, and set_forwards() only
    // avoids that because out() happens to size the incoming integral first. Seeded sized, the twin cannot hit it.
    Boundary<ad::Dual> in;
    in.time = in_time;
    in.value = ad::Dual(0.03, n + 1, n);
    in.slope = ad::Dual(0.0);
    in.slope.derivatives().setZero(n + 1);
    in.integral = ad::Dual(0.0);
    in.integral.derivatives().setZero(n + 1);
    in.has_predecessor = has_pred;
    twin->build(xd.data(), 0, n, in);
    (void)twin->out();  // as set_forwards() does after every build
  }
  void take(const ad::Dual& d, double* own, double& d_inv) const {
    if (d.derivatives().size() == n + 1) {
      for (int k = 0; k < n; ++k) own[k] = d.derivatives()[k];
      d_inv = d.derivatives()[n];
    } else {
      std::fill(own, own + n, 0.0);
      d_inv = 0.0;
    }
  }
  void integral(double t, double* own, double& d_inv) const {
    auto it = rows.find(t);
    if (it == rows.end()) {
      Row r;
      r.own.resize(static_cast<std::size_t>(n));
      take(twin->integral(t), r.own.data(), r.d_inv);
      it = rows.emplace(t, std::move(r)).first;
    }
    std::copy(it->second.own.begin(), it->second.own.end(), own);
    d_inv = it->second.d_inv;
  }
  void out_value(double* own, double& d_inv) const {
    if (!out_row) {
      out_row = std::make_unique<Row>();
      out_row->own.resize(static_cast<std::size_t>(n));
      take(twin->out().value, out_row->own.data(), out_row->d_inv);
    }
    std::copy(out_row->own.begin(), out_row->own.end(), own);
    d_inv = out_row->d_inv;
  }
};

// A policy is constructed from its knots, or from (knots, sigma) when it takes a tension parameter (Tension; the
// separate TensionHolder that existed for that one argument was folded in on 2026-11-01, review item 2.3).
template <class P>
P make_policy(std::vector<double> knots, double sigma) {
  if constexpr (std::is_constructible_v<P, std::vector<double>, double>) return P(std::move(knots), sigma);
  else return P(std::move(knots));
}

template <class S, template <class> class Policy>
struct RegionHolder final : RegionIface<S> {
  Policy<S> p;
  double sigma_;
  mutable LinearTangentTwin<Policy<ad::Dual>> twin_;  // built from p.knots() on the first tangent request
  double in_time_ = 0.0;
  bool has_pred_ = false;
  explicit RegionHolder(std::vector<double> knots, double sigma = 0.0) : p(make_policy<Policy<S>>(std::move(knots), sigma)), sigma_(sigma) {}
  int n_values() const override { return p.n_values(); }
  double t_end() const override { return p.t_end(); }
  bool linear() const override { return Policy<S>::is_linear_map; }
  // Policy::build indexes x[off+k]; a raw pointer satisfies that (pointer indexing), no copy of policy.
  void build(const S* x, int off, const Boundary<S>& in) override {
    in_time_ = in.time;
    has_pred_ = in.has_predecessor;
    p.build(x, off, p.n_values(), in);
  }
  bool has_tangent() const override {
    if constexpr (requires { Policy<S>::has_tangent; }) return Policy<S>::has_tangent;
    else return Policy<S>::is_linear_map;
  }
  void integral_tangent(double t, double* own, double& d_inv) const override {
    if constexpr (requires { p.integral_tangent(t, own, d_inv); }) p.integral_tangent(t, own, d_inv);
    else if constexpr (Policy<S>::is_linear_map) {
      twin_.ensure([this] { return std::make_unique<Policy<ad::Dual>>(make_policy<Policy<ad::Dual>>(p.knots(), sigma_)); }, p.n_values(), in_time_, has_pred_);
      twin_.integral(t, own, d_inv);
    } else throw std::logic_error("region: no tangent");
  }
  void out_value_tangent(double* own, double& d_inv) const override {
    if constexpr (requires { p.out_value_tangent(own, d_inv); }) p.out_value_tangent(own, d_inv);
    else if constexpr (Policy<S>::is_linear_map) {
      twin_.ensure([this] { return std::make_unique<Policy<ad::Dual>>(make_policy<Policy<ad::Dual>>(p.knots(), sigma_)); }, p.n_values(), in_time_, has_pred_);
      twin_.out_value(own, d_inv);
    } else throw std::logic_error("region: no tangent");
  }
  S forward(double t) const override { return p.forward(t); }
  S integral(double t) const override { return p.integral(t); }
  S forward_d1(double t) const override { return p.forward_d1(t); }
  S forward_d2(double t) const override { return p.forward_d2(t); }
  std::vector<double> pieces() const override { return p.pieces(); }
  Boundary<S> out() const override { return p.out(); }
  void pattern_into(std::vector<unsigned char>& out) const override {
    if constexpr (requires { p.pattern(); }) out.insert(out.end(), p.pattern().begin(), p.pattern().end());
  }
  void prefilter_into(std::vector<S>& out) const override {
    if constexpr (requires { p.prefilter(); }) out.insert(out.end(), p.prefilter().begin(), p.prefilter().end());
  }
  int prefilter_size() const override {
    if constexpr (requires { p.prefilter(); }) return static_cast<int>(p.prefilter().size());
    return 0;
  }
  void set_tangent_override(const S* m) override {
    if constexpr (requires { p.set_tangent_override(m); }) p.set_tangent_override(m);
  }
  int n_nodes() const override {
    if constexpr (requires { p.n_nodes(); }) return p.n_nodes();
    return 0;
  }
  int node_pattern_bytes() const override {
    if constexpr (requires { p.node_pattern_bytes(); }) return p.node_pattern_bytes();
    return 0;
  }
  void node_formula(int j, const unsigned char* pat, double* phi) const override {
    if constexpr (requires { p.node_formula(j, pat, phi); }) p.node_formula(j, pat, phi);
  }
  void pattern_from_prefilter(const double* z, std::vector<unsigned char>& out) const override {
    if constexpr (requires { p.pattern_from_prefilter(z, out); }) {
      thread_local std::vector<unsigned char> part;
      p.pattern_from_prefilter(z, part);
      out.insert(out.end(), part.begin(), part.end());
    }
  }
};

// MonotoneCubic is the one NON-LINEAR scheme here: RegionHolder<S,MonotoneCubic>::linear() returns false,
// so a ModularCurve containing it reports is_linear_map()==false -- the runtime signal to route that curve
// through the AAD engine rather than the W-cache (which its static_assert would reject at compile time).
// BSpline is linear too, but note its free values are CONTROL POINTS, which do not lie on the curve
// (docs/bezier-and-moments.md Part A) -- calibrated x are control points, mapped to forward-at-knot for
// a risk ladder by pricing::bspline_collocation. That is a property of the REGION, not of a curve type.
// Tension is the second LINEAR hyperbolic scheme (research note §3): with σ a FIXED hyperparameter its
// coefficients depend only on knot spacings, so is_linear_map = true -- it rides the W-cache exactly
// like NaturalCubic/BSpline, unlike the value-dependent MonotoneCubic. σ travels in CurveModule::sigma.
// MonotoneConvex (2026-10-06) is the second value-dependent scheme and the first that is NOT piecewise-linear:
// Hagan-West's four quadratic shapes have breakpoints at RATIOS of the knot values, so no branch cell has a
// constant W. It rides the AAD tier only (scheme_is_piecewise_linear below is what keeps it off the tier).
enum class Scheme { Flat, Linear, NaturalCubic, Hermite, MonotoneCubic, BSpline, Tension, MonotoneConvex };

// THE static answer to "does this scheme keep the curve a LINEAR MAP of its knot forwards" (so it rides the
// W-cache): every shipped scheme except the two value-dependent ones. The runtime truth is
// ModularCurve::is_linear_map() (the AND over the built regions' policies); the two are pinned equal per
// scheme in tests/kernel_pins_test.cpp. E6.1c (2026-09-10): this used to be re-decided by enum in
// hybrid_residual.hpp and api/bundle_api.cpp. Written as an INCLUSION list (2026-10-06): a new scheme is
// non-linear until it is named here, never the other way round.
inline constexpr bool scheme_is_linear(Scheme s) {
  return s == Scheme::Flat || s == Scheme::Linear || s == Scheme::NaturalCubic || s == Scheme::Hermite ||
         s == Scheme::BSpline || s == Scheme::Tension;
}
// Is the curve's integral PIECEWISE-linear in the knot values -- one constant W per branch cell (MonotoneCubic:
// the Hyman filter picks among linear tangent formulas), so the piecewise-linear W tier can track it? Linear
// schemes trivially (one cell). MonotoneConvex is not: its W varies continuously inside a shape cell, so a
// tier that froze W at a state would be silently wrong between syncs. Inclusion list, like scheme_is_linear.
inline constexpr bool scheme_is_piecewise_linear(Scheme s) { return scheme_is_linear(s) || s == Scheme::MonotoneCubic; }
// Does the scheme carry its integral's TANGENT (RegionIface::integral_tangent) -- a linear scheme's constant row, or a
// closed form for a value-dependent one (MonotoneConvex, 2026-10-31) -- so the compiled engine can re-take W at any
// state analytically? MonotoneCubic is tracked by the piecewise-linear tier instead. Inclusion list, like the others.
inline constexpr bool scheme_has_tangent(Scheme s) { return scheme_is_linear(s) || s == Scheme::MonotoneConvex; }

// The tripwire for adding a scheme (REVIEW FINDING 3, 2026-09-21): appending one moves the last enumerator and
// breaks this assert, which names what must learn about it: scheme_is_linear AND scheme_is_piecewise_linear
// (both inclusion lists: an unnamed scheme is non-linear and not piecewise-linear), ModularCurve::add, the
// codec's to/from string (api/codec.cpp), tests/region_combinatorial_test.cpp ALL_SCHEMES and kSchemeCount.
inline constexpr int kSchemeCount = 8;
static_assert(static_cast<int>(Scheme::MonotoneConvex) + 1 == kSchemeCount,
              "a Scheme was added or reordered: update scheme_is_linear / scheme_is_piecewise_linear / scheme_has_tangent "
              "(inclusion lists), ModularCurve::add, the codec's to/from string (api/codec.cpp), ALL_SCHEMES and kSchemeCount");

// Sort a region's knot times and drop duplicates: two instruments maturing on ONE date share ONE knot (a knot is a
// time, an instrument a row). A region constructor refuses a duplicate (a zero-length segment is a silent NaN), so
// whoever assembles knots from instruments -- the compile verb, a test fixture -- calls this (2026-10-12).
inline void sort_distinct_knots(std::vector<double>& knots) {
  std::sort(knots.begin(), knots.end());
  knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
}

// A region that says nothing about its smoothing (CurveModule::reg_lambda): no rows. The one spelling of "unset".
inline constexpr double kRegLambdaUnset = -1.0;
// One building block of a curve: the knot times of a region and the interpolation over them.
struct CurveModule {
  std::vector<double> knots;  // knot times (year fractions), ascending, within this region
  Scheme scheme = Scheme::Hermite;  // interpolation scheme over those knots (the one default; the JSON codec defers to it)
  // Tension hyperparameter (Scheme::Tension only): pulls the spline taut, σ→0 recovers NaturalCubic,
  // σ→∞ approaches piecewise-linear. Ignored by every other scheme. <=0 means "use the default 1.0".
  double sigma = 0.0;
  // This region's smoothing SWITCH -- SMOOTHING IS REGION DATA (owner's rule, 2026-10-13, ASSUMPTIONS K47) and a
  // TIE-BREAKER, never a trade against the quotes (K48): > 0 => the calibration chooses the minimum-curvature curve
  // among those the quotes allow over this region's knots (one curvature row per interior knot at the fixed
  // tie-breaker weight regularize.hpp kSmoothingTieBreak; the magnitude here is NOT read -- a strength would buy
  // smoothness with quote error); 0 or unset (kRegLambdaUnset) => the quotes alone. There is no bundle-level default a
  // region inherits. A Flat region needs none (a step level carries no curvature row). (A per-region regulariser σ,
  // `reg_sigma`, went with the tension-energy operator on 2026-09-22.)
  double reg_lambda = kRegLambdaUnset;
};

template <class S>
class ModularCurve {
 public:
  ModularCurve() = default;

  template <template <class> class Policy>
  ModularCurve& add(std::vector<double> knots, double sigma = 0.0) {  // sigma: read by a policy that takes one (Tension)
    regions_.push_back(std::make_unique<RegionHolder<S, Policy>>(std::move(knots), sigma));
    n_ += regions_.back()->n_values();
    return *this;
  }
  ModularCurve& add(const CurveModule& m) {
    switch (m.scheme) {
      case Scheme::Flat: return add<Flat>(m.knots);
      case Scheme::Linear: return add<Linear>(m.knots);
      case Scheme::NaturalCubic: return add<NaturalCubic>(m.knots);
      case Scheme::Hermite: return add<Hermite>(m.knots);
      case Scheme::MonotoneCubic: return add<MonotoneCubic>(m.knots);
      case Scheme::MonotoneConvex: return add<MonotoneConvex>(m.knots);
      case Scheme::BSpline: return add<BSpline>(m.knots);
      case Scheme::Tension: return add<Tension>(m.knots, m.sigma > 0.0 ? m.sigma : 1.0);
    }
    return *this;
  }

  int n_knots() const { return n_; }
  double max_time() const { return regions_.back()->t_end(); }
  bool is_linear_map() const {
    for (const auto& r : regions_)
      if (!r->linear()) return false;
    return true;
  }

  // x = all knot forwards, region by region; stitched left-to-right with a C0 boundary handoff.
  // buf_ is a REUSED member (sized once): on a streaming re-solve the same curve is set_forwards'd every
  // tick, so allocating a fresh contiguous buffer each call is pure churn. `resize` keeps capacity, so
  // after the first call this never reallocates.
  template <class Vec>
  void set_forwards(const Vec& x) {
    if (static_cast<int>(x.size()) != n_) throw std::invalid_argument("set_forwards: wrong size");
    buf_.resize(n_);  // contiguous scalar buffer so each region can index x[off+k]; reused across calls
    for (int i = 0; i < n_; ++i) buf_[i] = x[i];
    Boundary<S> b{};  // has_predecessor=false -> the FIRST region is LEADING (flat-extrapolates its 1st knot)
    int off = 0;
    for (auto& r : regions_) {
      r->build(buf_.data(), off, b);
      off += r->n_values();
      b = r->out();
      b.has_predecessor = true;  // every region after the first FOLLOWS its predecessor (C0 join, unchanged)
    }
  }

  // The i-th free value this curve was last built from (set_forwards' x[i]): a region's knot forward, a B-spline
  // control point, a MonotoneConvex discrete forward -- whatever the region's own parameterisation is. It is the
  // STATE a StateLinear calibration row reads (problem.hpp), with its AAD gradient when S is a dual. Requires
  // set_forwards() to have run (build_bundle_curves always does).
  S value(int i) const {
    if (i < 0 || i >= static_cast<int>(buf_.size())) throw std::out_of_range("ModularCurve::value: index out of range (call set_forwards first)");
    return buf_[static_cast<std::size_t>(i)];
  }

  S forward(double t) const {
    return locate(t, [](const RegionIface<S>& r, double u) { return r.forward(u); });
  }
  // Exact derivatives of the forward and the analytic-piece breakpoints of the whole composed curve (the
  // regulariser integrates (f'')² + σ²(f')² piece by piece; each piece is one polynomial or one tension piece).
  S forward_d1(double t) const {
    return locate(t, [](const RegionIface<S>& r, double u) { return r.forward_d1(u); });
  }
  S forward_d2(double t) const {
    return locate(t, [](const RegionIface<S>& r, double u) { return r.forward_d2(u); });
  }
  // Requires set_forwards() to have run at least once: a region's pieces include its C0 join with the predecessor
  // and (B-spline) its de Boor breakpoints, both fixed at build time. An unbuilt curve throws.
  std::vector<double> pieces() const {
    if (static_cast<int>(buf_.size()) != n_) throw std::logic_error("ModularCurve::pieces(): call set_forwards() first (pieces are fixed at build)");
    std::vector<double> bp;
    for (const auto& r : regions_) { const auto pr = r->pieces(); bp.insert(bp.end(), pr.begin(), pr.end()); }
    std::sort(bp.begin(), bp.end());
    bp.erase(std::unique(bp.begin(), bp.end(), [](double a, double b) { return std::abs(a - b) <= 1e-13 * (1.0 + std::abs(a)); }), bp.end());
    return bp;
  }
  // CONVENTION (E3 register D11): integral(t <= 0) == 0, i.e. discount(t) == 1 for ANY non-positive time.
  // Nothing in pricing may rely on a negative-time discount factor: a seasoned MtM reset in the past must
  // carry its fixed FX (cashflows.hpp xccy_mtm_leg_pv refuses one that does not), and the P&L roll floors
  // its shifted times at 0 by design. A negative time here is a stale-input smell, not a curve query.
  S integral(double t) const {
    if (t <= 0.0) return S(0.0);
    return locate(t, [](const RegionIface<S>& r, double u) { return r.integral(u); });
  }
  S discount(double t) const {
    using std::exp;
    return exp(-integral(t));
  }
  // THE CURVE'S TANGENT (2026-10-31): d integral(t) / d x over all n_knots() values at the last set_forwards, chained
  // front to back: a region's row over its own values, plus its sensitivity to the incoming forward times the
  // predecessor's outgoing-forward gradient, plus the incoming integral's gradient (coefficient 1). For a fully
  // linear curve this is the W row; with a MonotoneConvex region it is W at this state -- the analytic replacement
  // for the AAD sweep. integral(t <= 0) == 0 by convention, so its row is zero. Allocation-free after the first call.
  bool has_tangent() const {
    for (const auto& r : regions_)
      if (!r->has_tangent()) return false;
    return true;
  }
  void integral_tangent(double t, double* row) const {
    if (static_cast<int>(buf_.size()) != n_) throw std::logic_error("ModularCurve::integral_tangent: call set_forwards() first");
    std::fill(row, row + n_, 0.0);
    if (t <= 0.0) return;
    tg_dinv_.assign(static_cast<std::size_t>(n_), 0.0);  // d(incoming forward)/dx for the region at hand
    tg_dI_.assign(static_cast<std::size_t>(n_), 0.0);    // d(incoming integral)/dx
    tg_own_.resize(static_cast<std::size_t>(n_));
    tg_tmp_.resize(static_cast<std::size_t>(n_));
    int off = 0;
    const std::size_t nreg = regions_.size();
    for (std::size_t k = 0; k < nreg; ++k) {
      const RegionIface<S>& r = *regions_[k];
      const int nk = r.n_values();
      double d_inv = 0.0;
      if (t <= r.t_end() || k + 1 == nreg) {  // locate()'s rule: the first region reaching t, else the last
        r.integral_tangent(t, tg_own_.data(), d_inv);
        for (int i = 0; i < n_; ++i) row[i] = tg_dI_[static_cast<std::size_t>(i)] + d_inv * tg_dinv_[static_cast<std::size_t>(i)];
        for (int j = 0; j < nk; ++j) row[off + j] += tg_own_[static_cast<std::size_t>(j)];
        return;
      }
      // pass through: the outgoing integral is the integral at t_end, the outgoing forward is out().value
      r.integral_tangent(r.t_end(), tg_own_.data(), d_inv);
      for (int i = 0; i < n_; ++i) tg_dI_[static_cast<std::size_t>(i)] += d_inv * tg_dinv_[static_cast<std::size_t>(i)];
      for (int j = 0; j < nk; ++j) tg_dI_[static_cast<std::size_t>(off + j)] += tg_own_[static_cast<std::size_t>(j)];
      double dv_inv = 0.0;
      r.out_value_tangent(tg_own_.data(), dv_inv);
      for (int i = 0; i < n_; ++i) tg_tmp_[static_cast<std::size_t>(i)] = dv_inv * tg_dinv_[static_cast<std::size_t>(i)];
      for (int j = 0; j < nk; ++j) tg_tmp_[static_cast<std::size_t>(off + j)] += tg_own_[static_cast<std::size_t>(j)];
      tg_dinv_.swap(tg_tmp_);
      off += nk;
    }
  }
  // EXPERIMENT: the curve's branch pattern at the last set_forwards (concatenated region traces; empty for a
  // fully linear curve). A piecewise-linear curve is ONE linear map of x throughout each pattern's cell.
  void pattern_into(std::vector<unsigned char>& out) const {
    out.clear();
    for (const auto& r : regions_) r->pattern_into(out);
  }
  // Phase 2 (one-pass linear map): every region's filter inputs, concatenated -- linear in x, so a caller
  // takes G = d(prefilter)/dx ONCE (AAD) and afterwards reads patterns off z = G·x (pattern_from_prefilter).
  void prefilter_into(std::vector<S>& out) const {
    out.clear();
    for (const auto& r : regions_) r->prefilter_into(out);
  }
  // Analytic re-take (EXPERIMENT): supported for a curve with EXACTLY ONE value-dependent region (the one
  // whose tangents are overridden / decoded); otherwise the caller falls back to the AAD re-take.
  int n_value_dependent_regions() const {
    int k = 0;
    for (const auto& r : regions_) k += r->linear() ? 0 : 1;
    return k;
  }
  void set_tangent_override(const S* m) {
    for (auto& r : regions_) if (!r->linear()) r->set_tangent_override(m);
  }
  int pwl_nodes() const {  // after a set_forwards
    for (const auto& r : regions_) if (!r->linear()) return r->n_nodes();
    return 0;
  }
  // The single value-dependent region's cell description (the capability; 0 / no-op on a linear curve).
  int pwl_pattern_bytes() const {
    for (const auto& r : regions_) if (!r->linear()) return r->node_pattern_bytes();
    return 0;
  }
  int pwl_prefilter_size() const {
    int n = 0;
    for (const auto& r : regions_) n += r->prefilter_size();
    return n;
  }
  void pwl_node_formula(int j, const unsigned char* pat, double* phi) const {
    for (const auto& r : regions_) if (!r->linear()) { r->node_formula(j, pat, phi); return; }
  }
  // Requires one prior set_forwards (a region's node spacing includes its join time).
  void pattern_from_prefilter(const Eigen::VectorXd& z, std::vector<unsigned char>& out) const {
    out.clear();
    int o = 0;
    for (const auto& r : regions_) {
      const int k = r->prefilter_size();
      if (k) r->pattern_from_prefilter(z.data() + o, out);
      o += k;
    }
  }

 private:
  template <class Fn>
  S locate(double t, Fn fn) const {
    for (const auto& r : regions_)
      if (t <= r->t_end()) return fn(*r, t);
    return fn(*regions_.back(), t);  // beyond the last region -> extrapolate
  }
  std::vector<std::unique_ptr<RegionIface<S>>> regions_;
  int n_ = 0;
  std::vector<S> buf_;  // reused set_forwards scratch (see set_forwards) — sized once, no per-call alloc
  mutable std::vector<double> tg_dinv_, tg_dI_, tg_own_, tg_tmp_;  // integral_tangent scratch, sized once
};

// Cross-region join check, for ANY layout: each region must start strictly after the previous one ends.
// (Per-region strictly-increasing knots are enforced in the region ctors.) A zero-length first segment
// in a spline region produces NaN rather than an error, so this is checked up front -- it applies to
// user-composed region lists from the web composer exactly as it does to the shipped layouts.
inline void check_region_joins(const std::vector<CurveModule>& modules) {
  // E3-E1/E2 (2026-09-10): a curve with NO knots used to build an empty ModularCurve (n = 0) and crash
  // downstream (run_json `"regions": []` segfaulted); a knot at t <= 0 built silently and corrupted the
  // integral / W (a negative weight, ~49 bp of log-DF) while calibration "succeeded". Both are refused here,
  // the one factory every path (templated, AAD block, compiled W) builds curves through.
  std::size_t n_knots = 0;
  for (const auto& m : modules) n_knots += m.knots.size();
  if (n_knots == 0) throw std::invalid_argument("make_modular_curve: a curve needs at least one region with knots (regions: [] is not a curve)");
  for (std::size_t i = 0; i < modules.size(); ++i) {
    if (modules[i].knots.empty()) continue;
    const double t0 = modules[i].knots.front();
    if (!(t0 > 0.0))
      throw std::invalid_argument("make_modular_curve: region " + std::to_string(i) + " has a knot at t = " + std::to_string(t0) +
                                  "; every knot time must be > 0 (times are year fractions from the evaluation date)");
  }
  for (std::size_t i = 1; i < modules.size(); ++i) {
    const auto& prev = modules[i - 1].knots;
    const auto& cur = modules[i].knots;
    if (prev.empty() || cur.empty()) continue;
    if (!(cur.front() > prev.back()))
      throw std::invalid_argument("make_modular_curve: region " + std::to_string(i) + " starts at " +
                                  std::to_string(cur.front()) + ", which must exceed the previous region's "
                                  "last knot (" + std::to_string(prev.back()) + ")");
  }
}

// THE curve factory. Build a curve from a list of (knots, scheme) modules -- shipped layout or bespoke.
template <class S>
ModularCurve<S> make_modular_curve(const std::vector<CurveModule>& modules) {
  check_region_joins(modules);
  ModularCurve<S> c;
  for (const auto& m : modules) c.add(m);
  return c;
}

// ---- Named layouts -------------------------------------------------------------------------------
// A curve "flavour" is a module list, not a type. These are the shipped ones; anything else is just a
// different list. Empty regions are dropped so a knots-only-in-the-back curve is a one-region layout.

inline std::vector<CurveModule> two_region_layout(const std::vector<double>& front,
                                                  const std::vector<double>& back, Scheme back_scheme) {
  std::vector<CurveModule> m;
  if (!front.empty()) m.push_back({front, Scheme::Flat});
  if (!back.empty()) m.push_back({back, back_scheme});
  return m;
}

// The SHIPPED calibration layout: piecewise-flat meeting-date front + LOCAL C1 Hermite back.
//
// Hermite (vs a global natural cubic) keeps calibrated forwards local and non-oscillating -- a par swap
// constrains an INTEGRAL, so a global spline lets point forwards swing wildly while the rates still
// match; the local Hermite does not. It stays a LINEAR MAP of the knot values, so the W-cache /
// analytic-Jacobian / microsecond warm-recal fast path is preserved (CLAUDE.md §2). Validated
// end-to-end against QuantLib via the YieldTermStructure oracle.
inline std::vector<CurveModule> flat_hermite(const std::vector<double>& meeting,
                                             const std::vector<double>& back) {
  return two_region_layout(meeting, back, Scheme::Hermite);
}

// Control-point cubic B-spline back end (C2, convex-hull, positivity-friendly). Linear in the control
// points, so still W-cacheable -- but the free variables are control points (see Scheme::BSpline).
inline std::vector<CurveModule> flat_bspline(const std::vector<double>& meeting,
                                             const std::vector<double>& back) {
  return two_region_layout(meeting, back, Scheme::BSpline);
}

// Hyman-filtered MONOTONE cubic back end (matches QuantLib MonotonicCubicNaturalSpline). The filter is
// VALUE-dependent, so this layout reports is_linear_map()==false and calibrates on the AAD tier rather
// than the W-cache -- it is the concrete layout that exercises that fallback (CLAUDE.md §2).
inline std::vector<CurveModule> flat_monotone(const std::vector<double>& meeting,
                                              const std::vector<double>& back) {
  return two_region_layout(meeting, back, Scheme::MonotoneCubic);
}
// Hagan-West MONOTONE CONVEX back end (regions.hpp MonotoneConvex): the back knots' values are the DISCRETE forwards
// over the intervals between them (the first interval starts at the join). Value-dependent and not piecewise-linear
// -> the AAD tier only; the layout that exercises the tier's refusal (tests/monotone_convex_test.cpp).
inline std::vector<CurveModule> flat_monotone_convex(const std::vector<double>& meeting,
                                                     const std::vector<double>& back) {
  return two_region_layout(meeting, back, Scheme::MonotoneConvex);
}

// Tension-spline back end (research note §3): a flat meeting-date front + a spline-under-tension back
// with fixed hyperparameter σ. Like flat_hermite/flat_bspline it stays a LINEAR MAP of the knot forwards
// (σ fixed), so it is W-cacheable and analytic-Jacobian ready; σ trades smoothness (σ→0 == the natural
// cubic) for tautness (large σ suppresses overshoot). Validated end-to-end against QuantLib.
inline std::vector<CurveModule> flat_tension(const std::vector<double>& meeting,
                                             const std::vector<double>& back, double sigma) {
  std::vector<CurveModule> m;
  if (!meeting.empty()) m.push_back({meeting, Scheme::Flat, 0.0});
  if (!back.empty()) m.push_back({back, Scheme::Tension, sigma});
  return m;
}

}  // namespace swaps::curve
