#pragma once
// Region interpolation policies -- the math of each region, composed by ModularCurve (curve_module.hpp).
//
// A region maps its slice of the knot forwards to the instantaneous forward f(t) LINEARLY, and knows
// how to integrate it. Regions are stitched left-to-right by a Boundary handoff: C0 (level) by
// default -- a region pins its start to the previous region's end value -- with an optional C1 slope
// available for smooth region joins. Because every policy here is linear in the values, the whole
// curve's integral (log-discount) stays linear in x, so the W-cache / analytic-Jacobian / linear
// warm-recal fast path is preserved (CLAUDE.md §2).
//
// Each policy provides (Scalar-templated so AAD flows through):
//   int  n_values() const;                                  // free knot values it consumes
//   double t_end() const;                                   // region's end time
//   void build(const Vec& x, int off, int n, const Boundary&);   // linear in x[off..off+n)
//   Scalar forward(double t) const;  Scalar integral(double t) const;  // integral is from the origin
//   Boundary out() const;                                   // handoff to the next region
//   static constexpr bool is_linear_map = true;

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace swaps::curve {

// Knot times for a region must be finite and STRICTLY increasing: a duplicate (or unsorted) knot gives
// a zero-length segment -> a 0/0 in the interpolation -> silent NaN downstream. Reject at construction.
inline void require_increasing_knots(const std::vector<double>& k, const char* who) {
  if (k.empty()) throw std::invalid_argument(std::string(who) + ": empty knot set");
  for (std::size_t i = 0; i < k.size(); ++i) {
    if (!std::isfinite(k[i]))
      throw std::invalid_argument(std::string(who) + ": non-finite knot at index " + std::to_string(i));
    if (i && !(k[i] > k[i - 1]))
      throw std::invalid_argument(std::string(who) + ": knots must be strictly increasing (duplicate/unsorted at index " +
                                  std::to_string(i) + ": " + std::to_string(k[i - 1]) + ", " + std::to_string(k[i]) + ")");
  }
}

template <class Scalar>
struct Boundary {
  double time = 0.0;      // where this boundary sits
  Scalar value{0.0};      // forward level at the boundary (C0 handoff)
  Scalar slope{0.0};      // forward slope (optional C1 handoff)
  Scalar integral{0.0};   // cumulative integral_0^time f
  // FALSE only for the FIRST region of a curve (no predecessor): a LEADING region must NOT pin its start
  // to this (phantom, zero) value and ramp up to its first knot -- it flat-extrapolates its first FREE
  // knot value v1 = x[off] backwards, exactly mirroring how every scheme flat-extrapolates its LAST knot.
  // TRUE for every following region: pin the leading value to `value` for a genuine C0 join (unchanged
  // behaviour). Default false so the seed Boundary{} in set_forwards marks the first region leading; a
  // directly-constructed join boundary (unit tests) must set this true. set_forwards sets it centrally.
  bool has_predecessor = false;
};

// Piecewise-flat forward, breakpoints at the given times. f is constant on (t_{k-1}, t_k] and jumps
// at each internal breakpoint (the meeting-date front). Ignores the incoming value (it is self-
// contained); consumes one value per segment.
template <class Scalar>
class Flat {
 public:
  explicit Flat(std::vector<double> ends) : m_(std::move(ends)) { require_increasing_knots(m_, "Flat"); }
  int n_values() const { return static_cast<int>(m_.size()); }
  double t_end() const { return m_.back(); }
  static constexpr bool is_linear_map = true;

  template <class Vec>
  void build(const Vec& x, int off, int n, const Boundary<Scalar>& in) {
    t0_ = in.time;
    I0_ = in.integral;
    f_.resize(n);
    Icum_.resize(n);
    Scalar acc(0.0);
    double prev = t0_;
    for (int k = 0; k < n; ++k) {
      f_[k] = x[off + k];
      Icum_[k] = acc;  // integral from t0 to the START of segment k
      acc += f_[k] * (m_[k] - prev);
      prev = m_[k];
    }
    region_int_ = acc;
  }

  Scalar forward(double t) const { return t <= t0_ ? f_.front() : f_[seg(t)]; }
  // Derivatives of the forward (regulariser / energy; interior points only — the jumps at the breakpoints
  // are intentional and carry no energy). Width-preserving zeros for the AAD scalar types.
  Scalar forward_d1(double) const { return f_.front() * 0.0; }
  Scalar forward_d2(double) const { return f_.front() * 0.0; }
  std::vector<double> pieces() const { return m_; }  // the polynomial (here: constant) pieces break at the knots
  Scalar integral(double t) const {
    if (t <= t0_) return I0_;
    if (t >= m_.back()) return I0_ + region_int_ + f_.back() * (t - m_.back());
    const int k = seg(t);
    const double prev = (k == 0) ? t0_ : m_[k - 1];
    return I0_ + Icum_[k] + f_[k] * (t - prev);
  }
  Boundary<Scalar> out() const { return {m_.back(), f_.back(), Scalar(0.0), I0_ + region_int_}; }

 private:
  int seg(double t) const {
    auto it = std::lower_bound(m_.begin(), m_.end(), t);  // first m_k >= t
    return it == m_.end() ? static_cast<int>(m_.size()) - 1 : static_cast<int>(it - m_.begin());
  }
  std::vector<double> m_;
  double t0_ = 0.0;
  Scalar I0_{0.0}, region_int_{0.0};
  std::vector<Scalar> f_, Icum_;
};

// Piecewise-linear forward through (t_k, y_k). C0 across joins (its first knot = the incoming value),
// consumes one value per knot time.
template <class Scalar>
class Linear {
 public:
  explicit Linear(std::vector<double> knots) : s_(std::move(knots)) { require_increasing_knots(s_, "Linear"); }
  int n_values() const { return static_cast<int>(s_.size()); }
  double t_end() const { return s_.back(); }
  static constexpr bool is_linear_map = true;

  template <class Vec>
  void build(const Vec& x, int off, int n, const Boundary<Scalar>& in) {
    // LEADING (no predecessor): interpolate the n ACTUAL knots and flat-extrapolate the first value v1
    // backwards (forward(t<t1)=v1), mirroring the far end -- NO phantom (0,0) join, NO ramp from zero.
    // FOLLOWING: pin a leading node to the incoming boundary (in.time,in.value) for the C0 join (unchanged).
    const bool lead = !in.has_predecessor;
    const int N = lead ? n : n + 1;
    if (N < 2) throw std::invalid_argument("Linear: needs >= 2 nodes (a LEADING region needs >= 2 knots)");  // PATCH
    t_.resize(N);
    y_.resize(N);
    I_.resize(N);
    if (lead) {
      for (int i = 0; i < n; ++i) { t_[i] = s_[i]; y_[i] = x[off + i]; }
      I_[0] = in.integral + y_[0] * (t_[0] - in.time);  // flat pre-segment [in.time,t1] at v1
    } else {
      t_[0] = in.time;
      y_[0] = in.value;
      for (int i = 0; i < n; ++i) { t_[i + 1] = s_[i]; y_[i + 1] = x[off + i]; }
      I_[0] = in.integral;
    }
    for (int i = 0; i + 1 < N; ++i) {
      const double h = t_[i + 1] - t_[i];
      I_[i + 1] = I_[i] + 0.5 * (y_[i] + y_[i + 1]) * h;  // trapezoid = exact for linear f
    }
  }

  Scalar forward(double t) const {
    if (t <= t_.front()) return y_.front();  // flat pre-segment (leading) / clamp start (following: t=in.time)
    if (t >= t_.back()) return y_.back();
    const int i = seg(t);
    const double w = (t - t_[i]) / (t_[i + 1] - t_[i]);
    return y_[i] + (y_[i + 1] - y_[i]) * w;
  }
  Scalar forward_d1(double t) const {
    if (t <= t_.front() || t >= t_.back()) return y_.front() * 0.0;
    const int i = seg(t);
    return (y_[i + 1] - y_[i]) / (t_[i + 1] - t_[i]);
  }
  Scalar forward_d2(double) const { return y_.front() * 0.0; }
  std::vector<double> pieces() const { return t_; }
  Scalar integral(double t) const {
    if (t <= t_.front()) return I_.front() - y_.front() * (t_.front() - t);  // flat pre-segment at v1
    if (t >= t_.back()) return I_.back() + y_.back() * (t - t_.back());
    const int i = seg(t);
    const double u = t - t_[i], h = t_[i + 1] - t_[i];
    const Scalar slope = (y_[i + 1] - y_[i]) / h;
    return I_[i] + u * (y_[i] + 0.5 * slope * u);
  }
  Boundary<Scalar> out() const {
    const double h = t_.back() - t_[t_.size() - 2];
    return {t_.back(), y_.back(), (y_.back() - y_[t_.size() - 2]) / h, I_.back()};
  }

 private:
  int seg(double t) const {
    auto it = std::upper_bound(t_.begin(), t_.end(), t);
    return static_cast<int>(it - t_.begin()) - 1;
  }
  std::vector<double> s_, t_;
  std::vector<Scalar> y_, I_;
};

// Natural cubic spline on the forwards (C2 internally; C0 at the near join via the pinned leading
// value; natural zero-curvature at the far end). This reproduces the current back-end interpolation.
template <class Scalar>
class NaturalCubic {
 public:
  explicit NaturalCubic(std::vector<double> knots) : s_(std::move(knots)) { require_increasing_knots(s_, "NaturalCubic"); }
  int n_values() const { return static_cast<int>(s_.size()); }
  double t_end() const { return s_.back(); }
  static constexpr bool is_linear_map = true;

  template <class Vec>
  void build(const Vec& x, int off, int n, const Boundary<Scalar>& in) {
    // LEADING: spline the n actual knots, flat-extrapolate v1=x[off] for t<t1 (mirrors the far end); no
    // phantom (0,0). FOLLOWING: prepend the pinned (in.time,in.value) join point for C0 continuity (as before).
    const bool lead = !in.has_predecessor;
    const int N = lead ? n : n + 1;
    if (N < 2) throw std::invalid_argument("NaturalCubic: needs >= 2 nodes (a LEADING region needs >= 2 knots)");  // PATCH
    xs_.resize(N);
    ys_.resize(N);
    if (lead) {
      for (int i = 0; i < n; ++i) { xs_[i] = s_[i]; ys_[i] = x[off + i]; }
    } else {
      xs_[0] = in.time;
      ys_[0] = in.value;  // C0: leading value pinned to the incoming boundary
      for (int i = 0; i < n; ++i) { xs_[i + 1] = s_[i]; ys_[i + 1] = x[off + i]; }
    }

    const int nseg = N - 1;
    std::vector<double> h(nseg);
    for (int i = 0; i < nseg; ++i) h[i] = xs_[i + 1] - xs_[i];

    // natural: M[0]=M[N-1]=0; Thomas on the interior. WIDTH-PRESERVING zero (ys_[0]*0.0), not Scalar(0.0):
    // for a Dual, Scalar(0.0) has an EMPTY derivative vector and Eigen's expr(empty)+expr(M) is UB (it drops
    // the derivatives silently for N==2 and asserts in a debug build for N>=3).
    std::vector<Scalar> M(N, ys_[0] * 0.0);
    if (nseg >= 2) {
      const int k = nseg - 1;
      std::vector<double> lower(k), diag(k), upper(k);
      std::vector<Scalar> rhs(k);
      for (int i = 1; i <= k; ++i) {
        lower[i - 1] = h[i - 1];
        diag[i - 1] = 2.0 * (h[i - 1] + h[i]);
        upper[i - 1] = h[i];
        rhs[i - 1] = 6.0 * ((ys_[i + 1] - ys_[i]) / h[i] - (ys_[i] - ys_[i - 1]) / h[i - 1]);
      }
      for (int i = 1; i < k; ++i) {
        const double w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] = rhs[i] - w * rhs[i - 1];
      }
      std::vector<Scalar> mi(k);
      mi[k - 1] = rhs[k - 1] / diag[k - 1];
      for (int i = k - 1; i-- > 0;) mi[i] = (rhs[i] - upper[i] * mi[i + 1]) / diag[i];
      for (int i = 1; i <= k; ++i) M[i] = mi[i - 1];
    }

    a_.resize(nseg);
    b_.resize(nseg);
    c_.resize(nseg);
    d_.resize(nseg);
    for (int i = 0; i < nseg; ++i) {
      a_[i] = ys_[i];
      b_[i] = (ys_[i + 1] - ys_[i]) / h[i] - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0;
      c_[i] = M[i] / 2.0;
      d_[i] = (M[i + 1] - M[i]) / (6.0 * h[i]);
    }
    Is_.resize(N);
    Is_[0] = lead ? (in.integral + ys_[0] * (xs_[0] - in.time)) : in.integral;  // leading: flat pre-seg at v1
    for (int i = 0; i < nseg; ++i) {
      const double u = h[i];
      Is_[i + 1] = Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
    }
    const double hl = h[nseg - 1];
    end_slope_ = b_[nseg - 1] + hl * (2.0 * c_[nseg - 1] + 3.0 * d_[nseg - 1] * hl);
  }

  Scalar forward(double t) const {
    if (t <= xs_.front()) return ys_.front();  // flat pre-segment (leading) / clamp start (following)
    if (t >= xs_.back()) return ys_.back();
    const int i = seg(t);
    const double u = t - xs_[i];
    return a_[i] + u * (b_[i] + u * (c_[i] + u * d_[i]));
  }
  Scalar forward_d1(double t) const {  // exact derivatives of the piece cubic (flat outside the region)
    if (t <= xs_.front() || t >= xs_.back()) return ys_.front() * 0.0;
    const int i = seg(t);
    const double u = t - xs_[i];
    return b_[i] + u * (2.0 * c_[i] + 3.0 * u * d_[i]);
  }
  Scalar forward_d2(double t) const {
    if (t <= xs_.front() || t >= xs_.back()) return ys_.front() * 0.0;
    const int i = seg(t);
    const double u = t - xs_[i];
    return 2.0 * c_[i] + 6.0 * u * d_[i];
  }
  std::vector<double> pieces() const { return xs_; }  // the polynomial pieces break at the nodes (incl. the join)
  Scalar integral(double t) const {
    if (t <= xs_.front()) return Is_.front() - ys_.front() * (xs_.front() - t);  // flat pre-segment at v1
    if (t >= xs_.back()) return Is_.back() + ys_.back() * (t - xs_.back());
    const int i = seg(t);
    const double u = t - xs_[i];
    return Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
  }
  Boundary<Scalar> out() const { return {xs_.back(), ys_.back(), end_slope_, Is_.back()}; }

 private:
  int seg(double t) const {
    auto it = std::upper_bound(xs_.begin(), xs_.end(), t);
    return static_cast<int>(it - xs_.begin()) - 1;
  }
  std::vector<double> s_, xs_;
  std::vector<Scalar> ys_, a_, b_, c_, d_, Is_;
  Scalar end_slope_{0.0};
};

// Local C1 cubic (Hermite) with Bessel/parabolic tangents. Each node's tangent is a fixed linear
// combination of its neighbours' values, so f is C1, LOCAL (a knot influences only its adjacent
// intervals -> banded W, local deltas), and still a linear map of the knot values -> fast path
// preserved. Contrast NaturalCubic, which is C2 but GLOBAL (dense W). C0 at the near join (pinned
// leading value); one-sided parabolic tangents at the ends.
template <class Scalar>
class Hermite {
 public:
  explicit Hermite(std::vector<double> knots) : s_(std::move(knots)) { require_increasing_knots(s_, "Hermite"); }
  int n_values() const { return static_cast<int>(s_.size()); }
  double t_end() const { return s_.back(); }
  static constexpr bool is_linear_map = true;

  template <class Vec>
  void build(const Vec& x, int off, int n, const Boundary<Scalar>& in) {
    // LEADING: Hermite over the n actual knots, flat-extrapolate v1=x[off] for t<t1 (mirrors the far end);
    // no phantom (0,0). FOLLOWING: prepend the pinned (in.time,in.value) join point for C0 (unchanged).
    const bool lead = !in.has_predecessor;
    const int N = lead ? n : n + 1;
    if (N < 2) throw std::invalid_argument("Hermite: needs >= 2 nodes (a LEADING region needs >= 2 knots)");  // PATCH
    xs_.resize(N);
    ys_.resize(N);
    if (lead) {
      for (int i = 0; i < n; ++i) { xs_[i] = s_[i]; ys_[i] = x[off + i]; }
    } else {
      xs_[0] = in.time;
      ys_[0] = in.value;
      for (int i = 0; i < n; ++i) { xs_[i + 1] = s_[i]; ys_[i + 1] = x[off + i]; }
    }
    const int nseg = N - 1;
    std::vector<double> h(nseg);
    std::vector<Scalar> sec(nseg);
    for (int i = 0; i < nseg; ++i) {
      h[i] = xs_[i + 1] - xs_[i];
      sec[i] = (ys_[i + 1] - ys_[i]) / h[i];
    }
    // Bessel/parabolic tangents (linear in ys).
    std::vector<Scalar> m(N);
    for (int j = 1; j < N - 1; ++j)
      m[j] = (h[j] * sec[j - 1] + h[j - 1] * sec[j]) / (h[j - 1] + h[j]);
    if (nseg >= 2) {
      m[0] = ((2.0 * h[0] + h[1]) * sec[0] - h[0] * sec[1]) / (h[0] + h[1]);
      m[N - 1] = ((2.0 * h[nseg - 1] + h[nseg - 2]) * sec[nseg - 1] - h[nseg - 1] * sec[nseg - 2]) /
                 (h[nseg - 1] + h[nseg - 2]);
    } else {
      m[0] = sec[0];
      m[1] = sec[0];
    }

    a_.resize(nseg);
    b_.resize(nseg);
    c_.resize(nseg);
    d_.resize(nseg);
    for (int i = 0; i < nseg; ++i) {
      const double hi = h[i];
      a_[i] = ys_[i];
      b_[i] = m[i];
      c_[i] = 3.0 * sec[i] / hi - (2.0 * m[i] + m[i + 1]) / hi;
      d_[i] = (m[i] + m[i + 1]) / (hi * hi) - 2.0 * sec[i] / (hi * hi);
    }
    Is_.resize(N);
    Is_[0] = lead ? (in.integral + ys_[0] * (xs_[0] - in.time)) : in.integral;  // leading: flat pre-seg at v1
    for (int i = 0; i < nseg; ++i) {
      const double u = h[i];
      Is_[i + 1] = Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
    }
    end_slope_ = m[N - 1];
  }

  Scalar forward(double t) const {
    if (t <= xs_.front()) return ys_.front();  // flat pre-segment (leading) / clamp start (following)
    if (t >= xs_.back()) return ys_.back();
    const int i = seg(t);
    const double u = t - xs_[i];
    return a_[i] + u * (b_[i] + u * (c_[i] + u * d_[i]));
  }
  Scalar forward_d1(double t) const {  // exact derivatives of the piece cubic (flat outside the region)
    if (t <= xs_.front() || t >= xs_.back()) return ys_.front() * 0.0;
    const int i = seg(t);
    const double u = t - xs_[i];
    return b_[i] + u * (2.0 * c_[i] + 3.0 * u * d_[i]);
  }
  Scalar forward_d2(double t) const {
    if (t <= xs_.front() || t >= xs_.back()) return ys_.front() * 0.0;
    const int i = seg(t);
    const double u = t - xs_[i];
    return 2.0 * c_[i] + 6.0 * u * d_[i];
  }
  std::vector<double> pieces() const { return xs_; }  // the polynomial pieces break at the nodes (incl. the join)
  Scalar integral(double t) const {
    if (t <= xs_.front()) return Is_.front() - ys_.front() * (xs_.front() - t);  // flat pre-segment at v1
    if (t >= xs_.back()) return Is_.back() + ys_.back() * (t - xs_.back());
    const int i = seg(t);
    const double u = t - xs_[i];
    return Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
  }
  Boundary<Scalar> out() const { return {xs_.back(), ys_.back(), end_slope_, Is_.back()}; }

 private:
  int seg(double t) const {
    auto it = std::upper_bound(xs_.begin(), xs_.end(), t);
    return static_cast<int>(it - xs_.begin()) - 1;
  }
  std::vector<double> s_, xs_;
  std::vector<Scalar> ys_, a_, b_, c_, d_, Is_;
  Scalar end_slope_{0.0};
};

// Hyman-filtered MONOTONE cubic on the forward-at-knot values -- the first NON-LINEAR region policy
// (is_linear_map = false). It is a C2 natural cubic spline whose node first-derivatives are passed
// through Hyman's monotonicity filter, so the interpolated forward cannot overshoot / oscillate the way
// an unconstrained spline can. This is a standard production choice for rate curves; here it is also the
// concrete curve that EXERCISES the AAD fallback tier -- because the filter clamps the derivatives with
// data-dependent min/max/sign branches, the coefficients (and hence integral(t)) are NOT a linear map of
// the knot values, so the W-cache / analytic-Jacobian fast path does NOT apply. Calibration must (and
// does) route through the templated AAD engine instead; the min/max/abs branches are piecewise
// differentiable, so AutoDiffScalar carries a valid (one-sided at kinks) gradient through them.
//
// Transcribed to match QuantLib's MonotonicCubicNaturalSpline exactly (CubicInterpolation with da=Spline,
// monotonic=true, SecondDerivative=0 at both ends): same first-derivative tridiagonal, same Hyman filter,
// validated to ~1e-12 in tests/monotone_cubic_test.cpp. Node 0 is the pinned join value (C0 handoff, like
// NaturalCubic/Hermite). The per-segment cubic build + analytic quartic integral are identical to Hermite;
// only the tangents differ (spline + filter, vs Hermite's Bessel tangents).
template <class Scalar>
class MonotoneCubic {
 public:
  explicit MonotoneCubic(std::vector<double> knots) : s_(std::move(knots)) {
    require_increasing_knots(s_, "MonotoneCubic");
  }
  int n_values() const { return static_cast<int>(s_.size()); }
  double t_end() const { return s_.back(); }
  static constexpr bool is_linear_map = false;  // Hyman filter is value-dependent -> AAD tier, not W-cache

  template <class Vec>
  void build(const Vec& x, int off, int n, const Boundary<Scalar>& in) {
    // LEADING: filter+spline the n actual knots, flat-extrapolate v1=x[off] for t<t1 (mirrors the far end);
    // no phantom (0,0). FOLLOWING: prepend the pinned (in.time,in.value) join node for C0 (unchanged).
    const bool lead = !in.has_predecessor;
    const int N = lead ? n : n + 1;
    if (N < 2) throw std::invalid_argument("MonotoneCubic: needs >= 2 nodes (>=1 back knot when following)");
    xs_.resize(N);
    ys_.resize(N);
    if (lead) {
      for (int i = 0; i < n; ++i) { xs_[i] = s_[i]; ys_[i] = x[off + i]; }
    } else {
      xs_[0] = in.time;
      ys_[0] = in.value;
      for (int i = 0; i < n; ++i) { xs_[i + 1] = s_[i]; ys_[i + 1] = x[off + i]; }
    }
    const int nseg = N - 1;
    std::vector<double> h(nseg);
    std::vector<Scalar> S(nseg);  // secant slopes
    for (int i = 0; i < nseg; ++i) {
      h[i] = xs_[i + 1] - xs_[i];
      S[i] = (ys_[i + 1] - ys_[i]) / h[i];
    }

    // First derivatives m[0..N-1] from the C2 (Spline) tridiagonal with natural (2nd-deriv=0) ends --
    // QuantLib CubicInterpolation(Spline, SecondDerivative=0): the SAME spline as NaturalCubic, expressed
    // directly in first-derivative (Hermite) form so the Hyman filter can act on the tangents.
    //   row 0:   2 m0 + m1                       = 3 S0
    //   row i:   h[i] m_{i-1} + 2(h[i]+h[i-1]) m_i + h[i-1] m_{i+1} = 3(h[i] S_{i-1} + h[i-1] S_i)
    //   row N-1: m_{N-2} + 2 m_{N-1}             = 3 S_{N-2}
    std::vector<double> lower(N), diag(N), upper(N);
    std::vector<Scalar> rhs(N);
    diag[0] = 2.0; upper[0] = 1.0; rhs[0] = 3.0 * S[0];
    for (int i = 1; i < N - 1; ++i) {
      lower[i] = h[i];
      diag[i] = 2.0 * (h[i] + h[i - 1]);
      upper[i] = h[i - 1];
      rhs[i] = 3.0 * (h[i] * S[i - 1] + h[i - 1] * S[i]);
    }
    lower[N - 1] = 1.0; diag[N - 1] = 2.0; rhs[N - 1] = 3.0 * S[nseg - 1];
    std::vector<Scalar> m(N);
    thomas_solve(lower, diag, upper, rhs, m);

    // EXPERIMENT: keep the filter's INPUTS (linear in x) so a caller can precompute z = G·x once and read the
    // branch pattern without rebuilding the curve (pattern_from_prefilter).
    h_ = h;
    pre_.assign(S.begin(), S.end());
    pre_.insert(pre_.end(), m.begin(), m.end());

    // Hyman monotonicity filter (bit-for-bit QuantLib): clamp each tangent so no segment overshoots.
    // The `!= ` comparisons + min/max/sign are exactly the value-dependent branches that break linearity.
    hyman_filter(h, S, m, pattern_);
    // EXPERIMENT (analytic re-take): replace the filtered tangents by caller-supplied scalars. Seeded as
    // INDEPENDENT duals, one AAD pass then splits d integral(t) into [dx | dm] = [L_t | B_t] -- the curve is
    // linear in (x, m) jointly once m is freed from the filter. Values are irrelevant to those derivatives.
    if (m_override_)
      for (int j = 0; j < N; ++j) m[j] = m_override_[j];

    // Per-segment cubic f(u) = a + b u + c u^2 + d u^3, u = t - xs_[i]  (identical form to Hermite).
    a_.resize(nseg); b_.resize(nseg); c_.resize(nseg); d_.resize(nseg);
    for (int i = 0; i < nseg; ++i) {
      const double hi = h[i];
      a_[i] = ys_[i];
      b_[i] = m[i];
      c_[i] = 3.0 * S[i] / hi - (2.0 * m[i] + m[i + 1]) / hi;
      d_[i] = (m[i] + m[i + 1]) / (hi * hi) - 2.0 * S[i] / (hi * hi);
    }
    Is_.resize(N);
    Is_[0] = lead ? (in.integral + ys_[0] * (xs_[0] - in.time)) : in.integral;  // leading: flat pre-seg at v1
    for (int i = 0; i < nseg; ++i) {
      const double u = h[i];
      Is_[i + 1] = Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
    }
    end_slope_ = m[N - 1];
  }

  Scalar forward(double t) const {
    if (t <= xs_.front()) return ys_.front();  // flat pre-segment (leading) / clamp start (following)
    if (t >= xs_.back()) return ys_.back();
    const int i = seg(t);
    const double u = t - xs_[i];
    return a_[i] + u * (b_[i] + u * (c_[i] + u * d_[i]));
  }
  Scalar forward_d1(double t) const {  // exact derivatives of the piece cubic (flat outside the region)
    if (t <= xs_.front() || t >= xs_.back()) return ys_.front() * 0.0;
    const int i = seg(t);
    const double u = t - xs_[i];
    return b_[i] + u * (2.0 * c_[i] + 3.0 * u * d_[i]);
  }
  Scalar forward_d2(double t) const {
    if (t <= xs_.front() || t >= xs_.back()) return ys_.front() * 0.0;
    const int i = seg(t);
    const double u = t - xs_[i];
    return 2.0 * c_[i] + 6.0 * u * d_[i];
  }
  std::vector<double> pieces() const { return xs_; }  // the polynomial pieces break at the nodes (incl. the join)
  Scalar integral(double t) const {
    if (t <= xs_.front()) return Is_.front() - ys_.front() * (xs_.front() - t);  // flat pre-segment at v1
    if (t >= xs_.back()) return Is_.back() + ys_.back() * (t - xs_.back());
    const int i = seg(t);
    const double u = t - xs_[i];
    return Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
  }
  Boundary<Scalar> out() const { return {xs_.back(), ys_.back(), end_slope_, Is_.back()}; }

 private:
  int seg(double t) const {
    auto it = std::upper_bound(xs_.begin(), xs_.end(), t);
    return static_cast<int>(it - xs_.begin()) - 1;
  }
  // Thomas algorithm for a tridiagonal system (double bands, Scalar rhs/solution -> AAD flows via rhs).
  static void thomas_solve(std::vector<double>& lo, std::vector<double>& di, std::vector<double>& up,
                           std::vector<Scalar>& r, std::vector<Scalar>& out) {
    const int N = static_cast<int>(di.size());
    for (int i = 1; i < N; ++i) {
      const double w = lo[i] / di[i - 1];
      di[i] -= w * up[i - 1];
      r[i] = r[i] - w * r[i - 1];
    }
    out.resize(N);
    out[N - 1] = r[N - 1] / di[N - 1];
    for (int i = N - 1; i-- > 0;) out[i] = (r[i] - up[i] * out[i + 1]) / di[i];
  }
  // min/max on the SCALAR type via comparison (Eigen's AutoDiffScalar min/max overloads are ambiguous for
  // two duals; these pick a branch by value and return it by value, so the chosen dual's derivatives ride
  // along). abs() is the unary Eigen/std global (unambiguous), pulled in per-call with `using std::abs`.
  static Scalar smin(const Scalar& a, const Scalar& b) { return b < a ? b : a; }
  static Scalar smax(const Scalar& a, const Scalar& b) { return a < b ? b : a; }
  // QuantLib's Hyman monotonicity filter on the node tangents m, given segment lengths h and secants S.
  // QuantLib's Hyman monotonicity filter on the node tangents m, given segment lengths h and secants S.
  //
  // EXPERIMENT (exp/piecewise-linear-w): the filter's VALUES are built only from abs/min/max/sign and fixed
  // h-ratios (products appear only inside conditions), so each node ends as ONE of three linear formulas:
  //   0,   m_raw,   or   sign(m_raw) * c * |candidate|   (candidate one of S[i-1], S[i], pm, pd, pu, 3·S_end)
  // `pattern` records exactly that CANONICAL outcome per node -- two bytes {kind, candidate id + sign} -- NOT
  // the decision path. (A path trace over-reports: e.g. the pd/pu guard flips with the sign of a long-end
  // second difference even when the widened candidate then loses the max, which is the same formula. That
  // made the first version re-take W on almost every streaming tick.) Equal patterns <=> the same linear map
  // of x. Values are computed with exactly the original operations in the original order (bit-identical).
  template <class T>
  static void hyman_filter(const std::vector<double>& h, const std::vector<T>& S, std::vector<T>& m,
                           std::vector<unsigned char>& pattern) {
    using std::abs;
    struct C { T v; unsigned char code; };  // a value + the canonical formula it came from
    const auto cabs = [](const T& v, int id) { return C{abs(v), static_cast<unsigned char>(2 * id + (v < 0.0 ? 1 : 0))}; };
    const auto cmin = [](const C& a, const C& b) { return b.v < a.v ? b : a; };
    const auto cmax = [](const C& a, const C& b) { return a.v < b.v ? b : a; };
    pattern.clear();
    const int N = static_cast<int>(m.size());
    for (int i = 0; i < N; ++i) {
      T correction, pm, pu, pd;
      C M;
      bool gate;  // the outer sign test: false -> the node's tangent is zeroed
      if (i == 0 || i == N - 1) {
        const T& Se = (i == 0) ? S[0] : S[N - 2];
        gate = m[i] * Se > 0.0;
        M = cabs(3.0 * Se, kEnd);
      } else {
        pm = (S[i - 1] * h[i] + S[i] * h[i - 1]) / (h[i - 1] + h[i]);
        M = cmin(cmin(cabs(S[i - 1], kS0), cabs(S[i], kS1)), cabs(pm, kPm3));
        M.v = 3.0 * M.v;
        if (i > 1) {
          if ((S[i - 1] - S[i - 2]) * (S[i] - S[i - 1]) > 0.0) {
            pd = (S[i - 1] * (2.0 * h[i - 1] + h[i - 2]) - S[i - 2] * h[i - 1]) / (h[i - 2] + h[i - 1]);
            if (pm * pd > 0.0 && pm * (S[i - 1] - S[i - 2]) > 0.0) {
              C w = cmin(cabs(pm, kPm15), cabs(pd, kPd));
              w.v = T(1.5) * w.v;
              M = cmax(M, w);
            }
          }
        }
        if (i < N - 2) {
          if ((S[i] - S[i - 1]) * (S[i + 1] - S[i]) > 0.0) {
            pu = (S[i] * (2.0 * h[i] + h[i + 1]) - S[i + 1] * h[i]) / (h[i] + h[i + 1]);
            if (pm * pu > 0.0 && -pm * (S[i] - S[i - 1]) > 0.0) {
              C w = cmin(cabs(pm, kPm15), cabs(pu, kPu));
              w.v = T(1.5) * w.v;
              M = cmax(M, w);
            }
          }
        }
        gate = m[i] * pm > 0.0;
      }
      unsigned char kind, code = 0;
      if (gate) {
        const T am = abs(m[i]);
        const bool clamp = M.v < am;  // == smin(|m|, M) picking M
        correction = m[i] / am * (clamp ? M.v : am);
        kind = clamp ? static_cast<unsigned char>(m[i] < 0.0 ? 3 : 2) : 1;  // clamp keeps sign(m): part of the formula
        if (clamp) code = M.code;
      } else {
        correction = m[i] * 0.0;  // PATCH: width-preserving zero (Scalar(0.0) is an EMPTY dual -> size mismatch)
        kind = 0;
      }
      if (correction != m[i]) m[i] = correction;
      pattern.push_back(kind);
      pattern.push_back(code);
    }
  }

 public:
  // The branch pattern of the last build (see hyman_filter). Equal patterns => identical linear map of x.
  const std::vector<unsigned char>& pattern() const { return pattern_; }
  // The filter's inputs z = [S (N-1); m_raw (N)] at the last build -- LINEAR in x (they precede the filter).
  const std::vector<Scalar>& prefilter() const { return pre_; }
  // The branch pattern for inputs z (a slice of G·x), WITHOUT rebuilding: the same recorder on doubles.
  // Requires one prior build (h_ is the node spacing, fixed by the knots and the join time).
  // EXPERIMENT (analytic re-take).
  void set_tangent_override(const Scalar* m) { m_override_ = m; }  // nullptr restores the filter
  int n_nodes() const { return static_cast<int>(xs_.size()); }       // tangents N (valid after a build)
  const std::vector<double>& node_spacing() const { return h_; }
  // DECODE a node's canonical code into its formula over the filter inputs z = [S (N-1); m_raw (N)]:
  // m_j = phi · z, written into phi[0 .. 2N-2]. The exact inverse of the encoding in hyman_filter:
  //   kind 0 -> 0;  kind 1 -> m_raw_j;  kind 2/3 -> sign(m) * factor * sign(cand) * cand,
  // with cand one of S[i-1], S[i], pm, pd, pu (interior) or S_end (ends), factor 3 or 1.5.
  static void tangent_formula(unsigned char kind, unsigned char code, int i, const std::vector<double>& h, int N,
                              double* phi) {
    const int nz = 2 * N - 1, nS = N - 1;
    for (int k = 0; k < nz; ++k) phi[k] = 0.0;
    if (kind == 0) return;
    if (kind == 1) { phi[nS + i] = 1.0; return; }
    const double f = (kind == 3 ? -1.0 : 1.0) * ((code & 1) ? -1.0 : 1.0);
    const auto pm = [&](double c) {
      const double w = h[i - 1] + h[i];
      phi[i - 1] += c * h[i] / w;
      phi[i] += c * h[i - 1] / w;
    };
    switch (code >> 1) {
      case kS0: phi[i - 1] += 3.0 * f; break;
      case kS1: phi[i] += 3.0 * f; break;
      case kPm3: pm(3.0 * f); break;
      case kPm15: pm(1.5 * f); break;
      case kPd: {
        const double w = h[i - 2] + h[i - 1];
        phi[i - 1] += 1.5 * f * (2.0 * h[i - 1] + h[i - 2]) / w;
        phi[i - 2] -= 1.5 * f * h[i - 1] / w;
        break;
      }
      case kPu: {
        const double w = h[i] + h[i + 1];
        phi[i] += 1.5 * f * (2.0 * h[i] + h[i + 1]) / w;
        phi[i + 1] -= 1.5 * f * h[i] / w;
        break;
      }
      case kEnd: phi[i == 0 ? 0 : N - 2] += 3.0 * f; break;
      default: break;
    }
  }
  // THE piecewise-linear REGION CAPABILITY (2026-09-22): a value-dependent region describes its own branch cells --
  // `node_pattern_bytes()` per node in the pattern it records, and `node_formula(j, node_pattern, phi)`: node j's
  // filtered tangent as a row over this region's prefilter inputs z, for that node's pattern bytes. The pricing
  // layer's tracker (CompiledCurveSet) reads patterns off z = G·x and re-takes W by these rows; it names no scheme.
  int node_pattern_bytes() const { return 2; }
  void node_formula(int j, const unsigned char* node_pattern, double* phi) const {
    tangent_formula(node_pattern[0], node_pattern[1], j, h_, n_nodes(), phi);
  }
  void pattern_from_prefilter(const double* z, std::vector<unsigned char>& out) const {
    const int nseg = static_cast<int>(h_.size());
    std::vector<double>& S = scratch_S_;
    std::vector<double>& m = scratch_m_;
    S.assign(z, z + nseg);
    m.assign(z + nseg, z + 2 * nseg + 1);
    hyman_filter(h_, S, m, out);
  }

 private:
  // Canonical candidate ids of the filter's clamp formula (encoder: hyman_filter; decoder: tangent_formula).
  enum : int { kS0 = 1, kS1 = 2, kPm3 = 3, kPm15 = 4, kPd = 5, kPu = 6, kEnd = 7 };
  const Scalar* m_override_ = nullptr;
  std::vector<unsigned char> pattern_;
  std::vector<double> h_;
  std::vector<Scalar> pre_;
  mutable std::vector<double> scratch_S_, scratch_m_;
  std::vector<double> s_, xs_;
  std::vector<Scalar> ys_, a_, b_, c_, d_, Is_;
  Scalar end_slope_{0.0};
};

// Clamped cubic B-SPLINE, CONTROL-POINT parameterization (docs/bezier-and-moments.md, Part A).
// The `n` free values are control points P_1..P_n; P_0 is PINNED to the incoming boundary value for a
// C0 join, exactly like Hermite/NaturalCubic pin their leading value. Distinctive vs those:
//   * C2 (a cubic B-spline is automatically C2 across its interior knots) -- smoother than Hermite's C1;
//   * the CONVEX-HULL property -- f stays within [min P, max P], so forwards do not overshoot and
//     positivity is enforceable by keeping control points >= 0;
//   * the control points do NOT lie on the curve (except the clamped ends) -- so the calibrated x are
//     control points, mapped to forward-at-knot for risk by a fixed invertible collocation.
// Still a LINEAR MAP of the control points (de Boor is an affine combination with knot-only weights),
// so is_linear_map = true and the W-cache / analytic-Jacobian fast path is preserved.
//
// Interior breakpoints WERE uniform in time over the region until ce82d89; they are the de Boor knot averages now (see below) (the standard, well-conditioned default; a
// control-point B-spline is approximating, not interpolating, so instrument maturities need not be
// breakpoints). n free control points + the pinned P_0 => a clamped cubic over n-2 segments.
//
// `integral(t)` uses 2-point Gauss-Legendre per breakpoint segment: EXACT for a cubic (deg <= 3), not
// an approximation, and AAD-safe (nodes/weights are double, f(node) carries the derivatives). It is a
// setup-only cost -- the W-cache calls integral() once per node to build W, never on the hot path.
template <class Scalar>
class BSpline {
 public:
  explicit BSpline(std::vector<double> knots) : s_(std::move(knots)) {
    require_increasing_knots(s_, "BSpline");
  }
  int n_values() const { return static_cast<int>(s_.size()); }
  double t_end() const { return s_.back(); }
  static constexpr bool is_linear_map = true;

  template <class Vec>
  void build(const Vec& x, int off, int n, const Boundary<Scalar>& in) {
    if (n < 3) throw std::invalid_argument("BSpline: needs >= 3 back knots for a cubic B-spline");
    // LEADING (no predecessor): the pinned P_0 must NOT be the phantom zero join (which drags the clamped
    // start value f(start)=cp_[0] to 0 and ramps the whole short end). Tie P_0 to the first FREE control
    // point (cp_[0]=x[off]) so the clamp starts at a CALIBRATED value, and place the B-spline over [t1,te]
    // with a genuinely FLAT pre-segment [in.time,t1] at that value -- symmetric with the far-end clamp.
    // DOF-neutral: n_values() stays n (cp_[0] mirrors x[off], not a new variable).
    // FOLLOWING: P_0 = in.value is the real C0 join; the pre-segment is zero-length (t0_==pre_t_). Unchanged.
    const bool lead = !in.has_predecessor;
    pre_t_ = in.time;                       // start of the flat pre-segment (== t0_ for a following region)
    t0_ = lead ? s_.front() : in.time;      // clamped B-spline domain start
    I0_ = in.integral;
    const double te = s_.back();
    const int m = n + 1;  // control points: P_0 (pinned) + n free
    // Clamped cubic knot vector: 4x t0, (n-3) DATA-ADAPTED interior, 4x te  (length m+4 = n+5).
    // The interior knots are the de Boor AVERAGES of the control-point parameter SITES, not a uniform
    // grid over [t0,te]. Uniform interior ignores where the data actually is: with a dense front strip
    // (3M futures) and a sparse long end (annual swaps) the first uniform breakpoint lands far out, so
    // the whole near-join span is ONE stiff cubic Bezier that cannot leave the pinned P_0 (the front
    // level) promptly -- a spurious FLAT LIP just past the join. Averaging places breakpoints where the
    // knots are, so the basis is dense where the data is dense and the curve follows it from the join.
    // Sites: P_0 sits at t0 (the join for a following region, the first knot for a leading one); the n
    // free control points sit at the region's knots s_. Standard clamped-cubic averaging (de Boor,
    // Schoenberg-Whitney => invertible collocation): tau_[3+j] = mean(u[j],u[j+1],u[j+2]). Strictly
    // increasing in (t0,te) because s_ is strictly increasing and t0 <= s_[0].
    tau_.assign(m + 4, te);
    for (int i = 0; i < 4; ++i) tau_[i] = t0_;
    std::vector<double> u(m);
    u[0] = t0_;
    for (int i = 0; i < n; ++i) u[i + 1] = s_[i];
    for (int j = 1; j <= n - 3; ++j) tau_[3 + j] = (u[j] + u[j + 1] + u[j + 2]) / 3.0;
    cp_.resize(m);
    cp_[0] = lead ? x[off] : in.value;  // leading: calibrated clamp start; following: C0 pin to the join
    for (int i = 0; i < n; ++i) cp_[i + 1] = x[off + i];
    // A LEADING region receives a default Boundary whose value/integral carry an EMPTY AAD derivative,
    // while the free control points carry width-m ones. de Boor (below) combines control points via raw
    // expression-template adds that BYPASS Eigen's make_coherent, so mixing an empty derivative with a
    // width-m one asserts (undefined behaviour under -DNDEBUG -> the "leading BSpline" crash). Coerce the
    // pinned boundary to the free control points' width via `+= (x-x)` (a width-m zero): its in-place
    // make_coherent fills zeros when leading, and is a value-preserving no-op that KEEPS the join's real
    // sensitivities when this region FOLLOWS another. No-op for a plain double Scalar. (For leading, cp_[0]
    // already == x[off] carries the right width; the coerce keeps I0_ coherent and is a harmless no-op there.)
    { Scalar zero_w = x[off]; zero_w -= x[off]; cp_[0] += zero_w; I0_ += zero_w; }
    // Distinct breakpoints (t0, interior knots, te) for exact segment-wise integration.
    brk_.clear();
    brk_.push_back(t0_);
    for (int j = 1; j <= n - 3; ++j) brk_.push_back(tau_[3 + j]);
    brk_.push_back(te);
    // Whole-region integral, for out() and flat extrapolation. Seed from the first segment so the
    // accumulator carries derivatives (AAD SAFETY, top of this header).
    region_int_ = gauss2(brk_[0], brk_[1]);
    Ibrk_.resize(brk_.size());
    Ibrk_[0] = region_int_ * 0.0;  // width-preserving zero
    Ibrk_[1] = region_int_;
    for (std::size_t k = 1; k + 1 < brk_.size(); ++k) { region_int_ += gauss2(brk_[k], brk_[k + 1]); Ibrk_[k + 1] = region_int_; }
  }

  Scalar forward(double t) const {
    const double te = s_.back();
    if (t <= t0_) return cp_.front();  // flat pre-segment (leading) / clamp start (following: t0_==in.time)
    if (t >= te) return deboor(te);    // flat extrapolation beyond the region
    return deboor(t);
  }
  // Exact derivatives: the derivative of a degree-p B-spline is a degree-(p−1) B-spline on the knot vector with
  // its first and last knots dropped, with control points p·(c_{i+1} − c_i)/(τ_{i+p+1} − τ_{i+1}).
  Scalar forward_d1(double t) const {
    if (t <= t0_ || t >= s_.back()) return cp_.front() * 0.0;
    return deboor_general(deriv_cp(cp_, 3, 0), tau_.data() + 1, 2, t);
  }
  Scalar forward_d2(double t) const {
    if (t <= t0_ || t >= s_.back()) return cp_.front() * 0.0;
    return deboor_general(deriv_cp(deriv_cp(cp_, 3, 0), 2, 1), tau_.data() + 2, 1, t);
  }
  std::vector<double> pieces() const { return brk_; }  // the TRUE polynomial breakpoints (de Boor-averaged interior knots)
  Scalar integral(double t) const {
    // Flat pre-segment [pre_t_,t0_] at level cp_.front(): zero-length for a following region (pre_t_==t0_),
    // the calibrated flat short end for a leading one (pre_t_=in.time < t0_=t1).
    if (t <= t0_) return I0_ + cp_.front() * (t - pre_t_);
    const double te = s_.back();
    const Scalar base = I0_ + cp_.front() * (t0_ - pre_t_);  // integral accumulated up to t0_
    if (t >= te) return base + region_int_ + deboor(te) * (t - te);
    // PATCH: cumulative breakpoint integrals + one partial-segment gauss2 (O(log n) instead of O(n) de Boors)
    const std::size_t k = static_cast<std::size_t>(std::upper_bound(brk_.begin(), brk_.end(), t) - brk_.begin()) - 1;
    return base + Ibrk_[k] + gauss2(brk_[k], t);
  }
  Boundary<Scalar> out() const {
    const Scalar base = I0_ + cp_.front() * (t0_ - pre_t_);  // include the flat pre-segment (leading)
    return {s_.back(), deboor(s_.back()), Scalar(0.0), base + region_int_};
  }

  // Second moment ∫_a^b f(u)^2 du over this region -- the convexity term of the moment scheme
  // (docs/bezier-and-moments.md Part B). QUADRATIC in the control points (not linear): a separate
  // primitive beside the linear W. Integrated per breakpoint segment with 4-point Gauss-Legendre,
  // EXACT for f^2 (degree 6 <= 7). AAD-safe and setup-only. [a,b] is clamped to the region.
  Scalar integral2(double a, double b) const {
    a = std::max(a, t0_);
    b = std::min(b, s_.back());
    if (b <= a) return Scalar(0.0);
    Scalar acc{0.0};
    bool first = true;
    for (std::size_t k = 0; k + 1 < brk_.size(); ++k) {  // integrate each breakpoint cubic exactly
      const double lo = std::max(a, brk_[k]), hi = std::min(b, brk_[k + 1]);
      if (hi <= lo) continue;
      const Scalar seg = gauss4sq(lo, hi);
      if (first) { acc = seg; first = false; } else acc += seg;
    }
    return acc;
  }

 private:
  Scalar gauss4sq(double a, double b) const {  // ∫_a^b f^2, 4-pt Gauss (exact for f^2, a single cubic)
    if (b <= a) return Scalar(0.0);
    static const double gx[2] = {0.3399810435848563, 0.8611363115940526};
    static const double gw[2] = {0.6521451548625461, 0.3478548451374538};
    const double h = 0.5 * (b - a), c = 0.5 * (a + b);
    Scalar s{0.0};
    bool first = true;
    for (int i = 0; i < 2; ++i)
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        const Scalar f = deboor(c + sgn * gx[i] * h);
        const Scalar term = f * f * (gw[i] * h);
        if (first) { s = term; first = false; } else s += term;
      }
    return s;
  }
  int find_span(double t) const {
    const int m = static_cast<int>(cp_.size());
    if (t >= tau_[m]) return m - 1;
    int lo = 3, hi = m;  // clamped cubic: valid spans are [3, m-1]
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      (t < tau_[mid] ? hi : lo) = mid;
    }
    return lo;
  }
  Scalar deboor(double t) const {
    const int p = 3, k = find_span(t);
    Scalar d[4];
    for (int j = 0; j <= p; ++j) d[j] = cp_[k - p + j];
    for (int r = 1; r <= p; ++r)
      for (int j = p; j >= r; --j) {
        const double den = tau_[k + 1 + j - r] - tau_[k - p + j];
        const double a = den > 0.0 ? (t - tau_[k - p + j]) / den : 0.0;
        d[j] = d[j - 1] * (1.0 - a) + d[j] * a;  // affine in the control points; weights are knot-only
      }
    return d[p];
  }
  // 2-point Gauss-Legendre on [a,b] -- exact for the per-segment cubic.
  Scalar gauss2(double a, double b) const {
    if (b <= a) return Scalar(0.0);
    const double h = 0.5 * (b - a), c = 0.5 * (a + b), g = 0.5773502691896257 * h;  // g = h/sqrt(3)
    return (deboor(c - g) + deboor(c + g)) * h;
  }
  // Control points of the derivative spline: c has m points on the degree-p knot vector tau_ + shift (length
  // m+p+1); returns m−1 points on the degree-(p−1) vector tau_ + shift + 1.
  std::vector<Scalar> deriv_cp(const std::vector<Scalar>& c, int p, int shift) const {
    const int m = static_cast<int>(c.size());
    std::vector<Scalar> q(static_cast<std::size_t>(m - 1));
    for (int i = 0; i < m - 1; ++i) {
      const double den = tau_[shift + i + p + 1] - tau_[shift + i + 1];
      if (den > 0.0) q[i] = (c[i + 1] - c[i]) * (p / den);
      else q[i] = c[i] * 0.0;
    }
    return q;
  }
  // de Boor for a degree-p spline with control points c (m points) on the knot vector tau (m+p+1 entries).
  Scalar deboor_general(const std::vector<Scalar>& c, const double* tau, int p, double t) const {
    const int m = static_cast<int>(c.size());
    int k;
    if (t >= tau[m]) k = m - 1;
    else {
      int lo = p, hi = m;
      while (hi - lo > 1) { const int mid = (lo + hi) / 2; (t < tau[mid] ? hi : lo) = mid; }
      k = lo;
    }
    Scalar d[4];
    for (int j = 0; j <= p; ++j) d[j] = c[k - p + j];
    for (int r = 1; r <= p; ++r)
      for (int j = p; j >= r; --j) {
        const double den = tau[k + 1 + j - r] - tau[k - p + j];
        const double a = den > 0.0 ? (t - tau[k - p + j]) / den : 0.0;
        d[j] = d[j - 1] * (1.0 - a) + d[j] * a;
      }
    return d[p];
  }

  std::vector<double> s_, tau_, brk_;
  std::vector<Scalar> cp_, Ibrk_;
  double t0_ = 0.0;    // clamped B-spline domain start (== first knot when leading, in.time when following)
  double pre_t_ = 0.0;  // start of the flat pre-segment (in.time); == t0_ (zero-length) for a following region
  Scalar I0_{0.0}, region_int_{0.0};
};

// ---- Tension-spline shape functions (research note §1,§3,§6) --------------------------------------
// A spline under tension has f in span{1, t, sinh(σt), cosh(σt)} per interval. Assembling it needs a
// handful of scalar functions of (σ, h) and (σ, h, v) built from sinh/cosh. All are pure DOUBLE
// (structure-only: σ is a fixed hyperparameter, h/v are times) -- the AAD-carrying knot values enter
// only linearly, multiplying these coefficients (see class Tension). Every function is written to be
// numerically stable across the whole σh range:
//   * σh small: naive hyperbolic forms are 0/0 with catastrophic cancellation. Each function is
//     rewritten so the leading linear part cancels ANALYTICALLY, leaving a stable "minus-linear"
//     helper (sinhm1 = sinh(x)-x, coshm2 = cosh(x)-1-x²/2, xcoshm = x·cosh(x)-sinh(x)) evaluated by
//     Taylor SERIES for |x| < 0.5. In the σ→0 limit these reproduce the natural-cubic coefficients
//     EXACTLY (leading term), so Tension → NaturalCubic continuously.
//   * σh large (> 20): sinh/cosh overflow; use scaled-exponential asymptotics (coth→1, csch→0). The
//     interval is effectively taut there anyway.
// None of this touches the differentiated hot loop -- it runs once in Tension::build().
namespace tension_detail {

// sinh(x) - x = x³/6 + x⁵/120 + x⁷/5040 + ...   (stable for small x; direct subtraction is fine large)
inline double sinhm1(double x) {
  const double ax = std::abs(x);
  if (ax < 0.5) {
    const double x2 = x * x;
    // Horner in x² for x³/6 + x⁵/120 + x⁷/5040 + x⁹/362880 + x¹¹/39916800
    double s = 1.0 / 39916800.0;
    s = s * x2 + 1.0 / 362880.0;
    s = s * x2 + 1.0 / 5040.0;
    s = s * x2 + 1.0 / 120.0;
    s = s * x2 + 1.0 / 6.0;
    return s * x2 * x;  // x³·(...)
  }
  return std::sinh(x) - x;
}

// cosh(x) - 1 - x²/2 = x⁴/24 + x⁶/720 + x⁸/40320 + ...   (stable for small x)
inline double coshm2(double x) {
  const double ax = std::abs(x);
  if (ax < 0.5) {
    const double x2 = x * x;
    // Terms to x¹⁴/14!: the truncation error at the 0.5 branch point is then x¹⁶/16! ≈ 3e-16 RELATIVE to
    // x⁴/24 (the series used to stop at x¹²/12! with a 1/13! coefficient and no x¹⁰ term: 1e-7 relative).
    double s = 1.0 / 87178291200.0;  // x¹⁴/14!
    s = s * x2 + 1.0 / 479001600.0;  // x¹²/12!
    s = s * x2 + 1.0 / 3628800.0;    // x¹⁰/10!
    s = s * x2 + 1.0 / 40320.0;     // x⁸/8!
    s = s * x2 + 1.0 / 720.0;       // x⁶/6!
    s = s * x2 + 1.0 / 24.0;        // x⁴/4!
    return s * x2 * x2;             // x⁴·(...)
  }
  return std::cosh(x) - 1.0 - 0.5 * x * x;
}

// x·cosh(x) - sinh(x) = x³/3 + x⁵/30 + x⁷/840 + ...   (stable for small x)
inline double xcoshm(double x) {
  const double ax = std::abs(x);
  if (ax < 0.5) {
    const double x2 = x * x;
    // coefficient of x^(2k+1) is (2k)/(2k+1)! : 1/3, 1/30, 1/840, 1/45360, ...
    double s = 10.0 / 39916800.0;  // 10/11!
    s = s * x2 + 8.0 / 362880.0;   // 8/9!
    s = s * x2 + 6.0 / 5040.0;     // 6/7!
    s = s * x2 + 4.0 / 120.0;      // 4/5!
    s = s * x2 + 2.0 / 6.0;        // 2/3!
    return s * x2 * x;             // x³·(...)
  }
  return x * std::cosh(x) - std::sinh(x);
}

// p(σ,h) = [1/h - σ/sinh(σh)] / σ²  = sinhm1(σh) / (h·σ²·sinh(σh)).  σ→0: → h/6 (natural-cubic off-diag).
inline double p_coef(double sigma, double h) {
  const double x = sigma * h;
  if (x > 20.0) {  // sinh huge: 1/h - σ/sinh(σh) → 1/h;  p → 1/(h σ²) with an exp-small correction.
    const double e = std::exp(-2.0 * x);
    const double ratio = 2.0 * x * std::exp(-x) / (1.0 - e);  // = σh / sinh(σh)
    return (1.0 - ratio) / (h * sigma * sigma);
  }
  return sinhm1(x) / (h * sigma * sigma * std::sinh(x));
}

// q(σ,h) = [σ·coth(σh) - 1/h] / σ²  = xcoshm(σh) / (h·σ²·sinh(σh)).  σ→0: → h/3 (natural-cubic diag/side).
inline double q_coef(double sigma, double h) {
  const double x = sigma * h;
  if (x > 20.0) {  // coth → 1:  q → (σ - 1/h)/σ² = 1/σ - 1/(σ²h), plus exp-small.
    const double e = std::exp(-2.0 * x);
    const double coth = 1.0 + 2.0 * e / (1.0 - e);
    return (sigma * coth - 1.0 / h) / (sigma * sigma);
  }
  return xcoshm(x) / (h * sigma * sigma * std::sinh(x));
}

// Curvature shape function on [0,h]:  Φ(σ,h,v) = [sinh(σv)/sinh(σh) - v/h] / σ², with Φ(·,·,0)=Φ(·,·,h)=0.
// Stable "minus-linear" form: numerator h·sinh(σv) - v·sinh(σh) = h·sinhm1(σv) - v·sinhm1(σh) (the σv, σh
// terms cancel analytically). σ→0: Φ → (v³ - v h²)/(6h), the natural-cubic curvature shape.
inline double Phi(double sigma, double h, double v) {
  const double xh = sigma * h;
  if (xh > 20.0) {  // scaled-exponential: sinh(σv)/sinh(σh) = e^{-σ(h-v)}(1-e^{-2σv})/(1-e^{-2σh}).
    const double r = std::exp(-sigma * (h - v)) * (1.0 - std::exp(-2.0 * sigma * v)) /
                     (1.0 - std::exp(-2.0 * xh));
    return (r - v / h) / (sigma * sigma);
  }
  const double num = h * sinhm1(sigma * v) - v * sinhm1(xh);
  return num / (h * sigma * sigma * std::sinh(xh));
}

// ∫₀^v Φ(σ,h,w) dw = [(cosh(σv)-1)/(σ sinh(σh)) - v²/(2h)] / σ². Stable minus-quadratic form:
//   2h(cosh(σv)-1) - v²σ sinh(σh) = 2h·coshm2(σv) - v²·σ·sinhm1(σh)   (leading σ²v²h terms cancel).
// σ→0: Ψ → (v⁴ - 2v²h²)/(24h), the antiderivative of the natural-cubic curvature shape.
// d/dv Phi(sigma, h, v) = [sigma·cosh(sigma v)/sinh(sigma h) − 1/h]/sigma²   and   d²/dv² Phi = sinh(sigma v)/sinh(sigma h).
inline double Phi_d1(double sigma, double h, double v) {
  const double xh = sigma * h;
  if (xh > 20.0) {  // cosh(σv)/sinh(σh) = e^{-σ(h-v)}(1+e^{-2σv})/(1-e^{-2σh}) — no overflow
    const double r = std::exp(-sigma * (h - v)) * (1.0 + std::exp(-2.0 * sigma * v)) / (1.0 - std::exp(-2.0 * xh));
    return (sigma * r - 1.0 / h) / (sigma * sigma);
  }
  return (sigma * std::cosh(sigma * v) / std::sinh(xh) - 1.0 / h) / (sigma * sigma);
}
inline double Phi_d2(double sigma, double h, double v) {
  const double xh = sigma * h;
  if (xh > 20.0) return std::exp(-sigma * (h - v)) * (1.0 - std::exp(-2.0 * sigma * v)) / (1.0 - std::exp(-2.0 * xh));
  return std::sinh(sigma * v) / std::sinh(xh);
}

inline double Psi(double sigma, double h, double v) {
  const double xh = sigma * h;
  if (xh > 20.0) {  // (cosh(σv)-1)/(σ sinh(σh)) via scaled exponentials (all exponents ≤ 0, no overflow):
    // = [e^{σ(v-h)} + e^{-σ(v+h)} - 2e^{-σh}] / (σ(1-e^{-2σh})).
    const double term = std::exp(sigma * (v - h)) + std::exp(-sigma * (v + h)) - 2.0 * std::exp(-xh);
    const double first = term / (sigma * (1.0 - std::exp(-2.0 * xh)));
    return (first - v * v / (2.0 * h)) / (sigma * sigma);
  }
  const double num = 2.0 * h * coshm2(sigma * v) - v * v * sigma * sinhm1(xh);
  return num / (2.0 * h * sigma * sigma * sigma * std::sinh(xh));
}

}  // namespace tension_detail

// Spline under TENSION on the forward-at-knot values (research note §1,§3). Like NaturalCubic it is a
// C² interpolant with natural end curvature (f''=0 at both ends) and pins its leading value to the
// incoming boundary (C0 join). What differs is the basis: on each interval f ∈ span{1, t, sinh(σt),
// cosh(σt)} instead of a cubic, so a fixed hyperparameter σ ("tension") pulls the curve taut between
// knots -- σ→0 recovers the natural cubic exactly, σ→∞ approaches piecewise linear, killing the
// overshoot a plain cubic can produce (Hagan-West). Crucially, with σ FIXED the knot curvatures solve a
// tridiagonal system A(h,σ) z = B(h,σ) y whose matrices depend ONLY on the spacings h and σ, never on
// the values y, so z = A⁻¹B·y is a CONSTANT matrix times y and f (and ∫f) are LINEAR in the knot
// forwards. Therefore is_linear_map = true: the W-cache + analytic Jacobian ride it unchanged, and all
// the sinh/cosh + tridiagonal work happens ONCE here in build(), never in the differentiated hot loop
// (which stays DF = exp(-Wx)). This is what MonotoneCubic could not offer -- tension controls overshoot
// WITHOUT value-dependent branches, so it stays on the fast path.
template <class Scalar>
class Tension {
 public:
  explicit Tension(std::vector<double> knots, double sigma = 1.0)
      : s_(std::move(knots)), sigma_(sigma) {
    require_increasing_knots(s_, "Tension");
    if (!(sigma_ > 0.0) || !std::isfinite(sigma_))
      throw std::invalid_argument("Tension: sigma must be finite and > 0 (use NaturalCubic for σ=0)");
  }
  int n_values() const { return static_cast<int>(s_.size()); }
  double t_end() const { return s_.back(); }
  static constexpr bool is_linear_map = true;

  template <class Vec>
  void build(const Vec& x, int off, int n, const Boundary<Scalar>& in) {
    // LEADING: tension-spline the n actual knots, flat-extrapolate v1=x[off] for t<t1 (mirrors the far end);
    // no phantom (0,0). FOLLOWING: prepend the pinned (in.time,in.value) join point for C0 (unchanged).
    const bool lead = !in.has_predecessor;
    const int N = lead ? n : n + 1;
    if (N < 2) throw std::invalid_argument("Tension: needs >= 2 nodes (a LEADING region needs >= 2 knots)");  // PATCH
    xs_.resize(N);
    ys_.resize(N);
    if (lead) {
      for (int i = 0; i < n; ++i) { xs_[i] = s_[i]; ys_[i] = x[off + i]; }
    } else {
      xs_[0] = in.time;
      ys_[0] = in.value;  // C0 join
      for (int i = 0; i < n; ++i) { xs_[i + 1] = s_[i]; ys_[i + 1] = x[off + i]; }
    }
    const int nseg = N - 1;
    h_.resize(nseg);
    for (int i = 0; i < nseg; ++i) h_[i] = xs_[i + 1] - xs_[i];

    // Knot curvatures z (= f''(t_i)); natural BC z[0]=z[N-1]=0, interior from the tension tridiagonal.
    // Scaled by σ² out of both A and B so the system is the well-conditioned cubic-limit matrix:
    //   Ã_{i,i-1} = p(σ,h_{i-1}),  Ã_{i,i} = q(σ,h_{i-1}) + q(σ,h_i),  Ã_{i,i+1} = p(σ,h_i),
    //   B̃_i = (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1}   (linear in y).
    // p,q → natural-cubic h/6, h/3 as σ→0 (tension_detail), so z → the natural-cubic moments exactly.
    z_.assign(N, ys_[0] * 0.0);  // width-preserving zero (Scalar(0.0) is an EMPTY dual; see NaturalCubic)
    if (nseg >= 2) {
      const int k = nseg - 1;  // interior unknowns z[1..N-2]
      std::vector<double> lower(k), diag(k), upper(k);
      std::vector<Scalar> rhs(k);
      for (int i = 1; i <= k; ++i) {
        lower[i - 1] = tension_detail::p_coef(sigma_, h_[i - 1]);
        diag[i - 1] = tension_detail::q_coef(sigma_, h_[i - 1]) + tension_detail::q_coef(sigma_, h_[i]);
        upper[i - 1] = tension_detail::p_coef(sigma_, h_[i]);
        rhs[i - 1] = (ys_[i + 1] - ys_[i]) / h_[i] - (ys_[i] - ys_[i - 1]) / h_[i - 1];
      }
      // Thomas (double bands, Scalar rhs -> AAD flows through the values).
      for (int i = 1; i < k; ++i) {
        const double w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] = rhs[i] - w * rhs[i - 1];
      }
      std::vector<Scalar> zi(k);
      zi[k - 1] = rhs[k - 1] / diag[k - 1];
      for (int i = k - 1; i-- > 0;) zi[i] = (rhs[i] - upper[i] * zi[i + 1]) / diag[i];
      for (int i = 1; i <= k; ++i) z_[i] = zi[i - 1];
    }

    // Cumulative integral at each knot: full-segment ∫ = (y_i+y_{i+1})h/2 + (z_i+z_{i+1})·Ψ(σ,h,h).
    Is_.resize(N);
    Is_[0] = lead ? (in.integral + ys_[0] * (xs_[0] - in.time)) : in.integral;  // leading: flat pre-seg at v1
    for (int i = 0; i < nseg; ++i) {
      const double psih = tension_detail::Psi(sigma_, h_[i], h_[i]);
      Is_[i + 1] = Is_[i] + (ys_[i] + ys_[i + 1]) * (0.5 * h_[i]) + (z_[i] + z_[i + 1]) * psih;
    }
    // End slope for the C1 handoff: f'(t_end) = (y_N-y_{N-1})/h + z_{N-2}·p + z_{N-1}·q (z_{N-1}=0).
    const double hl = h_[nseg - 1];
    end_slope_ = (ys_[N - 1] - ys_[N - 2]) / hl + z_[N - 2] * tension_detail::p_coef(sigma_, hl) +
                 z_[N - 1] * tension_detail::q_coef(sigma_, hl);
  }

  Scalar forward(double t) const {
    if (t <= xs_.front()) return ys_.front();  // flat pre-segment (leading) / clamp start (following)
    if (t >= xs_.back()) return ys_.back();    // flat extrapolation
    const int i = seg(t);
    const double h = h_[i], u = t - xs_[i];
    // f = y_i(h-u)/h + y_{i+1}u/h + z_i·Φ(σ,h,h-u) + z_{i+1}·Φ(σ,h,u)   (linear in y and z).
    return ys_[i] * ((h - u) / h) + ys_[i + 1] * (u / h) +
           z_[i] * tension_detail::Phi(sigma_, h, h - u) + z_[i + 1] * tension_detail::Phi(sigma_, h, u);
  }
  // Exact derivatives of the tension piece: f = ys_i(h−u)/h + ys_{i+1}u/h + z_i Φ(h−u) + z_{i+1} Φ(u).
  Scalar forward_d1(double t) const {
    if (t <= xs_.front() || t >= xs_.back()) return ys_.front() * 0.0;
    const int i = seg(t);
    const double h = h_[i], u = t - xs_[i];
    return (ys_[i + 1] - ys_[i]) / h - z_[i] * tension_detail::Phi_d1(sigma_, h, h - u) +
           z_[i + 1] * tension_detail::Phi_d1(sigma_, h, u);
  }
  Scalar forward_d2(double t) const {
    if (t <= xs_.front() || t >= xs_.back()) return ys_.front() * 0.0;
    const int i = seg(t);
    const double h = h_[i], u = t - xs_[i];
    return z_[i] * tension_detail::Phi_d2(sigma_, h, h - u) + z_[i + 1] * tension_detail::Phi_d2(sigma_, h, u);
  }
  std::vector<double> pieces() const { return xs_; }
  double sigma() const { return sigma_; }
  Scalar integral(double t) const {
    if (t <= xs_.front()) return Is_.front() - ys_.front() * (xs_.front() - t);  // flat pre-segment at v1
    if (t >= xs_.back()) return Is_.back() + ys_.back() * (t - xs_.back());
    const int i = seg(t);
    const double h = h_[i], u = t - xs_[i];
    // ∫_{t_i}^{t} f = y_i(u - u²/2h) + y_{i+1}u²/2h + z_i[Ψ(h)-Ψ(h-u)] + z_{i+1}Ψ(u).
    const double lin_i = u - 0.5 * u * u / h, lin_ip1 = 0.5 * u * u / h;
    const double psi_h = tension_detail::Psi(sigma_, h, h);
    const double psi_hmu = tension_detail::Psi(sigma_, h, h - u);
    const double psi_u = tension_detail::Psi(sigma_, h, u);
    return Is_[i] + ys_[i] * lin_i + ys_[i + 1] * lin_ip1 + z_[i] * (psi_h - psi_hmu) + z_[i + 1] * psi_u;
  }
  Boundary<Scalar> out() const { return {xs_.back(), ys_.back(), end_slope_, Is_.back()}; }

 private:
  int seg(double t) const {
    auto it = std::upper_bound(xs_.begin(), xs_.end(), t);
    return static_cast<int>(it - xs_.begin()) - 1;
  }
  std::vector<double> s_, xs_, h_;
  double sigma_ = 1.0;
  std::vector<Scalar> ys_, z_, Is_;
  Scalar end_slope_{0.0};
};

}  // namespace swaps::curve
