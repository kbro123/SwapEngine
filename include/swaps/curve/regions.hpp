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
    t_.resize(n + 1);
    y_.resize(n + 1);
    t_[0] = in.time;
    y_[0] = in.value;
    for (int i = 0; i < n; ++i) {
      t_[i + 1] = s_[i];
      y_[i + 1] = x[off + i];
    }
    I_.resize(n + 1);
    I_[0] = in.integral;
    for (int i = 0; i < n; ++i) {
      const double h = t_[i + 1] - t_[i];
      I_[i + 1] = I_[i] + 0.5 * (y_[i] + y_[i + 1]) * h;  // trapezoid = exact for linear f
    }
  }

  Scalar forward(double t) const {
    if (t >= t_.back()) return y_.back();
    const int i = seg(t);
    const double w = (t - t_[i]) / (t_[i + 1] - t_[i]);
    return y_[i] + (y_[i + 1] - y_[i]) * w;
  }
  Scalar integral(double t) const {
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
    const int N = n + 1;  // spline points: [in.time, s_...]
    xs_.resize(N);
    ys_.resize(N);
    xs_[0] = in.time;
    ys_[0] = in.value;  // C0: leading value pinned to the incoming boundary
    for (int i = 0; i < n; ++i) {
      xs_[i + 1] = s_[i];
      ys_[i + 1] = x[off + i];
    }

    const int nseg = N - 1;
    std::vector<double> h(nseg);
    for (int i = 0; i < nseg; ++i) h[i] = xs_[i + 1] - xs_[i];

    std::vector<Scalar> M(N, Scalar(0.0));  // natural: M[0]=M[N-1]=0; Thomas on the interior
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
    Is_[0] = in.integral;
    for (int i = 0; i < nseg; ++i) {
      const double u = h[i];
      Is_[i + 1] = Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
    }
    const double hl = h[nseg - 1];
    end_slope_ = b_[nseg - 1] + hl * (2.0 * c_[nseg - 1] + 3.0 * d_[nseg - 1] * hl);
  }

  Scalar forward(double t) const {
    if (t >= xs_.back()) return ys_.back();  // flat extrapolation
    const int i = seg(t);
    const double u = t - xs_[i];
    return a_[i] + u * (b_[i] + u * (c_[i] + u * d_[i]));
  }
  Scalar integral(double t) const {
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
    const int N = n + 1;  // points: [in.time, s_...]
    xs_.resize(N);
    ys_.resize(N);
    xs_[0] = in.time;
    ys_[0] = in.value;
    for (int i = 0; i < n; ++i) {
      xs_[i + 1] = s_[i];
      ys_[i + 1] = x[off + i];
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
    Is_[0] = in.integral;
    for (int i = 0; i < nseg; ++i) {
      const double u = h[i];
      Is_[i + 1] = Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
    }
    end_slope_ = m[N - 1];
  }

  Scalar forward(double t) const {
    if (t >= xs_.back()) return ys_.back();
    const int i = seg(t);
    const double u = t - xs_[i];
    return a_[i] + u * (b_[i] + u * (c_[i] + u * d_[i]));
  }
  Scalar integral(double t) const {
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
    const int N = n + 1;  // spline nodes: [in.time, s_...]; node 0 pinned to the boundary value (C0 join)
    if (N < 2) throw std::invalid_argument("MonotoneCubic: needs >= 1 back knot");
    xs_.resize(N);
    ys_.resize(N);
    xs_[0] = in.time;
    ys_[0] = in.value;
    for (int i = 0; i < n; ++i) {
      xs_[i + 1] = s_[i];
      ys_[i + 1] = x[off + i];
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

    // Hyman monotonicity filter (bit-for-bit QuantLib): clamp each tangent so no segment overshoots.
    // The `!= ` comparisons + min/max/sign are exactly the value-dependent branches that break linearity.
    hyman_filter(h, S, m);

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
    Is_[0] = in.integral;
    for (int i = 0; i < nseg; ++i) {
      const double u = h[i];
      Is_[i + 1] = Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
    }
    end_slope_ = m[N - 1];
  }

  Scalar forward(double t) const {
    if (t >= xs_.back()) return ys_.back();
    const int i = seg(t);
    const double u = t - xs_[i];
    return a_[i] + u * (b_[i] + u * (c_[i] + u * d_[i]));
  }
  Scalar integral(double t) const {
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
  static void hyman_filter(const std::vector<double>& h, const std::vector<Scalar>& S,
                           std::vector<Scalar>& m) {
    using std::abs;
    const int N = static_cast<int>(m.size());
    for (int i = 0; i < N; ++i) {
      Scalar correction, pm, pu, pd, M;
      if (i == 0) {
        if (m[i] * S[0] > 0.0)
          correction = m[i] / abs(m[i]) * smin(abs(m[i]), abs(3.0 * S[0]));
        else
          correction = Scalar(0.0);
        if (correction != m[i]) m[i] = correction;
      } else if (i == N - 1) {
        if (m[i] * S[N - 2] > 0.0)
          correction = m[i] / abs(m[i]) * smin(abs(m[i]), abs(3.0 * S[N - 2]));
        else
          correction = Scalar(0.0);
        if (correction != m[i]) m[i] = correction;
      } else {
        pm = (S[i - 1] * h[i] + S[i] * h[i - 1]) / (h[i - 1] + h[i]);
        M = 3.0 * smin(smin(abs(S[i - 1]), abs(S[i])), abs(pm));
        if (i > 1) {
          if ((S[i - 1] - S[i - 2]) * (S[i] - S[i - 1]) > 0.0) {
            pd = (S[i - 1] * (2.0 * h[i - 1] + h[i - 2]) - S[i - 2] * h[i - 1]) / (h[i - 2] + h[i - 1]);
            if (pm * pd > 0.0 && pm * (S[i - 1] - S[i - 2]) > 0.0)
              M = smax(M, Scalar(1.5) * smin(abs(pm), abs(pd)));
          }
        }
        if (i < N - 2) {
          if ((S[i] - S[i - 1]) * (S[i + 1] - S[i]) > 0.0) {
            pu = (S[i] * (2.0 * h[i] + h[i + 1]) - S[i + 1] * h[i]) / (h[i] + h[i + 1]);
            if (pm * pu > 0.0 && -pm * (S[i] - S[i - 1]) > 0.0)
              M = smax(M, Scalar(1.5) * smin(abs(pm), abs(pu)));
          }
        }
        if (m[i] * pm > 0.0)
          correction = m[i] / abs(m[i]) * smin(abs(m[i]), M);
        else
          correction = Scalar(0.0);
        if (correction != m[i]) m[i] = correction;
      }
    }
  }

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
// Interior breakpoints are UNIFORM in time over the region (the standard, well-conditioned default; a
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
    t0_ = in.time;
    I0_ = in.integral;
    const double te = s_.back();
    const int m = n + 1;  // control points: P_0 (pinned) + n free
    // Clamped cubic knot vector: 4x t0, (n-3) uniform interior, 4x te  (length m+4 = n+5).
    tau_.assign(m + 4, te);
    for (int i = 0; i < 4; ++i) tau_[i] = t0_;
    for (int j = 1; j <= n - 3; ++j) tau_[3 + j] = t0_ + (te - t0_) * (static_cast<double>(j) / (n - 2));
    cp_.resize(m);
    cp_[0] = in.value;  // C0 pin
    for (int i = 0; i < n; ++i) cp_[i + 1] = x[off + i];
    // Distinct breakpoints (t0, interior knots, te) for exact segment-wise integration.
    brk_.clear();
    brk_.push_back(t0_);
    for (int j = 1; j <= n - 3; ++j) brk_.push_back(tau_[3 + j]);
    brk_.push_back(te);
    // Whole-region integral, for out() and flat extrapolation. Seed from the first segment so the
    // accumulator carries derivatives (AAD SAFETY, top of this header).
    region_int_ = gauss2(brk_[0], brk_[1]);
    for (std::size_t k = 1; k + 1 < brk_.size(); ++k) region_int_ += gauss2(brk_[k], brk_[k + 1]);
  }

  Scalar forward(double t) const {
    const double te = s_.back();
    if (t <= t0_) return cp_.front();
    if (t >= te) return deboor(te);  // flat extrapolation beyond the region
    return deboor(t);
  }
  Scalar integral(double t) const {
    if (t <= t0_) return I0_;
    const double te = s_.back();
    if (t >= te) return I0_ + region_int_ + deboor(te) * (t - te);
    Scalar acc = I0_;  // I0_ carries the front's derivatives
    for (std::size_t k = 0; k + 1 < brk_.size(); ++k) {
      const double lo = brk_[k], hi = brk_[k + 1];
      if (t <= lo) break;
      acc += gauss2(lo, t < hi ? t : hi);
      if (t <= hi) break;
    }
    return acc;
  }
  Boundary<Scalar> out() const { return {s_.back(), deboor(s_.back()), Scalar(0.0), I0_ + region_int_}; }

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

  std::vector<double> s_, tau_, brk_;
  std::vector<Scalar> cp_;
  double t0_ = 0.0;
  Scalar I0_{0.0}, region_int_{0.0};
};

}  // namespace swaps::curve
