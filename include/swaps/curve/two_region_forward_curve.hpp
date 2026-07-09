#pragma once
// Two-region instantaneous-forward curve (CLAUDE.md §2).
//
//   knots = union(CB meeting dates, par-swap maturity dates)
//
//   Front, t <= T (T = last meeting time):
//       f(t) = f_k   for t in (m_{k-1}, m_k],   m_0 := 0
//       piecewise-flat; jumps only at meeting dates.
//
//   Back, t > T:
//       f(t) = natural cubic spline through the knots
//                 (T, f_last), (s_1, g_1), ..., (s_n, g_n)
//       C1 and C2 continuous across the interior back knots; f'' = 0 at both ends.
//
//   Join constraint: the spline's value at T is pinned to the last front forward, so the
//   forward is LEVEL-continuous at the join. C1/C2 are deliberately NOT imposed across the
//   join — the front end is intentionally discontinuous.
//
// Free variables x = (f_1..f_Nf, g_1..g_Nb).  The spline's leading knot value is f_Nf, i.e. it
// is NOT a free variable.
//
// Differentiability: the natural-spline second derivatives solve a tridiagonal system whose
// right-hand side is linear in the knot values, so every coefficient is a linear map of x.
// All arithmetic is on the template Scalar, so substituting Eigen's AutoDiffScalar yields the
// exact d(curve)/dx with no code change (Phase 3).
//
// Both regions are integrated ANALYTICALLY (flat -> linear; cubic -> quartic), so
// discount(t) = exp(-integral(t)) is exact. No quadrature anywhere.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace swaps::curve {

template <class Scalar>
class TwoRegionForwardCurve {
 public:
  /// `meeting_times` and `back_times` are year fractions from the curve reference date,
  /// strictly increasing, with back_times.front() > meeting_times.back().
  /// Dates carry no derivative information, so these stay `double` even under AAD.
  TwoRegionForwardCurve(std::vector<double> meeting_times, std::vector<double> back_times)
      : m_(std::move(meeting_times)), s_(std::move(back_times)) {
    if (m_.empty() || s_.empty()) throw std::invalid_argument("curve: need >=1 knot per region");
    if (!std::is_sorted(m_.begin(), m_.end())) throw std::invalid_argument("meeting_times unsorted");
    if (!std::is_sorted(s_.begin(), s_.end())) throw std::invalid_argument("back_times unsorted");
    if (m_.front() <= 0.0) throw std::invalid_argument("meeting_times must be > 0");
    if (s_.front() <= m_.back()) throw std::invalid_argument("back_times must start after the join");

    // Spline abscissae: the join, then the back knots.
    xs_.reserve(s_.size() + 1);
    xs_.push_back(m_.back());
    xs_.insert(xs_.end(), s_.begin(), s_.end());
  }

  int n_front() const { return static_cast<int>(m_.size()); }
  int n_back() const { return static_cast<int>(s_.size()); }
  int n_knots() const { return n_front() + n_back(); }
  double join_time() const { return m_.back(); }
  double max_time() const { return s_.back(); }

  /// x = (f_1..f_Nf, g_1..g_Nb). Recomputes spline coefficients and cumulative integrals.
  /// Index-based access so `x` may be a std::vector, an Eigen vector, or an AutoDiffScalar vector.
  template <class Vec>
  void set_forwards(const Vec& x) {
    if (static_cast<int>(x.size()) != n_knots())
      throw std::invalid_argument("set_forwards: wrong size");
    const int nf = n_front();

    f_.resize(nf);
    for (int i = 0; i < nf; ++i) f_[i] = x[i];

    ys_.clear();
    ys_.reserve(s_.size() + 1);
    ys_.push_back(f_.back());  // <- the join constraint
    for (int i = nf; i < n_knots(); ++i) ys_.push_back(x[i]);

    build_front_integral();
    build_spline();
  }

  /// Instantaneous forward f(t). Exact (not a finite difference).
  Scalar forward(double t) const {
    if (t <= m_.back()) return f_[front_segment(t)];
    if (t >= xs_.back()) return ys_.back();  // flat extrapolation past the last knot
    const std::size_t i = back_segment(t);
    const double u = t - xs_[i];
    return a_[i] + u * (b_[i] + u * (c_[i] + u * d_[i]));
  }

  /// integral(t) = \int_0^t f(u) du, computed analytically.
  Scalar integral(double t) const {
    if (t <= 0.0) return Scalar(0.0);
    if (t <= m_.back()) {
      const std::size_t k = front_segment(t);
      const double prev = (k == 0) ? 0.0 : m_[k - 1];
      return If_[k] + f_[k] * (t - prev);
    }
    if (t >= xs_.back()) return Is_.back() + ys_.back() * (t - xs_.back());
    const std::size_t i = back_segment(t);
    const double u = t - xs_[i];
    // \int_0^u (a + b v + c v^2 + d v^3) dv
    return Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
  }

  Scalar discount(double t) const {
    using std::exp;
    return exp(-integral(t));
  }

  /// Continuously-compounded zero rate.
  Scalar zero(double t) const {
    if (t <= 0.0) return f_.front();
    return integral(t) / t;
  }

 private:
  // (m_{k-1}, m_k]  ->  index k. t==0 maps to segment 0.
  std::size_t front_segment(double t) const {
    auto it = std::lower_bound(m_.begin(), m_.end(), t);  // first m_k >= t
    if (it == m_.end()) return m_.size() - 1;
    return static_cast<std::size_t>(it - m_.begin());
  }

  // [xs_i, xs_{i+1}) -> i,  for xs_.front() <= t < xs_.back()
  std::size_t back_segment(double t) const {
    auto it = std::upper_bound(xs_.begin(), xs_.end(), t);  // first xs > t
    return static_cast<std::size_t>(it - xs_.begin()) - 1;
  }

  void build_front_integral() {
    If_.assign(m_.size(), Scalar(0.0));
    Scalar acc(0.0);
    for (std::size_t k = 0; k < m_.size(); ++k) {
      If_[k] = acc;  // integral up to the START of segment k
      const double prev = (k == 0) ? 0.0 : m_[k - 1];
      acc = acc + f_[k] * (m_[k] - prev);
    }
    join_integral_ = acc;  // \int_0^T f
  }

  // Natural cubic spline: M_0 = M_n = 0, C2 at interior knots.
  // Tridiagonal RHS is linear in ys_, so all coefficients are a linear map of the knot forwards.
  void build_spline() {
    const std::size_t n = xs_.size() - 1;  // number of spline segments
    std::vector<double> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = xs_[i + 1] - xs_[i];

    std::vector<Scalar> M(n + 1, Scalar(0.0));
    if (n >= 2) {
      // Thomas algorithm on the interior equations i = 1..n-1
      const std::size_t k = n - 1;
      std::vector<double> lower(k), diag(k), upper(k);
      std::vector<Scalar> rhs(k);
      for (std::size_t i = 1; i <= k; ++i) {
        lower[i - 1] = h[i - 1];
        diag[i - 1] = 2.0 * (h[i - 1] + h[i]);
        upper[i - 1] = h[i];
        rhs[i - 1] = 6.0 * ((ys_[i + 1] - ys_[i]) / h[i] - (ys_[i] - ys_[i - 1]) / h[i - 1]);
      }
      // forward sweep
      for (std::size_t i = 1; i < k; ++i) {
        const double w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] = rhs[i] - w * rhs[i - 1];
      }
      // back substitution
      std::vector<Scalar> mi(k);
      mi[k - 1] = rhs[k - 1] / diag[k - 1];
      for (std::size_t i = k - 1; i-- > 0;) mi[i] = (rhs[i] - upper[i] * mi[i + 1]) / diag[i];
      for (std::size_t i = 1; i <= k; ++i) M[i] = mi[i - 1];
    }

    a_.assign(n, Scalar(0.0));
    b_.assign(n, Scalar(0.0));
    c_.assign(n, Scalar(0.0));
    d_.assign(n, Scalar(0.0));
    for (std::size_t i = 0; i < n; ++i) {
      a_[i] = ys_[i];
      b_[i] = (ys_[i + 1] - ys_[i]) / h[i] - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0;
      c_[i] = M[i] / 2.0;
      d_[i] = (M[i + 1] - M[i]) / (6.0 * h[i]);
    }

    // Cumulative integral of the whole curve up to each spline abscissa.
    Is_.assign(xs_.size(), Scalar(0.0));
    Is_[0] = join_integral_;
    for (std::size_t i = 0; i < n; ++i) {
      const double u = h[i];
      Is_[i + 1] = Is_[i] + u * (a_[i] + u * (b_[i] / 2.0 + u * (c_[i] / 3.0 + u * d_[i] / 4.0)));
    }
  }

  std::vector<double> m_, s_, xs_;          // meeting times, back times, spline abscissae
  std::vector<Scalar> f_, ys_;              // front forwards; spline knot values (ys_[0] = f_.back())
  std::vector<Scalar> a_, b_, c_, d_;       // per-segment cubic coefficients
  std::vector<Scalar> If_, Is_;             // cumulative integrals (front starts, spline knots)
  Scalar join_integral_{0.0};               // \int_0^T f
};

}  // namespace swaps::curve
