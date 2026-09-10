#pragma once
// RESEARCH (moved out of include/ in E6.1, 2026-09-10): the reverse-mode tape behind research/gamma.hpp; no
// production consumer. Tested by tests/gamma_test.cpp and mutated by tools/mutate.py.
// Reverse-mode "tape" AAD scalar (CLAUDE.md §1 -- the ADDITIVE second-order companion to ad::Dual).
//
// The engine's forward-mode Dual (ad/dual.hpp) carries an M-vector of derivatives and yields one COLUMN
// of a Jacobian per output, in a single differentiated evaluation seeded over the M inputs. That is ideal
// for a WIDE-output / narrow-input Jacobian (the M-knot calibration Jacobian). A single SCALAR output with
// MANY inputs -- a book's NPV differentiated w.r.t. every knot forward -- is the mirror image, and the
// natural tool is REVERSE mode: evaluate once RECORDING a tape of elementary operations, then sweep the
// tape backwards ONCE to obtain d(output)/d(every input) -- the whole gradient for the cost of ~one primal.
//
// `Rev<T>` is a drop-in Scalar for the SAME templated kernels ad::Dual plugs into (float_leg_pv, annuity,
// MultiCurveBook::value, ModularCurve::discount): it satisfies the identical operator surface -- construct
// from double, +,-,*,/ (with Rev and with double, both sides), unary minus, compound assignment,
// comparisons, and the transcendentals the pricing/curve kernels call via ADL (exp; log/sqrt/abs are
// provided so the full region set -- incl. MonotoneCubic's Hyman filter -- instantiates). So a portfolio
// prices with Scalar = Rev<T> and, after ONE reverse sweep, yields the full curve-space gradient. Nothing
// existing is edited; this is a NEW scalar the existing generic code accepts unchanged.
//
// SECOND ORDER (curve-space gamma) -- why Rev is TEMPLATED on the value type T:
//   * T = double            -> a plain reverse gradient (the reverse analogue of a forward-AAD gradient).
//   * T = Tangent (a fwd    -> FORWARD-OVER-REVERSE: the value AND every recorded partial weight carry a
//     dual, below)             directional derivative, so ONE reverse sweep produces adjoints that are
//                              themselves Tangents -- adj[i].v = d(NPV)/dx_i and adj[i].d = d/dx_dir of
//                              that gradient = one Hessian-vector product (H·dir)_i. n seeds e_j give the
//                              full Hessian in n reverse sweeps (see research/gamma.hpp).
// Tangent is a SELF-CONTAINED 2-double forward dual (value + one directional derivative), deliberately NOT
// Eigen's AutoDiffScalar: a constant Tangent is {v, 0} -- it is never "empty", so the empty-derivative trap
// the cashflow kernel warns about (mixing a size-0 gradient with a size-M one) simply cannot arise on the
// second-order path. The reverse tape therefore never touches Eigen and never allocates a per-node vector.
//
// Header-only, thread-local tape. The active tape is a THREAD-LOCAL pointer per value-type T, so Rev's
// constructor matches ad::Dual's (value-from-double, no tape argument) and the generic kernels need no
// change; RevTapeScope sets/clears it around one differentiated evaluation.

#include <cmath>
#include <vector>

namespace swaps::ad {

// ---- Tangent: a minimal forward dual (value + ONE directional derivative) -----------------------
// Used as the value type T of Rev<T> for the second-order (forward-over-reverse) Hessian path. Every
// operation the pricing/curve kernels perform is defined here; a constant is {v, 0.0}, never empty.
struct Tangent {
  double v = 0.0;  // primal value
  double d = 0.0;  // directional derivative (d value / d seed direction)
  Tangent() = default;
  Tangent(double val) : v(val), d(0.0) {}          // constant: zero directional derivative (drop-in ctor)
  Tangent(double val, double dot) : v(val), d(dot) {}
};

inline Tangent operator+(const Tangent& a, const Tangent& b) { return {a.v + b.v, a.d + b.d}; }
inline Tangent operator-(const Tangent& a, const Tangent& b) { return {a.v - b.v, a.d - b.d}; }
inline Tangent operator*(const Tangent& a, const Tangent& b) { return {a.v * b.v, a.v * b.d + a.d * b.v}; }
inline Tangent operator/(const Tangent& a, const Tangent& b) {
  const double q = a.v / b.v;
  return {q, (a.d - q * b.d) / b.v};
}
inline Tangent operator+(const Tangent& a, double c) { return {a.v + c, a.d}; }
inline Tangent operator+(double c, const Tangent& a) { return {c + a.v, a.d}; }
inline Tangent operator-(const Tangent& a, double c) { return {a.v - c, a.d}; }
inline Tangent operator-(double c, const Tangent& a) { return {c - a.v, -a.d}; }
inline Tangent operator*(const Tangent& a, double c) { return {a.v * c, a.d * c}; }
inline Tangent operator*(double c, const Tangent& a) { return {c * a.v, c * a.d}; }
inline Tangent operator/(const Tangent& a, double c) { return {a.v / c, a.d / c}; }
inline Tangent operator/(double c, const Tangent& a) { return {c / a.v, -c * a.d / (a.v * a.v)}; }
inline Tangent operator-(const Tangent& a) { return {-a.v, -a.d}; }
inline Tangent& operator+=(Tangent& a, const Tangent& b) { a.v += b.v; a.d += b.d; return a; }
inline Tangent& operator-=(Tangent& a, const Tangent& b) { a.v -= b.v; a.d -= b.d; return a; }

inline Tangent exp(const Tangent& a) { const double e = std::exp(a.v); return {e, e * a.d}; }
inline Tangent log(const Tangent& a) { return {std::log(a.v), a.d / a.v}; }
inline Tangent sqrt(const Tangent& a) { const double s = std::sqrt(a.v); return {s, 0.5 * a.d / s}; }
inline Tangent abs(const Tangent& a) { return a.v >= 0.0 ? a : Tangent{-a.v, -a.d}; }

inline bool operator<(const Tangent& a, const Tangent& b) { return a.v < b.v; }
inline bool operator>(const Tangent& a, const Tangent& b) { return a.v > b.v; }
inline bool operator<=(const Tangent& a, const Tangent& b) { return a.v <= b.v; }
inline bool operator>=(const Tangent& a, const Tangent& b) { return a.v >= b.v; }
inline bool operator==(const Tangent& a, const Tangent& b) { return a.v == b.v; }
inline bool operator!=(const Tangent& a, const Tangent& b) { return a.v != b.v; }

// Scalar (primal) value extractor: the value that drives a branch selection (comparisons/abs sign).
inline double to_double(double x) { return x; }
inline double to_double(const Tangent& t) { return t.v; }

// ---- The tape: a flat list of elementary-operation nodes ----------------------------------------
// Each node records its (up to two) parents and the LOCAL partial derivatives w0 = d(node)/d(parent0),
// w1 = d(node)/d(parent1). Weights are of type T so the second-order (Tangent) path differentiates them.
// A parent index < 0 means "constant / no parent" and is skipped in the sweep. Nodes are appended in
// evaluation order, so a node's parents always precede it -> a single reverse pass respects dependencies.
template <class T>
struct RevNode {
  int p0;
  int p1;
  T w0;
  T w1;
};

template <class T>
struct RevTape {
  std::vector<RevNode<T>> nodes;
  void clear() { nodes.clear(); }
  std::size_t size() const { return nodes.size(); }
};

// Active tape: one THREAD-LOCAL pointer per value type T. A templated inline function owns the storage, so
// there is exactly one thread-local object per T across all translation units (templates are inline).
template <class T>
inline RevTape<T>*& rev_active_tape() {
  static thread_local RevTape<T>* tape = nullptr;
  return tape;
}

// Record an elementary op with the given parents/partials. Returns the new node index, or -1 when BOTH
// parents are constant (then the result is itself a constant and needs no tape node -- this keeps the tape
// lean: `x - 1.0`, weight rows and other constant sub-expressions never allocate a node).
template <class T>
inline int rev_record(int p0, const T& w0, int p1, const T& w1) {
  if (p0 < 0 && p1 < 0) return -1;
  RevTape<T>* tape = rev_active_tape<T>();
  tape->nodes.push_back(RevNode<T>{p0, p1, w0, w1});
  return static_cast<int>(tape->nodes.size()) - 1;
}

// ---- Rev<T>: the taped scalar --------------------------------------------------------------------
template <class T>
class Rev {
 public:
  T v{};          // primal value (a T so the Hessian path carries a directional derivative through it)
  int idx = -1;   // tape node index; -1 == a constant not recorded on the tape

  Rev() = default;
  Rev(double c) : v(T(c)), idx(-1) {}              // constant-from-double: the SAME ctor shape ad::Dual has
  Rev(const T& val, int node) : v(val), idx(node) {}

  Rev& operator+=(const Rev& b) { *this = *this + b; return *this; }
  Rev& operator-=(const Rev& b) { *this = *this - b; return *this; }
  Rev& operator*=(const Rev& b) { *this = *this * b; return *this; }
  Rev& operator/=(const Rev& b) { *this = *this / b; return *this; }
  Rev& operator+=(double c) { *this = *this + c; return *this; }
  Rev& operator-=(double c) { *this = *this - c; return *this; }
  Rev& operator*=(double c) { *this = *this * c; return *this; }
  Rev& operator/=(double c) { *this = *this / c; return *this; }
};

// Create an INDEPENDENT variable (a leaf) on the active tape with the given seed value. Its node has no
// parents; its adjoint after the reverse sweep is d(output)/d(this variable).
template <class T>
inline Rev<T> rev_leaf(const T& value) {
  RevTape<T>* tape = rev_active_tape<T>();
  tape->nodes.push_back(RevNode<T>{-1, -1, T(0.0), T(0.0)});
  return Rev<T>(value, static_cast<int>(tape->nodes.size()) - 1);
}

// -- arithmetic: value math in T, partials recorded as T --
template <class T>
inline Rev<T> operator+(const Rev<T>& a, const Rev<T>& b) {
  return Rev<T>(a.v + b.v, rev_record<T>(a.idx, T(1.0), b.idx, T(1.0)));
}
template <class T>
inline Rev<T> operator-(const Rev<T>& a, const Rev<T>& b) {
  return Rev<T>(a.v - b.v, rev_record<T>(a.idx, T(1.0), b.idx, T(-1.0)));
}
template <class T>
inline Rev<T> operator*(const Rev<T>& a, const Rev<T>& b) {
  return Rev<T>(a.v * b.v, rev_record<T>(a.idx, b.v, b.idx, a.v));
}
template <class T>
inline Rev<T> operator/(const Rev<T>& a, const Rev<T>& b) {
  const T inv = T(1.0) / b.v;
  const T val = a.v * inv;
  return Rev<T>(val, rev_record<T>(a.idx, inv, b.idx, -(val)*inv));  // d/da=1/b, d/db=-a/b^2
}
template <class T>
inline Rev<T> operator+(const Rev<T>& a, double c) {
  return Rev<T>(a.v + c, rev_record<T>(a.idx, T(1.0), -1, T(0.0)));
}
template <class T>
inline Rev<T> operator+(double c, const Rev<T>& a) { return a + c; }
template <class T>
inline Rev<T> operator-(const Rev<T>& a, double c) {
  return Rev<T>(a.v - c, rev_record<T>(a.idx, T(1.0), -1, T(0.0)));
}
template <class T>
inline Rev<T> operator-(double c, const Rev<T>& a) {
  return Rev<T>(c - a.v, rev_record<T>(a.idx, T(-1.0), -1, T(0.0)));
}
template <class T>
inline Rev<T> operator*(const Rev<T>& a, double c) {
  return Rev<T>(a.v * c, rev_record<T>(a.idx, T(c), -1, T(0.0)));
}
template <class T>
inline Rev<T> operator*(double c, const Rev<T>& a) { return a * c; }
template <class T>
inline Rev<T> operator/(const Rev<T>& a, double c) {
  return Rev<T>(a.v / c, rev_record<T>(a.idx, T(1.0 / c), -1, T(0.0)));
}
template <class T>
inline Rev<T> operator/(double c, const Rev<T>& a) {
  const T val = T(c) / a.v;
  return Rev<T>(val, rev_record<T>(a.idx, T(-c) / (a.v * a.v), -1, T(0.0)));
}
template <class T>
inline Rev<T> operator-(const Rev<T>& a) {
  return Rev<T>(-a.v, rev_record<T>(a.idx, T(-1.0), -1, T(0.0)));
}

// -- transcendentals (ADL-found: the curve does `using std::exp; exp(-integral(t))`) --
template <class T>
inline Rev<T> exp(const Rev<T>& a) {
  using std::exp;
  const T e = exp(a.v);
  return Rev<T>(e, rev_record<T>(a.idx, e, -1, T(0.0)));  // d exp/da = exp
}
template <class T>
inline Rev<T> log(const Rev<T>& a) {
  using std::log;
  return Rev<T>(log(a.v), rev_record<T>(a.idx, T(1.0) / a.v, -1, T(0.0)));
}
template <class T>
inline Rev<T> sqrt(const Rev<T>& a) {
  using std::sqrt;
  const T s = sqrt(a.v);
  return Rev<T>(s, rev_record<T>(a.idx, T(0.5) / s, -1, T(0.0)));
}
template <class T>
inline Rev<T> abs(const Rev<T>& a) {
  const double sign = to_double(a.v) >= 0.0 ? 1.0 : -1.0;
  return Rev<T>(sign >= 0.0 ? a.v : -a.v, rev_record<T>(a.idx, T(sign), -1, T(0.0)));
}

// -- comparisons: on the PRIMAL value (branch selection; the chosen branch's adjoint rides along, exactly
//    like AutoDiffScalar comparisons compare .value()) --
#define SWAPS_REV_CMP(OP)                                                                          \
  template <class T> inline bool operator OP(const Rev<T>& a, const Rev<T>& b) {                    \
    return to_double(a.v) OP to_double(b.v);                                                        \
  }                                                                                                 \
  template <class T> inline bool operator OP(const Rev<T>& a, double b) {                           \
    return to_double(a.v) OP b;                                                                     \
  }                                                                                                 \
  template <class T> inline bool operator OP(double a, const Rev<T>& b) {                           \
    return a OP to_double(b.v);                                                                     \
  }
SWAPS_REV_CMP(<)
SWAPS_REV_CMP(>)
SWAPS_REV_CMP(<=)
SWAPS_REV_CMP(>=)
SWAPS_REV_CMP(==)
SWAPS_REV_CMP(!=)
#undef SWAPS_REV_CMP

template <class T>
inline double to_double(const Rev<T>& r) { return to_double(r.v); }

// ---- The reverse sweep --------------------------------------------------------------------------
// Given a finished tape and the node index of a scalar output, accumulate adjoints back to front and
// return the adjoints of the first `n_leaves` nodes (== the seeded inputs, created first after clear()).
// For T = double this is the gradient; for T = Tangent each returned adjoint carries {gradient_i, (H·dir)_i}.
template <class T>
inline std::vector<T> rev_sweep(const RevTape<T>& tape, int out_idx, int n_leaves) {
  if (out_idx < 0) return std::vector<T>(n_leaves, T(0.0));  // output is a constant -> zero gradient
  std::vector<T> adj(tape.nodes.size(), T(0.0));
  adj[out_idx] = T(1.0);
  for (int i = static_cast<int>(tape.nodes.size()) - 1; i >= 0; --i) {
    const T a = adj[i];
    const RevNode<T>& nd = tape.nodes[i];
    if (nd.p0 >= 0) adj[nd.p0] = adj[nd.p0] + a * nd.w0;
    if (nd.p1 >= 0) adj[nd.p1] = adj[nd.p1] + a * nd.w1;
  }
  std::vector<T> g(n_leaves);
  for (int i = 0; i < n_leaves; ++i) g[i] = adj[i];
  return g;
}

// RAII: make `tape` the active tape for T and clear it; restore the previous active tape on scope exit.
template <class T>
struct RevTapeScope {
  RevTape<T>* prev;
  explicit RevTapeScope(RevTape<T>& tape) : prev(rev_active_tape<T>()) {
    rev_active_tape<T>() = &tape;
    tape.clear();
  }
  ~RevTapeScope() { rev_active_tape<T>() = prev; }
  RevTapeScope(const RevTapeScope&) = delete;
  RevTapeScope& operator=(const RevTapeScope&) = delete;
};

}  // namespace swaps::ad
