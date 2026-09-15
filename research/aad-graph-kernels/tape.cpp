// AAD-JIT spike: record the ENGINE'S OWN templated pricing (instrument_model_quote<Rec>) once per structure into a flat op
// list (constant folding + CSE at record time, value-dependent comparisons recorded as GUARDS), then replay value + Jacobian
// allocation-free. Compared against the engine's hot path (HybridBundleResidual) on the same fixture, in the same process.
// Research probe only -- not engine code.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <dlfcn.h>

#include <Eigen/Core>

namespace aj {
enum Op : uint8_t { CONST, INPUT, ADD, SUB, MUL, DIV, NEG, EXP, LOG, SQRT };
enum Cmp : uint8_t { LT, LE, GT, GE, EQ, NE };
struct Node { Op op; int a, b; double v; };
struct Guard { Cmp c; int a, b; bool expect; };

struct Recorder {
  std::vector<Node> nodes;
  std::vector<Guard> guards;
  std::vector<int> inputs;  // node id of input i
  std::unordered_map<uint64_t, int> konst;
  struct K { uint64_t h; bool operator==(const K& o) const { return h == o.h; } };
  std::unordered_map<std::string, int> cse;  // (op,a,b) -> node  (string key: probe code, not hot)
  long cse_hits = 0, folded = 0, recorded = 0;
  int constant(double v) {
    uint64_t bits; std::memcpy(&bits, &v, 8);
    auto it = konst.find(bits);
    if (it != konst.end()) return it->second;
    nodes.push_back({CONST, -1, -1, v});
    return konst[bits] = static_cast<int>(nodes.size()) - 1;
  }
  int emit(Op op, int a, int b, double v) {
    ++recorded;
    char key[16]; std::memcpy(key, &op, 1); std::memcpy(key + 1, &a, 4); std::memcpy(key + 5, &b, 4);
    std::string k(key, 9);
    auto it = cse.find(k);
    if (it != cse.end()) { ++cse_hits; return it->second; }
    nodes.push_back({op, a, b, v});
    return cse[k] = static_cast<int>(nodes.size()) - 1;
  }
};
inline thread_local Recorder* R = nullptr;

struct Rec {
  double v = 0.0;
  int id = -1;  // -1: constant (folded at record time)
  Rec() = default;
  Rec(double d) : v(d) {}
  Rec(int i) : v(i) {}
  Rec(double d, int i) : v(d), id(i) {}
  int slot() const { return id >= 0 ? id : R->constant(v); }
  Rec& operator+=(const Rec& o);
  Rec& operator-=(const Rec& o);
  Rec& operator*=(const Rec& o);
  Rec& operator/=(const Rec& o);
};
inline Rec bin(Op op, const Rec& x, const Rec& y, double v) {
  if (x.id < 0 && y.id < 0) { ++R->folded; return Rec(v); }
  return Rec(v, R->emit(op, x.slot(), y.slot(), v));
}
inline Rec operator+(const Rec& x, const Rec& y) { return bin(ADD, x, y, x.v + y.v); }
inline Rec operator-(const Rec& x, const Rec& y) { return bin(SUB, x, y, x.v - y.v); }
inline Rec operator*(const Rec& x, const Rec& y) { return bin(MUL, x, y, x.v * y.v); }
inline Rec operator/(const Rec& x, const Rec& y) { return bin(DIV, x, y, x.v / y.v); }
inline Rec operator-(const Rec& x) { return x.id < 0 ? Rec(-x.v) : Rec(-x.v, R->emit(NEG, x.id, -1, -x.v)); }
inline Rec exp(const Rec& x) { double v = std::exp(x.v); return x.id < 0 ? Rec(v) : Rec(v, R->emit(EXP, x.id, -1, v)); }
inline Rec log(const Rec& x) { double v = std::log(x.v); return x.id < 0 ? Rec(v) : Rec(v, R->emit(LOG, x.id, -1, v)); }
inline Rec sqrt(const Rec& x) { double v = std::sqrt(x.v); return x.id < 0 ? Rec(v) : Rec(v, R->emit(SQRT, x.id, -1, v)); }
inline bool cmp(Cmp c, const Rec& x, const Rec& y, bool res) {
  if (x.id >= 0 || y.id >= 0) R->guards.push_back({c, x.slot(), y.slot(), res});  // value-dependent branch: GUARD
  return res;
}
inline bool operator<(const Rec& x, const Rec& y) { return cmp(LT, x, y, x.v < y.v); }
inline bool operator<=(const Rec& x, const Rec& y) { return cmp(LE, x, y, x.v <= y.v); }
inline bool operator>(const Rec& x, const Rec& y) { return cmp(GT, x, y, x.v > y.v); }
inline bool operator>=(const Rec& x, const Rec& y) { return cmp(GE, x, y, x.v >= y.v); }
inline bool operator==(const Rec& x, const Rec& y) { return cmp(EQ, x, y, x.v == y.v); }
inline bool operator!=(const Rec& x, const Rec& y) { return cmp(NE, x, y, x.v != y.v); }
// |x| = guard(x >= 0) then x or -x: bit-identical value, and the branch becomes a recorded guard (affine under it).
inline Rec abs(const Rec& x) { if (x.id < 0) return Rec(std::abs(x.v)); return cmp(GE, x, Rec(0.0), x.v >= 0.0) ? x : -x; }
inline Rec& Rec::operator+=(const Rec& o) { return *this = *this + o; }
inline Rec& Rec::operator-=(const Rec& o) { return *this = *this - o; }
inline Rec& Rec::operator*=(const Rec& o) { return *this = *this * o; }
inline Rec& Rec::operator/=(const Rec& o) { return *this = *this / o; }
}  // namespace aj

namespace Eigen {
template <> struct NumTraits<aj::Rec> : NumTraits<double> {
  using Real = aj::Rec; using NonInteger = aj::Rec; using Nested = aj::Rec; using Literal = aj::Rec;
  enum { IsComplex = 0, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 1, AddCost = 3, MulCost = 3 };
};
}  // namespace Eigen

#include "shape_ladder.hpp"
#include "malloc_count.hpp"
#include "swaps/calibration/hybrid_residual.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
using clk = std::chrono::steady_clock;

namespace aj {
struct POp { uint8_t op; int32_t a, b; };

// The compiled kernel: [consts | inputs | ops], ops topologically ordered, dead nodes eliminated.
struct Kernel {
  int nconst = 0, nin = 0, nops = 0, base = 0;
  std::vector<POp> ops;
  std::vector<double> val, adj;
  std::vector<int> out;                 // slot of output j
  std::vector<Guard> guards;            // renumbered
  std::vector<std::vector<int>> cone;   // per output: op indices in its cone, DESCENDING
  std::vector<double> dot;              // forward-vector tangents, nin per op slot
  long cone_total = 0;

  void build(const Recorder& rec, const std::vector<int>& outs) {
    const int N = static_cast<int>(rec.nodes.size());
    std::vector<char> live(N, 0);
    std::vector<int> stack;
    auto mark = [&](int s) { if (s >= 0 && !live[s]) { live[s] = 1; stack.push_back(s); } };
    for (int o : outs) mark(o);
    for (const auto& g : rec.guards) { mark(g.a); mark(g.b); }
    while (!stack.empty()) { int s = stack.back(); stack.pop_back(); mark(rec.nodes[s].a); mark(rec.nodes[s].b); }
    for (int i : rec.inputs) live[i] = 1;
    std::vector<int> map(N, -1);
    for (int s = 0; s < N; ++s) if (live[s] && rec.nodes[s].op == CONST) map[s] = nconst++;
    nin = static_cast<int>(rec.inputs.size());
    for (int i = 0; i < nin; ++i) map[rec.inputs[i]] = nconst + i;
    base = nconst + nin;
    for (int s = 0; s < N; ++s) {
      const Node& nd = rec.nodes[s];
      if (!live[s] || nd.op == CONST || nd.op == INPUT) continue;
      map[s] = base + nops++;
      ops.push_back({static_cast<uint8_t>(nd.op), map[nd.a], nd.b >= 0 ? map[nd.b] : 0});
    }
    val.assign(base + nops, 0.0);
    for (int s = 0; s < N; ++s) if (live[s] && rec.nodes[s].op == CONST) val[map[s]] = rec.nodes[s].v;
    adj.assign(base + nops, 0.0);
    for (int o : outs) out.push_back(map[o]);
    for (auto g : rec.guards) { g.a = map[g.a]; g.b = map[g.b]; guards.push_back(g); }
    // per-output cones (for reverse sweeps restricted to what the row reads)
    std::vector<int> stamp(base + nops, -1);
    cone.resize(out.size());
    for (std::size_t j = 0; j < out.size(); ++j) {
      std::vector<int> st{out[j]};
      while (!st.empty()) {
        int s = st.back(); st.pop_back();
        if (s < base || stamp[s] == (int)j) continue;
        stamp[s] = static_cast<int>(j);
        cone[j].push_back(s - base);
        const POp& p = ops[s - base];
        st.push_back(p.a);
        if (p.op <= DIV) st.push_back(p.b);
      }
      std::sort(cone[j].begin(), cone[j].end(), std::greater<int>());
      cone_total += static_cast<long>(cone[j].size());
    }
    dot.assign(static_cast<std::size_t>(nops) * nin, 0.0);
  }

  // forward values; returns the number of guards whose outcome differs from the recording (0 = replay valid)
  int forward(const double* x) {
    double* __restrict v = val.data();
    std::memcpy(v + nconst, x, sizeof(double) * nin);
    const POp* __restrict p = ops.data();
    const int b0 = base;
    for (int k = 0; k < nops; ++k) {
      const double A = v[p[k].a], B = v[p[k].b];
      double r;
      switch (p[k].op) {
        case ADD: r = A + B; break;
        case SUB: r = A - B; break;
        case MUL: r = A * B; break;
        case DIV: r = A / B; break;
        case NEG: r = -A; break;
        case EXP: r = std::exp(A); break;
        case LOG: r = std::log(A); break;
        default: r = std::sqrt(A); break;
      }
      v[b0 + k] = r;
    }
    int flips = 0;
    for (const auto& g : guards) {
      const double A = v[g.a], B = v[g.b];
      bool res;
      switch (g.c) { case LT: res = A < B; break; case LE: res = A <= B; break; case GT: res = A > B; break;
                     case GE: res = A >= B; break; case EQ: res = A == B; break; default: res = A != B; }
      flips += (res != g.expect);
    }
    return flips;
  }
  // Jacobian by one reverse sweep per output over its cone (after forward()). J is m x nin, col-major.
  void jacobian_reverse(double* J) {
    const double* __restrict v = val.data();
    double* __restrict ad = adj.data();
    const POp* __restrict p = ops.data();
    const int m = static_cast<int>(out.size());
    for (int j = 0; j < m; ++j) {
      ad[out[j]] = 1.0;
      for (int k : cone[j]) {
        const int s = base + k;
        const double g = ad[s];
        ad[s] = 0.0;
        const int a = p[k].a, b = p[k].b;
        switch (p[k].op) {
          case ADD: ad[a] += g; ad[b] += g; break;
          case SUB: ad[a] += g; ad[b] -= g; break;
          case MUL: ad[a] += g * v[b]; ad[b] += g * v[a]; break;
          case DIV: ad[a] += g / v[b]; ad[b] -= g * v[s] / v[b]; break;
          case NEG: ad[a] -= g; break;
          case EXP: ad[a] += g * v[s]; break;
          case LOG: ad[a] += g / v[a]; break;
          default: ad[a] += g * 0.5 / v[s]; break;
        }
      }
      if (out[j] < nconst) ad[out[j]] = 0.0;  // a constant output (an input output is read + zeroed below)
      for (int i = 0; i < nin; ++i) { J[j + i * m] = ad[nconst + i]; ad[nconst + i] = 0.0; }
    }
  }
  // Jacobian by ONE forward-vector (tangent width nin) pass over the whole tape (after forward()).
  void jacobian_forward_vector(double* J) {
    const double* __restrict v = val.data();
    const POp* __restrict p = ops.data();
    const int n = nin, m = static_cast<int>(out.size());
    double* __restrict D = dot.data();
    auto tan = [&](int slot) -> const double* { return slot >= base ? D + static_cast<std::size_t>(slot - base) * n : nullptr; };
    static thread_local std::vector<double> zero, unit;
    if ((int)zero.size() != n) { zero.assign(n, 0.0); unit.assign(static_cast<std::size_t>(n) * n, 0.0); for (int i = 0; i < n; ++i) unit[i * n + i] = 1.0; }
    auto T = [&](int slot) -> const double* {
      if (slot >= base) return tan(slot);
      if (slot >= nconst) return unit.data() + static_cast<std::size_t>(slot - nconst) * n;
      return zero.data();
    };
    for (int k = 0; k < nops; ++k) {
      double* __restrict o = D + static_cast<std::size_t>(k) * n;
      const int a = p[k].a, b = p[k].b, s = base + k;
      const double* __restrict ta = T(a);
      switch (p[k].op) {
        case ADD: { const double* __restrict tb = T(b); for (int i = 0; i < n; ++i) o[i] = ta[i] + tb[i]; break; }
        case SUB: { const double* __restrict tb = T(b); for (int i = 0; i < n; ++i) o[i] = ta[i] - tb[i]; break; }
        case MUL: { const double* __restrict tb = T(b); const double va = v[a], vb = v[b]; for (int i = 0; i < n; ++i) o[i] = ta[i] * vb + tb[i] * va; break; }
        case DIV: { const double* __restrict tb = T(b); const double ib = 1.0 / v[b], q = v[s]; for (int i = 0; i < n; ++i) o[i] = (ta[i] - q * tb[i]) * ib; break; }
        case NEG: for (int i = 0; i < n; ++i) o[i] = -ta[i]; break;
        case EXP: { const double e = v[s]; for (int i = 0; i < n; ++i) o[i] = ta[i] * e; break; }
        case LOG: { const double ia = 1.0 / v[a]; for (int i = 0; i < n; ++i) o[i] = ta[i] * ia; break; }
        default: { const double c = 0.5 / v[s]; for (int i = 0; i < n; ++i) o[i] = ta[i] * c; break; }
      }
    }
    for (int j = 0; j < m; ++j) { const double* t = T(out[j]); for (int i = 0; i < n; ++i) J[j + i * m] = t[i]; }
  }

  // Emit the forward-value kernel as straight-line C++ (constants inlined as hex floats) -- the codegen option.
  void emit_cpp(const std::string& path) const {
    std::ofstream f(path);
    f << "#include <cmath>\nextern \"C\" int kern(const double* x, double* v) {\n";
    auto S = [&](int s) -> std::string {
      if (s < nconst) { char buf[40]; std::snprintf(buf, 40, "(%a)", val[s]); return buf; }
      if (s < base) return "x[" + std::to_string(s - nconst) + "]";
      return "v[" + std::to_string(s - base) + "]";
    };
    for (int k = 0; k < nops; ++k) {
      const auto& p = ops[k];
      f << " v[" << k << "]=";
      switch (p.op) {
        case ADD: f << S(p.a) << "+" << S(p.b); break;
        case SUB: f << S(p.a) << "-" << S(p.b); break;
        case MUL: f << S(p.a) << "*" << S(p.b); break;
        case DIV: f << S(p.a) << "/" << S(p.b); break;
        case NEG: f << "-" << S(p.a); break;
        case EXP: f << "std::exp(" << S(p.a) << ")"; break;
        case LOG: f << "std::log(" << S(p.a) << ")"; break;
        default: f << "std::sqrt(" << S(p.a) << ")"; break;
      }
      f << ";\n";
    }
    f << " int fl=0;\n";
    static const char* cs[] = {"<", "<=", ">", ">=", "==", "!="};
    for (const auto& g : guards) f << " fl+=((" << S(g.a) << cs[g.c] << S(g.b) << ")!=" << (g.expect ? 1 : 0) << ");\n";
    f << " return fl;\n}\n";
  }
};

// AFFINE COLLAPSE: under fixed guards every node built from inputs by +,-,neg and x*const, x/const is affine in the inputs.
// Collapse every affine node that feeds a nonlinear op (or an output) into one sparse row of a matrix W: exactly the
// engine's hand-built W-cache, discovered from the tape. Returns the count of rows, nnz and remaining nonlinear ops.
struct AffineStats { int rows = 0; long nnz = 0; int nonlinear = 0; int affine_nodes = 0; };
inline AffineStats affine_analysis(const Kernel& K) {
  const int n = K.nin, L = K.nops;
  std::vector<char> isaff(L, 0);
  std::vector<std::vector<double>> row(L);  // dense coefficient rows (probe only)
  auto aff_of = [&](int s, std::vector<double>& r, double& c) -> bool {
    r.assign(n, 0.0); c = 0.0;
    if (s < K.nconst) { c = K.val[s]; return true; }
    if (s < K.base) { r[s - K.nconst] = 1.0; return true; }
    if (!isaff[s - K.base]) return false;
    r = row[s - K.base]; return true;
  };
  AffineStats st;
  std::vector<double> ra, rb; double ca, cb;
  for (int k = 0; k < L; ++k) {
    const auto& p = K.ops[k];
    bool a_is_const = p.a < K.nconst, b_is_const = p.b < K.nconst;
    bool ok = false;
    switch (p.op) {
      case ADD: case SUB: ok = aff_of(p.a, ra, ca) && aff_of(p.b, rb, cb); if (ok) { row[k].resize(n); for (int i = 0; i < n; ++i) row[k][i] = p.op == ADD ? ra[i] + rb[i] : ra[i] - rb[i]; } break;
      case NEG: ok = aff_of(p.a, ra, ca); if (ok) { row[k].resize(n); for (int i = 0; i < n; ++i) row[k][i] = -ra[i]; } break;
      case MUL: if (a_is_const || b_is_const) { ok = aff_of(a_is_const ? p.b : p.a, ra, ca); double c = K.val[a_is_const ? p.a : p.b]; if (ok) { row[k].resize(n); for (int i = 0; i < n; ++i) row[k][i] = ra[i] * c; } } break;
      case DIV: if (b_is_const) { ok = aff_of(p.a, ra, ca); if (ok) { row[k].resize(n); for (int i = 0; i < n; ++i) row[k][i] = ra[i] / K.val[p.b]; } } break;
      default: break;
    }
    isaff[k] = ok;
    st.affine_nodes += ok;
    if (!ok) ++st.nonlinear;
  }
  // frontier: affine nodes read by a nonlinear op or an output
  std::vector<char> front(L, 0);
  for (int k = 0; k < L; ++k) {
    if (isaff[k]) continue;
    const auto& p = K.ops[k];
    if (p.a >= K.base && isaff[p.a - K.base]) front[p.a - K.base] = 1;
    if (p.op <= DIV && p.b >= K.base && isaff[p.b - K.base]) front[p.b - K.base] = 1;
  }
  for (int o : K.out) if (o >= K.base && isaff[o - K.base]) front[o - K.base] = 1;
  for (int k = 0; k < L; ++k) if (front[k]) { ++st.rows; for (double w : row[k]) st.nnz += (w != 0.0); }
  return st;
}
// The COLLAPSED kernel: slots [consts | inputs | affine frontier rows (CSR over inputs) | nonlinear ops]. Forward = one sparse
// GEMV + the nonlinear ops; Jacobian = reverse over each output's NONLINEAR cone, then scatter row adjoints through W (the
// engine's -(G diag(DF)) W product, derived instead of hand-written). Rounding-level parity with the interpreted tape.
struct Collapsed {
  int nconst = 0, nin = 0, nrows = 0, nnl = 0, brow = 0, bnl = 0;
  std::vector<double> val, adj, c0, w;
  std::vector<int> rp, ci, out;
  std::vector<POp> ops;
  std::vector<Guard> guards;
  std::vector<std::vector<int>> cone, crow;
  int forward(const double* x) {
    double* __restrict v = val.data();
    std::memcpy(v + nconst, x, sizeof(double) * nin);
    const int* __restrict RP = rp.data();
    const int* __restrict CI = ci.data();
    const double* __restrict W = w.data();
    const double* __restrict C0 = c0.data();
    const double* __restrict xx = x;
    for (int r = 0; r < nrows; ++r) {
      double acc = C0[r];
      for (int k = RP[r]; k < RP[r + 1]; ++k) acc += W[k] * xx[CI[k]];
      v[brow + r] = acc;
    }
    const POp* __restrict p = ops.data();
    const int b0 = bnl;
    for (int k = 0; k < nnl; ++k) {
      const double A = v[p[k].a], B = v[p[k].b];
      double r;
      switch (p[k].op) {
        case ADD: r = A + B; break;
        case SUB: r = A - B; break;
        case MUL: r = A * B; break;
        case DIV: r = A / B; break;
        case NEG: r = -A; break;
        case EXP: r = std::exp(A); break;
        case LOG: r = std::log(A); break;
        default: r = std::sqrt(A); break;
      }
      v[b0 + k] = r;
    }
    int flips = 0;
    for (const auto& g : guards) {
      const double A = v[g.a], B = v[g.b];
      bool res;
      switch (g.c) { case LT: res = A < B; break; case LE: res = A <= B; break; case GT: res = A > B; break;
                     case GE: res = A >= B; break; case EQ: res = A == B; break; default: res = A != B; }
      flips += (res != g.expect);
    }
    return flips;
  }
  void jacobian_reverse(double* J) {
    const int m = static_cast<int>(out.size());
    std::fill(J, J + static_cast<std::size_t>(m) * nin, 0.0);
    const double* __restrict v = val.data();
    double* __restrict ad = adj.data();
    const POp* __restrict p = ops.data();
    const int* __restrict RP = rp.data();
    const int* __restrict CI = ci.data();
    const double* __restrict W = w.data();
    for (int j = 0; j < m; ++j) {
      ad[out[j]] += 1.0;
      for (int k : cone[j]) {
        const int s = bnl + k;
        const double g = ad[s];
        ad[s] = 0.0;
        const int a = p[k].a, b = p[k].b;
        switch (p[k].op) {
          case ADD: ad[a] += g; ad[b] += g; break;
          case SUB: ad[a] += g; ad[b] -= g; break;
          case MUL: ad[a] += g * v[b]; ad[b] += g * v[a]; break;
          case DIV: ad[a] += g / v[b]; ad[b] -= g * v[s] / v[b]; break;
          case NEG: ad[a] -= g; break;
          case EXP: ad[a] += g * v[s]; break;
          case LOG: ad[a] += g / v[a]; break;
          default: ad[a] += g * 0.5 / v[s]; break;
        }
      }
      for (int r : crow[j]) {
        const double g = ad[brow + r];
        ad[brow + r] = 0.0;
        for (int k = RP[r]; k < RP[r + 1]; ++k) J[j + CI[k] * m] += g * W[k];
      }
      for (int i = 0; i < nin; ++i) { J[j + i * m] += ad[nconst + i]; ad[nconst + i] = 0.0; }
      if (out[j] < nconst) ad[out[j]] = 0.0;
    }
  }
};

inline Collapsed collapse(const Kernel& K) {
  const int n = K.nin, L = K.nops;
  std::vector<char> isaff(L, 0);
  std::vector<std::vector<double>> row(L);
  std::vector<double> cst(L, 0.0);
  std::vector<double> ra, rb;
  double ca = 0, cb = 0;
  auto fetch = [&](int s, std::vector<double>& r, double& c) -> bool {
    if (s < K.nconst) { r.assign(n, 0.0); c = K.val[s]; return true; }
    if (s < K.base) { r.assign(n, 0.0); r[s - K.nconst] = 1.0; c = 0.0; return true; }
    const int k = s - K.base;
    if (!isaff[k]) return false;
    r = row[k]; c = cst[k]; return true;
  };
  for (int k = 0; k < L; ++k) {
    const auto& p = K.ops[k];
    const bool ac = p.a < K.nconst, bc = p.b < K.nconst;
    bool ok = false;
    switch (p.op) {
      case ADD: case SUB:
        ok = fetch(p.a, ra, ca) && fetch(p.b, rb, cb);
        if (ok) { row[k].resize(n); for (int i = 0; i < n; ++i) row[k][i] = p.op == ADD ? ra[i] + rb[i] : ra[i] - rb[i]; cst[k] = p.op == ADD ? ca + cb : ca - cb; }
        break;
      case NEG:
        ok = fetch(p.a, ra, ca);
        if (ok) { row[k].resize(n); for (int i = 0; i < n; ++i) row[k][i] = -ra[i]; cst[k] = -ca; }
        break;
      case MUL:
        if (ac != bc) {
          ok = fetch(ac ? p.b : p.a, ra, ca);
          const double cv = K.val[ac ? p.a : p.b];
          if (ok) { row[k].resize(n); for (int i = 0; i < n; ++i) row[k][i] = ra[i] * cv; cst[k] = ca * cv; }
        }
        break;
      case DIV:
        if (bc && !ac) {
          ok = fetch(p.a, ra, ca);
          const double cv = K.val[p.b];
          if (ok) { row[k].resize(n); for (int i = 0; i < n; ++i) row[k][i] = ra[i] / cv; cst[k] = ca / cv; }
        }
        break;
      default: break;
    }
    isaff[k] = ok;
  }
  std::vector<char> front(L, 0);
  auto mark = [&](int s) { if (s >= K.base && isaff[s - K.base]) front[s - K.base] = 1; };
  for (int k = 0; k < L; ++k) {
    if (isaff[k]) continue;
    mark(K.ops[k].a);
    if (K.ops[k].op <= DIV) mark(K.ops[k].b);
  }
  for (int o : K.out) mark(o);
  for (const auto& g : K.guards) { mark(g.a); mark(g.b); }
  Collapsed C;
  C.nconst = K.nconst; C.nin = n; C.brow = K.base;
  std::vector<int> ns(K.base + L, -1);
  for (int s = 0; s < K.base; ++s) ns[s] = s;
  C.rp.push_back(0);
  for (int k = 0; k < L; ++k) {
    if (!front[k]) continue;
    ns[K.base + k] = C.brow + C.nrows++;
    for (int i = 0; i < n; ++i) if (row[k][i] != 0.0) { C.ci.push_back(i); C.w.push_back(row[k][i]); }
    C.rp.push_back(static_cast<int>(C.ci.size()));
    C.c0.push_back(cst[k]);
  }
  C.bnl = C.brow + C.nrows;
  for (int k = 0; k < L; ++k) if (!isaff[k]) ns[K.base + k] = C.bnl + C.nnl++;
  for (int k = 0; k < L; ++k) {
    if (isaff[k]) continue;
    POp p = K.ops[k];
    p.a = ns[p.a];
    p.b = p.op <= DIV ? ns[p.b] : 0;
    if (p.a < 0 || p.b < 0) { std::fprintf(stderr, "collapse: dangling operand\n"); std::abort(); }
    C.ops.push_back(p);
  }
  C.val.assign(C.bnl + C.nnl, 0.0);
  for (int s = 0; s < K.nconst; ++s) C.val[s] = K.val[s];
  C.adj.assign(C.val.size(), 0.0);
  for (int o : K.out) C.out.push_back(ns[o]);
  for (auto g : K.guards) { g.a = ns[g.a]; g.b = ns[g.b]; C.guards.push_back(g); }
  C.cone.resize(C.out.size()); C.crow.resize(C.out.size());
  std::vector<int> stamp(C.val.size(), -1);
  for (std::size_t j = 0; j < C.out.size(); ++j) {
    std::vector<int> st{C.out[j]};
    while (!st.empty()) {
      const int s = st.back(); st.pop_back();
      if (s < C.brow || stamp[s] == (int)j) continue;
      stamp[s] = static_cast<int>(j);
      if (s < C.bnl) { C.crow[j].push_back(s - C.brow); continue; }
      C.cone[j].push_back(s - C.bnl);
      st.push_back(C.ops[s - C.bnl].a);
      if (C.ops[s - C.bnl].op <= DIV) st.push_back(C.ops[s - C.bnl].b);
    }
    std::sort(C.cone[j].begin(), C.cone[j].end(), std::greater<int>());
  }
  return C;
}
}  // namespace aj

struct Timing { double med; };
template <class F> double time_us(F&& f, int reps) {
  std::vector<double> t;
  for (int b = 0; b < 7; ++b) {
    auto t0 = clk::now();
    for (int r = 0; r < reps; ++r) f();
    t.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps);
  }
  std::sort(t.begin(), t.end());
  return t[3];
}

int main(int argc, char** argv) {
  const bool do_codegen = argc > 1 && std::string(argv[1]) == "codegen";
  std::printf("%-14s %5s %6s %7s %7s %6s %6s | %8s %8s %8s | %8s %8s %8s %8s | %9s %9s | %5s %5s %5s | %s\n", "shape", "m", "rec_ms",
              "rec_ops", "live", "cse%", "guard", "eng_res", "tmpl_res", "tape_res", "eng_J", "tapeJrev", "tapeJfwd", "cone", "maxdq_bit",
              "maxdJ", "fl01", "fl25", "flBig", "affine(rows/nnz/nonlin)");
  for (const auto& s : swaps::shapes::ladder()) {
    if (s.name != "ois_nolag" && s.name != "ibor_multicurve" && s.name != "averaged" && s.name != "mixed_scheme" &&
        s.name != "fx_xccy" && s.name != "desk" && s.name != "desk_mixed" && s.name != "all_schemes" && s.name != "turns")
      continue;
    const auto& p = s.prob;
    const int n = p.n_knots(), m = p.n_residuals();
    // ---- record once ----
    aj::Recorder rec;
    aj::R = &rec;
    auto t0 = clk::now();
    std::vector<aj::Rec> X(n);
    for (int i = 0; i < n; ++i) {
      rec.nodes.push_back({aj::INPUT, -1, -1, s.x_true[i]});
      rec.inputs.push_back(static_cast<int>(rec.nodes.size()) - 1);
      X[i] = aj::Rec(s.x_true[i], rec.inputs.back());
    }
    const auto C = cal::build_bundle_curves<aj::Rec>(p.curves, [&](int c, int i) { return X[p.offset(c) + i]; });
    const auto of = [&](int i) -> const px::CurveHandle<aj::Rec>& { return *C[i]; };
    std::vector<int> outs;
    for (const auto& ins : p.instruments) outs.push_back(cal::instrument_model_quote<aj::Rec>(ins, of).slot());
    aj::Kernel K;
    K.build(rec, outs);
    const double rec_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    aj::R = nullptr;

    // ---- parity: tape replay vs the engine's templated double kernel (same ops => expect bit-identical) and vs hot path ----
    cal::HybridBundleResidual h(p);
    auto templ_quotes = [&](const Eigen::VectorXd& x) {
      const auto Cd = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
      const auto ofd = [&](int i) -> const px::CurveHandle<double>& { return *Cd[i]; };
      Eigen::VectorXd q(m);
      for (int r = 0; r < m; ++r) q[r] = cal::instrument_model_quote<double>(p.instruments[r], ofd);
      return q;
    };
    Eigen::VectorXd x1 = s.x_true;
    for (int i = 0; i < n; ++i) x1[i] += 1e-5 * std::sin(1.3 * i + 0.2);  // an x the tape was NOT recorded at
    int fl01 = K.forward(x1.data());
    const Eigen::VectorXd qt = templ_quotes(x1);
    double maxdq_bit = 0;
    for (int r = 0; r < m; ++r) maxdq_bit = std::max(maxdq_bit, std::abs(K.val[K.out[r]] - qt[r]));
    // Jacobian parity vs engine jacobian (unbanded, non-FX rows only: engine J is dr/dx)
    Eigen::MatrixXd Jt(m, n), Je, Jf(m, n);
    K.jacobian_reverse(Jt.data());
    K.forward(x1.data());
    K.jacobian_forward_vector(Jf.data());
    h.jacobian_vs_into(x1, s.q0, Je);
    const Eigen::VectorXd mr = h.model_rates(x1);
    double maxdJ = 0;
    for (int r = 0; r < m; ++r) {
      const auto& ins = p.instruments[r];
      double scale = 1.0;
      if (ins.quote == cal::QuoteKind::FxForward) scale = 1.0 / (mr[r] * ins.fx_time);
      else if (ins.band_upper > ins.band_lower) scale = cal::band_slope(mr[r], ins.band_lower, ins.band_upper, ins.band_decay);
      for (int i = 0; i < n; ++i) {
        maxdJ = std::max(maxdJ, std::abs(Jt(r, i) * scale - Je(r, i)));
        maxdJ = std::max(maxdJ, std::abs(Jf(r, i) * scale - Je(r, i)));
      }
    }
    // guard flips at bigger moves
    Eigen::VectorXd x25 = s.x_true, xb = s.x_true;
    for (int i = 0; i < n; ++i) { x25[i] += 25e-4 * (0.9 + 0.2 * i / n); xb[i] += 25e-4 * std::sin(0.7 * i + 0.3); }
    int fl25 = K.forward(x25.data());
    int flb = K.forward(xb.data());

    // ---- timings (same process, same load) ----
    Eigen::VectorXd xa = x1, xb2 = s.x_true;
    bool flip = false;
    int reps_small = std::max(20, static_cast<int>(20000.0 / (K.nops + 1)));
    double eng_res = time_us([&] { flip = !flip; volatile double d = h.residuals_vs(flip ? xa : xb2, s.q0)[0]; (void)d; }, reps_small * 5);
    double tmpl_res = time_us([&] { flip = !flip; volatile double d = templ_quotes(flip ? xa : xb2)[0]; (void)d; }, std::max(5, reps_small));
    double tape_res = time_us([&] { flip = !flip; volatile int f = K.forward((flip ? xa : xb2).data()); (void)f; }, reps_small * 5);
    Eigen::MatrixXd JJ;
    double eng_J = time_us([&] { flip = !flip; h.jacobian_vs_into(flip ? xa : xb2, s.q0, JJ); }, std::max(3, reps_small / 5));
    double tJr = time_us([&] { flip = !flip; K.forward((flip ? xa : xb2).data()); K.jacobian_reverse(Jt.data()); }, std::max(3, reps_small / 5));
    double tJf = time_us([&] { flip = !flip; K.forward((flip ? xa : xb2).data()); K.jacobian_forward_vector(Jf.data()); }, std::max(3, reps_small / 5));
    const auto aff = aj::affine_analysis(K);
    std::printf("%-14s %5d %6.1f %7ld %7d %5.1f%% %6zu | %8.2f %8.2f %8.2f | %8.1f %8.1f %8.1f %8ld | %9.1e %9.1e | %5d %5d %5d | %d/%ld/%d\n",
                s.name.c_str(), m, rec_ms, rec.recorded, K.nops, 100.0 * rec.cse_hits / std::max(1L, rec.recorded), K.guards.size(), eng_res,
                tmpl_res, tape_res, eng_J, tJr, tJf, K.cone_total, maxdq_bit, maxdJ, fl01, fl25, flb, aff.rows, aff.nnz, aff.nonlinear);
    std::fflush(stdout);
    {
      auto tc0 = clk::now();
      auto KC = aj::collapse(K);
      const double coll_ms = std::chrono::duration<double, std::milli>(clk::now() - tc0).count();
      Eigen::MatrixXd Jw(m, n);
      unsigned long allocs_interp = 0, allocs_coll = 0;
      {
        swaps::testing::AllocScope a;
        for (int r = 0; r < 50; ++r) { K.forward(x1.data()); K.jacobian_reverse(Jw.data()); }
        allocs_interp = a.allocs();
      }
      {
        swaps::testing::AllocScope a;
        for (int r = 0; r < 50; ++r) { KC.forward(x1.data()); KC.jacobian_reverse(Jw.data()); }
        allocs_coll = a.allocs();
      }
      std::printf("  record %.1f ms + collapse %.1f ms (a guard flip re-records); allocs over 50 fwd+J replays: interp %lu, collapsed %lu\n",
                  rec_ms, coll_ms, allocs_interp, allocs_coll);
      K.forward(x1.data());
      const int flc = KC.forward(x1.data());
      double mdq = 0;
      for (int r = 0; r < m; ++r) mdq = std::max(mdq, std::abs(KC.val[KC.out[r]] - K.val[K.out[r]]));
      Eigen::MatrixXd Jc(m, n);
      KC.jacobian_reverse(Jc.data());
      K.jacobian_reverse(Jt.data());
      const double mdJ = (Jc - Jt).cwiseAbs().maxCoeff();
      double c_res = time_us([&] { flip = !flip; volatile int f = KC.forward((flip ? xa : xb2).data()); (void)f; }, reps_small * 5);
      double c_J = time_us([&] { flip = !flip; KC.forward((flip ? xa : xb2).data()); KC.jacobian_reverse(Jc.data()); }, std::max(3, reps_small / 5));
      std::printf("  collapsed %-14s rows %4d nnz %6zu nonlinear %6d | res %7.2f us (interp %7.2f, engine %7.2f) | fwd+J %7.1f us (engine J %7.1f) | maxdq %.1e maxdJ %.1e flips %d\n",
                  s.name.c_str(), KC.nrows, KC.w.size(), KC.nnl, c_res, tape_res, eng_res, c_J, eng_J, mdq, mdJ, flc);
      std::fflush(stdout);
    }

    if (do_codegen && (s.name == "mixed_scheme" || s.name == "desk_mixed")) {
      const std::string dir = "/private/tmp/claude-501/-Users-kevinbroughton-Desktop-Claude-Projects-SwapEngineWeb/646b1604-5e27-4aa9-ac4d-00179134a0f6/scratchpad/aadjit/";
      const std::string src = dir + "gen_" + s.name + ".cpp", so = dir + "gen_" + s.name + ".so";
      K.emit_cpp(src);
      for (const char* opt : {"-O1", "-O2"}) {
        auto c0 = clk::now();
        const std::string cmd = std::string("clang++ -std=c++17 ") + opt + " -march=x86-64-v3 -fno-math-errno -shared -fPIC -isysroot $(xcrun --show-sdk-path) '" + src + "' -o '" + so + "'";
        int rc = std::system(cmd.c_str());
        const double cms = std::chrono::duration<double, std::milli>(clk::now() - c0).count();
        void* hd = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (rc != 0 || !hd) { std::printf("  codegen %s failed rc=%d\n", opt, rc); continue; }
        using KF = int (*)(const double*, double*);
        KF kf = reinterpret_cast<KF>(dlsym(hd, "kern"));
        std::vector<double> vv(K.nops);
        int fl = kf(x1.data(), vv.data());
        K.forward(x1.data());
        double md = 0;
        for (int k = 0; k < K.nops; ++k) md = std::max(md, std::abs(vv[k] - K.val[K.base + k]));
        double tcg = time_us([&] { flip = !flip; volatile int f = kf((flip ? xa : xb2).data(), vv.data()); (void)f; }, reps_small * 5);
        std::printf("  codegen %-3s %-12s compile %.0f ms  eval %.3f us  (interp %.3f us)  flips %d  max|d| vs interp %.1e\n", opt,
                    s.name.c_str(), cms, tcg, tape_res, fl, md);
        dlclose(hd);
      }
    }
  }
}
