#pragma once
#include <functional>
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
