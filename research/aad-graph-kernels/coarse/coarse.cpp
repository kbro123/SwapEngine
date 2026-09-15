// COARSE-GRAINED lowering of the recorded pricing graph (research probe, no JIT).
// The recorded scalar op list (tape.cpp's recorder, unchanged) is lowered to a handful of PRE-COMPILED kernels that run
// over SoA index/weight arrays -- "the problem as data, fixed engine code":
//   ROWS  : affine rows over the knots, dense per column-span block (W-cache generalised; fast mode only)
//   SUM   : left-fold gather-and-signed-sum rows (legs, annuities, Σ sub-periods) -- bit-identical to the scalar fold
//   ADD/SUB/MUL/DIV/NEG/EXP/LOG/SQRT/INV : one tight loop per (level, op type), batched across instruments
// Dispatch = one switch per GROUP, not per scalar op.
// Jacobians: (A) owner-partitioned reverse sweep through the same groups, ending in the rows' W analytically
// (J of exp(row) is DF·w), (B) per-row cone reverse on the scalar collapsed tape (tape.cpp), (C) forward-vector.
#include "rec.hpp"
#include "shape_ladder.hpp"
#include "malloc_count.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "kernel.hpp"
#include "util.hpp"

#include <map>
#include <set>

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
using clk = std::chrono::steady_clock;

namespace co {
constexpr uint8_t INV = 20, SUMR = 21;
inline bool binary(uint8_t t) { return t >= aj::ADD && t <= aj::DIV; }

// ---- a common source form: [consts | inputs | rows | ops] ----
struct Src {
  int nconst = 0, nin = 0, nrow = 0, base = 0;
  std::vector<double> cval;
  std::vector<int> rp{0}, ci;
  std::vector<double> w, c0;
  std::vector<aj::POp> ops;
  std::vector<int> out;
  std::vector<aj::Guard> guards;
};
inline Src from_kernel(const aj::Kernel& K) {
  Src s;
  s.nconst = K.nconst; s.nin = K.nin; s.nrow = 0; s.base = K.base;
  s.cval.assign(K.val.begin(), K.val.begin() + K.nconst);
  s.ops = K.ops; s.out = K.out; s.guards = K.guards;
  return s;
}
inline Src from_collapsed(const aj::Collapsed& C) {
  Src s;
  s.nconst = C.nconst; s.nin = C.nin; s.nrow = C.nrows; s.base = C.bnl;
  s.cval.assign(C.val.begin(), C.val.begin() + C.nconst);
  s.rp = C.rp; s.ci = C.ci; s.w = C.w; s.c0 = C.c0;
  s.ops = C.ops; s.out = C.out; s.guards = C.guards;
  return s;
}
// a/b -> a * inv(b), inv shared per b (the engine's INV trick; 1-ulp, rounding-level)
inline Src share_reciprocals(const Src& s) {
  Src r = s;
  r.ops.clear();
  std::vector<int> map(s.ops.size(), -1);
  std::map<int, int> inv;
  auto ns = [&](int old) { return old < s.base ? old : map[old - s.base]; };
  for (std::size_t k = 0; k < s.ops.size(); ++k) {
    const auto& p = s.ops[k];
    const int a = ns(p.a), b = binary(p.op) ? ns(p.b) : 0;
    if (p.op == aj::DIV) {
      int is;
      auto it = inv.find(b);
      if (it == inv.end()) { r.ops.push_back({INV, b, 0}); is = r.base + static_cast<int>(r.ops.size()) - 1; inv[b] = is; }
      else is = it->second;
      r.ops.push_back({aj::MUL, a, is});
    } else {
      r.ops.push_back({p.op, a, b});
    }
    map[k] = r.base + static_cast<int>(r.ops.size()) - 1;
  }
  for (auto& o : r.out) o = ns(o);
  for (auto& g : r.guards) { g.a = ns(g.a); g.b = ns(g.b); }
  return r;
}

struct Plan {
  int nconst = 0, nin = 0, nrow = 0, rbase = 0, opbase = 0, nval = 0;
  std::vector<double> V;
  struct Blk { int row0, nr, lo, span; Eigen::MatrixXd B; std::vector<double> c0; };
  std::vector<Blk> blks;
  std::vector<int> rrp{0}, rci;
  std::vector<double> rw;
  struct Group { uint8_t t; int base, n, off; };
  std::vector<Group> groups;
  std::vector<int> A, B, TS;
  std::vector<double> TW;
  std::vector<int> out;
  std::vector<aj::Guard> guards;
  struct NodeRec { uint8_t t; int slot, a, b, level; };
  std::vector<NodeRec> nodes;
  long elements = 0;
  long dispatch() const { return static_cast<long>(blks.size() + groups.size()) + (guards.empty() ? 0 : 1); }

  int forward(const double* x) {
    double* const vb = V.data();
    std::memcpy(vb + nconst, x, sizeof(double) * nin);
    for (auto& b : blks) {
      double* o = vb + b.row0;
      if (b.span > 0) {
        Eigen::Map<Eigen::VectorXd> ov(o, b.nr);
        ov.noalias() = b.B * Eigen::Map<const Eigen::VectorXd>(x + b.lo, b.span);
        for (int i = 0; i < b.nr; ++i) o[i] += b.c0[i];
      } else {
        for (int i = 0; i < b.nr; ++i) o[i] = b.c0[i];
      }
    }
    const double* __restrict v = vb;
    const double* __restrict tw = TW.data();
    const int* __restrict ts = TS.data();
    for (const Group& g : groups) {
      double* __restrict o = vb + g.base;
      const int* __restrict a = A.data() + g.off;
      const int* __restrict b = B.data() + g.off;
      const int n = g.n;
      switch (g.t) {
        case aj::ADD: for (int i = 0; i < n; ++i) o[i] = v[a[i]] + v[b[i]]; break;
        case aj::SUB: for (int i = 0; i < n; ++i) o[i] = v[a[i]] - v[b[i]]; break;
        case aj::MUL: for (int i = 0; i < n; ++i) o[i] = v[a[i]] * v[b[i]]; break;
        case aj::DIV: for (int i = 0; i < n; ++i) o[i] = v[a[i]] / v[b[i]]; break;
        case aj::NEG: for (int i = 0; i < n; ++i) o[i] = -v[a[i]]; break;
        case aj::EXP: for (int i = 0; i < n; ++i) o[i] = std::exp(v[a[i]]); break;
        case aj::LOG: for (int i = 0; i < n; ++i) o[i] = std::log(v[a[i]]); break;
        case aj::SQRT: for (int i = 0; i < n; ++i) o[i] = std::sqrt(v[a[i]]); break;
        case INV: for (int i = 0; i < n; ++i) o[i] = 1.0 / v[a[i]]; break;
        case SUMR:
          for (int i = 0; i < n; ++i) {
            int t = a[i];
            const int t1 = b[i];
            double acc = tw[t] * v[ts[t]];
            for (++t; t < t1; ++t) acc += tw[t] * v[ts[t]];
            o[i] = acc;
          }
          break;
      }
    }
    int flips = 0;
    for (const auto& g : guards) {
      const double x0 = vb[g.a], x1 = vb[g.b];
      bool res;
      switch (g.c) { case aj::LT: res = x0 < x1; break; case aj::LE: res = x0 <= x1; break; case aj::GT: res = x0 > x1; break;
                     case aj::GE: res = x0 >= x1; break; case aj::EQ: res = x0 == x1; break; default: res = x0 != x1; }
      flips += (res != g.expect);
    }
    return flips;
  }
};

inline Plan lower(const Src& s, bool fold) {
  const int L = static_cast<int>(s.ops.size());
  const int rbase = s.nconst + s.nin;
  std::vector<int> uses(L, 0);
  auto use = [&](int slot) { if (slot >= s.base) ++uses[slot - s.base]; };
  for (const auto& p : s.ops) { use(p.a); if (binary(p.op)) use(p.b); }
  for (int o : s.out) use(o);
  for (const auto& g : s.guards) { use(g.a); use(g.b); }
  auto is_addsub = [](uint8_t t) { return t == aj::ADD || t == aj::SUB; };
  std::vector<char> absorbed(L, 0);
  if (fold)
    for (int k = 0; k < L; ++k) {
      const auto& p = s.ops[k];
      if (is_addsub(p.op) && p.a >= s.base && is_addsub(s.ops[p.a - s.base].op) && uses[p.a - s.base] == 1) absorbed[p.a - s.base] = 1;
    }
  std::vector<std::vector<std::pair<int, double>>> terms(L);
  std::vector<uint8_t> type(L);
  std::vector<int> lvl(L, 0);
  auto lv = [&](int slot) { return slot < s.base ? 0 : lvl[slot - s.base]; };
  for (int k = 0; k < L; ++k) {
    const auto& p = s.ops[k];
    if (is_addsub(p.op) && fold) {
      if (p.a >= s.base && absorbed[p.a - s.base]) terms[k] = std::move(terms[p.a - s.base]);
      else terms[k] = {{p.a, 1.0}};
      terms[k].push_back({p.b, p.op == aj::ADD ? 1.0 : -1.0});
    }
    if (absorbed[k]) continue;
    if (fold && is_addsub(p.op) && terms[k].size() >= 3) {
      type[k] = SUMR;
      int m = 0;
      for (const auto& t : terms[k]) m = std::max(m, lv(t.first));
      lvl[k] = m + 1;
    } else {
      type[k] = p.op;
      lvl[k] = 1 + std::max(lv(p.a), binary(p.op) ? lv(p.b) : 0);
    }
  }
  std::vector<int> order;
  for (int k = 0; k < L; ++k) if (!absorbed[k]) order.push_back(k);
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return lvl[x] != lvl[y] ? lvl[x] < lvl[y] : type[x] < type[y]; });

  Plan P;
  P.nconst = s.nconst; P.nin = s.nin; P.nrow = s.nrow; P.rbase = rbase; P.opbase = rbase + s.nrow;
  // rows -> dense column-span blocks
  std::vector<int> rowmap(s.nrow, -1);
  {
    std::map<std::pair<int, int>, int> key;
    std::vector<std::vector<int>> members;
    for (int r = 0; r < s.nrow; ++r) {
      int lo = s.nin, hi = 0;
      for (int k = s.rp[r]; k < s.rp[r + 1]; ++k) { lo = std::min(lo, s.ci[k]); hi = std::max(hi, s.ci[k] + 1); }
      if (hi <= lo) { lo = 0; hi = 0; }
      auto it = key.find({lo, hi});
      if (it == key.end()) { key[{lo, hi}] = static_cast<int>(members.size()); members.push_back({r}); }
      else members[it->second].push_back(r);
    }
    int cur = 0;
    for (const auto& mem : members) {
      int lo = s.nin, hi = 0;
      for (int r : mem) for (int k = s.rp[r]; k < s.rp[r + 1]; ++k) { lo = std::min(lo, s.ci[k]); hi = std::max(hi, s.ci[k] + 1); }
      if (hi <= lo) { lo = 0; hi = 0; }
      Plan::Blk b{rbase + cur, static_cast<int>(mem.size()), lo, hi - lo, Eigen::MatrixXd::Zero(static_cast<int>(mem.size()), hi - lo), {}};
      for (std::size_t i = 0; i < mem.size(); ++i) {
        const int r = mem[i];
        rowmap[r] = cur++;
        for (int k = s.rp[r]; k < s.rp[r + 1]; ++k) b.B(static_cast<int>(i), s.ci[k] - lo) = s.w[k];
        b.c0.push_back(s.c0[r]);
      }
      P.blks.push_back(std::move(b));
    }
    std::vector<int> inv(s.nrow);
    for (int r = 0; r < s.nrow; ++r) inv[rowmap[r]] = r;
    for (int q = 0; q < s.nrow; ++q) {
      const int r = inv[q];
      for (int k = s.rp[r]; k < s.rp[r + 1]; ++k) { P.rci.push_back(s.ci[k]); P.rw.push_back(s.w[k]); }
      P.rrp.push_back(static_cast<int>(P.rci.size()));
    }
  }
  std::vector<int> opmap(L, -1);
  int cur = P.opbase;
  for (std::size_t i = 0; i < order.size();) {
    std::size_t j = i;
    while (j < order.size() && lvl[order[j]] == lvl[order[i]] && type[order[j]] == type[order[i]]) ++j;
    P.groups.push_back({type[order[i]], cur, static_cast<int>(j - i), static_cast<int>(i)});
    for (std::size_t q = i; q < j; ++q) opmap[order[q]] = cur++;
    i = j;
  }
  P.nval = cur;
  auto ns = [&](int slot) -> int {
    if (slot < rbase) return slot;
    if (slot < s.base) return rbase + rowmap[slot - rbase];
    return opmap[slot - s.base];
  };
  for (int k : order) {
    const auto& p = s.ops[k];
    if (type[k] == SUMR) {
      P.A.push_back(static_cast<int>(P.TS.size()));
      for (const auto& t : terms[k]) { P.TS.push_back(ns(t.first)); P.TW.push_back(t.second); }
      P.B.push_back(static_cast<int>(P.TS.size()));
    } else {
      P.A.push_back(ns(p.a));
      P.B.push_back(binary(p.op) ? ns(p.b) : 0);
    }
    P.nodes.push_back({type[k], opmap[k], P.A.back(), P.B.back(), lvl[k]});
  }
  P.elements = static_cast<long>(order.size());
  P.V.assign(P.nval, 0.0);
  for (int i = 0; i < s.nconst; ++i) P.V[i] = s.cval[i];
  for (int o : s.out) P.out.push_back(ns(o));
  for (auto g : s.guards) { g.a = ns(g.a); g.b = ns(g.b); P.guards.push_back(g); }
  return P;
}

// ---- (A) owner-partitioned reverse sweep --------------------------------------------------------------------------
// Every node is owned by the set of output rows whose cone contains it. A node with ONE owner keeps one adjoint; a node with
// several owners (a DF read by many rows, a shared curve-layer op) keeps one adjoint PER OWNER. One sweep over the level
// groups (descending) then yields every row's Jacobian at once; row slots finish analytically through W.
struct Adj {
  struct AG { uint8_t t; int n, off; };
  std::vector<AG> groups;
  std::vector<int> SELF, S, VA, VB, TA, TB, TT;
  std::vector<double> TTW, ad;
  std::vector<int> seed, rp_adj, rp_owner, rp_row, ip_adj, ip_owner, ip_in;
  int trash = 0;
  long elements = 0, pairs = 0;

  void jacobian(const Plan& P, double* J) {
    const int m = static_cast<int>(P.out.size()), n = P.nin;
    std::fill(J, J + static_cast<std::size_t>(m) * n, 0.0);
    double* __restrict a = ad.data();
    const double* __restrict v = P.V.data();
    for (int j = 0; j < m; ++j) a[seed[j]] += 1.0;
    const double* __restrict ttw = TTW.data();
    const int* __restrict tt = TT.data();
    for (const AG& g : groups) {
      const int* __restrict self = SELF.data() + g.off;
      const int* __restrict s = S.data() + g.off;
      const int* __restrict va = VA.data() + g.off;
      const int* __restrict vb = VB.data() + g.off;
      const int* __restrict ta = TA.data() + g.off;
      const int* __restrict tb = TB.data() + g.off;
      const int N = g.n;
      switch (g.t) {
        case aj::ADD: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; a[ta[i]] += x; a[tb[i]] += x; } break;
        case aj::SUB: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; a[ta[i]] += x; a[tb[i]] -= x; } break;
        case aj::MUL: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; a[ta[i]] += x * v[vb[i]]; a[tb[i]] += x * v[va[i]]; } break;
        case aj::DIV: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; const double ib = 1.0 / v[vb[i]]; a[ta[i]] += x * ib; a[tb[i]] -= x * v[s[i]] * ib; } break;
        case aj::NEG: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; a[ta[i]] -= x; } break;
        case aj::EXP: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; a[ta[i]] += x * v[s[i]]; } break;
        case aj::LOG: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; a[ta[i]] += x / v[va[i]]; } break;
        case aj::SQRT: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; a[ta[i]] += x * 0.5 / v[s[i]]; } break;
        case INV: for (int i = 0; i < N; ++i) { const double x = a[self[i]]; a[self[i]] = 0; a[ta[i]] -= x * v[s[i]] * v[s[i]]; } break;
        case SUMR:
          for (int i = 0; i < N; ++i) {
            const double x = a[self[i]];
            a[self[i]] = 0;
            for (int t = ta[i]; t < tb[i]; ++t) a[tt[t]] += ttw[t] * x;
          }
          break;
      }
    }
    const int* __restrict rrp = P.rrp.data();
    const int* __restrict rci = P.rci.data();
    const double* __restrict rw = P.rw.data();
    for (std::size_t p = 0; p < rp_adj.size(); ++p) {
      const double x = a[rp_adj[p]];
      a[rp_adj[p]] = 0;
      if (x == 0.0) continue;
      const int r = rp_row[p], j = rp_owner[p];
      for (int k = rrp[r]; k < rrp[r + 1]; ++k) J[j + rci[k] * m] += x * rw[k];
    }
    for (std::size_t p = 0; p < ip_adj.size(); ++p) {
      J[ip_owner[p] + ip_in[p] * m] += a[ip_adj[p]];
      a[ip_adj[p]] = 0;
    }
    a[trash] = 0;
  }
};

inline Adj build_adj(const Plan& P) {
  const int m = static_cast<int>(P.out.size());
  std::vector<std::vector<int>> own(P.nval);
  for (int j = 0; j < m; ++j) own[P.out[j]].push_back(j);
  auto merge = [&](int u, const std::vector<int>& src) {
    auto& d = own[u];
    std::vector<int> r;
    r.reserve(d.size() + src.size());
    std::set_union(d.begin(), d.end(), src.begin(), src.end(), std::back_inserter(r));
    d.swap(r);
  };
  for (int q = static_cast<int>(P.nodes.size()) - 1; q >= 0; --q) {
    const auto& nd = P.nodes[q];
    const std::vector<int> o = own[nd.slot];
    if (o.empty()) continue;
    if (nd.t == SUMR) for (int t = nd.a; t < nd.b; ++t) merge(P.TS[t], o);
    else { merge(nd.a, o); if (binary(nd.t)) merge(nd.b, o); }
  }
  Adj A;
  std::vector<int> poff(P.nval, -1);
  int cursor = P.nval;
  for (int u = P.nconst; u < P.nval; ++u) {
    const bool boundary = u < P.opbase;
    if (!own[u].empty() && (boundary || own[u].size() >= 2)) { poff[u] = cursor; cursor += static_cast<int>(own[u].size()); A.pairs += static_cast<long>(own[u].size()); }
  }
  A.trash = cursor++;
  A.ad.assign(cursor, 0.0);
  auto rank = [&](int u, int j) { return static_cast<int>(std::lower_bound(own[u].begin(), own[u].end(), j) - own[u].begin()); };
  auto target = [&](int u, int j) -> int {
    if (u < P.nconst) return A.trash;
    if (poff[u] >= 0) return poff[u] + rank(u, j);
    return u;
  };
  struct E { int level; uint8_t t; int self, s, va, vb, ta, tb; };
  std::vector<E> es;
  std::vector<int> TT;
  std::vector<double> TTW;
  for (const auto& nd : P.nodes) {
    const auto& o = own[nd.slot];
    if (o.empty()) continue;
    const bool single = poff[nd.slot] < 0;
    const std::size_t cnt = single ? 1 : o.size();
    for (std::size_t qi = 0; qi < cnt; ++qi) {
      const int j = o[qi];
      const int self = single ? nd.slot : poff[nd.slot] + static_cast<int>(qi);
      if (nd.t == SUMR) {
        const int t0 = static_cast<int>(TT.size());
        for (int t = nd.a; t < nd.b; ++t) { TT.push_back(target(P.TS[t], j)); TTW.push_back(P.TW[t]); }
        es.push_back({nd.level, nd.t, self, nd.slot, 0, 0, t0, static_cast<int>(TT.size())});
      } else {
        es.push_back({nd.level, nd.t, self, nd.slot, nd.a, binary(nd.t) ? nd.b : 0, target(nd.a, j), binary(nd.t) ? target(nd.b, j) : A.trash});
      }
    }
  }
  std::stable_sort(es.begin(), es.end(), [](const E& x, const E& y) { return x.level != y.level ? x.level > y.level : x.t < y.t; });
  for (std::size_t i = 0; i < es.size();) {
    std::size_t j = i;
    while (j < es.size() && es[j].level == es[i].level && es[j].t == es[i].t) ++j;
    A.groups.push_back({es[i].t, static_cast<int>(j - i), static_cast<int>(i)});
    i = j;
  }
  for (const auto& e : es) { A.SELF.push_back(e.self); A.S.push_back(e.s); A.VA.push_back(e.va); A.VB.push_back(e.vb); A.TA.push_back(e.ta); A.TB.push_back(e.tb); }
  A.TT = std::move(TT); A.TTW = std::move(TTW);
  A.elements = static_cast<long>(es.size());
  for (int j = 0; j < m; ++j) { const int u = P.out[j]; A.seed.push_back(target(u, j)); }
  for (int u = P.rbase; u < P.opbase; ++u)
    if (poff[u] >= 0) for (std::size_t q = 0; q < own[u].size(); ++q) { A.rp_adj.push_back(poff[u] + static_cast<int>(q)); A.rp_owner.push_back(own[u][q]); A.rp_row.push_back(u - P.rbase); }
  for (int u = P.nconst; u < P.rbase; ++u)
    if (poff[u] >= 0) for (std::size_t q = 0; q < own[u].size(); ++q) { A.ip_adj.push_back(poff[u] + static_cast<int>(q)); A.ip_owner.push_back(own[u][q]); A.ip_in.push_back(u - P.nconst); }
  return A;
}

// ---- (C) forward-vector (tangent width = n knots) through the same groups ------------------------------------------
struct FwdVec {
  int n = 0, rbase = 0, nconst = 0;
  std::vector<double> T, zero, unit;
  explicit FwdVec(const Plan& P) : n(P.nin), rbase(P.rbase), nconst(P.nconst) {
    T.assign(static_cast<std::size_t>(P.nval - P.rbase) * n, 0.0);
    for (int r = 0; r < P.nrow; ++r)
      for (int k = P.rrp[r]; k < P.rrp[r + 1]; ++k) T[static_cast<std::size_t>(r) * n + P.rci[k]] = P.rw[k];
    zero.assign(n, 0.0);
    unit.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) unit[static_cast<std::size_t>(i) * n + i] = 1.0;
  }
  const double* tan(int slot) const {
    if (slot < nconst) return zero.data();
    if (slot < rbase) return unit.data() + static_cast<std::size_t>(slot - nconst) * n;
    return T.data() + static_cast<std::size_t>(slot - rbase) * n;
  }
  void jacobian(const Plan& P, double* J) {
    const double* v = P.V.data();
    const int N = n;
    for (const auto& g : P.groups) {
      const int* a = P.A.data() + g.off;
      const int* b = P.B.data() + g.off;
      for (int i = 0; i < g.n; ++i) {
        const int s = g.base + i;
        double* __restrict o = T.data() + static_cast<std::size_t>(s - rbase) * N;
        switch (g.t) {
          case aj::ADD: { const double* __restrict x = tan(a[i]); const double* __restrict y = tan(b[i]); for (int k = 0; k < N; ++k) o[k] = x[k] + y[k]; break; }
          case aj::SUB: { const double* __restrict x = tan(a[i]); const double* __restrict y = tan(b[i]); for (int k = 0; k < N; ++k) o[k] = x[k] - y[k]; break; }
          case aj::MUL: { const double* __restrict x = tan(a[i]); const double* __restrict y = tan(b[i]); const double va = v[a[i]], vb = v[b[i]]; for (int k = 0; k < N; ++k) o[k] = x[k] * vb + y[k] * va; break; }
          case aj::DIV: { const double* __restrict x = tan(a[i]); const double* __restrict y = tan(b[i]); const double ib = 1.0 / v[b[i]], q = v[s]; for (int k = 0; k < N; ++k) o[k] = (x[k] - q * y[k]) * ib; break; }
          case aj::NEG: { const double* __restrict x = tan(a[i]); for (int k = 0; k < N; ++k) o[k] = -x[k]; break; }
          case aj::EXP: { const double* __restrict x = tan(a[i]); const double e = v[s]; for (int k = 0; k < N; ++k) o[k] = x[k] * e; break; }
          case aj::LOG: { const double* __restrict x = tan(a[i]); const double ia = 1.0 / v[a[i]]; for (int k = 0; k < N; ++k) o[k] = x[k] * ia; break; }
          case aj::SQRT: { const double* __restrict x = tan(a[i]); const double c = 0.5 / v[s]; for (int k = 0; k < N; ++k) o[k] = x[k] * c; break; }
          case INV: { const double* __restrict x = tan(a[i]); const double c = -v[s] * v[s]; for (int k = 0; k < N; ++k) o[k] = x[k] * c; break; }
          case SUMR: {
            for (int k = 0; k < N; ++k) o[k] = 0.0;
            for (int t = a[i]; t < b[i]; ++t) { const double* __restrict x = tan(P.TS[t]); const double w = P.TW[t]; for (int k = 0; k < N; ++k) o[k] += w * x[k]; }
            break;
          }
        }
      }
    }
    const int m = static_cast<int>(P.out.size());
    for (int j = 0; j < m; ++j) { const double* t = tan(P.out[j]); for (int k = 0; k < N; ++k) J[j + k * m] = t[k]; }
  }
};
}  // namespace co

int main(int argc, char** argv) {
  std::vector<std::string> want = {"ois_nolag", "fx_xccy", "desk", "mixed_scheme", "desk_mixed"};
  if (argc > 1) want.assign(argv + 1, argv + argc);
  std::printf("# coarse kernels probe. engine = HybridBundleResidual; all times µs, median of 7 batches, same process\n");
  for (const auto& s : swaps::shapes::ladder()) {
    if (std::find(want.begin(), want.end(), s.name) == want.end()) continue;
    const auto& p = s.prob;
    const int n = p.n_knots(), m = p.n_residuals();
    // ---- record (tape.cpp recorder, unchanged) ----
    aj::Recorder rec;
    aj::R = &rec;
    std::vector<aj::Rec> X(n);
    for (int i = 0; i < n; ++i) {
      rec.nodes.push_back({aj::INPUT, -1, -1, s.x_true[i]});
      rec.inputs.push_back(static_cast<int>(rec.nodes.size()) - 1);
      X[i] = aj::Rec(s.x_true[i], rec.inputs.back());
    }
    std::vector<int> outs_tl;
    {
      const auto C = cal::build_bundle_curves<aj::Rec>(p.curves, [&](int c, int i) { return X[p.offset(c) + i]; });
      const auto of = [&](int i) -> const px::CurveHandle<aj::Rec>& { return *C[i]; };
      for (const auto& ins : p.instruments) outs_tl.push_back(cal::instrument_model_quote<aj::Rec>(ins, of).slot());
      aj::R = nullptr;
    }
    aj::Kernel K;
    const auto tb0 = clk::now();
    K.build(rec, outs_tl);
    aj::Collapsed KC = aj::collapse(K);
    co::Plan PE = co::lower(co::from_kernel(K), true);                       // exact: no rows, no reciprocals
    co::Plan PF = co::lower(co::from_collapsed(KC), true);                   // fast: W rows + groups
    co::Plan PR = co::lower(co::share_reciprocals(co::from_collapsed(KC)), true);  // fast + shared reciprocals
    co::Adj AR = co::build_adj(PR);
    co::FwdVec FV(PR);
    const double build_ms = std::chrono::duration<double, std::milli>(clk::now() - tb0).count();

    // ---- parity ----
    cal::HybridBundleResidual h(p);
    auto templ_quotes = [&](const Eigen::VectorXd& x) {
      const auto Cd = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
      const auto ofd = [&](int i) -> const px::CurveHandle<double>& { return *Cd[i]; };
      Eigen::VectorXd q(m);
      for (int r = 0; r < m; ++r) q[r] = cal::instrument_model_quote<double>(p.instruments[r], ofd);
      return q;
    };
    Eigen::VectorXd x1 = s.x_true;
    for (int i = 0; i < n; ++i) x1[i] += 1e-5 * std::sin(1.3 * i + 0.2);
    const Eigen::VectorXd qt = templ_quotes(x1);
    const Eigen::VectorXd mr = h.model_rates(x1);
    auto maxdiff = [&](co::Plan& P, const Eigen::VectorXd& ref) { double d = 0; for (int r = 0; r < m; ++r) d = std::max(d, std::abs(P.V[P.out[r]] - ref[r])); return d; };
    int flE = PE.forward(x1.data()); const double dE_t = maxdiff(PE, qt);
    int flF = PF.forward(x1.data()); const double dF_e = maxdiff(PF, mr);
    int flR = PR.forward(x1.data()); const double dR_e = maxdiff(PR, mr), dR_t = maxdiff(PR, qt);
    Eigen::MatrixXd JA(m, n), JC(m, n), Je;
    AR.jacobian(PR, JA.data());
    FV.jacobian(PR, JC.data());
    h.jacobian_vs_into(x1, s.q0, Je);
    double dJA = 0, dJC = 0;
    for (int r = 0; r < m; ++r) {
      const auto& ins = p.instruments[r];
      double sc = 1.0;
      if (ins.quote == cal::QuoteKind::FxForward) sc = 1.0 / (mr[r] * ins.fx_time);
      else if (ins.band_upper > ins.band_lower) sc = cal::band_slope(mr[r], ins.band_lower, ins.band_upper, ins.band_decay);
      for (int i = 0; i < n; ++i) { dJA = std::max(dJA, std::abs(JA(r, i) * sc - Je(r, i))); dJC = std::max(dJC, std::abs(JC(r, i) * sc - Je(r, i))); }
    }
    unsigned long allocs = 0;
    {
      swaps::testing::AllocScope a;
      for (int r = 0; r < 20; ++r) { PE.forward(x1.data()); PR.forward(x1.data()); AR.jacobian(PR, JA.data()); FV.jacobian(PR, JC.data()); }
      allocs = a.allocs();
    }
    std::printf("\n== %s  m=%d n=%d  build(record-free part: K+collapse+3 plans+adjoint) %.1f ms  allocs/20 replays %lu\n", s.name.c_str(), m, n, build_ms, allocs);
    std::printf("   dispatch/eval: scalar interp %d | scalar collapsed %d (+rows) | coarse exact %ld (%ld groups over %ld elements) | coarse fast %ld | fast+inv %ld\n",
                K.nops, KC.nnl, PE.dispatch(), static_cast<long>(PE.groups.size()), PE.elements, PF.dispatch(), PR.dispatch());
    std::printf("   rows %d in %zu span blocks | adjoint: %ld elements, %ld owner-pairs, %zu groups\n", PR.nrow, PR.blks.size(), AR.elements, AR.pairs, AR.groups.size());
    std::printf("   parity: exact-coarse vs templated %.1e (flips %d) | fast vs engine %.1e | fast+inv vs engine %.1e vs templated %.1e (flips %d/%d) | J owner-rev vs engine %.1e | J fwd-vec vs engine %.1e\n",
                dE_t, flE, dF_e, dR_e, dR_t, flF, flR, dJA, dJC);
    std::fflush(stdout);

    if (std::getenv("AADJIT_NO_TIMING")) continue;  // counts-only run: parity/dispatch/allocs above, no timing loops
    // ---- timings ----
    double load0 = 0, load1 = 0, t_eng = 0, t_mr = 0, t_si = 0, t_sc = 0, t_ce = 0, t_cf = 0, t_cr = 0, t_eJ = 0, t_scJ = 0, t_A = 0, t_C = 0;
    util::guarded([&] {
    load0 = util::load1();
    Eigen::VectorXd xa = x1, xb = s.x_true;
    bool flip = false;
    const int reps = std::max(20, static_cast<int>(40000.0 / (K.nops + 1)));
    Eigen::MatrixXd JJ(m, n);
    t_eng = util::time_us([&] { flip = !flip; volatile double d = h.residuals_vs(flip ? xa : xb, s.q0)[0]; (void)d; }, reps * 5);
    t_mr = util::time_us([&] { flip = !flip; volatile double d = h.model_rates(flip ? xa : xb)[0]; (void)d; }, reps * 5);
    t_si = util::time_us([&] { flip = !flip; volatile int f = K.forward((flip ? xa : xb).data()); (void)f; }, reps * 2);
    t_sc = util::time_us([&] { flip = !flip; volatile int f = KC.forward((flip ? xa : xb).data()); (void)f; }, reps * 5);
    t_ce = util::time_us([&] { flip = !flip; volatile int f = PE.forward((flip ? xa : xb).data()); (void)f; }, reps * 2);
    t_cf = util::time_us([&] { flip = !flip; volatile int f = PF.forward((flip ? xa : xb).data()); (void)f; }, reps * 5);
    t_cr = util::time_us([&] { flip = !flip; volatile int f = PR.forward((flip ? xa : xb).data()); (void)f; }, reps * 5);
    const int rj = std::max(5, reps / 5);
    t_eJ = util::time_us([&] { flip = !flip; h.jacobian_vs_into(flip ? xa : xb, s.q0, JJ); }, rj);
    t_scJ = util::time_us([&] { flip = !flip; KC.forward((flip ? xa : xb).data()); KC.jacobian_reverse(JJ.data()); }, rj);
    t_A = util::time_us([&] { flip = !flip; PR.forward((flip ? xa : xb).data()); AR.jacobian(PR, JJ.data()); }, rj);
    t_C = util::time_us([&] { flip = !flip; PR.forward((flip ? xa : xb).data()); FV.jacobian(PR, JJ.data()); }, rj);
    load1 = util::load1();
    });
    std::printf("   load %.2f -> %.2f\n", load0, load1);
    std::printf("   VALUE µs : engine residuals_vs %.2f (model_rates %.2f) | scalar interp %.2f | scalar collapsed %.2f | coarse exact %.2f | coarse fast %.2f | coarse fast+inv %.2f\n",
                t_eng, t_mr, t_si, t_sc, t_ce, t_cf, t_cr);
    std::printf("   VALUE+J µs: engine jacobian_vs_into %.1f (J only) | scalar collapsed cone-rev %.1f | coarse owner-rev %.1f | coarse fwd-vector %.1f\n",
                t_eJ, t_scJ, t_A, t_C);
    std::fflush(stdout);
  }
}
