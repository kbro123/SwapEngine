// COUNTS-ONLY probe for the main line's q_small hypothesis (2026-09-15): does V1's per-row noise flip a Hyman clamp between x(q0) and
// x(V1), and does the flip coincide with the refresh? variant() V1/V4 copied verbatim from scratchpad/dm/probe_dm_stall_qsmall.cpp;
// protocol as there: calibrate, break-even pinned 64, 6 warm ticks alternating q0 <-> V1 from the committed state.
// Every x the streamer evaluates goes through (i) a NAMED predicate evaluator for SOFR's MonotoneCubic region (the engine's
// hyman_filter re-expressed predicate by predicate: node, knot interval, truth, margin, and whether the engine evaluates it), and
// (ii) the recorded desk_mixed tape's guard vector (the engine's own ops) as a cross-check.
#include "rec.hpp"
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "kernel.hpp"

#include <cstdio>
#include <string>

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
using swaps::shapes::Shape;

static double maturity(const cal::Instrument& ins) {
  switch (ins.quote) {
    case cal::QuoteKind::FxForward: return ins.fx_time;
    case cal::QuoteKind::Rate: return ins.obs.sub_end.empty() ? 0.0 : ins.obs.sub_end.back();
    default: return ins.fixed.coupons.empty() ? 0.0 : ins.fixed.coupons.back().pay;
  }
}
static Eigen::VectorXd variant(const Shape& s, int v) {  // verbatim (V1, V4 used)
  Eigen::VectorXd q = s.q0;
  for (int i = 0; i < q.size(); ++i) {
    const auto& ins = s.prob.instruments[static_cast<std::size_t>(i)];
    const bool fx = ins.quote == cal::QuoteKind::FxForward;
    const bool basis = ins.quote == cal::QuoteKind::XccyMtmBasis || ins.quote == cal::QuoteKind::ParSpread;
    const bool spread = ins.quote == cal::QuoteKind::Portfolio || ins.quote == cal::QuoteKind::TurnJump;
    const double bump = std::sin(0.7 * i + 0.3);
    switch (v) {
      case 0: q[i] = s.q_small[i]; break;
      case 1: q[i] = fx ? ins.market : ins.market + 1e-5 * bump; break;
      case 2: q[i] = fx ? ins.market : ins.market + (ins.quote == cal::QuoteKind::XccyMtmBasis ? 1e-6 : 1e-5) * bump; break;
      case 3: q[i] = fx ? ins.market * std::exp(1e-5 * bump * maturity(ins)) : ins.market + 1e-5 * bump; break;
      case 4: {
        const double tilt = 0.9 + 0.2 * std::min(maturity(ins), 30.0) / 30.0;
        q[i] = (fx || spread) ? ins.market : ins.market + (basis ? 1e-6 : 1e-5) * tilt;
        break;
      }
    }
  }
  return q;
}

// ---- named Hyman predicates of one MonotoneCubic region (following region: node 0 = the join) ----
struct Pred { std::string name; bool evaluated; bool truth; double margin; };
struct Region {
  std::vector<double> xs;   // node times: join, then the region knots
  int x0 = 0;               // state index of the join value (the previous region's last knot)
  std::vector<Pred> eval(const Eigen::VectorXd& x) const {
    const int N = static_cast<int>(xs.size()), nseg = N - 1;
    std::vector<double> ys(N), h(nseg), S(nseg);
    for (int i = 0; i < N; ++i) ys[i] = x[x0 + i];
    for (int i = 0; i < nseg; ++i) { h[i] = xs[i + 1] - xs[i]; S[i] = (ys[i + 1] - ys[i]) / h[i]; }
    std::vector<double> lo(N), di(N), up(N), r(N), m(N);
    di[0] = 2.0; up[0] = 1.0; r[0] = 3.0 * S[0];
    for (int i = 1; i < N - 1; ++i) { lo[i] = h[i]; di[i] = 2.0 * (h[i] + h[i - 1]); up[i] = h[i - 1]; r[i] = 3.0 * (h[i] * S[i - 1] + h[i - 1] * S[i]); }
    lo[N - 1] = 1.0; di[N - 1] = 2.0; r[N - 1] = 3.0 * S[nseg - 1];
    for (int i = 1; i < N; ++i) { const double w = lo[i] / di[i - 1]; di[i] -= w * up[i - 1]; r[i] = r[i] - w * r[i - 1]; }
    m[N - 1] = r[N - 1] / di[N - 1];
    for (int i = N - 1; i-- > 0;) m[i] = (r[i] - up[i] * m[i + 1]) / di[i];
    std::vector<Pred> P;
    auto add = [&](int i, const char* what, bool ev, bool t, double mg) {
      char buf[160];
      const double tl = i > 0 ? xs[i - 1] : xs[0], tr = i < N - 1 ? xs[i + 1] : xs[N - 1];
      std::snprintf(buf, sizeof buf, "SOFR node %d (t=%.2f, tangent spans [%.2f,%.2f]) %s", i, xs[i], tl, tr, what);
      P.push_back({buf, ev, t, mg});
    };
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N - 1) {
        const double Se = i == 0 ? S[0] : S[N - 2];
        const bool sign = m[i] * Se > 0.0;
        add(i, "end: m*S > 0", true, sign, m[i] * Se);
        add(i, "end: clamp |3S| < |m|", sign, std::abs(3.0 * Se) < std::abs(m[i]), std::abs(m[i]) - std::abs(3.0 * Se));
      } else {
        const double pm = (S[i - 1] * h[i] + S[i] * h[i - 1]) / (h[i - 1] + h[i]);
        const bool c_s = std::abs(S[i]) < std::abs(S[i - 1]);
        add(i, "min: |S_i| < |S_i-1|", true, c_s, std::abs(S[i - 1]) - std::abs(S[i]));
        const double t0 = c_s ? std::abs(S[i]) : std::abs(S[i - 1]);
        const bool c_p = std::abs(pm) < t0;
        add(i, "min: |pm| < min|S|", true, c_p, t0 - std::abs(pm));
        double M = 3.0 * (c_p ? std::abs(pm) : t0);
        if (i > 1) {
          const double run = (S[i - 1] - S[i - 2]) * (S[i] - S[i - 1]);
          add(i, "down-run (S_i-1 - S_i-2)(S_i - S_i-1) > 0", true, run > 0.0, run);
          const double pd = (S[i - 1] * (2.0 * h[i - 1] + h[i - 2]) - S[i - 2] * h[i - 1]) / (h[i - 2] + h[i - 1]);
          const bool e1 = run > 0.0;
          add(i, "pm*pd > 0", e1, pm * pd > 0.0, pm * pd);
          const bool e2 = e1 && pm * pd > 0.0;
          add(i, "pm*(S_i-1 - S_i-2) > 0", e2, pm * (S[i - 1] - S[i - 2]) > 0.0, pm * (S[i - 1] - S[i - 2]));
          const bool e3 = e2 && pm * (S[i - 1] - S[i - 2]) > 0.0;
          const double mn = std::abs(pd) < std::abs(pm) ? std::abs(pd) : std::abs(pm);
          add(i, "min: |pd| < |pm|", e3, std::abs(pd) < std::abs(pm), std::abs(pm) - std::abs(pd));
          add(i, "max: M < 1.5*min(|pm|,|pd|)", e3, M < 1.5 * mn, 1.5 * mn - M);
          if (e3 && M < 1.5 * mn) M = 1.5 * mn;
        }
        if (i < N - 2) {
          const double run = (S[i] - S[i - 1]) * (S[i + 1] - S[i]);
          add(i, "up-run (S_i - S_i-1)(S_i+1 - S_i) > 0", true, run > 0.0, run);
          const double pu = (S[i] * (2.0 * h[i] + h[i + 1]) - S[i + 1] * h[i]) / (h[i] + h[i + 1]);
          const bool e1 = run > 0.0;
          add(i, "pm*pu > 0", e1, pm * pu > 0.0, pm * pu);
          const bool e2 = e1 && pm * pu > 0.0;
          add(i, "-pm*(S_i - S_i-1) > 0", e2, -pm * (S[i] - S[i - 1]) > 0.0, -pm * (S[i] - S[i - 1]));
          const bool e3 = e2 && -pm * (S[i] - S[i - 1]) > 0.0;
          const double mn = std::abs(pu) < std::abs(pm) ? std::abs(pu) : std::abs(pm);
          add(i, "min: |pu| < |pm|", e3, std::abs(pu) < std::abs(pm), std::abs(pm) - std::abs(pu));
          add(i, "max: M < 1.5*min(|pm|,|pu|)", e3, M < 1.5 * mn, 1.5 * mn - M);
          if (e3 && M < 1.5 * mn) M = 1.5 * mn;
        }
        const bool sign = m[i] * pm > 0.0;
        add(i, "sign: m*pm > 0", true, sign, m[i] * pm);
        add(i, "clamp: M < |m|", sign, M < std::abs(m[i]), std::abs(m[i]) - M);
      }
    }
    return P;
  }
};

struct Trace {
  const Region* reg = nullptr;
  aj::Kernel* K = nullptr;
  std::vector<Pred> prev;
  std::vector<char> prev_bits;
  long evals = 0;
  bool on = false;
  std::vector<char> bits(const Eigen::VectorXd& x) {
    K->forward(x.data());
    std::vector<char> b(K->guards.size());
    for (std::size_t g = 0; g < K->guards.size(); ++g) {
      const auto& gd = K->guards[g];
      const double a = K->val[gd.a], c = K->val[gd.b];
      bool r;
      switch (gd.c) { case aj::LT: r = a < c; break; case aj::LE: r = a <= c; break; case aj::GT: r = a > c; break;
                      case aj::GE: r = a >= c; break; case aj::EQ: r = a == c; break; default: r = a != c; }
      b[g] = r;
    }
    return b;
  }
  void note(char kind, const Eigen::VectorXd& x) {
    if (!on) return;
    ++evals;
    const auto P = reg->eval(x);
    const auto B = bits(x);
    int tape_flips = 0;
    if (!prev_bits.empty()) for (std::size_t g = 0; g < B.size() && g < prev_bits.size(); ++g) tape_flips += B[g] != prev_bits[g];
    std::printf("    [%c eval %ld: tape guards %zu, changed vs previous eval %d%s]\n", kind, evals, B.size(),
                prev_bits.empty() ? 0 : tape_flips + static_cast<int>(B.size() != prev_bits.size()), B.size() != prev_bits.size() && !prev_bits.empty() ? " (guard COUNT changed)" : "");
    if (!prev.empty())
      for (std::size_t k = 0; k < P.size(); ++k) {
        const bool was = prev[k].evaluated && prev[k].truth, now = P[k].evaluated && P[k].truth;
        if (prev[k].evaluated != P[k].evaluated || (P[k].evaluated && prev[k].truth != P[k].truth))
          std::printf("        FLIP %s: %s%s -> %s%s  margin %+.3e -> %+.3e\n", P[k].name.c_str(), prev[k].evaluated ? "" : "(not evaluated) ",
                      prev[k].evaluated ? (was ? "true" : "false") : "", P[k].evaluated ? "" : "(not evaluated) ", P[k].evaluated ? (now ? "true" : "false") : "",
                      prev[k].margin, P[k].margin);
      }
    prev = P;
    prev_bits = B;
  }
};
inline Trace g_tr;

struct LProb : cal::BundleProblem {
  LProb() = default;
  explicit LProb(const cal::BundleProblem& p) : cal::BundleProblem(p) {}
};
struct LEngine {
  cal::HybridBundleResidual h;
  explicit LEngine(const cal::BundleProblem& p) : h(p) {}
  int n_residuals() const { return h.n_residuals(); }
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const { g_tr.note('r', x); return h.residuals_vs(x, q); }
  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const { return h.residuals(x); }
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J) const { g_tr.note('J', x); h.jacobian_vs_into(x, q, J); }
  Eigen::MatrixXd jacobian_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const { return h.jacobian_vs(x, q); }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const { return h.jacobian(x); }
  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const { return h.model_rates(x); }
  void set_quotes(const cal::BundleProblem& p) { h.set_quotes(p); }
  void set_quote(int row, double m, double lo, double up, double dc) { h.set_quote(row, m, lo, up, dc); }
  void set_market(const Eigen::VectorXd& q) { h.set_market(q); }
};
namespace swaps::calibration {
template <> struct residual_engine<LProb> { using type = LEngine; };
}  // namespace swaps::calibration

#define SWAPS_STREAM_TRACE 1
#include "swaps/calibration/streaming.hpp"

static void table_diff(const char* tag, const std::vector<Pred>& A, const std::vector<Pred>& B) {
  int n = 0;
  for (std::size_t k = 0; k < A.size(); ++k) {
    const bool changed = A[k].evaluated != B[k].evaluated || (B[k].evaluated && A[k].truth != B[k].truth);
    if (!changed) continue;
    ++n;
    std::printf("    %s  %s: %s -> %s   margin %+.3e -> %+.3e\n", tag, B[k].name.c_str(),
                A[k].evaluated ? (A[k].truth ? "true" : "false") : "(n/e)", B[k].evaluated ? (B[k].truth ? "true" : "false") : "(n/e)", A[k].margin, B[k].margin);
  }
  if (!n) std::printf("    %s  no Hyman predicate differs\n", tag);
}

int main() {
  static const auto L = swaps::shapes::ladder();
  const Shape* dm = nullptr;
  for (const auto& s : L) if (s.name == "desk_mixed") dm = &s;
  const Shape& s = *dm;
  const auto& p = s.prob;
  const int n = p.n_knots();
  // SOFR is curve 0 (offset 0): Hermite front, MonotoneCubic back. The join node is the front's last knot.
  Region reg;
  const auto& R = p.curves[0].regions;
  reg.xs.push_back(R[0].knots.back());
  for (double t : R[1].knots) reg.xs.push_back(t);
  reg.x0 = static_cast<int>(R[0].knots.size()) - 1;
  std::printf("# desk_mixed: the ONLY value-dependent region is SOFR (curve 0) knots %d..%d, MonotoneCubic over t = %.2f..%.2f (join at %.2f)\n",
              reg.x0 + 1, reg.x0 + static_cast<int>(R[1].knots.size()), R[1].knots.front(), R[1].knots.back(), reg.xs.front());
  for (std::size_t c = 1; c < p.curves.size(); ++c)
    for (const auto& r : p.curves[c].regions)
      if (!swaps::curve::scheme_is_linear(r.scheme)) std::printf("# (unexpected) curve %zu also has a value-dependent region\n", c);

  const Eigen::VectorXd x = cal::calibrate(p, s.x0).x;
  const Eigen::VectorXd qV1 = variant(s, 1), qV4 = variant(s, 4);
  LProb lp(p);
  LEngine eng(p);
  using SC = cal::StreamingCalibrator<LProb>;
  SC::Options o;
  o.breakeven_steps = 64;
  SC st(eng, lp, x, s.q0, o);
  bool f = false;
  std::printf("#### warm: 6 ticks alternating q0 <-> V1 (untraced)\n");
  for (int i = 0; i < 6; ++i) { f = !f; st.update(f ? qV1 : s.q0); }
  const Eigen::VectorXd xq0 = st.current();

  // tape at x(q0)
  aj::Recorder rec;
  aj::R = &rec;
  std::vector<aj::Rec> X(n);
  for (int i = 0; i < n; ++i) {
    rec.nodes.push_back({aj::INPUT, -1, -1, xq0[i]});
    rec.inputs.push_back(static_cast<int>(rec.nodes.size()) - 1);
    X[i] = aj::Rec(xq0[i], rec.inputs.back());
  }
  std::vector<int> outs;
  {
    const auto C = cal::build_bundle_curves<aj::Rec>(p.curves, [&](int c, int i) { return X[p.offset(c) + i]; });
    const auto of = [&](int i) -> const px::CurveHandle<aj::Rec>& { return *C[i]; };
    for (const auto& ins : p.instruments) outs.push_back(cal::instrument_model_quote<aj::Rec>(ins, of).slot());
  }
  aj::R = nullptr;
  aj::Kernel K;
  K.build(rec, outs);
  g_tr.reg = &reg;
  g_tr.K = &K;

  auto run_tick = [&](const char* name, const Eigen::VectorXd& q) {
    std::printf("#### TICK %s from the committed state\n", name);
    g_tr.prev.clear(); g_tr.prev_bits.clear(); g_tr.on = true;
    const cal::StreamTick t = st.update(q);
    g_tr.on = false;
    std::printf("#### end %s: conv %d steps %d refreshes %d rescales %d\n", name, static_cast<int>(t.converged), t.newton_steps, t.refreshes, t.rescales);
    return st.current();
  };
  const Eigen::VectorXd xV1 = run_tick("V1", qV1);
  run_tick("q0", s.q0);
  const Eigen::VectorXd xq0b = st.current();
  const Eigen::VectorXd xV4 = run_tick("V4", qV4);
  run_tick("q0", s.q0);
  run_tick("V1 (again)", qV1);

  auto knotmove = [&](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    Eigen::Index k = 0;
    const double d = (a - b).cwiseAbs().maxCoeff(&k);
    std::printf("|dx|_inf %.2e at knot %d (curve %s)", d, static_cast<int>(k), k < p.offset(1) ? "SOFR" : "other");
  };
  std::printf("\n#### committed states: x(q0) vs x(V1): "); knotmove(xV1, xq0); std::printf(";  x(q0) vs x(V4): "); knotmove(xV4, xq0b); std::printf("\n");
  const auto P0 = reg.eval(xq0), P1 = reg.eval(xV1), P4 = reg.eval(xV4);
  std::printf("#### Hyman predicates, committed x(q0) -> x(V1):\n"); table_diff("q0->V1", P0, P1);
  std::printf("#### Hyman predicates, committed x(q0) -> x(V4):\n"); table_diff("q0->V4", P0, P4);
  std::printf("#### tape guard vectors: |x(q0)| %zu guards, |x(V1)| %zu, |x(V4)| %zu\n", g_tr.bits(xq0).size(), g_tr.bits(xV1).size(), g_tr.bits(xV4).size());
  // the smallest EVALUATED margins at x(q0): how close the committed curve sits to each kink
  std::vector<std::pair<double, std::size_t>> mg;
  for (std::size_t k = 0; k < P0.size(); ++k) if (P0[k].evaluated) mg.push_back({std::abs(P0[k].margin), k});
  std::sort(mg.begin(), mg.end());
  std::printf("#### smallest evaluated margins at x(q0):\n");
  for (int i = 0; i < 6 && i < static_cast<int>(mg.size()); ++i) std::printf("    %.3e  %s (%s)\n", mg[i].first, P0[mg[i].second].name.c_str(), P0[mg[i].second].truth ? "true" : "false");
}
