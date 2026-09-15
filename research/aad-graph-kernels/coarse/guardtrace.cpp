// COUNTS-ONLY probe: is desk_mixed's per-tick false-stall refresh (FLK2 half step, then the adaptive stall reads rho = 1) a Hyman
// guard flipping between Newton iterates? A logging engine (a BundleProblem subclass selects it through residual_engine<>) records
// every x the StreamingCalibrator evaluates; each x is replayed through the RECORDED desk_mixed tape, whose 78 guards are exactly
// the MonotoneCubic region's Hyman branches, and the guard outcomes are printed between the streamer's own trace lines.
#include "rec.hpp"
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "kernel.hpp"

#include <cstdio>
#include <map>
#include <set>

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

struct GuardLog {
  aj::Kernel* K = nullptr;
  std::vector<char> anchor, prev;
  std::vector<double> prev_margin;
  long evals = 0;
  std::set<std::vector<char>> patterns;
  int kink_knot = -1;
  void note(char kind, const Eigen::VectorXd& x) {
    if (!K) return;
    ++evals;
    K->forward(x.data());
    const std::size_t G = K->guards.size();
    std::vector<char> bits(G);
    std::vector<double> margin(G);
    for (std::size_t g = 0; g < G; ++g) {
      const auto& gd = K->guards[g];
      const double a = K->val[gd.a], b = K->val[gd.b];
      bool res;
      switch (gd.c) { case aj::LT: res = a < b; break; case aj::LE: res = a <= b; break; case aj::GT: res = a > b; break;
                      case aj::GE: res = a >= b; break; case aj::EQ: res = a == b; break; default: res = a != b; }
      bits[g] = res;
      margin[g] = a - b;
    }
    patterns.insert(bits);
    int vs_anchor = 0, vs_prev = 0;
    std::string flipped;
    for (std::size_t g = 0; g < G; ++g) {
      vs_anchor += bits[g] != anchor[g];
      if (!prev.empty() && bits[g] != prev[g]) {
        ++vs_prev;
        if (vs_prev <= 6) {
          char buf[96];
          std::snprintf(buf, sizeof buf, " g%zu[%s](%+.2e->%+.2e)", g, (K->guards[g].c == aj::NE || K->guards[g].c == aj::EQ) ? "!=" : "cmp",
                        prev_margin[g], margin[g]);
          flipped += buf;
        }
      }
    }
    std::printf("    [%c eval %ld: guards != anchor %d, != previous eval %d%s%s; x[%d]=%.7f]\n", kind, evals, vs_anchor, vs_prev,
                vs_prev ? ":" : "", flipped.c_str(), kink_knot, x[kink_knot]);
    prev = bits;
    prev_margin = margin;
  }
};
inline GuardLog g_log;

struct LProb : cal::BundleProblem {
  LProb() = default;
  explicit LProb(const cal::BundleProblem& p) : cal::BundleProblem(p) {}
};
struct LEngine {
  cal::HybridBundleResidual h;
  explicit LEngine(const cal::BundleProblem& p) : h(p) {}
  int n_residuals() const { return h.n_residuals(); }
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const { g_log.note('r', x); return h.residuals_vs(x, q); }
  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const { return h.residuals(x); }
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J) const { g_log.note('J', x); h.jacobian_vs_into(x, q, J); }
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

int main() {
  static const auto L = swaps::shapes::ladder();
  const swaps::shapes::Shape* dm = nullptr;
  for (const auto& s : L) if (s.name == "desk_mixed") dm = &s;
  const auto& s = *dm;
  const auto& p = s.prob;
  const int n = p.n_knots();
  const Eigen::VectorXd x = cal::calibrate(p, s.x0).x;
  g_log.kink_knot = 11;  // the FLK2 note: knot 11 alternating 0.0341513 / 0.0342819

  LProb lp(p);
  LEngine eng(p);
  using SC = cal::StreamingCalibrator<LProb>;
  SC::Options o;
  o.breakeven_steps = 64;  // pinned, as in scratchpad/audit/probes/probe_dm_trace.cpp (wall-clock independent)
  SC st(eng, lp, x, s.q0, o);
  bool f = false;
  std::printf("#### warm-up (4 ticks, not traced for guards)\n");
  for (int i = 0; i < 4; ++i) { f = !f; st.update(f ? s.q_small : s.q0); }

  // record the tape at the COMMITTED state; its guard outcomes are the anchor pattern
  const Eigen::VectorXd xc = st.current();
  aj::Recorder rec;
  aj::R = &rec;
  std::vector<aj::Rec> X(n);
  for (int i = 0; i < n; ++i) {
    rec.nodes.push_back({aj::INPUT, -1, -1, xc[i]});
    rec.inputs.push_back(static_cast<int>(rec.nodes.size()) - 1);
    X[i] = aj::Rec(xc[i], rec.inputs.back());
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
  K.forward(xc.data());
  g_log.K = &K;
  g_log.anchor.resize(K.guards.size());
  for (std::size_t g = 0; g < K.guards.size(); ++g) {
    const auto& gd = K.guards[g];
    const double a = K.val[gd.a], b = K.val[gd.b];
    bool res;
    switch (gd.c) { case aj::LT: res = a < b; break; case aj::LE: res = a <= b; break; case aj::GT: res = a > b; break;
                    case aj::GE: res = a >= b; break; case aj::EQ: res = a == b; break; default: res = a != b; }
    g_log.anchor[g] = res;
  }
  std::printf("#### tape recorded at the committed state: %zu guards (all in SOFR's MonotoneCubic region)\n", K.guards.size());

  for (int i = 0; i < 6; ++i) {
    f = !f;
    std::printf("#### SMALL TICK %d (%s)\n", i, f ? "q_small" : "q0");
    g_log.prev.clear();
    const cal::StreamTick t = st.update(f ? s.q_small : s.q0);
    std::printf("#### end: conv %d steps %d refreshes %d rescales %d\n", static_cast<int>(t.converged), t.newton_steps, t.refreshes, t.rescales);
  }
  // the committed states of the two ticks, and the 2-cycle's two iterates, compared guard by guard
  std::printf("#### distinct guard patterns seen over %ld evaluations: %zu\n", g_log.evals, g_log.patterns.size());
  // guard margins at the committed state: the smallest |a - b| (how close the committed curve sits to a Hyman kink)
  K.forward(st.current().data());
  std::vector<std::pair<double, std::size_t>> mg;
  for (std::size_t g = 0; g < K.guards.size(); ++g) mg.push_back({std::abs(K.val[K.guards[g].a] - K.val[K.guards[g].b]), g});
  std::sort(mg.begin(), mg.end());
  std::printf("#### smallest guard margins at the final committed state:");
  for (int i = 0; i < 6 && i < static_cast<int>(mg.size()); ++i) std::printf(" g%zu %.2e", mg[i].second, mg[i].first);
  std::printf("\n");
}
