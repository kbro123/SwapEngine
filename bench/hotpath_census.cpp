// hotpath_census — E3.1: per PRODUCTION path (the BundleSession calls a client makes), how many heap
// allocations, bytes and frees each call performs after warm-up, and its time. Counts EVERY allocation in the
// process through libmalloc's logger hook (malloc_logger), not Eigen's guard alone — std::vector, std::string,
// boost::json and any std::function inside the kernels all show up. Ours-only, QuantLib-free, desk-scale
// fixtures shared with the gated benches (8-curve x 26-knot spread chain + 200-swap book; the 17-cell SOFR
// swaption cube). Prints a table and writes JSON to argv[1] (default build/perf/hotpath_census.json).
//
// A path that claims "allocation-free per tick" (CLAUDE.md §5) must show allocs_per_call == 0 here; the T4
// hot-path-invariant tests (E5) turn those rows into assertions. Run on a quiesced machine.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"

extern "C" {
typedef void(malloc_logger_t)(uint32_t type, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t result,
                              uint32_t num_hot_frames_to_skip);
extern malloc_logger_t* malloc_logger;  // libmalloc's per-allocation hook (all zones, all threads)
}

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {

struct Counters { unsigned long allocs = 0, bytes = 0, frees = 0; };
Counters g_c;
void logger(uint32_t type, uintptr_t, uintptr_t a2, uintptr_t a3, uintptr_t, uint32_t) {
  if (type & 2) { ++g_c.allocs; g_c.bytes += (type & 4) ? a3 : a2; }  // realloc carries size in arg3
  if (type & 4) ++g_c.frees;
}

struct Row {
  std::string path;
  double allocs = 0, bytes = 0, frees = 0, us_min = 0, us_mean = 0;
  int calls = 0;
  std::string note;
};

// Warm up `warm` times, then measure `n` calls: allocation counters bracket each call (the logger is armed
// only inside the call so the harness's own vectors are not counted); time is wall-clock per call.
template <class F>
Row census(const std::string& name, F&& f, int warm = 3, int n = 20, std::string note = "") {
  for (int i = 0; i < warm; ++i) f();
  Row r; r.path = name; r.calls = n; r.note = std::move(note);
  double sum = 0, mn = 1e300;
  Counters tot;
  for (int i = 0; i < n; ++i) {
    g_c = Counters{};
    const auto t0 = std::chrono::steady_clock::now();
    malloc_logger = logger;
    f();
    malloc_logger = nullptr;
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    tot.allocs += g_c.allocs; tot.bytes += g_c.bytes; tot.frees += g_c.frees;
    sum += us; mn = std::min(mn, us);
  }
  r.allocs = double(tot.allocs) / n; r.bytes = double(tot.bytes) / n; r.frees = double(tot.frees) / n;
  r.us_min = mn; r.us_mean = sum / n;
  std::printf("  %-44s allocs/call %9.1f  bytes/call %11.0f  frees/call %9.1f  us min %10.1f  mean %10.1f  %s\n",
              r.path.c_str(), r.allocs, r.bytes, r.frees, r.us_min, r.us_mean, r.note.c_str());
  return r;
}

// ---- desk-scale spread-chain fixture (identical to bench/session_warm_bench.cpp) ----------------------------
constexpr int NC = 8, NK = 26;
constexpr double MAX_T = 30.0;
struct Legs { std::vector<px::FloatCoupon> flt; std::vector<px::FixedCoupon> fix; };
Legs annual(double T) {
  Legs L; double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c; c.obs.sub_start = {prev}; c.obs.sub_end = {u}; c.obs.tau_index = u - prev; c.pay = u; c.tau_pay = u - prev;
    L.flt.push_back(c); L.fix.push_back({u, u - prev}); prev = u;
  }
  return L;
}
cal::Instrument par_inst(double T, int fc, int dc) {
  Legs L = annual(T); cal::Instrument in; in.quote = cal::QuoteKind::ParRate; in.fwd = {L.flt, fc, dc}; in.fixed = {L.fix, dc}; return in;
}
cal::Instrument basis_inst(double T, int fc, int bc, int dc) {
  Legs L = annual(T); cal::Instrument in; in.quote = cal::QuoteKind::ParSpread; in.fwd = {L.flt, fc, dc}; in.bench = {L.flt, bc, dc}; in.fixed = {L.fix, dc}; return in;
}
struct Fixture {
  cal::BundleProblem prob, pert;
  Eigen::VectorXd x0, q0, q1, q_big;
  api::RegSpec tension;
  swaps::portfolio::MultiCurveBook book;
  Fixture() {
    std::vector<double> meeting{0.25}, back;
    for (int i = 1; i <= NK - 1; ++i) back.push_back(MAX_T * i / (NK - 1));
    prob.curves.resize(NC);
    prob.curves[0] = px::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};
    for (int c = 1; c < NC; ++c) prob.curves[c] = px::CurveStructure{.base = c - 1, .regions = swaps::curve::flat_hermite(meeting, back)};
    std::vector<double> mats;
    for (double T = 1.0; T <= MAX_T + 1e-9; T += 1.0) mats.push_back(T);
    for (double T : mats) prob.instruments.push_back(par_inst(T, 0, 0));
    for (int c = 1; c < NC; ++c) for (double T : mats) prob.instruments.push_back(basis_inst(T, c, c - 1, 0));
    Eigen::VectorXd x_true(NC * NK);
    for (int c = 0; c < NC; ++c) for (int i = 0; i < NK; ++i) x_true[c * NK + i] = (c == 0) ? 0.040 + 0.0005 * i : 0.0020 + 0.0001 * i;
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < int(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];
    x0.resize(NC * NK);
    for (int c = 0; c < NC; ++c) for (int i = 0; i < NK; ++i) x0[c * NK + i] = (c == 0) ? 0.040 : 0.0020;
    pert = prob;
    for (int i = 0; i < int(pert.instruments.size()); ++i) pert.instruments[i].market += 1e-4 * std::sin(0.7 * i + 0.3);
    const int m = int(prob.instruments.size());
    q0.resize(m); q1.resize(m); q_big.resize(m);
    for (int i = 0; i < m; ++i) { q0[i] = prob.instruments[i].market; q1[i] = q0[i] + 1e-5 * std::sin(0.7 * i + 0.3); q_big[i] = q0[i] + 25e-4 * std::sin(0.3 * i + 0.1); }
    tension.lambda = 1e-3; tension.tension = true; tension.sigma = 0.5;
    for (int c = 0; c < NC; ++c) tension.curves.push_back(c);
    for (int i = 0; i < 200; ++i) {
      const double T = 1.0 + (i % 30); Legs L = annual(T);
      swaps::portfolio::MultiCurveBook::Position p;
      p.kind = swaps::portfolio::MultiCurveBook::Kind::Swap; p.notional = (i % 2 ? 1.0 : -1.0) * (1.0 + 0.01 * i);
      p.float_coupons = L.flt; p.fixed_coupons = L.fix; p.fwd_curve = i % NC; p.disc_curve = 0; p.fixed_curve = 0; p.fixed_rate = 0.04;
      book.positions.push_back(std::move(p));
    }
  }
};

// ---- the SOFR swaption-cube fixture (bench/vol_cube_bench.cpp) ----------------------------------------------
std::string sofr_bundle_json() { std::ifstream f(SOFR_BUNDLE_JSON); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
api::VolCubeSpec cube_spec() {
  api::VolCubeSpec s; s.value_date = "2026-07-08"; s.index = "USD-SOFR";
  api::VolCubeCell smile; smile.expiry = "1Y"; smile.tenor = "5Y"; smile.has_sabr = true; smile.sabr_alpha = 0.009; smile.sabr_rho = -0.25; smile.sabr_nu = 0.45;
  for (int bp = -150; bp <= 150; bp += 15) smile.moneyness_bp.push_back(bp);
  s.cells.push_back(std::move(smile));
  const char* exps[] = {"1Y", "2Y", "5Y", "10Y"}; const char* tens[] = {"2Y", "5Y", "10Y", "30Y"};
  for (const char* e : exps) for (const char* t : tens) {
    api::VolCubeCell c; c.expiry = e; c.tenor = t; c.has_sabr = true; c.sabr_alpha = 0.009; c.sabr_rho = -0.25; c.sabr_nu = 0.45; c.atm = true; s.cells.push_back(std::move(c));
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out_path = argc > 1 ? argv[1] : "build/perf/hotpath_census.json";
  const Fixture f;
  std::vector<Row> rows;
  std::printf("hotpath_census — chain8x26 (%d instruments, %d knots), book200, sofr swaption cube17\n",
              int(f.prob.instruments.size()), f.prob.n_knots());

  // Cold: construct + calibrate (allocation is expected here; the number is the reference for 'warm' claims).
  rows.push_back(census("chain8x26_cold_build_calibrate", [&] { api::BundleSession s(f.prob); s.calibrate(f.x0); }, 1, 5));

  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  bool flip = false;
  rows.push_back(census("chain8x26_rebind_warm", [&] { flip = !flip; sess.rebind(flip ? f.pert : f.prob); }));
  {
    api::BundleSession st(f.prob);
    st.calibrate(f.x0, f.tension);
    bool fl = false;
    rows.push_back(census("chain8x26_rebind_tension_warm", [&] { fl = !fl; st.rebind(fl ? f.pert : f.prob, f.tension); }));
  }
  rows.push_back(census("chain8x26_jacobian_tension", [&] { volatile double k = sess.jacobian(f.tension)(0, 0); (void)k; }));
  rows.push_back(census("chain8x26_risk_operator_tension", [&] { volatile double k = sess.risk_operator(f.tension)(0, 0); (void)k; }));
  rows.push_back(census("chain8x26_sample_6_times", [&] { auto s = sess.sample({0.5, 1, 2, 5, 10, 30}); volatile double k = s[0].discount.empty() ? 0.0 : s[0].discount[0]; (void)k; }));

  rows.push_back(census("chain8x26_book200_price_portfolio_oneshot", [&] { volatile double k = sess.price_portfolio(f.book).npv; (void)k; }));
  sess.bind_portfolio(f.book);
  rows.push_back(census("chain8x26_book200_reprice_bound", [&] { volatile double k = sess.reprice_bound().npv; (void)k; }));
  rows.push_back(census("chain8x26_book200_price_portfolio_risk", [&] { volatile double k = sess.price_portfolio_risk(f.book, f.tension).npv; (void)k; }, 2, 10));

  {
    api::BundleSession ss(f.prob);
    ss.calibrate(f.x0);
    ss.start_streaming();
    bool fl = false;
    auto tick_note = [&] { return "last tick: newton=" + std::to_string(ss.last_newton_steps()) + " refreshes=" + std::to_string(ss.last_refreshes()) + " rescales=" + std::to_string(ss.last_rescales()); };
    for (int i = 0; i < 5; ++i) { fl = !fl; ss.stream_update(fl ? f.q1 : f.q0); }
    rows.push_back(census("chain8x26_stream_tick_0p1bp", [&] { fl = !fl; volatile double k = ss.stream_update(fl ? f.q1 : f.q0)[0]; (void)k; }, 0, 50, tick_note()));
    bool big = false;
    for (int i = 0; i < 2; ++i) { big = !big; ss.stream_update(big ? f.q_big : f.q0); }
    rows.push_back(census("chain8x26_stream_tick_25bp_move", [&] { big = !big; volatile double k = ss.stream_update(big ? f.q_big : f.q0)[0]; (void)k; }, 0, 20, tick_note()));
  }

  {
    api::BundleSession vs(api::bundle_from_json(boost::json::parse(sofr_bundle_json())));
    api::RegSpec reg; reg.tension = true; reg.lambda = 0.02; reg.curves = {0};
    vs.calibrate(api::flat_x0(vs.problem()), reg);
    const api::VolCubeSpec spec = cube_spec();
    rows.push_back(census("sofr_swaption_cube17_price_vol_cube_warm", [&] { volatile double k = vs.price_vol_cube(spec).price[0]; (void)k; }));
    rows.push_back(census("sofr_swaption_cube17_cold_build_calibrate_price", [&] {
      api::BundleSession c(api::bundle_from_json(boost::json::parse(sofr_bundle_json())));
      c.calibrate(api::flat_x0(c.problem()), reg);
      volatile double k = c.price_vol_cube(spec).price[0]; (void)k; }, 1, 5, "includes JSON parse"));
  }

  boost::json::object o;
  for (const auto& r : rows)
    o[r.path] = boost::json::object{{"allocs_per_call", r.allocs}, {"bytes_per_call", r.bytes}, {"frees_per_call", r.frees},
                                    {"us_min", r.us_min}, {"us_mean", r.us_mean}, {"calls", r.calls}, {"note", r.note}};
  std::ofstream(out_path) << boost::json::serialize(o) << "\n";
  std::printf("wrote %s\n", out_path.c_str());
  return 0;
}
