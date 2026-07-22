// Full multi-currency day simulation for the phone artifact. Streams the 7-curve compiled bundle
// (SOFR + FF + PRIME + ESTR + EUR3M + EUR6M + EONIA) at 1 tick/second for a 23h trading day, recording
// the COMPILED calibration time every tick. Every 10 simulated minutes it snapshots the SOFR + ESTR
// forward curves (knot-aware grid, so the piecewise-flat policy-step front renders sharp), reprices a
// 10,000-swap portfolio (40% USD SOFR book, 40% EUR ESTR book -- both on the vectorized CompiledPortfolio
// kernel -- and 20% EURUSD FX-forwards), and records the portfolio NPV + the reprice time. Emits full_day.json.
//
// Build (not in CMake; links the vendored QuantLib static lib). On the Mac Pro set the 14.5-SDK libc++:
//   export CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX14.5.sdk/usr/include/c++/v1
//   c++ -std=c++20 -O3 -march=native -DEIGEN_ENABLE_AVX512 \
//       -I include -I tests -I third_party/eigen -I third_party/boost -I third_party/quantlib/install/include \
//       tools/full_day_sim.cpp third_party/quantlib/install/lib/libQuantLib.a -o full_day_sim
//   ./full_day_sim
#include <ql/quantlib.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "reference_multicurrency.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/portfolio.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

int main() {
  // 7-curve W-cacheable bundle with the FULL-RANK EUR block (outright-pinned) so it streams on the
  // compiled fast path (the coupled basis-only trio is rank-deficient -> cannot stream; see the artifact note).
  rb::MultiCcyBundle b = rb::build_full_multicurrency(/*include_xccy=*/false, /*coupled_eur=*/false);
  const cal::BundleProblem& prob = b.prob;
  const int NC = b.n_curves();
  const int N = prob.n_knots();
  const int SOFR = 0, ESTR = 3, EUR3M = 4, EUR6M = 5;
  const char* names[] = {"SOFR", "FF", "PRIME", "ESTR", "EUR3M", "EUR6M"};
  const double S = b.fx_spot, kEurUsdBasis = -0.0015;  // EUR-in-USD ~= ESTR - 15bp (fixed for the fx sleeve)
  const DayCounter dc = b.dc;

  cal::CompiledBundleResidual engine(prob);
  Eigen::VectorXd x_true = b.x_true;                     // self-consistent (r(x_true)=0)
  Eigen::VectorXd x_drive = x_true;                      // the "true" market curve that random-walks
  const Eigen::VectorXd q0 = engine.model_rates(x_true);
  cal::StreamingCalibrator<cal::BundleProblem> sc(prob, x_true, q0,
                                                  cal::StreamingCalibrator<cal::BundleProblem>::Options{});

  // Per-curve global offsets + knot times.
  std::vector<int> off(NC, 0);
  for (int c = 1; c < NC; ++c) off[c] = off[c - 1] + prob.curves[c - 1].n_knots();
  auto block = [&](const Eigen::VectorXd& x, int c) { return x.segment(off[c], prob.curves[c].n_knots()); };

  // ------------------------- Build the 10,000-swap portfolio -------------------------
  Settings::instance().evaluationDate() = b.today;
  std::mt19937 rng(20260721u);
  // A currency sleeve as an OIS book on one self-discounting curve (SOFR or ESTR).
  auto make_book = [&](const ext::shared_ptr<OvernightIndex>& idx, int role, int n) {
    struct Tmpl { std::vector<px::FloatCoupon> fl; std::vector<px::FixedCoupon> fx; double par; };
    std::vector<Tmpl> tm;
    for (int T : {1, 2, 3, 5, 7, 10, 15, 20, 30}) {
      auto o = ext::shared_ptr<OvernightIndexedSwap>(
          MakeOIS(T * Years, idx, 0.03).withDiscountingTermStructure(b.h[role]));
      o->deepUpdate();
      const auto fl = swaps::qlx::extract_float_leg(o->overnightLeg(), b.today, dc);
      const auto fx = swaps::qlx::extract_fixed_leg(o->fixedLeg(), b.today, dc);
      const double par = px::float_leg_pv<double>(fl, *b.curve_handles[role], *b.curve_handles[role]) /
                         px::annuity<double>(fx, *b.curve_handles[role]);
      tm.push_back({fl, fx, par});
    }
    std::uniform_int_distribution<int> pick(0, static_cast<int>(tm.size()) - 1);
    std::uniform_real_distribution<double> notl(1e6, 1e7), u01(0.0, 1.0);
    pf::Portfolio p;
    for (int i = 0; i < n; ++i) {
      const auto& t = tm[pick(rng)];
      const double sgn = (u01(rng) < 0.55) ? 1.0 : -1.0;  // 55% payer / 45% receiver => modest net long duration
      p.positions.push_back({t.fl, t.fx, t.par, sgn * notl(rng)});  // at-par => NPV starts ~0, PnL from rate moves
    }
    return p;
  };
  const int N_USD = 4000, N_EUR = 4000, N_FX = 2000;
  pf::Portfolio usd_pf = make_book(b.sofr, SOFR, N_USD);
  pf::Portfolio eur_pf = make_book(b.estr, ESTR, N_EUR);
  pf::CompiledPortfolio usd_book(prob.curves[SOFR].meeting, prob.curves[SOFR].back, usd_pf);
  pf::CompiledPortfolio eur_book(prob.curves[ESTR].meeting, prob.curves[ESTR].back, eur_pf);
  // FX-forward sleeve: strike = the forward at x_true (so each starts ~0 and moves as SOFR/ESTR move).
  struct FxFwd { double t, K, notl; };
  std::vector<FxFwd> fxbook;
  auto fx_fwd = [&](double t, const std::vector<std::unique_ptr<cal::CurveHandle<double>>>& C) {
    return S * (C[ESTR]->discount(t) * std::exp(-kEurUsdBasis * t)) / C[SOFR]->discount(t);  // S·DF_c/DF_S
  };
  {
    auto C0 = cal::build_bundle_curves<double>(prob.curves, [&](int c, int i) { return x_true[off[c] + i]; });
    std::uniform_real_distribution<double> ften(0.25, 10.0), fnotl(1e6, 1e7), u01(0.0, 1.0);
    for (int i = 0; i < N_FX; ++i) {
      const double tt = ften(rng);
      const double sgn = (u01(rng) < 0.55) ? 1.0 : -1.0;
      fxbook.push_back({tt, fx_fwd(tt, C0), sgn * fnotl(rng)});
    }
  }
  auto reprice = [&](const Eigen::VectorXd& x, double& usd, double& eur, double& fxv) {
    const Eigen::VectorXd xs = block(x, SOFR), xe = block(x, ESTR);
    usd = usd_book.total_npv(xs);
    eur = S * eur_book.total_npv(xe);  // EUR NPV -> USD at spot
    auto C = cal::build_bundle_curves<double>(prob.curves, [&](int c, int i) { return x[off[c] + i]; });
    fxv = 0;
    for (const auto& f : fxbook) fxv += f.notl * (fx_fwd(f.t, C) - f.K) * C[SOFR]->discount(f.t);
  };

  // ------------------------- Display grid (knot-aware, for the sharp flat front) -------------------------
  auto knots_of = [&](int c) {
    std::vector<double> k;
    for (double t : prob.curves[c].meeting) k.push_back(t);
    for (double t : prob.curves[c].back) k.push_back(t);
    return k;
  };
  const double DAY = 1.0 / 365.0, WEEK = 7.0 / 365.0;
  std::vector<double> allk;
  for (int c : {SOFR, ESTR}) for (double t : knots_of(c)) allk.push_back(t);
  std::sort(allk.begin(), allk.end());
  allk.erase(std::unique(allk.begin(), allk.end(), [](double a, double c) { return std::abs(a - c) < 1e-9; }), allk.end());
  std::vector<double> grid{DAY};
  for (std::size_t k = 0; k < allk.size(); ++k) {
    const double kt = allk[k];
    if (kt <= 1.6) { grid.push_back(std::max(DAY, kt - DAY)); grid.push_back(kt + DAY); }
    else grid.push_back(kt);
    const double next = (k + 1 < allk.size()) ? allk[k + 1] : 30.0;
    const double gap = next - kt, step = gap < 0.6 ? WEEK : (gap < 2.5 ? 3 * WEEK : gap / 2.2);
    for (double t = kt + step; t < next - 0.6 * step; t += step) grid.push_back(t);
  }
  std::sort(grid.begin(), grid.end());
  grid.erase(std::unique(grid.begin(), grid.end(), [](double a, double c) { return std::abs(a - c) < 1e-9; }), grid.end());
  auto fwd_on_grid = [&](const Eigen::VectorXd& x, int c) {
    auto C = cal::build_bundle_curves<double>(prob.curves, [&](int cc, int i) { return x[off[cc] + i]; });
    std::vector<double> out(grid.size());
    for (std::size_t g = 0; g < grid.size(); ++g) out[g] = 100.0 * C[c]->forward(grid[g]);
    return out;
  };

  // ------------------------- Run the day -------------------------
  const int TICKS = 23 * 3600;       // 23h at 1 tick/s
  const int snap_every = TICKS / 138;  // ~ every 10 simulated minutes
  const double bp = 1e-4, kappa = 0.0006;
  std::normal_distribution<double> level_move(0.15, 0.25), slope_move(0.0, 0.05), spread_move(0.0, 0.04);
  std::uniform_int_distribution<int> sign(0, 1);
  double level = 0.0;
  std::vector<double> spread(NC, 0.0);
  std::vector<double> knot_t0 = knots_of(SOFR);  // for slope normalization
  double maxT = 0;
  for (int c = 0; c < NC; ++c) for (double t : knots_of(c)) maxT = std::max(maxT, t);

  std::vector<double> cal_us;
  cal_us.reserve(TICKS);
  double sum_us = 0, min_us = 1e18, max_us = 0, sum_rep = 0, min_rep = 1e18, max_rep = 0;
  struct Snap { double hour, npv, cal_us, rep_us, npv_usd, npv_eur, npv_fx; std::vector<double> sofr_fwd, estr_fwd; };
  std::vector<Snap> snaps;
  std::vector<double> ser_hour, ser_npv, ser_cal, ser_level;

  for (int tk = 0; tk < TICKS; ++tk) {
    // Drive SOFR with a mean-reverting level + gentle slope; each spread curve a small reverting move.
    const double dL = (sign(rng) ? 1.0 : -1.0) * level_move(rng) - kappa * level, dS = slope_move(rng);
    level += dL;
    for (int c = 0; c < NC; ++c) {
      const bool base = (c == SOFR || c == ESTR);
      const double ds = base ? dL : (spread_move(rng) - kappa * spread[c]);
      const double dslope = base ? dS : 0.4 * slope_move(rng);
      if (!base) spread[c] += ds;
      auto kt = knots_of(c);
      for (int i = 0; i < prob.curves[c].n_knots(); ++i) {
        const double tn = kt[i] / maxT;
        x_drive[off[c] + i] += (ds + dslope * (tn - 0.5) * 2.0) * bp;
      }
    }
    const Eigen::VectorXd q = engine.model_rates(x_drive);
    const auto t0 = std::chrono::steady_clock::now();
    sc.update(q);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    cal_us.push_back(us);
    sum_us += us; min_us = std::min(min_us, us); max_us = std::max(max_us, us);

    if (tk % snap_every == 0 && static_cast<int>(snaps.size()) < 138) {
      const Eigen::VectorXd x = sc.current();
      double u, e, f;
      const auto r0 = std::chrono::steady_clock::now();
      reprice(x, u, e, f);
      const double rep = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - r0).count();
      sum_rep += rep; min_rep = std::min(min_rep, rep); max_rep = std::max(max_rep, rep);
      const double hour = 23.0 * tk / TICKS, npv = u + e + f;
      snaps.push_back({hour, npv, us, rep, u, e, f, fwd_on_grid(x, SOFR), fwd_on_grid(x, ESTR)});
      ser_hour.push_back(hour); ser_npv.push_back(npv); ser_cal.push_back(us); ser_level.push_back(level);
    }
  }

  std::vector<double> sorted = cal_us;
  std::sort(sorted.begin(), sorted.end());
  auto pct = [&](double p) { return sorted[std::min<std::size_t>(sorted.size() - 1, (std::size_t)(p * sorted.size()))]; };

  // ------------------------- Emit JSON -------------------------
  std::ofstream fjs("full_day.json");
  fjs.setf(std::ios::fixed);
  fjs << "{\n  \"ticks\":" << TICKS << ",\"hours\":23,\"snap_every_min\":10,\n";
  fjs << "  \"curves\":[\"SOFR\",\"FF\",\"PRIME\",\"ESTR\",\"EUR3M\",\"EUR6M\"],\"n_knots\":" << N
      << ",\"n_instruments\":" << prob.n_residuals() << ",\n";
  fjs << "  \"portfolio\":{\"n\":" << (N_USD + N_EUR + N_FX) << ",\"usd\":" << N_USD << ",\"eur\":" << N_EUR
      << ",\"fx\":" << N_FX << "},\n";
  fjs.precision(3);
  fjs << "  \"cal_us\":{\"min\":" << min_us << ",\"avg\":" << sum_us / TICKS << ",\"median\":" << pct(0.5)
      << ",\"p99\":" << pct(0.99) << ",\"max\":" << max_us << ",\"refreshes\":" << sc.refresh_count() - 1 << "},\n";
  fjs << "  \"reprice_us\":{\"min\":" << min_rep << ",\"avg\":" << sum_rep / snaps.size() << ",\"max\":" << max_rep << "},\n";
  auto arr = [&](const std::vector<double>& v) {
    fjs << "[";
    for (std::size_t i = 0; i < v.size(); ++i) fjs << (i ? "," : "") << v[i];
    fjs << "]";
  };
  fjs.precision(5);
  fjs << "  \"grid\":"; arr(grid); fjs << ",\n";
  fjs << "  \"sofr_knots\":"; arr(knots_of(SOFR)); fjs << ",\n";
  fjs << "  \"estr_knots\":"; arr(knots_of(ESTR)); fjs << ",\n";
  fjs.precision(4);
  fjs << "  \"snapshots\":[\n";
  for (std::size_t s = 0; s < snaps.size(); ++s) {
    const auto& sn = snaps[s];
    fjs << "    {\"hour\":" << sn.hour << ",\"npv\":" << sn.npv / 1e6 << ",\"npv_usd\":" << sn.npv_usd / 1e6
        << ",\"npv_eur\":" << sn.npv_eur / 1e6 << ",\"npv_fx\":" << sn.npv_fx / 1e6 << ",\"cal_us\":" << sn.cal_us
        << ",\"rep_us\":" << sn.rep_us << ",\"sofr\":"; arr(sn.sofr_fwd); fjs << ",\"estr\":"; arr(sn.estr_fwd);
    fjs << "}" << (s + 1 < snaps.size() ? "," : "") << "\n";
  }
  fjs << "  ],\n";
  fjs << "  \"series\":{\"hour\":"; arr(ser_hour); fjs << ",\"npv\":";
  { std::vector<double> v; for (double x : ser_npv) v.push_back(x / 1e6); arr(v); }
  fjs << ",\"cal_us\":"; arr(ser_cal); fjs << "}\n}\n";
  fjs.close();

  std::printf("ticks=%d  cal_us min=%.2f avg=%.2f median=%.2f p99=%.2f max=%.2f  refreshes=%d\n",
              TICKS, min_us, sum_us / TICKS, pct(0.5), pct(0.99), max_us, sc.refresh_count() - 1);
  std::printf("portfolio=%d (USD %d + EUR %d + FX %d)  reprice_us min=%.1f avg=%.1f max=%.1f  snapshots=%zu\n",
              N_USD + N_EUR + N_FX, N_USD, N_EUR, N_FX, min_rep, sum_rep / snaps.size(), max_rep, snaps.size());
  return 0;
}
