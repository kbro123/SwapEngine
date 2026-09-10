// Drive 10k chained market ticks through the StreamingCalibrator and emit JSON for visualization.
// Proves: (a) it uses the linear fast path while the market stays inside the anchor envelope and
// re-anchors (refreshes the Jacobian) when it drifts out; (b) the streamed curve stays accurate vs
// an independent exact solve; (c) the per-tick time is nanoseconds on the fast path with occasional
// microsecond refresh spikes.
#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <random>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/curve_module.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 10000;
  const std::string out = argc > 2 ? argv[2] : "stream.json";
  const std::string qdump = argc > 3 ? argv[3] : "";  // optional: dump full quote trajectory here

  RelinkableHandle<YieldTermStructure> h;
  auto mk = rb::build_market(h);
  auto prob = rb::build_square_problem(mk);
  const int M = prob.n_knots(), R = prob.n_residuals();

  cal::CompiledResidual cr(prob);
  const int nf = static_cast<int>(prob.meeting_times.size());  // front (meeting) knots
  // Base curve = a realistic policy path, NOT a flat line. The front knots are the instantaneous
  // forward in each inter-meeting segment (the priced-in expected overnight rate), so distinct levels
  // give the FRONT its characteristic staircase -- flat between meetings, jumping at each meeting by
  // the priced expected move. Here: a gentle easing cycle. The back is a smooth mild term-premium rise.
  const double front_base[6] = {0.0433, 0.0422, 0.0412, 0.0404, 0.0398, 0.0393};
  Eigen::VectorXd x0(M);
  for (int k = 0; k < nf; ++k) x0[k] = front_base[k];
  for (int j = 0; j < M - nf; ++j) x0[nf + j] = 0.0390 + 0.0015 * (double(j) / (M - nf - 1));
  const Eigen::VectorXd q0 = cr.model_rates(x0);  // exact anchor: residual(x0)=0 at q0

  cal::StreamingCalibrator::Options opt;  // exact frozen-Newton (the LINEAR mode was deleted in E6.1, 2026-09-10)
  const char* mode = "EXACT (frozen-Newton)";
  cal::StreamingCalibrator sc(prob, x0, q0, opt);

  // Exact reference (fresh AAD Jacobian each Newton step) for accuracy validation.
  auto exact = [&](const Eigen::VectorXd& q, Eigen::VectorXd x) {
    for (int k = 0; k < 8; ++k) {
      const Eigen::VectorXd r = cr.model_rates(x) - q;
      const Eigen::MatrixXd J = cal::aad_jacobian(prob, x);
      const Eigen::VectorXd dx = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(J).solve(r);
      x -= dx;
      if (dx.cwiseAbs().maxCoeff() < 1e-12) break;
    }
    return x;
  };

  // One realistic trading DAY driven by CURVE FACTORS -- level, slopes (2s10s, 10s30s) and butterflies
  // (5-7-10, 10-20-30, 10-15-30) -- so moves are correlated across tenors like a real market, with
  // each rate's intraday high-low range staying ~<=25bp. Each factor loads onto instruments by tenor.
  std::vector<double> qmin(R, 1e9), qmax(R, -1e9);  // per-instrument intraday range

  // We drive a SMOOTH true forward curve x_true (flat 4% + factor moves on the KNOT forwards) and let
  // the instruments quote off it: q = model_rates(x_true). The calibration then recovers x_true exactly
  // (unique solution), so the streamed curve is smooth and stable -- no ill-conditioning wiggle.
  // Knot tenors, in x order (meeting-date front, then back knots).
  std::vector<double> ktau = prob.meeting_times;
  ktau.insert(ktau.end(), prob.back_times.begin(), prob.back_times.end());

  // Factor loadings: piecewise-linear on knot tenor (years), flat outside; flies have zero shoulders.
  auto load = [](double t, const std::vector<std::pair<double, double>>& p) {
    if (t <= p.front().first) return p.front().second;
    for (std::size_t k = 1; k < p.size(); ++k)
      if (t <= p[k].first)
        return p[k - 1].second +
               (p[k].second - p[k - 1].second) * (t - p[k - 1].first) / (p[k].first - p[k - 1].first);
    return p.back().second;
  };
  // Slopes/flies are zero on the front (<1.5y); the FRONT staircase is driven only by policy factors
  // (parallel shift + easing tilt + per-meeting jiggle), applied directly to the meeting knots below.
  const std::vector<std::pair<double, double>>
      LVL{{0, 1}},                                            // parallel (all knots)
      S210{{1.5, 0}, {2, -1}, {10, 1}, {30, 1}},              // 2s10s steepener (front untouched)
      S1030{{10, 0}, {30, 1}},                                // 10s30s (<=10y untouched)
      // Butterflies: gentle, WIDE bumps (soft ramps over several years) so forwards stay smooth --
      // a rate fly is a broad curvature change, not a sharp forward kink.
      F5_7_10{{2, 0}, {7, 1}, {12, 0}}, F10_20_30{{7, 0}, {20, 1}, {30, 0}},
      F10_15_30{{7, 0}, {15, 1}, {28, 0}};

  // Two display series with DIFFERENT grids.
  //  - Par swap rates: a smooth term grid (1y..30y).
  //  - 1-day forward rates: sampled just before/after every meeting date so the piecewise-flat policy
  //    staircase renders with genuinely VERTICAL jumps (two x-points ~1bp-of-a-year apart at each knot).
  const std::vector<double> ptenors = {1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30};
  std::vector<double> ftenors{0.004};
  for (double m : prob.meeting_times) {
    ftenors.push_back(m - 0.004);  // end of the segment BEFORE the meeting
    ftenors.push_back(m + 0.004);  // start of the segment AFTER  the meeting -> vertical jump at m
  }
  for (double t = 0.7; t <= 30.0 + 1e-9; t += (t < 3 ? 0.25 : t < 10 ? 0.5 : 1.0)) ftenors.push_back(t);

  std::vector<std::string> frames;

  // Par swap rates and 1-day forward rates off the actual (Hermite) calibration curve.
  auto curve_json = [&](const Eigen::VectorXd& x) {
    auto c = swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(prob.meeting_times, prob.back_times));
    c.set_forwards(x);
    // Par rate of a T-year annual-fixed OIS: (1 - DF(T)) / sum_i alpha_i DF(t_i).
    auto par = [&](double T) {
      double ann = 0, prev = 0;
      for (double tp = 1.0; tp < T - 1e-9; tp += 1.0) { ann += (tp - prev) * c.discount(tp); prev = tp; }
      ann += (T - prev) * c.discount(T);  // final period (full or stub)
      return (1.0 - c.discount(T)) / ann;
    };
    const double h = 1.0 / 360.0;  // one business day
    auto fwd1d = [&](double t) { return (c.discount(t) / c.discount(t + h) - 1.0) / h; };
    std::string p = "\"p\":[", f = "\"f\":[";
    for (std::size_t k = 0; k < ptenors.size(); ++k) p += (k ? "," : "") + std::to_string(par(ptenors[k]));
    for (std::size_t k = 0; k < ftenors.size(); ++k) f += (k ? "," : "") + std::to_string(fwd1d(ftenors[k]));
    return p + "]," + f + "]";
  };

  // Realistic tick-level feed: the dominant factors random-walk ~0.1-0.2bp PER TICK (mean-reverting so
  // the day stays range-bound), riding a gentle smooth trend for the slower shape factors. This is the
  // granularity a real market-data feed delivers -- not a 0.005bp/tick crawl -- and it is what determines
  // how often the cached Jacobian goes stale.
  std::mt19937 rng(20260711);
  std::normal_distribution<double> z(0.0, 1.0);
  double nL = 0, nFr = 0, n210 = 0;                // OU states (rate units)
  const double kappa = 2.3e-4, sig = 0.13e-4;      // ~0.13bp/tick innovation, ~6bp mean-reverting band
  // Trending day: a large slow DIRECTIONAL move so cumulative drift repeatedly crosses the ~30bp
  // staleness envelope and fires genuine exact-mode Jacobian recomputes. env SWAPS_TREND_BP sets the swing.
  double trendA = 32e-4;
  if (const char* e = std::getenv("SWAPS_TREND_BP")) trendA = std::atof(e) * 1e-4;

  // ---- Phase 1: generate ONE trending day of quotes at a realistic tick granularity ----
  std::vector<Eigen::VectorXd> qs;
  qs.reserve(N);
  std::vector<double> moves;
  for (int t = 0; t < N; ++t) {
    const double ph = 2 * M_PI * t / N;  // slow smooth trend (curve shape morphs across the day)
    const double aF1 = 2e-4 * std::sin(2.1 * ph + 0.5), aF2 = 2e-4 * std::sin(1.7 * ph + 3.0),
                 aF3 = 1.5e-4 * std::sin(2.6 * ph + 4.0), a1030 = 1.5e-4 * std::sin(0.8 * ph + 2.0),
                 aTl = 1.5e-4 * std::sin(0.9 * ph + 1.5);
    nL = (1 - kappa) * nL + sig * z(rng);  // per-tick random-walk innovations (level, front, 2s10s)
    nFr = (1 - kappa) * nFr + 0.7 * sig * z(rng);
    n210 = (1 - kappa) * n210 + 0.6 * sig * z(rng);
    const double aL = 3e-4 * std::sin(ph) + nL, aFr = 2e-4 * std::sin(1.1 * ph + 0.3) + nFr,
                 a210 = 2e-4 * std::sin(1.3 * ph + 1.0) + n210;
    Eigen::VectorXd xt = x0;
    for (int k = nf; k < M; ++k)  // back knots: level + slopes + butterflies (stay smooth)
      xt[k] += aL * load(ktau[k], LVL) + a210 * load(ktau[k], S210) + a1030 * load(ktau[k], S1030) +
               aF1 * load(ktau[k], F5_7_10) + aF2 * load(ktau[k], F10_20_30) + aF3 * load(ktau[k], F10_15_30);
    for (int k = 0; k < nf; ++k)  // front (meeting) knots: policy shift + easing tilt -> staircase
      xt[k] += aL + aFr + aTl * (k - (nf - 1) / 2.0);
    xt.array() += trendA * (0.6 * std::sin(ph) + 0.4 * std::sin(1.7 * ph + 0.6));  // directional trend
    const Eigen::VectorXd q = cr.model_rates(xt);
    for (int i = 0; i < R; ++i) { qmin[i] = std::min(qmin[i], q[i]); qmax[i] = std::max(qmax[i], q[i]); }
    if (!qs.empty()) moves.push_back((q - qs.back()).cwiseAbs().maxCoeff());
    qs.push_back(q);
  }
  double maxrange = 0;
  for (int i = 0; i < R; ++i) maxrange = std::max(maxrange, qmax[i] - qmin[i]);
  std::sort(moves.begin(), moves.end());
  const double move_med = moves[moves.size() / 2], move_p90 = moves[int(0.9 * moves.size())];

  // ---- Phase 2: stream the SAME market through the exact frozen-Newton streamer ----
  const int frame_every = std::max(1, N / 160), time_every = std::max(1, N / 2400);
  struct ModeStats {
    int recalcs = 0, max_steps = 0;
    long total_steps = 0;
    double p50 = 0, p99 = 0, mx = 0, total_ns = 0, max_rt = 0, max_err = 0;
    std::string timeline;
    std::vector<int> recalc_ticks;  // exact tick indices where the Jacobian was recomputed
  };
  auto run_mode = [&](bool want_frames) {
    cal::StreamingCalibrator sc(prob, x0, q0, opt);
    ModeStats st;
    std::vector<double> ns_all;
    ns_all.reserve(N);
    if (want_frames) frames.clear();
    for (int t = 0; t < N; ++t) {
      const auto a = clk::now();
      const cal::StreamTick tk = sc.update(qs[t]);
      const double ns = std::chrono::duration<double, std::nano>(clk::now() - a).count();
      ns_all.push_back(ns);
      st.total_ns += ns;
      st.total_steps += tk.newton_steps;
      st.max_steps = std::max(st.max_steps, tk.newton_steps);
      const double rt = (cr.model_rates(sc.current()) - qs[t]).cwiseAbs().maxCoeff();
      st.max_rt = std::max(st.max_rt, rt);
      if (tk.refreshed) st.recalc_ticks.push_back(t);
      if (t % 5 == 0)
        st.max_err = std::max(st.max_err, (sc.current() - exact(qs[t], sc.current())).cwiseAbs().maxCoeff());
      if (t % time_every == 0)
        st.timeline += (st.timeline.empty() ? "" : ",\n") + std::string("{\"t\":") + std::to_string(t) +
                       ",\"s\":" + std::to_string(tk.newton_steps) + ",\"m\":" + (tk.refreshed ? "1" : "0") +
                       ",\"d\":" + std::to_string(tk.drift) + "}";
      if (want_frames && t % frame_every == 0)
        frames.push_back("{\"t\":" + std::to_string(t) + ",\"r\":" + (tk.refreshed ? "1" : "0") + "," +
                         curve_json(sc.current()) + "}");
    }
    st.recalcs = sc.refresh_count() - 1;
    std::sort(ns_all.begin(), ns_all.end());
    auto pc = [&](double p) { return ns_all[std::min<size_t>(ns_all.size() - 1, size_t(p * ns_all.size()))]; };
    st.p50 = pc(0.5);
    st.p99 = pc(0.99);
    st.mx = pc(1.0);
    return st;
  };
  const ModeStats EX = run_mode(true);  // exact frozen-Newton (+ curve frames)

  // Reference: one full-from-scratch LM solve, to state the equivalent all-cold cost.
  const double full_lm_us = [&] {
    const auto a = clk::now();
    volatile double sink = cal::calibrate(prob, Eigen::VectorXd::Constant(M, 0.035), true).x.sum();
    (void)sink;
    return std::chrono::duration<double, std::micro>(clk::now() - a).count();
  }();

  auto sci = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.3e", v); return std::string(b); };
  auto mode_json = [&](const char* name, const ModeStats& s) {
    std::string o = std::string("\"") + name + "\":{\"recalcs\":" + std::to_string(s.recalcs) +
                    ",\"avg_steps\":" + std::to_string(double(s.total_steps) / N) +
                    ",\"max_steps\":" + std::to_string(s.max_steps) +
                    ",\"p50_ns\":" + std::to_string(s.p50) + ",\"p99_ns\":" + std::to_string(s.p99) +
                    ",\"max_ns\":" + std::to_string(s.mx) + ",\"total_ms\":" + std::to_string(s.total_ns / 1e6) +
                    ",\"max_rt_bp\":" + sci(s.max_rt * 1e4) + ",\"max_err_bp\":" + sci(s.max_err * 1e4) +
                    ",\"recalc_ticks\":[";
    for (std::size_t k = 0; k < s.recalc_ticks.size(); ++k) o += (k ? "," : "") + std::to_string(s.recalc_ticks[k]);
    o += "],\"timeline\":[\n" + s.timeline + "\n]}";
    return o;
  };

  std::ofstream f(out);
  f.setf(std::ios::fixed);
  f << "{\n";
  f << "\"n_ticks\":" << N << ",\"trend_bp\":" << trendA * 1e4 << ",\"stale_bp\":30,\n";
  f << "\"move_median_bp\":" << move_med * 1e4 << ",\"move_p90_bp\":" << move_p90 * 1e4
    << ",\"range_bp\":" << maxrange * 1e4 << ",\n";
  f << "\"full_lm_us\":" << full_lm_us << ",\"full_lm_equiv_s\":" << full_lm_us * N / 1e6 << ",\n";
  f << "\"ptenors\":[";
  for (std::size_t k = 0; k < ptenors.size(); ++k) f << (k ? "," : "") << ptenors[k];
  f << "],\n\"ftenors\":[";
  for (std::size_t k = 0; k < ftenors.size(); ++k) f << (k ? "," : "") << ftenors[k];
  f << "],\n\"frames\":[\n";
  for (std::size_t k = 0; k < frames.size(); ++k) f << (k ? ",\n" : "") << frames[k];
  f << "\n],\n" << mode_json("exact", EX) << "\n}\n";

  std::printf("trending day: trend=%.0fbp  range=%.1fbp  per-tick move median=%.3fbp p90=%.3fbp\n",
              trendA * 1e4, maxrange * 1e4, move_med * 1e4, move_p90 * 1e4);
  std::printf("EXACT : recalcs=%d  avg_steps=%.2f  max_steps=%d  p50=%.0fns p99=%.0fns  round-trip=%.1e bp  err=%.1e bp\n",
              EX.recalcs, double(EX.total_steps) / N, EX.max_steps, EX.p50, EX.p99, EX.max_rt * 1e4, EX.max_err * 1e4);
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
