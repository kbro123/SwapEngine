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
#include <fstream>
#include <random>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/two_region_forward_curve.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 10000;
  const std::string out = argc > 2 ? argv[2] : "stream.json";

  RelinkableHandle<YieldTermStructure> h;
  auto mk = rb::build_market(h);
  auto prob = rb::build_square_problem(mk);
  const int M = prob.n_knots(), R = prob.n_residuals();

  cal::CompiledResidual cr(prob);
  Eigen::VectorXd x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(M, 0.035), true).x;
  const Eigen::VectorXd q0 = cr.model_rates(x0);  // exact anchor: residual(x0)=0 at q0

  cal::StreamingCalibrator::Options opt;
  opt.envelope = 0.35e-4;
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

  // Driven market: slow level + slope oscillation (visible curve motion) + small random-walk noise.
  std::mt19937 rng(12345);
  std::normal_distribution<double> step(0.0, 0.035e-4);  // ~0.08bp/tick noise
  std::vector<double> s(R), rw(R, 0.0);
  for (int i = 0; i < R; ++i) s[i] = 2.0 * i / (R - 1) - 1.0;  // -1 front .. +1 back (twist weight)

  // Curve sampling grid (year fractions) for the animation.
  std::vector<double> tenors;
  for (double t = 0.08; t <= 30.0; t += (t < 2 ? 0.08 : 0.5)) tenors.push_back(t);

  const int n_frames = 160, frame_every = std::max(1, N / n_frames);
  const int n_time = 2400, time_every = std::max(1, N / n_time);

  std::vector<std::string> frames, timeline;
  long n_linear = 0, n_refresh = 0;
  double total_ns = 0, lin_ns = 0, ref_ns = 0, max_err = 0;

  auto fwd_json = [&](const Eigen::VectorXd& x) {
    swaps::curve::TwoRegionForwardCurve<double> c(prob.meeting_times, prob.back_times);
    c.set_forwards(x);
    std::string j = "[";
    for (std::size_t k = 0; k < tenors.size(); ++k)
      j += (k ? "," : "") + std::to_string(c.forward(tenors[k]));
    return j + "]";
  };

  for (int t = 0; t < N; ++t) {
    const double phase = 2 * M_PI * t / 5000.0;
    const double level = 22e-4 * std::sin(phase);
    const double slope = 13e-4 * std::sin(1.3 * phase + 0.6);
    Eigen::VectorXd q(R);
    for (int i = 0; i < R; ++i) {
      rw[i] += step(rng);
      q[i] = q0[i] + level + slope * s[i] + rw[i];
    }

    const auto a = clk::now();
    const cal::StreamTick tick = sc.update(q);
    const double ns = std::chrono::duration<double, std::nano>(clk::now() - a).count();

    total_ns += ns;
    if (tick.refreshed) {
      ++n_refresh;
      ref_ns += ns;
    } else {
      ++n_linear;
      lin_ns += ns;
    }

    // accuracy vs exact (every 5th tick to bound validation cost)
    double err = -1;
    if (t % 5 == 0) {
      err = (sc.current() - exact(q, sc.current())).cwiseAbs().maxCoeff();
      max_err = std::max(max_err, err);
    }

    if (t % frame_every == 0)
      frames.push_back("{\"t\":" + std::to_string(t) + ",\"r\":" + (tick.refreshed ? "1" : "0") +
                       ",\"f\":" + fwd_json(sc.current()) + "}");
    if (t % time_every == 0)
      timeline.push_back("{\"t\":" + std::to_string(t) + ",\"ns\":" + std::to_string(ns) +
                         ",\"m\":" + (tick.refreshed ? "1" : "0") + ",\"d\":" +
                         std::to_string(tick.drift) +
                         ",\"e\":" + std::to_string(err) + "}");
  }

  // Reference: one full-from-scratch LM solve, to state the equivalent all-cold cost.
  const double full_lm_us = [&] {
    const auto a = clk::now();
    volatile double sink = cal::calibrate(prob, Eigen::VectorXd::Constant(M, 0.035), true).x.sum();
    (void)sink;
    return std::chrono::duration<double, std::micro>(clk::now() - a).count();
  }();

  std::ofstream f(out);
  f.setf(std::ios::fixed);
  f << "{\n";
  f << "\"n_ticks\":" << N << ",\"n_linear\":" << n_linear << ",\"n_refresh\":" << n_refresh << ",\n";
  f << "\"stream_total_ms\":" << total_ns / 1e6 << ",\"avg_linear_ns\":" << (lin_ns / std::max(1L, n_linear))
    << ",\"avg_refresh_us\":" << (ref_ns / std::max(1L, n_refresh)) / 1e3 << ",\n";
  f << "\"max_error\":" << max_err << ",\"envelope_bp\":" << opt.envelope * 1e4 << ",\n";
  f << "\"full_lm_us\":" << full_lm_us << ",\"full_lm_equiv_s\":" << full_lm_us * N / 1e6 << ",\n";
  f << "\"tenors\":[";
  for (std::size_t k = 0; k < tenors.size(); ++k) f << (k ? "," : "") << tenors[k];
  f << "],\n\"frames\":[\n";
  for (std::size_t k = 0; k < frames.size(); ++k) f << (k ? ",\n" : "") << frames[k];
  f << "\n],\n\"timeline\":[\n";
  for (std::size_t k = 0; k < timeline.size(); ++k) f << (k ? ",\n" : "") << timeline[k];
  f << "\n]}\n";

  std::printf("ticks=%d  linear=%ld (%.2f%%)  refresh=%ld\n", N, n_linear, 100.0 * n_linear / N,
              n_refresh);
  std::printf("stream total=%.2f ms   avg linear=%.0f ns   avg refresh=%.1f us\n", total_ns / 1e6,
              lin_ns / std::max(1L, n_linear), (ref_ns / std::max(1L, n_refresh)) / 1e3);
  std::printf("max error vs exact=%.2e   all-cold full-LM equiv=%.1f s (%.0fx)\n", max_err,
              full_lm_us * N / 1e6, (full_lm_us * N) / (total_ns / 1e3));
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
