// Cold-calibration time comparison: SELF-CONSISTENT vs REALISTIC (over-determined + noisy) feeds,
// for a SINGLE curve (SOFR) and a BUNDLE (the 6-curve multi-currency streamable bundle).
//
// The point (user request): a self-consistent market (quotes = model quotes at a known curve) has a
// reachable zero-residual optimum, while a realistic strip is over-determined AND inconsistent -- the
// futures and swaps don't agree, so the least-squares fit lands with a NON-ZERO residual (the tension).
// This tool times a cold LM solve of both feeds, both problem sizes, and prints iterations, the achieved
// rms residual (bp) and first-order optimality ||Jᵀr||∞, so the calibration times are directly comparable.
//
// Build (not in CMake; links the vendored QuantLib static lib). On the Mac Pro set the 14.5-SDK libc++:
//   export CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX14.5.sdk/usr/include/c++/v1
//   clang++ -std=c++20 -O2 -march=native -DNDEBUG \
//       -I include -I tests -I build/generated -I third_party/eigen -I third_party/boost \
//       -I third_party/quantlib/install/include \
//       tools/cal_times.cpp third_party/quantlib/install/lib/libQuantLib.a -o build/cal_times
//   ./build/cal_times
#include <ql/quantlib.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "reference_curve.hpp"
#include "reference_multicurrency.hpp"
#include "swaps/calibration/lm.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace rm = swaps::refmkt;
namespace cal = swaps::calibration;

namespace {

// The reference SOFR forwards (6 front meetings + 17 back knots).
Eigen::VectorXd reference_x() {
  Eigen::VectorXd x(rm::n_knots);
  int i = 0;
  for (double f : rm::reference_front_forwards) x[i++] = f;
  for (double g : rm::reference_back_forwards) x[i++] = g;
  return x;
}

// Median wall-clock (µs) of `reps` cold calibrations from the SAME x0 (the solve is deterministic, so
// this measures timing scatter only). AAD/compiled engine path by default (use_aad = the fast W-cache).
template <class Problem>
double median_cold_us(const Problem& prob, const Eigen::VectorXd& x0, int reps, cal::CalibrationResult& last) {
  std::vector<double> t;
  t.reserve(reps);
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    last = cal::calibrate(prob, x0);
    t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

void row(const char* label, int knots, int instr, double us, const cal::CalibrationResult& r) {
  std::printf("  %-30s %5d %6d   %8.0f   %5d      %9.3f    %10.2e\n", label, knots, instr, us,
              r.iterations, r.rms_residual * 1e4, r.stationarity);
}

}  // namespace

int main() {
  std::mt19937 rng(20260722);
  std::normal_distribution<double> noise(0.0, 0.4e-4);  // ~0.4bp independent per-instrument mispricing
  const int REPS = 25;

  std::printf("Cold calibration: self-consistent vs realistic (over-determined + noisy) feed\n");
  std::printf("  %-30s %5s %6s   %8s   %5s      %9s    %10s\n", "feed", "knots", "instr", "median µs",
              "iters", "rms(bp)", "||Jᵀr||∞");
  std::printf("  %s\n", std::string(92, '-').c_str());

  // ---------------- SINGLE CURVE: SOFR (futures + swaps, over-determined 29 vs 23) ----------------
  {
    RelinkableHandle<YieldTermStructure> h;
    rb::Market mk = rb::build_market(h);
    const Eigen::VectorXd xref = reference_x();

    // Self-consistent: quotes = model quotes at xref (bump each market by its residual at xref -> r(xref)=0).
    cal::CalibrationProblem sc = rb::build_problem(mk);
    const Eigen::VectorXd r0 = sc.residuals<double>(xref);
    for (int i = 0; i < static_cast<int>(sc.instruments.size()); ++i) sc.instruments[i].market += r0[i];

    // Realistic: the self-consistent quotes + independent 0.4bp noise -> the strip no longer agrees.
    cal::CalibrationProblem ns = sc;
    for (auto& ins : ns.instruments) ins.market += noise(rng);

    // Same start well away from the answer (+40bp parallel + alternating tilt) for both.
    Eigen::VectorXd x0 = xref;
    for (int i = 0; i < x0.size(); ++i) x0[i] += 0.0040 + (i % 2 ? 0.0015 : -0.0015);

    cal::CalibrationResult rr;
    const double us_sc = median_cold_us(sc, x0, REPS, rr);
    row("single SOFR  · self-consistent", sc.n_knots(), sc.n_residuals(), us_sc, rr);
    const double us_ns = median_cold_us(ns, x0, REPS, rr);
    row("single SOFR  · realistic+noise", ns.n_knots(), ns.n_residuals(), us_ns, rr);
  }

  // ---------------- BUNDLE: the 6-curve multi-currency streamable bundle (SOFR+FF+PRIME+ESTR+3M+6M) ----
  {
    rb::MultiCcyBundle b = rb::build_full_multicurrency(/*include_xccy=*/false, /*coupled_eur=*/false);
    cal::BundleProblem sc = b.prob;  // built self-consistent from x_true (r(x_true)=0)

    cal::BundleProblem ns = sc;
    for (auto& ins : ns.instruments) ins.market += noise(rng);

    cal::CalibrationResult rr;
    const double us_sc = median_cold_us(sc, b.x0, REPS, rr);
    row("bundle 6-ccy · self-consistent", sc.n_knots(), sc.n_residuals(), us_sc, rr);
    const double us_ns = median_cold_us(ns, b.x0, REPS, rr);
    row("bundle 6-ccy · realistic+noise", ns.n_knots(), ns.n_residuals(), us_ns, rr);
  }

  std::printf("\n  Notes: self-consistent feeds reach the zero-residual optimum (rms→0); realistic feeds keep a\n");
  std::printf("  non-zero least-squares residual (the tension) yet are first-order optimal (||Jᵀr||∞≈0).\n");
  return 0;
}
