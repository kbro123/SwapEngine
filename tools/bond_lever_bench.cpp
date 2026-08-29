// Indicative micro-benchmark of the ALGORITHMIC LEVER behind portfolio/bond_universe.hpp.
//
//   READ THIS BEFORE QUOTING ANY NUMBER IT PRINTS.
//
// This is NOT a gate benchmark and NOT a comparison against any external library. It links neither
// QuantLib nor Eigen; it compares two hand-written evaluation ORDERS of the same street math:
//
//   (A) bond-major   — per bond, per cashflow std::pow, and a scalar Newton solve per bond. This is the
//                      SHAPE of a per-bond library yield loop, written by us. It deliberately omits the
//                      virtual dispatch, shared_ptr indirection, date arithmetic and observer machinery a
//                      real library carries, so it is a CONSERVATIVE stand-in — a real library loop is
//                      slower than this, by an amount nobody in this repo has yet measured.
//
//   (B) cashflow-major — the BondUniverse shape: the coupon-polynomial (Horner) sweep across bond lanes
//                      with synthetic differentiation, one pow(v,w) per bond, and ONE batched Newton for
//                      the whole universe. Replicates BondUniverse::horner_pass / yields_from_clean.
//
// What it legitimately shows: the cost of the factoring itself (B*K transcendentals -> B), and that the
// factoring is exact (both paths agree to ~1e-13 on yields, ~1e-15 on prices).
// What it does NOT show: any speedup over QuantLib, Rateslib or FinancePy. See
// docs/bond-pricing-perf-handoff.md for the benchmark that would.
//
// Build (standalone, no deps):
//   c++ -std=c++20 -O3 -march=native tools/bond_lever_bench.cpp -o bond_lever_bench && ./bond_lever_bench
//
// Perf-gate integrity (CLAUDE.md section 3): numbers from this program are per-machine like any other and
// must not be compared across fingerprints. Run it on the machine you intend to quote.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

// A deterministic heterogeneous universe: B bonds, 1..60 semiannual cashflows (0.5y..30y), coupons
// 0.5%..8%, settlement offsets w spread through the coupon period. Column-major padded B x K matrices,
// matching Eigen's MatrixXd layout so column k is contiguous across bond lanes (as in BondUniverse).
struct Universe {
  int B = 0, K = 0;
  std::vector<int> n;                  // cashflows per bond
  std::vector<double> w, cpn;          // current-period fraction, annual coupon
  std::vector<double> A, E;            // B x K, column-major: coefficients and exponents

  double& a(int b, int k) { return A[std::size_t(k) * B + b]; }
  double& e(int b, int k) { return E[std::size_t(k) * B + b]; }
  const double* col(int k) const { return &A[std::size_t(k) * B]; }

  explicit Universe(int bonds) : B(bonds), n(bonds), w(bonds), cpn(bonds) {
    for (int b = 0; b < B; ++b) {
      n[b] = 1 + (b * 7919) % 60;
      w[b] = 0.02 + 0.96 * double((b * 4703) % 1000) / 1000.0;
      cpn[b] = 0.005 + 0.075 * double((b * 3571) % 1000) / 1000.0;
      K = std::max(K, n[b]);
    }
    A.assign(std::size_t(B) * K, 0.0);
    E.assign(std::size_t(B) * K, 0.0);
    for (int b = 0; b < B; ++b)
      for (int k = 0; k < n[b]; ++k) {
        a(b, k) = cpn[b] / 2.0 + (k == n[b] - 1 ? 1.0 : 0.0);  // coupon, plus redemption at maturity
        e(b, k) = w[b] + k;                                    // E_i = w + i (regular bond)
      }
  }
};

// (A) One bond, one evaluation: dirty(y) and d dirty/dy by pow() per cashflow.
double naive_value(const Universe& u, int b, double y, double& d1) {
  const double base = 1.0 + y / 2.0;
  double p = 0.0;
  d1 = 0.0;
  for (int k = 0; k < u.n[b]; ++k) {
    const double ex = u.E[std::size_t(k) * u.B + b];
    const double t = u.A[std::size_t(k) * u.B + b] * std::pow(base, -ex);
    p += t;
    d1 += -(ex / 2.0) * t / base;
  }
  return p;
}

}  // namespace

int main() {
  const int B = 5000;
  Universe u(B);
  std::printf("universe: %d bonds, max %d cashflows\n\n", u.B, u.K);

  // Target dirty prices implied by known yields, so both paths solve genuine roots.
  std::vector<double> target(B);
  for (int b = 0; b < B; ++b) {
    const double y = 0.0325 + 0.02 * double((b * 6151) % 1000) / 1000.0;
    double d;
    target[b] = naive_value(u, b, y, d);
  }

  // ================= 1. Solve yield from price across the universe =================
  std::vector<double> yA(B, 0.05);
  int iters_A = 0;
  const auto t0 = clk::now();
  for (int b = 0; b < B; ++b) {
    double y = 0.05;
    for (int it = 0; it < 100; ++it) {
      double d1;
      const double r = naive_value(u, b, y, d1) - target[b];
      ++iters_A;
      if (std::fabs(r) <= 1e-13) break;
      y -= r / d1;
    }
    yA[b] = y;
  }
  const auto t1 = clk::now();

  std::vector<double> yB(B, 0.05), v(B), q(B), dq(B), ddq(B), vw(B), dirty(B), dP(B);
  int iters_B = 0;
  const auto t2 = clk::now();
  for (int it = 0; it < 100; ++it) {
    for (int b = 0; b < B; ++b) v[b] = 1.0 / (1.0 + yB[b] / 2.0);
    std::fill(q.begin(), q.end(), 0.0);
    std::fill(dq.begin(), dq.end(), 0.0);
    std::fill(ddq.begin(), ddq.end(), 0.0);
    for (int k = u.K - 1; k >= 0; --k) {  // one FMA sweep per cashflow column; no transcendental
      const double* Ak = u.col(k);
      for (int b = 0; b < B; ++b) {
        ddq[b] = ddq[b] * v[b] + dq[b];   // synthetic differentiation: update Q''/2, then Q', then Q
        dq[b] = dq[b] * v[b] + q[b];
        q[b] = q[b] * v[b] + Ak[b];
      }
    }
    for (int b = 0; b < B; ++b) vw[b] = std::pow(v[b], u.w[b]);  // ONE transcendental per bond
    double maxr = 0.0;
    for (int b = 0; b < B; ++b) {
      const double Pv = vw[b] * (u.w[b] * q[b] / v[b] + dq[b]);
      dirty[b] = vw[b] * q[b];
      dP[b] = Pv * (-(v[b] * v[b]) / 2.0);  // chain rule v -> y
      maxr = std::max(maxr, std::fabs(dirty[b] - target[b]));
    }
    ++iters_B;
    if (maxr <= 1e-13) break;
    for (int b = 0; b < B; ++b) yB[b] -= (dirty[b] - target[b]) / dP[b];  // converged bonds self-arrest
  }
  const auto t3 = clk::now();

  double max_dy = 0.0;
  for (int b = 0; b < B; ++b) max_dy = std::max(max_dy, std::fabs(yA[b] - yB[b]));
  const double ms_A = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_B = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::printf("[1] solve yield from price\n");
  std::printf("    (A) bond-major  pow + per-bond Newton : %8.3f ms   (%d bond-iterations)\n", ms_A, iters_A);
  std::printf("    (B) batched Horner + batched Newton   : %8.3f ms   (%d universe-iterations)\n", ms_B, iters_B);
  std::printf("    ratio %.2fx      max |y_A - y_B| = %.3e\n\n", ms_A / ms_B, max_dy);

  // ================= 2. Re-mark: yield -> price, value only =================
  const int REPS = 200;
  std::vector<double> y(B), pA(B), pB(B);
  for (int b = 0; b < B; ++b) y[b] = 0.0325 + 0.02 * double((b * 6151) % 1000) / 1000.0;

  const auto t4 = clk::now();
  for (int r = 0; r < REPS; ++r)
    for (int b = 0; b < B; ++b) {
      const double base = 1.0 + (y[b] + r * 1e-9) / 2.0;  // perturb so nothing is hoisted
      double p = 0.0;
      for (int k = 0; k < u.n[b]; ++k)
        p += u.A[std::size_t(k) * B + b] * std::pow(base, -u.E[std::size_t(k) * B + b]);
      pA[b] = p;
    }
  const auto t5 = clk::now();
  for (int r = 0; r < REPS; ++r) {
    for (int b = 0; b < B; ++b) v[b] = 1.0 / (1.0 + (y[b] + r * 1e-9) / 2.0);
    std::fill(q.begin(), q.end(), 0.0);
    for (int k = u.K - 1; k >= 0; --k) {
      const double* Ak = u.col(k);
      for (int b = 0; b < B; ++b) q[b] = q[b] * v[b] + Ak[b];
    }
    for (int b = 0; b < B; ++b) vw[b] = std::pow(v[b], u.w[b]);
    for (int b = 0; b < B; ++b) pB[b] = vw[b] * q[b];
  }
  const auto t6 = clk::now();

  double max_dp = 0.0;
  for (int b = 0; b < B; ++b) max_dp = std::max(max_dp, std::fabs(pA[b] - pB[b]));
  const double ms_A2 = std::chrono::duration<double, std::milli>(t5 - t4).count() / REPS;
  const double ms_B2 = std::chrono::duration<double, std::milli>(t6 - t5).count() / REPS;

  std::printf("[2] re-mark (yield -> dirty price, value only)\n");
  std::printf("    (A) bond-major pow per cashflow       : %8.3f ms\n", ms_A2);
  std::printf("    (B) batched Horner value pass         : %8.3f ms   (%.1f ns/bond)\n", ms_B2,
              ms_B2 * 1e6 / B);
  std::printf("    ratio %.2fx      max |P_A - P_B| = %.3e\n\n", ms_A2 / ms_B2, max_dp);

  std::printf("Reminder: (A) is OUR construction of a per-bond loop, not any library's code.\n");
  std::printf("No external library was executed or timed. Not a gate benchmark.\n");
  return 0;
}
