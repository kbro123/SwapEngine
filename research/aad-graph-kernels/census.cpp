// Census of the shape ladder: sizes that decide how big a recorded tape would be, plus engine eval/Jacobian timings.
#include <chrono>
#include <cstdio>
#include <vector>
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

struct Counting : px::CurveHandle<double> {
  const px::CurveHandle<double>* real = nullptr;
  mutable long* nd, *ni, *nf;
  double discount(double t) const override { ++*nd; return real->discount(t); }
  double integral(double t) const override { ++*ni; return real->integral(t); }
  double forward(double t) const override { ++*nf; return real->forward(t); }
  double turn_jump(int j) const override { return real->turn_jump(j); }
  void pieces_into(std::vector<double>& o) const override { real->pieces_into(o); }
  void set_forwards(const Eigen::VectorXd&) override {}
};

static void legs(const cal::Instrument& in, long& fix, long& flt, long& sub) {
  auto L = [&](const cal::FloatLeg& l) { for (auto& c : l.coupons) { ++flt; sub += (long)c.obs.sub_start.size(); } };
  if (in.quote == cal::QuoteKind::Portfolio) { for (auto& c : in.combination) legs(c.instrument, fix, flt, sub); return; }
  if (in.quote == cal::QuoteKind::Rate) { sub += (long)in.obs.sub_start.size(); return; }
  L(in.fwd); L(in.bench); L(in.mtm); fix += (long)in.fixed.coupons.size();
}

int main() {
  using clk = std::chrono::steady_clock;
  std::printf("%-20s %4s %4s %5s %6s %7s %6s %7s %7s %4s %4s %9s %9s\n", "shape", "m", "n", "fixc", "fltc", "subs", "Tcomp",
              "disc", "integ", "cmp", "aadw", "res_us", "jac_us");
  for (const auto& s : swaps::shapes::ladder()) {
    const auto& p = s.prob;
    long fix = 0, flt = 0, sub = 0;
    for (auto& in : p.instruments) legs(in, fix, flt, sub);
    const int NC = p.n_curves();
    const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return s.x_true[p.offset(c) + i]; });
    long nd = 0, ni = 0, nf = 0;
    std::vector<Counting> H(NC);
    for (int c = 0; c < NC; ++c) { H[c].real = C[c].get(); H[c].nd = &nd; H[c].ni = &ni; H[c].nf = &nf; }
    const auto of = [&](int i) -> const px::CurveHandle<double>& { return H[i]; };
    for (auto& in : p.instruments) (void)cal::instrument_residual<double>(in, of);
    cal::HybridBundleResidual h(p);
    Eigen::MatrixXd J;
    const int R = 2000;
    double acc = 0;
    auto t0 = clk::now();
    for (int k = 0; k < R; ++k) { Eigen::VectorXd x = s.x_true; x[k % x.size()] += 1e-9 * ((k & 1) ? 1 : -1); acc += h.residuals_vs(x, s.q_small)[0]; }
    double res_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / R;
    int RJ = 200;
    t0 = clk::now();
    for (int k = 0; k < RJ; ++k) { Eigen::VectorXd x = s.x_true; x[k % x.size()] += 1e-9; h.jacobian_vs_into(x, s.q_small, J); }
    double jac_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / RJ;
    std::printf("%-20s %4d %4d %5ld %6ld %7ld %6d %7ld %7ld %4d %4d %9.2f %9.2f\n", s.name.c_str(), p.n_residuals(), p.n_knots(), fix,
                flt, sub, h.n_times(), nd, ni + nf, h.n_compiled_rows(), h.aad_width(), res_us, jac_us);
    (void)acc;
  }
}
