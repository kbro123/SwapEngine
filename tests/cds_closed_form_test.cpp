// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// T5 (E5.3 2026-09-10): the CDS protection leg against the CONTINUOUS closed form. On a flat hazard h and a flat
// rate r the exact protection PV is P = (1−R)·h/(r+h)·(1 − e^{−(r+h)T}) and the risky annuity of the engine's
// synthetic quarterly schedule is A = Σᵢ τᵢ e^{−(r+h) tᵢ} (tᵢ = i/4 exactly, τᵢ = 0.25·365/360): the engine's par
// spread s = P/A must converge to P_exact/A as its protection grid is refined -- at FIRST order (the documented
// default-paid-at-period-end rule), so the error halves when the grid doubles. Before this file the only
// closed-form check was the credit triangle at 5e-4 (2.8 % of the spread; review-2026-09-08 finding 8), which
// says nothing about the discretisation actually converging to the integral it claims to approximate.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <vector>

#include "swaps/build/credit_instruments.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/hazard.hpp"

namespace b = swaps::build;
namespace curve = swaps::curve;

namespace {
struct FlatDf {
  double r;
  double operator()(double t) const { return std::exp(-r * t); }
};
}  // namespace

TEST(CdsClosedForm, ParSpreadConvergesToTheExactProtectionIntegralAtFirstOrder) {
  const double h = 0.03, r = 0.02, R = 0.40, T = 5.0;
  const std::vector<double> knots{1.0, 2.0, 3.0, 5.0, 10.0};
  auto hz = curve::make_modular_curve<double>({{knots, curve::Scheme::Flat}});
  hz.set_forwards(Eigen::VectorXd::Constant(static_cast<int>(knots.size()), h));
  const curve::SurvivalCurve<double> surv{&hz};

  const double ratio = 365.0 / 360.0;
  double A = 0.0;
  for (int i = 1; i <= 20; ++i) A += 0.25 * ratio * std::exp(-(r + h) * (0.25 * i));
  const double P_exact = (1.0 - R) * h / (r + h) * (1.0 - std::exp(-(r + h) * T));
  const double s_exact = P_exact / A;

  double prev_err = 0.0;
  for (int steps : {1, 4, 16, 64, 256, 1024}) {
    const auto cds = b::make_cds(T, 0.0, R, FlatDf{r}, 4, steps, ratio);
    const double s = cds.model_quote<double>(surv);
    const double err = std::abs(s - s_exact);
    std::cout << "  [cds] prot_steps/quarter " << steps << ": s = " << s << " exact " << s_exact << " |err| = " << err
              << (prev_err > 0 ? " ratio " + std::to_string(prev_err / err) : std::string()) << "\n";
    if (steps == 1) EXPECT_LT(err, 1.5e-4) << "the quarterly right-end rule sits within the credit-triangle band";
    if (prev_err > 0.0) EXPECT_NEAR(prev_err / err, 4.0, 0.6) << "first-order convergence: 4x fewer errors per 4x finer grid";
    if (steps == 1024) EXPECT_LT(err, 3e-7) << "the refined grid reproduces the exact integral";
    prev_err = err;
  }
  // and every discretisation UNDERSTATES the protection leg (a default paid at the interval's right end is paid
  // later than the continuous payment, so it is discounted more) -- the discretised spread sits below the exact one.
  EXPECT_LT(b::make_cds(T, 0.0, R, FlatDf{r}, 4, 1, ratio).model_quote<double>(surv), s_exact);
}
