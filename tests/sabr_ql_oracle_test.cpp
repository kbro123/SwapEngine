// @oracle-test — validates against QuantLib::sabrVolatility (Hagan 2002 B.65a, general β) point-for-point.
// DO NOT DELETE OR WEAKEN without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5.3 (2026-09-10): the 2026-09-08 audit found NO off-ATM SABR value pinned to anything outside sabr.hpp
// (the two "references" in swaps_tests are transcriptions of the implementation). This pins the general-β
// BLACK (lognormal) expansion sabr_black_vol against QuantLib across β ∈ {0, ¼, ½, ¾, 1}, four expiries and a
// ±180 bp strike grid (ATM included -- both sides take their ATM branch there). The NORMAL-vol expansion
// (sabr_normal_vol, Obłój arc-length backbone) has no QuantLib counterpart in 1.35 and stays a regression
// freeze (rates_fx_grid_test) -- that is an OPEN oracle gap, recorded in the E5 plan.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <cmath>

#include "swaps/vol/sabr.hpp"

namespace v = swaps::vol;

TEST(SabrOracle, GeneralBetaBlackVolMatchesQuantLib) {
  const double F = 0.030, rho = -0.30, nu = 0.45;
  double worst = 0.0;
  for (double beta : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const double alpha = 0.20 * std::pow(F, 1.0 - beta);  // a ~20 % lognormal level on every backbone
    for (double T : {0.5, 2.0, 5.0, 10.0}) {
      for (int bp = -180; bp <= 180; bp += 12) {
        const double K = F + bp / 1e4;
        if (K <= 0.0) continue;
        const double ours = v::sabr_black_vol<double>(F, K, T, alpha, rho, nu, beta);
        const double ql = QuantLib::sabrVolatility(K, F, T, alpha, beta, nu, rho);
        worst = std::max(worst, std::abs(ours - ql));
        EXPECT_NEAR(ours, ql, 1e-13) << "beta=" << beta << " T=" << T << " K=" << K;
      }
    }
  }
  std::cout << "  [sabr-oracle] max |sabr_black_vol - QuantLib::sabrVolatility| = " << worst << "\n";
}
