// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
// QuantLib CORRECTNESS ORACLE for the Bachelier (normal) options kernel (vol/bachelier.hpp) — the layer the
// swaption vol_cube prices with. Proves our closed forms EQUAL QuantLib's, across a strike/vol/expiry grid:
//   * price  vs QuantLib::bachelierBlackFormula                         (exact, ~1e-12)
//   * delta  vs QuantLib::bachelierBlackFormulaForwardDerivative        (dPrice/dForward, exact)
//   * vega   vs QuantLib::bachelierBlackFormulaStdDevDerivative * sqrt(T)(dPrice/dvol, exact)
//   * gamma  vs a central finite-difference of QuantLib's forward derivative (no QL 2nd-deriv exists)
//   * implied vol: our safeguarded inversion recovers the input, and agrees with QuantLib's approximation.
// QuantLib's `discount` argument carries our ANNUITY (the swaption numeraire), so this oracles the exact
// (forward, strike, vol, expiry, annuity) -> price/greeks kernel. The curve -> (forward, annuity) step is
// separately oracled: the curve itself against QuantLib's bootstrap (build_square_ql / compile_parity), and
// the forward-swap arithmetic Σ τ_i·DF_i in tests/vol_swaption_test.cpp — so the composition is QuantLib-clean.
#include <gtest/gtest.h>

#include <ql/quantlib.hpp>

#include <cmath>

#include "swaps/vol/bachelier.hpp"

namespace v = swaps::vol;
namespace QL = QuantLib;

namespace {
QL::Option::Type qtype(v::Payoff cp) {
  return cp == v::Payoff::Payer ? QL::Option::Call : QL::Option::Put;  // payer = call on the rate
}
// A realistic SOFR-ish grid: inverted-to-normal forwards, ITM/ATM/OTM strikes, low-to-high normal vols, and
// short-to-long expiries. Annuity ~ a 5y-ish swap numeraire.
const double kF[] = {0.020, 0.030, 0.035, 0.045};
const double kK[] = {0.010, 0.030, 0.040, 0.055};
const double kVol[] = {0.0035, 0.0080, 0.0140};
const double kT[] = {0.25, 1.0, 5.0, 10.0};
constexpr double kAnnuity = 4.37;
}  // namespace

TEST(VolBachelierQLOracle, PriceDeltaVegaEqualQuantLib) {
  for (double F : kF)
    for (double K : kK)
      for (double vol : kVol)
        for (double T : kT)
          for (v::Payoff cp : {v::Payoff::Payer, v::Payoff::Receiver}) {
            const double sd = vol * std::sqrt(T);
            const QL::Option::Type ot = qtype(cp);
            const double scale = std::max(1.0, kAnnuity);

            const double ql_px = QL::bachelierBlackFormula(ot, K, F, sd, kAnnuity);
            const double our_px = v::bachelier_price<double>(F, K, vol, T, kAnnuity, cp);
            EXPECT_NEAR(our_px, ql_px, 1e-12 * scale) << "price F=" << F << " K=" << K << " vol=" << vol << " T=" << T;

            const double ql_delta = QL::bachelierBlackFormulaForwardDerivative(ot, K, F, sd, kAnnuity);
            const double our_delta = v::bachelier_delta<double>(F, K, vol, T, kAnnuity, cp);
            EXPECT_NEAR(our_delta, ql_delta, 1e-11 * scale) << "delta F=" << F << " K=" << K;

            const double ql_vega = QL::bachelierBlackFormulaStdDevDerivative(K, F, sd, kAnnuity) * std::sqrt(T);
            const double our_vega = v::bachelier_vega<double>(F, K, vol, T, kAnnuity);
            EXPECT_NEAR(our_vega, ql_vega, 1e-11 * scale) << "vega F=" << F << " K=" << K;
          }
}

TEST(VolBachelierQLOracle, GammaMatchesFiniteDiffOfQuantLibDelta) {
  const double h = 1e-7;  // forward bump for the central FD of QL's delta
  for (double F : kF)
    for (double K : kK)
      for (double vol : kVol)
        for (double T : kT) {
          const double sd = vol * std::sqrt(T);
          const QL::Option::Type ot = QL::Option::Call;  // gamma is payer/receiver symmetric
          const double dP = QL::bachelierBlackFormulaForwardDerivative(ot, K, F + h, sd, kAnnuity);
          const double dM = QL::bachelierBlackFormulaForwardDerivative(ot, K, F - h, sd, kAnnuity);
          const double ql_gamma = (dP - dM) / (2.0 * h);
          const double our_gamma = v::bachelier_gamma<double>(F, K, vol, T, kAnnuity);
          EXPECT_NEAR(our_gamma, ql_gamma, 1e-4 * std::max(1.0, std::abs(ql_gamma)))
              << "gamma F=" << F << " K=" << K << " vol=" << vol << " T=" << T;
        }
}

TEST(VolBachelierQLOracle, ImpliedVolInvertsAndAgreesWithQuantLib) {
  for (double F : kF)
    for (double K : kK)
      for (double vol : kVol)
        for (double T : kT)
          for (v::Payoff cp : {v::Payoff::Payer, v::Payoff::Receiver}) {
            const double sd = vol * std::sqrt(T);
            // Only test where the option has RECOVERABLE time value: beyond a few std devs it is almost all
            // intrinsic (deep ITM) or almost worthless (deep OTM), the time value drops below double epsilon,
            // and vol is not recoverable from price for ANY method (ours correctly returns 0 there).
            if (std::abs(F - K) > 4.0 * sd) continue;
            const double px = v::bachelier_price<double>(F, K, vol, T, kAnnuity, cp);
            // Our safeguarded Newton-bisection recovers the input vol — the strong correctness claim (our
            // inversion is exact, not an approximation).
            const double ours = v::bachelier_implied_vol(px, F, K, T, kAnnuity, cp);
            EXPECT_NEAR(ours, vol, 1e-8) << "our inversion F=" << F << " K=" << K << " vol=" << vol << " T=" << T;
            // Cross-check against QuantLib's inverter. QL's is an ANALYTIC APPROXIMATION (Choi-Kim-Kwak),
            // accurate near the money, so we compare only within ~1 std dev of the forward.
            if (std::abs(F - K) <= sd) {
              const double ql = QL::bachelierBlackFormulaImpliedVol(qtype(cp), K, F, T, px, kAnnuity);
              EXPECT_NEAR(ours, ql, 1e-4 * std::max(1.0, vol)) << "vs QL approx F=" << F << " K=" << K;
            }
          }
}
