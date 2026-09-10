// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Phase-1 gate for the options/vol layer: the Bachelier (normal) analytics. Pure closed-form identities —
// no curve, no QuantLib — so it runs in the QuantLib-free swaps_tests binary. These are the invariants the
// build plan names as the acceptance gate: put-call parity, the zero-vol intrinsic limit, an exact ATM
// reference, vega vs finite difference, and a vol->price->vol round-trip.
#include <gtest/gtest.h>

#include <cmath>

#include "swaps/vol/bachelier.hpp"

namespace v = swaps::vol;

namespace {
constexpr double F = 0.032;    // forward swap rate
constexpr double A = 4.5;      // annuity (Σ τ_i DF_i)
constexpr double T = 2.0;      // expiry (years)
}  // namespace

TEST(Bachelier, PutCallParity) {
  // V_pay - V_rec = A * (F - K) for every strike and vol.
  for (double K : {0.010, 0.032, 0.050, -0.005}) {
    for (double vol : {0.0040, 0.0120}) {
      const double pay = v::bachelier_price<double>(F, K, vol, T, A, v::Payoff::Payer);
      const double rec = v::bachelier_price<double>(F, K, vol, T, A, v::Payoff::Receiver);
      EXPECT_NEAR(pay - rec, A * (F - K), 1e-12) << "K=" << K << " vol=" << vol;
    }
  }
}

TEST(Bachelier, ZeroVolIsDiscountedIntrinsic) {
  for (double K : {0.010, 0.032, 0.050}) {
    EXPECT_NEAR(v::bachelier_price<double>(F, K, 0.0, T, A, v::Payoff::Payer),
                A * std::max(F - K, 0.0), 1e-14);
    EXPECT_NEAR(v::bachelier_price<double>(F, K, 0.0, T, A, v::Payoff::Receiver),
                A * std::max(K - F, 0.0), 1e-14);
  }
}

TEST(Bachelier, AtmClosedForm) {
  // ATM (K=F): V = A * σ√T * φ(0) = A * σ√T / √(2π), identical payer/receiver.
  const double vol = 0.0085;
  const double expected = A * vol * std::sqrt(T) * v::kInvSqrt2Pi;
  EXPECT_NEAR(v::bachelier_price<double>(F, F, vol, T, A, v::Payoff::Payer), expected, 1e-14);
  EXPECT_NEAR(v::bachelier_price<double>(F, F, vol, T, A, v::Payoff::Receiver), expected, 1e-14);
}

TEST(Bachelier, VegaMatchesFiniteDifference) {
  const double vol = 0.0095, h = 1e-7;
  for (double K : {0.020, 0.032, 0.045}) {
    const double up = v::bachelier_price<double>(F, K, vol + h, T, A, v::Payoff::Payer);
    const double dn = v::bachelier_price<double>(F, K, vol - h, T, A, v::Payoff::Payer);
    EXPECT_NEAR(v::bachelier_vega<double>(F, K, vol, T, A), (up - dn) / (2 * h), 1e-4) << "K=" << K;
  }
}

TEST(Bachelier, DeltaMatchesFiniteDifference) {
  const double vol = 0.0095, h = 1e-8;
  for (double K : {0.020, 0.032, 0.045}) {
    for (auto cp : {v::Payoff::Payer, v::Payoff::Receiver}) {
      const double up = v::bachelier_price<double>(F + h, K, vol, T, A, cp);
      const double dn = v::bachelier_price<double>(F - h, K, vol, T, A, cp);
      EXPECT_NEAR(v::bachelier_delta<double>(F, K, vol, T, A, cp), (up - dn) / (2 * h), 1e-5) << "K=" << K;
    }
  }
}

TEST(Bachelier, ImpliedVolRoundTrip) {
  for (double K : {0.015, 0.032, 0.048}) {
    for (double vol : {0.0030, 0.0075, 0.0150}) {
      for (auto cp : {v::Payoff::Payer, v::Payoff::Receiver}) {
        const double price = v::bachelier_price<double>(F, K, vol, T, A, cp);
        const double iv = v::bachelier_implied_vol(price, F, K, T, A, cp);
        EXPECT_NEAR(iv, vol, 1e-10) << "K=" << K << " vol=" << vol;
      }
    }
  }
}

// R1 hot-path: the one-pass bachelier_greeks() must equal the six individual analytics EXACTLY (it just
// shares sqrt/d/pdf/cdf). Pins the streaming/cube optimization to the oracle-validated closed forms.
TEST(Bachelier, GreeksOnePassMatchesIndividual) {
  const double A = 4.25;
  for (double F : {-0.002, 0.0, 0.015, 0.03, 0.06}) {
    for (double K : {F - 0.02, F - 0.005, F, F + 0.005, F + 0.02}) {
      for (double vol : {0.0020, 0.0075, 0.015}) {
        for (double T : {0.08, 1.0, 5.0, 10.0}) {
          for (auto cp : {v::Payoff::Payer, v::Payoff::Receiver}) {
            const v::BachelierGreeks<double> g = v::bachelier_greeks<double>(F, K, vol, T, A, cp);
            EXPECT_NEAR(g.price, v::bachelier_price<double>(F, K, vol, T, A, cp), 1e-15) << "price";
            EXPECT_NEAR(g.vega, v::bachelier_vega<double>(F, K, vol, T, A), 1e-15) << "vega";
            EXPECT_NEAR(g.delta, v::bachelier_delta<double>(F, K, vol, T, A, cp), 1e-15) << "delta";
            EXPECT_NEAR(g.gamma, v::bachelier_gamma<double>(F, K, vol, T, A), 1e-15) << "gamma";
            EXPECT_NEAR(g.vanna, v::bachelier_vanna<double>(F, K, vol, T, A), 1e-15) << "vanna";
            EXPECT_NEAR(g.volga, v::bachelier_volga<double>(F, K, vol, T, A), 1e-15) << "volga";
          }
        }
      }
    }
  }
}
