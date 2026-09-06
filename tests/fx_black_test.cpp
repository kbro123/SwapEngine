// Gate for the FX lognormal layer: Garman-Kohlhagen closed-form analytics (vol/fx_black.hpp) + the
// delta-quoted FX smile (vol/fx_vol_surface.hpp). Pure closed-form identities — no curve, no QuantLib — so
// this runs in the QuantLib-free swaps_tests binary. Invariants: put-call parity, the zero-vol intrinsic
// limit, a known Garman-Kohlhagen benchmark value (Haug), every Greek vs central finite difference, the
// one-pass gk_greeks == the individual analytics, an implied-vol price round-trip, and the FX smile's
// {ATM,RR,BF} -> strikes -> vols / delta round-trips.
#include <gtest/gtest.h>

#include <cmath>

#include "swaps/vol/fx_black.hpp"
#include "swaps/vol/fx_vol_surface.hpp"

namespace v = swaps::vol;

namespace {
constexpr double S = 1.10;    // spot (domestic per foreign)
constexpr double RD = 0.030;  // domestic rate
constexpr double RF = 0.010;  // foreign rate
constexpr double T = 0.75;    // expiry (years)
}  // namespace

TEST(FxBlack, PutCallParity) {
  // Call - Put = S·e^{-r_f T} - K·e^{-r_d T} = df_dom·(F - K), every strike and vol.
  const double df_dom = std::exp(-RD * T), df_for = std::exp(-RF * T);
  for (double K : {0.90, 1.05, 1.10, 1.25, 1.40}) {
    for (double vol : {0.06, 0.12, 0.20}) {
      const double call = v::gk_price<double>(S, K, vol, T, RD, RF, v::CallPut::Call);
      const double put = v::gk_price<double>(S, K, vol, T, RD, RF, v::CallPut::Put);
      EXPECT_NEAR(call - put, S * df_for - K * df_dom, 1e-13) << "K=" << K << " vol=" << vol;
    }
  }
}

TEST(FxBlack, ZeroVolIsDiscountedIntrinsic) {
  const double df_dom = std::exp(-RD * T);
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  for (double K : {0.95, 1.10, 1.30}) {
    EXPECT_NEAR(v::gk_price<double>(S, K, 0.0, T, RD, RF, v::CallPut::Call), df_dom * std::max(fwd - K, 0.0),
                1e-14);
    EXPECT_NEAR(v::gk_price<double>(S, K, 0.0, T, RD, RF, v::CallPut::Put), df_dom * std::max(K - fwd, 0.0),
                1e-14);
  }
}

TEST(FxBlack, GarmanKohlhagenBenchmark) {
  // Haug, "Complete Guide to Option Pricing Formulas": S=1.56, K=1.60, T=0.5, r_dom=0.06, r_for=0.08,
  // sigma=0.12 -> call = 0.0291.
  const double call = v::gk_price<double>(1.56, 1.60, 0.12, 0.5, 0.06, 0.08, v::CallPut::Call);
  EXPECT_NEAR(call, 0.0291, 1e-3);
}

TEST(FxBlack, DeltaMatchesFiniteDifference) {
  const double vol = 0.14, h = 1e-5;
  for (double K : {0.95, 1.10, 1.30}) {
    for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
      const double up = v::gk_price<double>(S + h, K, vol, T, RD, RF, cp);
      const double dn = v::gk_price<double>(S - h, K, vol, T, RD, RF, cp);
      EXPECT_NEAR(v::gk_delta<double>(S, K, vol, T, RD, RF, cp), (up - dn) / (2 * h), 1e-6) << "K=" << K;
    }
  }
}

TEST(FxBlack, VegaMatchesFiniteDifference) {
  const double vol = 0.14, h = 1e-6;
  for (double K : {0.95, 1.10, 1.30}) {
    const double up = v::gk_price<double>(S, K, vol + h, T, RD, RF, v::CallPut::Call);
    const double dn = v::gk_price<double>(S, K, vol - h, T, RD, RF, v::CallPut::Call);
    EXPECT_NEAR(v::gk_vega<double>(S, K, vol, T, RD, RF), (up - dn) / (2 * h), 1e-6) << "K=" << K;
  }
}

TEST(FxBlack, GammaMatchesFiniteDifference) {
  const double vol = 0.14, h = 1e-4;
  for (double K : {0.95, 1.10, 1.30}) {
    const double up = v::gk_price<double>(S + h, K, vol, T, RD, RF, v::CallPut::Call);
    const double mid = v::gk_price<double>(S, K, vol, T, RD, RF, v::CallPut::Call);
    const double dn = v::gk_price<double>(S - h, K, vol, T, RD, RF, v::CallPut::Call);
    EXPECT_NEAR(v::gk_gamma<double>(S, K, vol, T, RD, RF), (up - 2 * mid + dn) / (h * h), 1e-4) << "K=" << K;
  }
}

TEST(FxBlack, ThetaMatchesFiniteDifference) {
  // theta = dV/dt = -dV/dT.
  const double vol = 0.14, h = 1e-6;
  for (double K : {0.95, 1.10, 1.30}) {
    for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
      const double up = v::gk_price<double>(S, K, vol, T + h, RD, RF, cp);
      const double dn = v::gk_price<double>(S, K, vol, T - h, RD, RF, cp);
      EXPECT_NEAR(v::gk_theta<double>(S, K, vol, T, RD, RF, cp), -(up - dn) / (2 * h), 1e-5) << "K=" << K;
    }
  }
}

TEST(FxBlack, RhoDomMatchesFiniteDifference) {
  const double vol = 0.14, h = 1e-7;
  for (double K : {0.95, 1.10, 1.30}) {
    for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
      const double up = v::gk_price<double>(S, K, vol, T, RD + h, RF, cp);
      const double dn = v::gk_price<double>(S, K, vol, T, RD - h, RF, cp);
      EXPECT_NEAR(v::gk_rho_dom<double>(S, K, vol, T, RD, RF, cp), (up - dn) / (2 * h), 1e-5) << "K=" << K;
    }
  }
}

TEST(FxBlack, RhoForMatchesFiniteDifference) {
  const double vol = 0.14, h = 1e-7;
  for (double K : {0.95, 1.10, 1.30}) {
    for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
      const double up = v::gk_price<double>(S, K, vol, T, RD, RF + h, cp);
      const double dn = v::gk_price<double>(S, K, vol, T, RD, RF - h, cp);
      EXPECT_NEAR(v::gk_rho_for<double>(S, K, vol, T, RD, RF, cp), (up - dn) / (2 * h), 1e-5) << "K=" << K;
    }
  }
}

TEST(FxBlack, GreeksOnePassMatchesIndividual) {
  for (double K : {0.85, 1.00, 1.10, 1.25, 1.45}) {
    for (double vol : {0.05, 0.12, 0.25}) {
      for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
        const v::GkGreeks<double> g = v::gk_greeks<double>(S, K, vol, T, RD, RF, cp);
        EXPECT_NEAR(g.price, v::gk_price<double>(S, K, vol, T, RD, RF, cp), 1e-15) << "price";
        EXPECT_NEAR(g.delta, v::gk_delta<double>(S, K, vol, T, RD, RF, cp), 1e-15) << "delta";
        EXPECT_NEAR(g.gamma, v::gk_gamma<double>(S, K, vol, T, RD, RF), 1e-15) << "gamma";
        EXPECT_NEAR(g.vega, v::gk_vega<double>(S, K, vol, T, RD, RF), 1e-15) << "vega";
        EXPECT_NEAR(g.theta, v::gk_theta<double>(S, K, vol, T, RD, RF, cp), 1e-15) << "theta";
        EXPECT_NEAR(g.rho_dom, v::gk_rho_dom<double>(S, K, vol, T, RD, RF, cp), 1e-15) << "rho_dom";
        EXPECT_NEAR(g.rho_for, v::gk_rho_for<double>(S, K, vol, T, RD, RF, cp), 1e-15) << "rho_for";
      }
    }
  }
}

TEST(FxBlack, ImpliedVolRoundTrip) {
  for (double K : {0.90, 1.05, 1.10, 1.25, 1.40}) {
    for (double vol : {0.04, 0.10, 0.18, 0.30}) {
      for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
        const double price = v::gk_price<double>(S, K, vol, T, RD, RF, cp);
        const double iv = v::gk_implied_vol(price, S, K, T, RD, RF, cp);
        // Price inversion is well-posed; recovering vol on the low-vega wings (K far from the forward,
        // vega -> 0) is not, so state the round-trip where it is exact -- the re-priced value -- and only
        // sanity-band the vol itself.
        EXPECT_NEAR(v::gk_price<double>(S, K, iv, T, RD, RF, cp), price, 1e-12) << "K=" << K << " vol=" << vol;
        EXPECT_NEAR(iv, vol, 1e-4) << "K=" << K << " vol=" << vol;
      }
    }
  }
}

TEST(FxSmile, DeltaStrikeRoundTrip) {
  // A strike placed at |Δ|=0.25 must read back a 0.25 spot delta under the same vol.
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  const double df_for = std::exp(-RF * T);
  const double vol = 0.13;
  for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
    const double K = v::fx_strike_from_delta(fwd, T, df_for, vol, 0.25, cp);
    const double delta = v::gk_delta<double>(S, K, vol, T, RD, RF, cp);
    EXPECT_NEAR(std::abs(delta), 0.25, 1e-8) << (cp == v::CallPut::Call ? "call" : "put");
  }
}

TEST(FxSmile, AtmRrBfRoundTrip) {
  // Build a smile from {ATM,RR25,BF25,RR10,BF10}; the reconstructed wing strikes must read back the wing vols.
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  const double df_for = std::exp(-RF * T);
  v::FxDeltaQuotes q;
  q.atm = 0.11;
  q.rr25 = -0.015;  // put skew
  q.bf25 = 0.0030;
  q.has10 = true;
  q.rr10 = -0.028;
  q.bf10 = 0.0095;
  const v::FxVolSurface surf = v::FxVolSurface::from_delta_quotes(q, fwd, T, df_for);

  const double v25c = q.atm + q.bf25 + 0.5 * q.rr25;
  const double v25p = q.atm + q.bf25 - 0.5 * q.rr25;
  const double v10c = q.atm + q.bf10 + 0.5 * q.rr10;
  const double v10p = q.atm + q.bf10 - 0.5 * q.rr10;
  // ATM (delta-neutral straddle) knot reads back exactly.
  const double k_atm = fwd * std::exp(0.5 * q.atm * q.atm * T);
  EXPECT_NEAR(surf.vol_at_strike(k_atm), q.atm, 1e-12);
  // Wing knots read back exactly (PCHIP interpolates its knots).
  EXPECT_NEAR(surf.vol_at_strike(v::fx_strike_from_delta(fwd, T, df_for, v25c, 0.25, v::CallPut::Call)), v25c,
              1e-12);
  EXPECT_NEAR(surf.vol_at_strike(v::fx_strike_from_delta(fwd, T, df_for, v25p, 0.25, v::CallPut::Put)), v25p,
              1e-12);
  EXPECT_NEAR(surf.vol_at_strike(v::fx_strike_from_delta(fwd, T, df_for, v10c, 0.10, v::CallPut::Call)), v10c,
              1e-12);
  EXPECT_NEAR(surf.vol_at_strike(v::fx_strike_from_delta(fwd, T, df_for, v10p, 0.10, v::CallPut::Put)), v10p,
              1e-12);
}

TEST(FxSmile, BatchRepriceMatchesScalar) {
  // The SoA batch reprice must equal per-option gk_greeks, and by-delta options must resolve to a ~0.25 delta.
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  const double df_for = std::exp(-RF * T);
  v::FxDeltaQuotes q;
  q.atm = 0.12;
  q.rr25 = 0.010;
  q.bf25 = 0.0025;
  const v::FxVolSurface surf = v::FxVolSurface::from_delta_quotes(q, fwd, T, df_for);
  const v::FxSurfaceMarket mkt{S, RD, RF};

  std::vector<v::FxOption> book;
  book.push_back({"EURUSD", 1.05, 0.0, T, v::CallPut::Put, 1e6, false});
  book.push_back({"EURUSD", 1.10, 0.0, T, v::CallPut::Call, 2e6, false});
  book.push_back({"EURUSD", 0.0, 0.25, T, v::CallPut::Call, 1e6, true});   // by delta
  book.push_back({"EURUSD", 0.0, 0.25, T, v::CallPut::Put, 1e6, true});    // by delta

  const v::FxVolCube cube = v::price_fx_book(mkt, surf, book);
  ASSERT_EQ(cube.n, 4);
  for (int i = 0; i < cube.n; ++i) {
    const double vol = surf.vol_at_strike(cube.strike[i]);
    const v::GkGreeks<double> g =
        v::gk_greeks<double>(S, cube.strike[i], vol, T, RD, RF, book[i].cp);
    EXPECT_NEAR(cube.price[i], g.price, 1e-14) << "i=" << i;
    EXPECT_NEAR(cube.delta[i], g.delta, 1e-14) << "i=" << i;
    EXPECT_NEAR(cube.notional_price[i], g.price * book[i].notional, 1e-6) << "i=" << i;
  }
  EXPECT_NEAR(std::abs(cube.delta[2]), 0.25, 1e-6);  // by-delta call
  EXPECT_NEAR(std::abs(cube.delta[3]), 0.25, 1e-6);  // by-delta put
}
