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

// ------------------------------------------------------------------------------------------------------
// FOUR delta/ATM conventions (audit item #8). Frozen copies of the ORIGINAL single-convention formulas so we
// can prove the default paths are byte-identical (EXPECT_DOUBLE_EQ) after generalisation.
namespace {
double frozen_gk_delta(double spot, double strike, double vol, double expiry, double r_dom, double r_for,
                       v::CallPut cp) {
  const double sgn = v::cp_sign(cp);
  const double df_for = std::exp(-r_for * expiry);
  const double stddev = vol * std::sqrt(expiry);
  const double fwd = spot * std::exp((r_dom - r_for) * expiry);
  if (stddev <= 0.0) return df_for * sgn * ((sgn * (fwd - strike) > 0.0) ? 1.0 : 0.0);
  const double d1 = (std::log(fwd / strike) + 0.5 * stddev * stddev) / stddev;
  return df_for * sgn * v::normal_cdf(sgn * d1);
}
double frozen_strike_from_delta(double forward, double expiry, double df_for, double vol, double delta_mag,
                                v::CallPut cp) {
  const double stddev = vol * std::sqrt(expiry);
  const double d1 = v::cp_sign(cp) * v::norm_inv(delta_mag / df_for);
  return forward * std::exp(-d1 * stddev + 0.5 * stddev * stddev);
}
// Reference deltas straight from the convention definitions (independent of the header's dispatch).
double ref_delta(double spot, double strike, double vol, double expiry, double r_dom, double r_for,
                 v::CallPut cp, v::DeltaConv conv) {
  const double sgn = v::cp_sign(cp);
  const double df_for = std::exp(-r_for * expiry);
  const double stddev = vol * std::sqrt(expiry);
  const double fwd = spot * std::exp((r_dom - r_for) * expiry);
  const double d1 = (std::log(fwd / strike) + 0.5 * stddev * stddev) / stddev;
  const double d2 = d1 - stddev;
  const double disc = v::delta_is_spot(conv) ? df_for : 1.0;
  if (!v::delta_is_pa(conv)) return disc * sgn * v::normal_cdf(sgn * d1);
  return disc * sgn * (strike / fwd) * v::normal_cdf(sgn * d2);
}
}  // namespace

TEST(FxDeltaConv, DefaultPathByteIdentical) {
  // The generalised gk_delta(...,SpotUnadj), the 7-arg gk_delta, and the SpotUnadj strike inversion must all
  // be BIT-for-BIT equal to the frozen originals (and to each other) across a strike/vol grid.
  const double df_for = std::exp(-RF * T);
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  for (double K : {0.85, 1.00, 1.10, 1.25, 1.45}) {
    for (double vol : {0.05, 0.12, 0.25}) {
      for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
        const double frozen = frozen_gk_delta(S, K, vol, T, RD, RF, cp);
        EXPECT_DOUBLE_EQ(v::gk_delta<double>(S, K, vol, T, RD, RF, cp), frozen);  // 7-arg untouched
        EXPECT_DOUBLE_EQ(v::gk_delta<double>(S, K, vol, T, RD, RF, cp, v::DeltaConv::SpotUnadj), frozen);
      }
    }
  }
  for (double vol : {0.05, 0.12, 0.25}) {
    for (double dm : {0.10, 0.25, 0.40}) {
      for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
        const double frozen = frozen_strike_from_delta(fwd, T, df_for, vol, dm, cp);
        EXPECT_DOUBLE_EQ(v::fx_strike_from_delta(fwd, T, df_for, vol, dm, cp), frozen);  // 6-arg default
        EXPECT_DOUBLE_EQ(v::fx_strike_from_delta(fwd, T, df_for, vol, dm, cp, v::DeltaConv::SpotUnadj), frozen);
      }
    }
  }
}

TEST(FxDeltaConv, DeltaIdentitiesAtBenchmark) {
  // At one benchmark point the four deltas satisfy their defining relations: spot = df_for·forward for each
  // {unadjusted, premium-adjusted} family, and each equals its closed-form reference.
  const double vol = 0.12, K = 1.15;
  const double df_for = std::exp(-RF * T);
  for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
    const double su = v::gk_delta<double>(S, K, vol, T, RD, RF, cp, v::DeltaConv::SpotUnadj);
    const double fu = v::gk_delta<double>(S, K, vol, T, RD, RF, cp, v::DeltaConv::FwdUnadj);
    const double sp = v::gk_delta<double>(S, K, vol, T, RD, RF, cp, v::DeltaConv::SpotPA);
    const double fp = v::gk_delta<double>(S, K, vol, T, RD, RF, cp, v::DeltaConv::FwdPA);
    EXPECT_NEAR(su, df_for * fu, 1e-14);  // spot = df_for · forward (unadjusted)
    EXPECT_NEAR(sp, df_for * fp, 1e-14);  // spot = df_for · forward (premium-adjusted)
    for (auto conv : {v::DeltaConv::SpotUnadj, v::DeltaConv::FwdUnadj, v::DeltaConv::SpotPA,
                      v::DeltaConv::FwdPA})
      EXPECT_NEAR(v::gk_delta<double>(S, K, vol, T, RD, RF, cp, conv),
                  ref_delta(S, K, vol, T, RD, RF, cp, conv), 1e-14);
  }
  // gk_delta with SpotUnadj is exactly the price sensitivity ∂V/∂S (already FD-checked); forward-unadjusted
  // is that divided by df_for — i.e. N(d1) for a call.
  const double fu_call = v::gk_delta<double>(S, K, vol, T, RD, RF, v::CallPut::Call, v::DeltaConv::FwdUnadj);
  const double stddev = vol * std::sqrt(T);
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  const double d1 = (std::log(fwd / K) + 0.5 * stddev * stddev) / stddev;
  EXPECT_NEAR(fu_call, v::normal_cdf(d1), 1e-14);
}

TEST(FxDeltaConv, StrikeDeltaRoundTripAllConventions) {
  // For every convention, strike -> |delta| -> strike round-trips on the OTM (market) branch: OTM call K>F,
  // OTM put K<F -- the lower-|log-moneyness| root the premium-adjusted solver returns.
  const double vol = 0.13;
  const double df_for = std::exp(-RF * T);
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  for (auto conv : {v::DeltaConv::SpotUnadj, v::DeltaConv::FwdUnadj, v::DeltaConv::SpotPA,
                    v::DeltaConv::FwdPA}) {
    const double Kc = 1.15, Kp = 1.05;  // OTM call (K>F) and OTM put (K<F)
    const double dc = std::abs(v::gk_delta<double>(S, Kc, vol, T, RD, RF, v::CallPut::Call, conv));
    const double dp = std::abs(v::gk_delta<double>(S, Kp, vol, T, RD, RF, v::CallPut::Put, conv));
    EXPECT_NEAR(v::fx_strike_from_delta(fwd, T, df_for, vol, dc, v::CallPut::Call, conv), Kc, 1e-9);
    EXPECT_NEAR(v::fx_strike_from_delta(fwd, T, df_for, vol, dp, v::CallPut::Put, conv), Kp, 1e-9);
  }
}

TEST(FxDeltaConv, PremiumAdjustedTwoStrikeBranch) {
  // Premium-adjusted CALL delta is non-monotone in strike: it rises to an interior maximum, then falls. So a
  // target |Δ| below the max is met by TWO strikes; the solver must return the OTM one (smaller |log-moneyness|).
  const double vol = 0.13;
  const double df_for = std::exp(-RF * T);
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  const auto pa = [&](double K) {
    return v::gk_delta<double>(S, K, vol, T, RD, RF, v::CallPut::Call, v::DeltaConv::SpotPA);
  };
  // Scan log-moneyness for the maximum and the two roots of pa == target.
  double xstar = 0.0, dmax = -1.0;
  for (double x = -1.5; x <= 1.5; x += 1e-4) {
    const double d = pa(fwd * std::exp(x));
    if (d > dmax) { dmax = d; xstar = x; }
  }
  ASSERT_GT(dmax, 0.0);
  const double target = 0.85 * dmax;
  // Left (ITM, x<xstar) and right (OTM, x>xstar) roots by scanning.
  double xL = xstar, xR = xstar;
  for (double x = -1.5; x < xstar; x += 1e-4)
    if (pa(fwd * std::exp(x)) >= target) { xL = x; break; }
  for (double x = 1.5; x > xstar; x -= 1e-4)
    if (pa(fwd * std::exp(x)) >= target) { xR = x; break; }
  ASSERT_LT(xL, xstar);
  ASSERT_GT(xR, xstar);
  EXPECT_GT(std::abs(xL), std::abs(xR));  // the ITM partner is farther from the forward
  // Both strikes genuinely share the target delta (proving the two-strike ambiguity is real).
  EXPECT_NEAR(pa(fwd * std::exp(xL)), target, 5e-4);
  EXPECT_NEAR(pa(fwd * std::exp(xR)), target, 5e-4);
  // The solver returns the OTM (lower-|log-moneyness|) branch and round-trips there.
  const double Ksol = v::fx_strike_from_delta(fwd, T, df_for, vol, target, v::CallPut::Call, v::DeltaConv::SpotPA);
  EXPECT_NEAR(std::log(Ksol / fwd), xR, 2e-4);
  EXPECT_LT(std::abs(std::log(Ksol / fwd)), std::abs(xL));  // NOT the deep-ITM partner
  EXPECT_NEAR(pa(Ksol), target, 1e-9);
}

TEST(FxDeltaConv, AtmStrikeConventions) {
  const double vol = 0.12;
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  // ATM-forward is K = F under either premium flag.
  EXPECT_DOUBLE_EQ(v::fx_atm_strike(fwd, T, vol, v::AtmConv::Forward, false), fwd);
  EXPECT_DOUBLE_EQ(v::fx_atm_strike(fwd, T, vol, v::AtmConv::Forward, true), fwd);
  // DNS unadjusted = F·e^{+½σ²T} (byte-identical to the historical literal) and is spot-delta-neutral.
  const double k_dns_u = v::fx_atm_strike(fwd, T, vol, v::AtmConv::DeltaNeutral, false);
  EXPECT_DOUBLE_EQ(k_dns_u, fwd * std::exp(0.5 * vol * vol * T));
  EXPECT_GT(k_dns_u, fwd);
  const double dnc_u = v::gk_delta<double>(S, k_dns_u, vol, T, RD, RF, v::CallPut::Call, v::DeltaConv::SpotUnadj);
  const double dnp_u = v::gk_delta<double>(S, k_dns_u, vol, T, RD, RF, v::CallPut::Put, v::DeltaConv::SpotUnadj);
  EXPECT_NEAR(dnc_u + dnp_u, 0.0, 1e-12);
  // DNS premium-adjusted = F·e^{−½σ²T} and is PA-delta-neutral.
  const double k_dns_p = v::fx_atm_strike(fwd, T, vol, v::AtmConv::DeltaNeutral, true);
  EXPECT_DOUBLE_EQ(k_dns_p, fwd * std::exp(-0.5 * vol * vol * T));
  EXPECT_LT(k_dns_p, fwd);
  const double dnc_p = v::gk_delta<double>(S, k_dns_p, vol, T, RD, RF, v::CallPut::Call, v::DeltaConv::SpotPA);
  const double dnp_p = v::gk_delta<double>(S, k_dns_p, vol, T, RD, RF, v::CallPut::Put, v::DeltaConv::SpotPA);
  EXPECT_NEAR(dnc_p + dnp_p, 0.0, 1e-12);
}

TEST(FxDeltaConv, SmileDefaultBuildByteIdentical) {
  // from_delta_quotes with default conventions must reproduce the historical knots bit-for-bit.
  const double fwd = v::gk_forward<double>(S, T, RD, RF);
  const double df_for = std::exp(-RF * T);
  v::FxDeltaQuotes q;
  q.atm = 0.11; q.rr25 = -0.015; q.bf25 = 0.0030; q.has10 = true; q.rr10 = -0.028; q.bf10 = 0.0095;
  const v::FxVolSurface a = v::FxVolSurface::from_delta_quotes(q, fwd, T, df_for);
  const v::FxVolSurface b =
      v::FxVolSurface::from_delta_quotes(q, fwd, T, df_for, v::DeltaConv::SpotUnadj, v::AtmConv::DeltaNeutral);
  ASSERT_EQ(a.logm_knots().size(), b.logm_knots().size());
  for (std::size_t i = 0; i < a.logm_knots().size(); ++i) {
    EXPECT_DOUBLE_EQ(a.logm_knots()[i], b.logm_knots()[i]);
    EXPECT_DOUBLE_EQ(a.vol_knots()[i], b.vol_knots()[i]);
  }
}
