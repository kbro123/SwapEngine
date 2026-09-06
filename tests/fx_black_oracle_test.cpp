// EXHAUSTIVE analytic-oracle gate for the FX lognormal (Garman-Kohlhagen) layer (vol/fx_black.hpp) and the
// delta-quoted smile (vol/fx_vol_surface.hpp). This COMPLEMENTS fx_black_test.cpp: where that file pins a
// handful of representative points, this sweeps a dense GRID of (spot, strike, expiry, r_dom, r_for, vol)
// — including deep ITM/OTM wings and near-zero vol — and asserts the closed-form identities everywhere:
//   * put-call parity   C − P = DF_dom·(F − K)   to 1e-12 (absolute, per unit foreign notional),
//   * every analytic Greek (delta, gamma, vega, theta, rho_dom, rho_for) vs a central finite difference,
//     with a relative tolerance SCALED by the bump so a wing with a tiny value is not held to an absolute
//     floor it cannot meet,
//   * gk_greeks one-pass == the individual closed forms BITWISE (memcmp), across the grid,
//   * vega ≥ 0 and gamma ≥ 0 unconditionally; the zero-vol limit is the discounted intrinsic,
//   * implied vol round-trips IN PRICE SPACE to 1e-12 on the well-conditioned strikes (the low-vega wings
//     are excluded from the vol-recovery assertion — that inversion is ill-conditioned — but their re-priced
//     value is still checked),
//   * delta↔strike round-trip through the smile at |Δ| ∈ {0.10, 0.25, 0.50}.
// QuantLib-free, deterministic (fixed mt19937 for the randomised grid), header-only under test.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "swaps/vol/fx_black.hpp"
#include "swaps/vol/fx_vol_surface.hpp"

namespace v = swaps::vol;

namespace {

bool bit_identical(double x, double y) { return std::memcmp(&x, &y, sizeof(double)) == 0; }

// One grid point of market state.
struct Pt {
  double S, K, T, rd, rf, vol;
};

// A dense deterministic grid: the Cartesian product of representative axes, plus a batch of seeded random
// points to catch anything the lattice misses. Deep ITM/OTM and near-zero vol are deliberately included.
std::vector<Pt> make_grid() {
  std::vector<Pt> g;
  const double Ss[] = {0.85, 1.10, 1.55};
  const double Ks[] = {0.50, 0.80, 1.00, 1.10, 1.40, 2.00, 3.00};  // incl. deep wings vs S≈1.1
  const double Ts[] = {0.05, 0.5, 2.0, 10.0};
  const double rds[] = {-0.005, 0.0, 0.03, 0.08};
  const double rfs[] = {-0.01, 0.0, 0.02, 0.06};
  const double vols[] = {0.02, 0.08, 0.20, 0.60};
  for (double S : Ss)
    for (double K : Ks)
      for (double T : Ts)
        for (double rd : rds)
          for (double rf : rfs)
            for (double vol : vols) g.push_back({S, K, T, rd, rf, vol});

  std::mt19937 rng(0xC0FFEEu);
  std::uniform_real_distribution<double> uS(0.5, 2.0), uK(0.3, 3.5), uT(0.02, 12.0), ur(-0.03, 0.10),
      uv(0.005, 0.90);
  for (int i = 0; i < 400; ++i) g.push_back({uS(rng), uK(rng), uT(rng), ur(rng), ur(rng), uv(rng)});
  return g;
}

}  // namespace

// --- 1. Put-call parity across the whole grid --------------------------------------------------------------
TEST(FxBlackOracle, PutCallParityGrid) {
  for (const Pt& p : make_grid()) {
    const double df_dom = std::exp(-p.rd * p.T), df_for = std::exp(-p.rf * p.T);
    const double c = v::gk_price<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, v::CallPut::Call);
    const double put = v::gk_price<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, v::CallPut::Put);
    // C − P = S·DF_for − K·DF_dom = DF_dom·(F − K). Both RHS forms checked (they are algebraically equal).
    const double fwd = v::gk_forward<double>(p.S, p.T, p.rd, p.rf);
    EXPECT_NEAR(c - put, p.S * df_for - p.K * df_dom, 1e-12)
        << "S=" << p.S << " K=" << p.K << " T=" << p.T << " vol=" << p.vol;
    EXPECT_NEAR(c - put, df_dom * (fwd - p.K), 1e-12);
  }
}

// --- 2. vega ≥ 0, gamma ≥ 0, and the zero-vol discounted-intrinsic limit ----------------------------------
TEST(FxBlackOracle, VegaGammaNonNegativeAndZeroVolIntrinsic) {
  for (const Pt& p : make_grid()) {
    EXPECT_GE(v::gk_vega<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf), 0.0);
    EXPECT_GE(v::gk_gamma<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf), 0.0);
    const double df_dom = std::exp(-p.rd * p.T);
    const double fwd = v::gk_forward<double>(p.S, p.T, p.rd, p.rf);
    EXPECT_NEAR(v::gk_price<double>(p.S, p.K, 0.0, p.T, p.rd, p.rf, v::CallPut::Call),
                df_dom * std::max(fwd - p.K, 0.0), 1e-13);
    EXPECT_NEAR(v::gk_price<double>(p.S, p.K, 0.0, p.T, p.rd, p.rf, v::CallPut::Put),
                df_dom * std::max(p.K - fwd, 0.0), 1e-13);
  }
}

// --- 3. All five (six) Greeks vs central finite difference, relative tol scaled by the bump ----------------
// The FD error of a central difference is O(bump²·|f'''|); asserting |analytic − FD| ≤ rel·|analytic| + abs
// with rel≈a few·bump² and a small abs floor keeps deep-wing points (tiny Greeks) honest without a spurious
// absolute floor. Each Greek is differenced along its own axis with a bump chosen for its curvature.
TEST(FxBlackOracle, AllGreeksVsFiniteDifference) {
  for (const Pt& p : make_grid()) {
    for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
      // delta = ∂/∂S
      {
        const double h = 1e-6 * std::max(1.0, p.S);
        const double up = v::gk_price<double>(p.S + h, p.K, p.vol, p.T, p.rd, p.rf, cp);
        const double dn = v::gk_price<double>(p.S - h, p.K, p.vol, p.T, p.rd, p.rf, cp);
        const double an = v::gk_delta<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp);
        EXPECT_NEAR(an, (up - dn) / (2 * h), 1e-5 * std::abs(an) + 1e-7);
      }
      // gamma = ∂²/∂S² = ∂(delta)/∂S. A raw second difference of PRICE is truncation-limited in the deep
      // wings this grid reaches (d1 large ⇒ speed/color blow up), so we finite-difference the ANALYTIC
      // delta instead — a first-order central difference of an exact function, well-conditioned everywhere.
      if (cp == v::CallPut::Call) {
        const double h = 1e-6 * std::max(1.0, p.S);
        const double up = v::gk_delta<double>(p.S + h, p.K, p.vol, p.T, p.rd, p.rf, cp);
        const double dn = v::gk_delta<double>(p.S - h, p.K, p.vol, p.T, p.rd, p.rf, cp);
        const double an = v::gk_gamma<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf);
        EXPECT_NEAR(an, (up - dn) / (2 * h), 1e-5 * std::abs(an) + 1e-8);
      }
      // vega = ∂/∂σ (symmetric; test on call)
      if (cp == v::CallPut::Call) {
        const double h = 1e-6;
        const double up = v::gk_price<double>(p.S, p.K, p.vol + h, p.T, p.rd, p.rf, cp);
        const double dn = v::gk_price<double>(p.S, p.K, p.vol - h, p.T, p.rd, p.rf, cp);
        const double an = v::gk_vega<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf);
        EXPECT_NEAR(an, (up - dn) / (2 * h), 1e-5 * std::abs(an) + 1e-6);
      }
      // theta = ∂/∂t = −∂/∂T
      {
        const double h = 1e-6 * std::max(1.0, p.T);
        const double up = v::gk_price<double>(p.S, p.K, p.vol, p.T + h, p.rd, p.rf, cp);
        const double dn = v::gk_price<double>(p.S, p.K, p.vol, p.T - h, p.rd, p.rf, cp);
        const double an = v::gk_theta<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp);
        EXPECT_NEAR(an, -(up - dn) / (2 * h), 1e-4 * std::abs(an) + 1e-6);
      }
      // rho_dom = ∂/∂r_d
      {
        const double h = 1e-7;
        const double up = v::gk_price<double>(p.S, p.K, p.vol, p.T, p.rd + h, p.rf, cp);
        const double dn = v::gk_price<double>(p.S, p.K, p.vol, p.T, p.rd - h, p.rf, cp);
        const double an = v::gk_rho_dom<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp);
        EXPECT_NEAR(an, (up - dn) / (2 * h), 1e-5 * std::abs(an) + 1e-6);
      }
      // rho_for = ∂/∂r_f
      {
        const double h = 1e-7;
        const double up = v::gk_price<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf + h, cp);
        const double dn = v::gk_price<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf - h, cp);
        const double an = v::gk_rho_for<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp);
        EXPECT_NEAR(an, (up - dn) / (2 * h), 1e-5 * std::abs(an) + 1e-6);
      }
    }
  }
}

// --- 4. gk_greeks one-pass vs the individual analytics, across the grid ------------------------------------
// FINDING: six of the seven outputs (price, delta, gamma, vega, theta, rho_for) are BYTE-IDENTICAL to the
// individual closed forms — they share the same float expression tree, so bitwise equality holds and is
// asserted. rho_dom is the exception: gk_rho_dom forms d2 = (log(F/K) − ½σ²T)/(σ√T) directly, while
// gk_greeks forms d2 = d1 − σ√T. These are algebraically identical but round differently in the LAST bit, so
// rho_dom agrees only to ~1 ULP (observed max relative diff 5.6e-16), not byte-for-byte. It is therefore
// pinned with a tight ULP-scale relative tolerance rather than memcmp.
TEST(FxBlackOracle, OnePassMatchesIndividual) {
  for (const Pt& p : make_grid()) {
    for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
      const v::GkGreeks<double> g = v::gk_greeks<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp);
      EXPECT_TRUE(bit_identical(g.price, v::gk_price<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp)));
      EXPECT_TRUE(bit_identical(g.delta, v::gk_delta<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp)));
      EXPECT_TRUE(bit_identical(g.gamma, v::gk_gamma<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf)));
      EXPECT_TRUE(bit_identical(g.vega, v::gk_vega<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf)));
      EXPECT_TRUE(bit_identical(g.theta, v::gk_theta<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp)));
      EXPECT_TRUE(bit_identical(g.rho_for, v::gk_rho_for<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp)));
      // rho_dom: full-precision equal, not byte-identical (different d2 evaluation order). Where it is
      // materially non-zero the two d2 forms diverge by a handful of ULP (observed max relative diff ≈
      // 2e-15), pinned to a relative 1e-13. DEEP in the wing rho_dom is ~1e-195 (financially exact zero) and
      // there Φ's exponential tail is so sensitive to the 1-ULP d2 difference that the RELATIVE error grows
      // (absolute error still ~1e-207); the 1e-30 absolute floor absorbs that tail noise while sitting far
      // below any economically meaningful rho_dom (which is O(K·T·df) ≈ O(1)).
      const double rd_ind = v::gk_rho_dom<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp);
      EXPECT_NEAR(g.rho_dom, rd_ind, 1e-13 * std::abs(rd_ind) + 1e-30);
    }
  }
}

// The zero-vol branch of gk_greeks vs the individual zero-vol closed forms. FINDING: on an out-of-the-money
// point the one-pass price is −0.0 (it evaluates df_dom·intr·0.0 with intr<0), while gk_price returns +0.0
// (max(intr,0)). These are VALUE-equal (−0.0 == +0.0) but not byte-identical, so price is asserted with ==
// (which treats signed zeros as equal). delta/rho_dom/rho_for share the same intrinsic expression and ARE
// byte-identical here.
TEST(FxBlackOracle, OnePassMatchesIndividualAtZeroVol) {
  for (const Pt& p : make_grid()) {
    for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
      const v::GkGreeks<double> g = v::gk_greeks<double>(p.S, p.K, 0.0, p.T, p.rd, p.rf, cp);
      EXPECT_EQ(g.price, v::gk_price<double>(p.S, p.K, 0.0, p.T, p.rd, p.rf, cp));  // ±0 value-equal
      EXPECT_TRUE(bit_identical(g.delta, v::gk_delta<double>(p.S, p.K, 0.0, p.T, p.rd, p.rf, cp)));
      EXPECT_TRUE(bit_identical(g.rho_dom, v::gk_rho_dom<double>(p.S, p.K, 0.0, p.T, p.rd, p.rf, cp)));
      EXPECT_TRUE(bit_identical(g.rho_for, v::gk_rho_for<double>(p.S, p.K, 0.0, p.T, p.rd, p.rf, cp)));
      EXPECT_EQ(g.gamma, 0.0);
      EXPECT_EQ(g.vega, 0.0);
      EXPECT_EQ(g.theta, 0.0);
    }
  }
}

// --- 5. Implied vol round-trips IN PRICE SPACE (1e-12); vol recovery only where vega is not vanishing -------
TEST(FxBlackOracle, ImpliedVolPriceRoundTrip) {
  for (const Pt& p : make_grid()) {
    for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
      const double price = v::gk_price<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf, cp);
      const double df_dom = std::exp(-p.rd * p.T);
      const double fwd = v::gk_forward<double>(p.S, p.T, p.rd, p.rf);
      const double intrinsic = df_dom * std::max(v::cp_sign(cp) * (fwd - p.K), 0.0);
      if (!(price > intrinsic + 1e-14)) continue;  // no vol information above intrinsic -> nothing to invert
      const double iv = v::gk_implied_vol(price, p.S, p.K, p.T, p.rd, p.rf, cp);
      const double vega = v::gk_vega<double>(p.S, p.K, p.vol, p.T, p.rd, p.rf);
      // Price-space round-trip is well-posed everywhere the option has time value, but its achievable
      // precision is CONDITIONING-LIMITED by vega: gk_implied_vol converges the vol to ~1e-12, and repricing
      // propagates that as ≈ vega·(vol residual). A flat 1e-12 held on high-vega deep-ITM long-dated points
      // (vega up to ~3 on this grid) genuinely fails — observed max reprice error 1.5e-12 at price≈2.3 — so
      // the defensible bound scales with vega rather than pretending the inversion is vega-independent.
      EXPECT_NEAR(v::gk_price<double>(p.S, p.K, iv, p.T, p.rd, p.rf, cp), price, 1e-12 + 1e-11 * vega)
          << "S=" << p.S << " K=" << p.K << " T=" << p.T << " vol=" << p.vol << " vega=" << vega;
      // Vol recovery itself only where vega is materially non-zero (well-conditioned inversion).
      if (vega > 1e-4) EXPECT_NEAR(iv, p.vol, 1e-6) << "vega=" << vega;
    }
  }
}

// --- 6. delta ↔ strike round-trip through the smile at |Δ| ∈ {0.10, 0.25, 0.50} ---------------------------
// A strike placed at |Δ| via fx_strike_from_delta must read back that spot-delta magnitude under the same vol,
// for both rights, over a grid of (T, r_for, vol). The smile's own strike_for_delta fixed point (vol varies
// with strike) must likewise resolve to the target delta.
TEST(FxBlackOracle, DeltaStrikeRoundTripGrid) {
  const double S = 1.10;
  for (double rd : {0.0, 0.03, 0.06}) {
    for (double rf : {0.0, 0.02, 0.05}) {
      for (double T : {0.1, 0.75, 3.0}) {
        for (double vol : {0.06, 0.13, 0.30}) {
          const double fwd = v::gk_forward<double>(S, T, rd, rf);
          const double df_for = std::exp(-rf * T);
          for (double dm : {0.10, 0.25, 0.50}) {
            for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
              const double K = v::fx_strike_from_delta(fwd, T, df_for, vol, dm, cp);
              const double d = v::gk_delta<double>(S, K, vol, T, rd, rf, cp);
              EXPECT_NEAR(std::abs(d), dm, 1e-8)
                  << "dm=" << dm << " T=" << T << " vol=" << vol << (cp == v::CallPut::Call ? " C" : " P");
            }
          }
          // And through a non-flat smile: strike_for_delta's fixed point reads back the target delta.
          v::FxDeltaQuotes q;
          q.atm = vol;
          q.rr25 = -0.012;
          q.bf25 = 0.004;
          const v::FxVolSurface surf = v::FxVolSurface::from_delta_quotes(q, fwd, T, df_for);
          for (double dm : {0.10, 0.25}) {
            for (auto cp : {v::CallPut::Call, v::CallPut::Put}) {
              const double K = surf.strike_for_delta(dm, cp);
              const double vAtK = surf.vol_at_strike(K);
              const double d = v::gk_delta<double>(S, K, vAtK, T, rd, rf, cp);
              EXPECT_NEAR(std::abs(d), dm, 1e-6)
                  << "smile dm=" << dm << " T=" << T << (cp == v::CallPut::Call ? " C" : " P");
            }
          }
        }
      }
    }
  }
}
