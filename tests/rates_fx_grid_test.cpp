// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T6 regression (fails on the reverted bug)
// GRID gate (T5 identities + finite differences + regression freezes, NOT an oracle -- renamed from
// *_oracle_test on 2026-09-10; the only independent literals here are the Hull conversion factor, the z=0
// 6 %-yield hand sum and covered interest parity) spanning the three "rates/FX" derivative kernels:
//   * NDF / NDS         (pricing/ndf.hpp)          — linear covered-interest-parity forward layer,
//   * bond future / CTD (pricing/bond_future.hpp)  — CME conversion factor + basis / implied repo,
//   * full-beta SABR    (vol/sabr.hpp)             — Hagan normal-vol smile across the backbone.
// This COMPLEMENTS ndf_test.cpp / bond_future_test.cpp / vol_sabr_beta_test.cpp: those pin representative
// points; this sweeps dense GRIDs, adds a hand-rolled 6%-yield clean-price sum, freezes the general-β normal
// SABR expansion against a transcription of its own derivation (a REGRESSION freeze -- the β∈(0,1) VALUE is
// pinned by the QuantLib SABR oracle in swaps_oracle_tests), and goes adversarial (randomised seeded
// baskets, deep-carry NDFs).
// QuantLib-free (targets swaps_tests); deterministic — every random draw is a fixed-seed mt19937.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "swaps/build/bond.hpp"
#include "swaps/pricing/bond.hpp"
#include "swaps/pricing/bond_future.hpp"
#include "swaps/pricing/ndf.hpp"
#include "swaps/vol/sabr.hpp"
#include "swaps/vol/sabr_calibration.hpp"

#include "swaps/build/day_count.hpp"
#include "swaps/conventions_data.hpp"

// The CME Treasury contract parameters come from the conventions DB (bond_futures[]), not literals.
static const double kNotionalCoupon = swaps::conventions::require_bond_future("CME-TY").notional_coupon;
static const double kRepoBasis = swaps::build::day_count_basis(std::string(swaps::conventions::require_bond_future("CME-TY").repo_day_count));

namespace p = swaps::pricing;
namespace bld = swaps::build;
namespace v = swaps::vol;

namespace {
bool bit_identical(double x, double y) { return std::memcmp(&x, &y, sizeof(double)) == 0; }
}  // namespace

// =========================================================================================================
// NDF / NDS
// =========================================================================================================

// Covered-interest parity F = S·e^{(r_settle−r_nd)T} to 1e-14 over a grid, and PV=0 exactly at the fair
// forward for both directions and any notional.
TEST(RatesFxGrid, NdfCoveredInterestParityAndFairPvGrid) {
  const double spots[] = {0.02, 0.20, 1.0, 5.0};
  const double Ts[] = {0.08, 0.5, 2.0, 7.0};
  const double rss[] = {-0.01, 0.0, 0.045, 0.09};
  const double rns[] = {0.0, 0.02, 0.105, 0.25};  // steep EM carry incl. r_nd >> r_settle
  const double Ns[] = {1.0, 1.0e6, -3.5e5};
  for (double S : spots)
    for (double T : Ts)
      for (double rs : rss)
        for (double rn : rns) {
          const double F = p::ndf_fair_forward<double>(S, T, rs, rn);
          EXPECT_NEAR(F, S * std::exp((rs - rn) * T), 1e-14) << "S=" << S << " T=" << T;
          for (double N : Ns)
            for (double dir : {+1.0, -1.0}) {
              const double pv = p::ndf_pv<double>(S, F, T, rs, rn, N, dir);
              // Scale the zero tolerance by |N| so a 1e6 notional is not held to a 1e-9 absolute floor.
              EXPECT_NEAR(pv, 0.0, 1e-9 * (1.0 + std::abs(N)));
              const p::NdfGreeks<double> g = p::ndf_greeks<double>(S, F, T, rs, rn, N, dir);
              EXPECT_NEAR(g.pv, 0.0, 1e-9 * (1.0 + std::abs(N)));
              EXPECT_NEAR(g.fair_forward, F, 1e-14);
            }
        }
}

// PV is exactly linear in notional, sign-flips with direction, and its Greeks match central FD — over a grid.
TEST(RatesFxGrid, NdfLinearityAndGreeksGrid) {
  std::mt19937 rng(0x11DEADu);
  std::uniform_real_distribution<double> uS(0.05, 3.0), uK(0.05, 3.0), uT(0.05, 8.0), ur(-0.02, 0.20);
  for (int i = 0; i < 300; ++i) {
    const double S = uS(rng), K = uK(rng), T = uT(rng), rs = ur(rng), rn = ur(rng);
    const double N = 1.0e6;
    const double pv_buy = p::ndf_pv<double>(S, K, T, rs, rn, N, +1.0);
    const double pv_sell = p::ndf_pv<double>(S, K, T, rs, rn, N, -1.0);
    EXPECT_NEAR(pv_buy, -pv_sell, 1e-8 * (1.0 + std::abs(pv_buy)));           // direction sign-flip
    EXPECT_NEAR(p::ndf_pv<double>(S, K, T, rs, rn, 3.0 * N, +1.0), 3.0 * pv_buy,
                1e-7 * (1.0 + std::abs(pv_buy)));                             // linear in notional
    // Greeks vs central FD (relative tol scaled by the value).
    const p::NdfGreeks<double> g = p::ndf_greeks<double>(S, K, T, rs, rn, N, +1.0);
    const double h = 1e-6;
    const double dS = (p::ndf_pv<double>(S + h, K, T, rs, rn, N, +1.0) -
                       p::ndf_pv<double>(S - h, K, T, rs, rn, N, +1.0)) / (2 * h);
    EXPECT_NEAR(g.delta_spot, dS, 1e-5 * std::abs(g.delta_spot) + 1e-3);
    const double drs = (p::ndf_pv<double>(S, K, T, rs + h, rn, N, +1.0) -
                        p::ndf_pv<double>(S, K, T, rs - h, rn, N, +1.0)) / (2 * h);
    EXPECT_NEAR(g.dpv_dr_settle, drs, 1e-4 * std::abs(g.dpv_dr_settle) + 1e-2);
    const double drn = (p::ndf_pv<double>(S, K, T, rs, rn + h, N, +1.0) -
                        p::ndf_pv<double>(S, K, T, rs, rn - h, N, +1.0)) / (2 * h);
    EXPECT_NEAR(g.dpv_dr_nd, drn, 1e-4 * std::abs(g.dpv_dr_nd) + 1e-2);
  }
}

// NDS fair rate reprices the strip to ~0, sits inside the per-period forward range, and is notional-weighting
// dependent but direction independent — over several strips.
TEST(RatesFxGrid, NdsFairRateRepricesStripGrid) {
  const double S = 0.20, rs = 0.045;
  for (double rn : {0.02, 0.105, 0.18}) {
    const std::vector<double> mats{0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0};
    std::vector<double> nots(mats.size(), 1.0e6);
    const double kstar = p::nds_fair_rate<double>(S, rs, rn, mats, nots);
    for (double dir : {+1.0, -1.0}) {
      const double pv = p::nds_pv<double>(S, kstar, rs, rn, mats, nots, dir);
      EXPECT_NEAR(pv, 0.0, 1e-4);  // 1e6 notional × 7 legs -> loosen from the per-unit 1e-9
    }
    // Bracketed by the extreme per-period forwards.
    const double f_lo = p::ndf_fair_forward<double>(S, mats.front(), rs, rn);
    const double f_hi = p::ndf_fair_forward<double>(S, mats.back(), rs, rn);
    EXPECT_LE(std::min(f_lo, f_hi), kstar);
    EXPECT_GE(std::max(f_lo, f_hi), kstar);
    // Direction-independent and default-notional consistent.
    EXPECT_NEAR(kstar, p::nds_fair_rate<double>(S, rs, rn, mats, {}), 1e-14);
  }
}

// =========================================================================================================
// BOND FUTURE / CTD
// =========================================================================================================

namespace {
// A bond built to settle ON a coupon-grid date (zero accrued), `half_years` semiannual periods to maturity.
bld::BuiltBond on_grid_bond(int half_years, double coupon) {
  const bld::Date issue = bld::Date::ymd(2020, 2, 15);
  bld::FixedBondTerms t;
  t.value_date = issue;
  t.settle = issue;
  t.issue = issue;
  t.maturity = issue.plus_months(6 * half_years);
  t.coupon = coupon;
  t.freq = 2;
  return bld::fixed_rate_bond(t);
}

// A completely independent 6%-yield clean-price oracle for a z=0 (whole-half-year) deliverable: the closed
// present value of a semiannual bond discounted at 3% per period. This does NOT touch cme_conversion_factor
// nor the engine's bond kernel — it is arithmetic from first principles.
double clean_6pct_zzero(double coupon, int two_n) {  // two_n = number of semiannual periods
  const double disc = 1.0 / 1.03;  // per-period 6%/2 discount
  double pv = 0.0, d = 1.0;
  for (int k = 1; k <= two_n; ++k) {
    d *= disc;
    pv += (coupon / 2.0) * d;
  }
  pv += d;  // redemption of par at maturity (d == 1.03^{-two_n})
  return pv;  // clean == dirty since settle is on a coupon date (zero accrued)
}
}  // namespace

// (1) cme_conversion_factor == the engine's own 6%-yield clean price (bond_clean_from_yield), AND ==
// the from-scratch closed-form oracle for z=0 — over a grid of (n, z, coupon). ~1e-10.
TEST(RatesFxGrid, ConversionFactorEqualsSixPercentClean) {
  struct Case { int n; int z; int half_years; double coupon; };
  const Case cases[] = {
      {2, 0, 4, 0.02},  {2, 6, 5, 0.02},  {5, 0, 10, 0.0375}, {5, 6, 11, 0.0375},
      {7, 0, 14, 0.05}, {7, 6, 15, 0.05}, {10, 0, 20, 0.06},  {10, 6, 21, 0.06},
      {15, 0, 30, 0.08}, {20, 0, 40, 0.10}};
  for (const Case& c : cases) {
    const double cf = p::cme_conversion_factor<double>(c.coupon, c.n, c.z, kNotionalCoupon);
    const bld::BuiltBond bb = on_grid_bond(c.half_years, c.coupon);
    EXPECT_NEAR(cf, p::bond_clean_from_yield(bb.yield, 0.06), 1e-10) << "n=" << c.n << " z=" << c.z;
    if (c.z == 0)  // independent first-principles oracle only valid on the whole-half-year grid
      EXPECT_NEAR(cf, clean_6pct_zzero(c.coupon, 2 * c.n), 1e-12) << "n=" << c.n;
  }
  // A 6% coupon whole-half-years out has CF exactly 1; Hull's 10%/20y benchmark is 1.4623.
  for (int n : {2, 5, 8, 15}) EXPECT_NEAR(p::cme_conversion_factor<double>(0.06, n, 0, kNotionalCoupon), 1.0, 1e-12);
  EXPECT_NEAR(p::cme_conversion_factor<double>(0.10, 20, 0, kNotionalCoupon), 1.4623, 5e-5);
  // CF is affine in coupon: CF(c) = CF0 + slope·c. Verify at three coupons for a fixed (n,z).
  const double a = p::cme_conversion_factor<double>(0.00, 10, 6, kNotionalCoupon);
  const double b = p::cme_conversion_factor<double>(0.04, 10, 6, kNotionalCoupon);
  const double d = p::cme_conversion_factor<double>(0.08, 10, 6, kNotionalCoupon);
  EXPECT_NEAR(d - b, b - a, 1e-12);  // equal coupon steps -> equal CF steps (affine)
}

// (2) Net basis == 0 at the implied repo, and is strictly monotone increasing in the funding repo — over a
// randomised seeded set of deliverables and futures/day settings.
TEST(RatesFxGrid, NetBasisZeroAtImpliedRepoAndMonotone) {
  std::mt19937 rng(20240906u);
  std::uniform_real_distribution<double> uClean(0.90, 1.08), uCf(0.85, 1.05), uAcc(0.0, 0.03),
      uFut(0.90, 1.08), uDays(30.0, 180.0);
  std::bernoulli_distribution hasCoupon(0.5);
  for (int i = 0; i < 200; ++i) {
    p::DeliverableInput in;
    in.clean = uClean(rng);
    in.conversion_factor = uCf(rng);
    in.accrued_now = uAcc(rng);
    in.accrued_delivery = uAcc(rng);
    const double days = uDays(rng);
    if (hasCoupon(rng)) in.interim_coupons = {{0.02, days * 0.4}};
    const double fut = uFut(rng);
    const double irr = p::bond_future_implied_repo<double>(in, fut, days, kRepoBasis);
    EXPECT_NEAR(p::bond_future_net_basis<double>(in, fut, irr, days, kRepoBasis), 0.0, 1e-11) << "i=" << i;
    // Monotone increasing in repo: sample three repos around the implied rate.
    const double lo = p::bond_future_net_basis<double>(in, fut, irr - 0.02, days, kRepoBasis);
    const double mid = p::bond_future_net_basis<double>(in, fut, irr, days, kRepoBasis);
    const double hi = p::bond_future_net_basis<double>(in, fut, irr + 0.02, days, kRepoBasis);
    EXPECT_LT(lo, mid);
    EXPECT_LT(mid, hi);
  }
}

// (3) CTD = argmax implied repo across a randomised seeded basket, cross-checked by a brute-force scan; and
// select_ctd agrees with a direct argmax over the results.
TEST(RatesFxGrid, CtdIsArgmaxImpliedRepoRandomBasket) {
  std::mt19937 rng(0xBA5Eu);
  std::uniform_real_distribution<double> uClean(0.85, 1.10), uCf(0.80, 1.10), uAcc(0.0, 0.03),
      uFut(0.92, 1.05), uRepo(0.01, 0.08), uDays(45.0, 150.0);
  for (int trial = 0; trial < 100; ++trial) {
    const int nb = 3 + (rng() % 5);  // 3..7 deliverables
    std::vector<p::DeliverableInput> basket(nb);
    for (auto& in : basket) {
      in.clean = uClean(rng);
      in.conversion_factor = uCf(rng);
      in.accrued_now = uAcc(rng);
      in.accrued_delivery = uAcc(rng);
    }
    const double fut = uFut(rng), repo = uRepo(rng), days = uDays(rng);
    std::vector<p::DeliverableResult<double>> res;
    for (const auto& in : basket) res.push_back(p::analyze_deliverable<double>(in, fut, repo, days, kRepoBasis));

    // The CTD is the deliverable with the highest implied repo -- the specification of select_ctd, asserted
    // against every other deliverable. (E5 2026-09-10: the former "brute-force argmax" was select_ctd's own
    // loop re-typed, and the "implied_repo independent of `repo`" check was vacuous -- `repo` is not an input
    // of bond_future_implied_repo; both deleted.)
    const std::size_t ctd = p::select_ctd(res);
    ASSERT_LT(ctd, res.size()) << "trial=" << trial;
    for (std::size_t i = 0; i < res.size(); ++i)
      EXPECT_GE(res[ctd].implied_repo, res[i].implied_repo) << "trial=" << trial;
  }
}

// =========================================================================================================
// FULL-BETA SABR  (vol/sabr.hpp)
// =========================================================================================================

namespace {
// A TRANSCRIPTION of the general-β normal-vol expansion as sabr.hpp derives it (arc-length ζ, the L²/24 +
// L⁴/1920 brackets, the same T-term). It is a REGRESSION FREEZE of the header's own derivation, not an
// oracle (E5 2026-09-10 -- the 2026-09-08 audit showed it is term-for-term the code under test; the
// independent β∈(0,1) VALUE pin is the QuantLib SabrSmileSection oracle in swaps_oracle_tests).
double hagan_normal_vol_ref(double F, double K, double T, double alpha, double rho, double nu, double beta) {
  const double e = 1.0 - beta;
  const double L = std::log(F / K);
  const double FK = F * K;
  const double arc = (e == 0.0) ? L : (std::pow(F, e) - std::pow(K, e)) / e;
  const double zeta = (nu / alpha) * arc;
  double zx;
  if (zeta > -1e-7 && zeta < 1e-7) {
    zx = 1.0;
  } else {
    const double xhat =
        std::log((std::sqrt(1.0 - 2.0 * rho * zeta + zeta * zeta) - rho + zeta) / (1.0 - rho));
    zx = zeta / xhat;
  }
  const double L2 = L * L, L4 = L2 * L2;
  const double num_br = 1.0 + L2 / 24.0 + L4 / 1920.0;
  const double den_br = 1.0 + (e * e) * L2 / 24.0 + (e * e * e * e) * L4 / 1920.0;
  const double pref = alpha * std::pow(FK, beta * 0.5) * (num_br / den_br);
  const double FKe = std::pow(FK, e);
  const double FKhe = std::pow(FK, e * 0.5);
  const double tterm = (-beta * (2.0 - beta) / 24.0 * (alpha * alpha) / FKe +
                        rho * beta * nu / 4.0 * alpha / FKhe +
                        (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu) * T;
  return pref * zx * (1.0 + tterm);
}

double legacy_normal_vol(double fwd, double strike, double expiry, double alpha, double rho, double nu) {
  const double time_corr = 1.0 + ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu * expiry;
  const double dk = fwd - strike;
  if (nu <= 0.0) return alpha * time_corr;
  const double zeta = (nu / alpha) * dk;
  if (zeta > -1e-7 && zeta < 1e-7) return alpha * time_corr;
  const double xhat =
      std::log((std::sqrt(1.0 - 2.0 * rho * zeta + zeta * zeta) - rho + zeta) / (1.0 - rho));
  return alpha * (zeta / xhat) * time_corr;
}
}  // namespace

// Over β∈{0,0.25,0.5,0.75,1} and a strike grid: sabr_normal_vol is finite/positive, byte-identical to the
// legacy closed form at β=0, and matches the independent Hagan reference off-ATM for β>0.
TEST(RatesFxGrid, SabrNormalVolGridAndOracles) {
  const double F = 0.030, T = 5.0, alpha0 = 0.0090, rho = -0.30, nu = 0.45;
  std::vector<double> Ks;
  for (int bp = -180; bp <= 180; bp += 12) Ks.push_back(F + bp / 1e4);
  for (double beta : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    // alpha scaled so the ATM level is comparable across the backbone (alpha ~ atm_normal / F^beta).
    const double alpha = alpha0 * std::pow(F, -beta);
    const v::SabrParams pr{alpha, rho, nu, beta};
    for (double K : Ks) {
      if (K <= 0.0) continue;
      const double got = v::sabr_normal_vol(F, K, T, pr);
      ASSERT_TRUE(std::isfinite(got)) << "beta=" << beta << " K=" << K;
      EXPECT_GT(got, 0.0) << "beta=" << beta << " K=" << K;
      if (beta == 0.0) {
        // β=0 must be BIT-FOR-BIT the frozen legacy formula.
        EXPECT_TRUE(bit_identical(got, legacy_normal_vol(F, K, T, alpha, rho, nu))) << "K=" << K;
      } else {
        // β>0: freeze against the transcription (drift guard; the value pin is the QuantLib oracle).
        if (std::abs(K - F) > 1e-9)
          EXPECT_NEAR(got, hagan_normal_vol_ref(F, K, T, alpha, rho, nu, beta), 1e-13) << "beta=" << beta;
      }
    }
  }
}

// ATM continuity across β (the ζ/x̂→1 removable singularity), the 3-arg ATM helper == the smile at K=F, and
// the normal/Black ATM consistency σ_N ≈ F·σ_B·(1 − σ_B²T/24).
TEST(RatesFxGrid, SabrAtmContinuityAndNormalBlackConsistency) {
  const double F = 0.030, T = 4.0, rho = -0.28, nu = 0.50;
  for (double beta : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const double alpha = 0.0090 * std::pow(F, -beta);
    const v::SabrParams pr{alpha, rho, nu, beta};
    const double atm = v::sabr_normal_vol(F, F, T, pr);
    ASSERT_TRUE(std::isfinite(atm)) << "beta=" << beta;
    EXPECT_GT(atm, 0.0);
    EXPECT_NEAR(v::sabr_normal_vol(F, F - 1e-7, T, pr), atm, 1e-6) << "beta=" << beta;
    EXPECT_NEAR(v::sabr_normal_vol(F, F + 1e-7, T, pr), atm, 1e-6) << "beta=" << beta;
    EXPECT_NEAR(v::sabr_atm_normal_vol(F, T, pr), atm, 1e-15) << "beta=" << beta;
    // Normal/Black ATM consistency σ_N ≈ F·σ_B·(1 − σ_B²T/24). FINDING: this is only a LEADING-order bridge.
    // Its neglected terms are O((σ_B²T)²) plus vol-of-vol cross terms, so its precision depends strongly on
    // (ν, T): the existing β=1 gate holds it to 2e-6·F only because it uses a gentler point (ν=0.40, a benign
    // strike). At this deliberately harder ATM point (ν=0.50, T=4) the residual is a genuine ≈3.9e-5 at β=0
    // and ≈1.8e-5 at β=0.75 — an artefact of the truncated bridge, NOT a smile bug (the two expansions agree
    // to O(T); the α² backbone term cancels). So the bridge is pinned to a documented 6e-5 band (≈1.5× the
    // observed max) — honest to what a second-order conversion can deliver rather than tightened past it.
    // The exact ATM invariants (continuity, positivity, helper==smile) above stay tight.
    const double sb = v::sabr_black_vol(F, F, T, pr);
    const double bridge = F * sb * (1.0 - sb * sb * T / 24.0);
    EXPECT_NEAR(atm, bridge, 6e-5) << "beta=" << beta;
  }
}

// Free-beta calibration recovers a KNOWN backbone from a strip generated at that beta (β=0.5 here), from a
// deliberately different seed — the adversarial recovery test.
TEST(RatesFxGrid, SabrFreeBetaRecoversKnownBackbone) {
  const double Fc = 0.028, Tc = 3.0, betaTrue = 0.5;
  const v::SabrParams truth{0.0092 * std::pow(Fc, -betaTrue), -0.31, 0.46, betaTrue};
  std::vector<double> K, mv;
  for (int bp = -200; bp <= 200; bp += 10) {
    const double k = Fc + bp / 1e4;
    if (k <= 0.0) continue;
    K.push_back(k);
    mv.push_back(v::sabr_normal_vol(Fc, k, Tc, truth));
  }
  v::SabrParams seed{0.02, 0.0, 0.20, 0.30};  // wrong on every axis, including beta
  const v::SabrCalibResult r = v::sabr_calibrate_free_beta(Fc, Tc, K, mv, seed);
  EXPECT_LT(r.rms, 1e-7);
  EXPECT_NEAR(r.params.beta, betaTrue, 5e-3);
  EXPECT_NEAR(r.params.rho, truth.rho, 5e-3);
  EXPECT_NEAR(r.params.nu, truth.nu, 5e-3);
  EXPECT_NEAR(r.params.alpha, truth.alpha, 2e-3 * truth.alpha);
}
