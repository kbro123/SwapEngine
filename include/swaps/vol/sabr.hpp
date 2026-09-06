#pragma once
// SABR implied-vol smile (Hagan et al. 2002), FULL-beta CEV backbone. The model is
//   dF = α F^β dW1,   dα = ν α dW2,   dW1·dW2 = ρ dt,
// with β ∈ [0,1] selecting the backbone: β=0 is normal SABR (the SOFR default — vols quoted normal/bp,
// robust at low/negative strikes without a shift), β=1 is lognormal SABR, and the CEV lives in between.
// Given (α, ρ, ν, β) this returns the Bachelier (normal) implied vol at any strike; feed that vol into
// vol/bachelier.hpp to price the swaption/caplet. A companion `sabr_black_vol` returns the Black
// (lognormal) implied vol for the lognormal-quoting desks. This is the smile layer the CMS static-
// replication (phase 3) integrates over, and the calibration target the model's local-vol skin fits.
//
// Normal-vol expansion (Hagan A.67a), general β:
//   σ_N(K) = α·(FK)^{β/2}·R_N(F,K)·(ζ/x̂(ζ))·[1 + ( −β(2−β)α²/(24(FK)^{1−β})
//                                                  + ρβνα/(4(FK)^{(1−β)/2})
//                                                  + (2−3ρ²)ν²/24 )·T],
//   R_N = [1 + L²/24 + L⁴/1920] / [1 + (1−β)²L²/24 + (1−β)⁴L⁴/1920],   L = log(F/K).
// ζ is the SABR backbone variable. We use the arc-length form (Obłój 2008), which is both more accurate
// than Hagan's original ζ=(ν/α)(FK)^{(1−β)/2}L AND collapses EXACTLY to the legacy β=0 code:
//   ζ = (ν/α)·(F^{1−β} − K^{1−β})/(1−β)   →   β=0: (ν/α)(F−K);   β=1: (ν/α)log(F/K).
//   x̂(ζ) = log( (√(1 − 2ρζ + ζ²) − ρ + ζ) / (1 − ρ) ).   At K→F, ζ/x̂ → 1 (removable singularity).
//
// α>0 (level; ≈ ATM normal vol / F^β), ρ∈(−1,1) (skew), ν≥0 (vol-of-vol / curvature), β∈[0,1] (backbone).
// Scalar-templated so the calibration and the model fit can differentiate through it (β included).
//
// BYTE-IDENTICAL β=0: `beta` defaults to 0.0 everywhere, and the β=0 evaluation takes a verbatim copy of
// the legacy normal-SABR code path (bit-for-bit unchanged), so every existing caller/test is untouched.
// For β>0 the CEV backbone requires F>0 and K>0 (the general path assumes positive forward/strike).
#include <cmath>

namespace swaps::vol {

struct SabrParams {
  double alpha = 0.0;  // level (~ ATM normal vol at beta=0)
  double rho = 0.0;    // correlation / skew, in (-1, 1)
  double nu = 0.0;     // vol-of-vol, >= 0
  double beta = 0.0;   // CEV backbone in [0,1]; 0 = normal (default), 1 = lognormal
};

// Normal (Bachelier) implied vol under general-β SABR. `expiry` in years. `beta` defaults to 0.0, in which
// case this is BIT-FOR-BIT the legacy normal-SABR smile (the branch below is a verbatim copy).
template <class S>
inline S sabr_normal_vol(const S& fwd, const S& strike, double expiry, const S& alpha, const S& rho,
                         const S& nu, const S& beta = S(0.0)) {
  using std::log;
  using std::sqrt;
  // ---- β = 0: legacy normal SABR, kept BYTE-IDENTICAL (do not refactor) ----------------------------
  if (beta == S(0.0)) {
    const S time_corr = S(1.0) + ((S(2.0) - S(3.0) * rho * rho) / S(24.0)) * nu * nu * S(expiry);
    const S dk = fwd - strike;
    if (nu <= S(0.0)) return alpha * time_corr;
    const S zeta = (nu / alpha) * dk;
    if (zeta > S(-1e-7) && zeta < S(1e-7)) return alpha * time_corr;  // ζ/x̂(ζ) -> 1 at the ATM point
    const S xhat = log((sqrt(S(1.0) - S(2.0) * rho * zeta + zeta * zeta) - rho + zeta) / (S(1.0) - rho));
    return alpha * (zeta / xhat) * time_corr;
  }
  // ---- general β ∈ (0,1]: full Hagan CEV backbone (requires F>0, K>0) -------------------------------
  const auto pw = [](const S& x, const S& y) {  // x^y via exp(y·log x) — AD-safe for a Dual exponent y
    using std::exp;
    using std::log;
    return exp(y * log(x));
  };
  const S one(1.0), two(2.0);
  const S L = log(fwd / strike);
  const S FK = fwd * strike;
  const S e = one - beta;  // 1 − β
  // Backbone variable ζ = (ν/α)·arc-length; arc-length = (F^{1−β} − K^{1−β})/(1−β), → log(F/K) at β=1.
  const S arc = (e == S(0.0)) ? L : (pw(fwd, e) - pw(strike, e)) / e;
  const S zeta = (nu / alpha) * arc;
  S zx;  // ζ/x̂(ζ), with the removable singularity guarded exactly as the legacy path
  if (zeta > S(-1e-7) && zeta < S(1e-7)) {
    zx = one;
  } else {
    const S xhat = log((sqrt(one - two * rho * zeta + zeta * zeta) - rho + zeta) / (one - rho));
    zx = zeta / xhat;
  }
  const S L2 = L * L, L4 = L2 * L2;
  const S num_br = one + (one / S(24.0)) * L2 + (one / S(1920.0)) * L4;
  const S den_br = one + (e * e / S(24.0)) * L2 + (e * e * e * e / S(1920.0)) * L4;
  const S pref = alpha * pw(FK, beta * S(0.5)) * (num_br / den_br);  // α·(FK)^{β/2}·R_N
  const S FKe = pw(FK, e);            // (FK)^{1−β}
  const S FKhe = pw(FK, e * S(0.5));  // (FK)^{(1−β)/2}
  const S tterm = (-beta * (two - beta) / S(24.0) * (alpha * alpha) / FKe +
                   rho * beta * nu / S(4.0) * alpha / FKhe +
                   (two - S(3.0) * rho * rho) / S(24.0) * nu * nu) *
                  S(expiry);
  return pref * zx * (one + tterm);
}

inline double sabr_normal_vol(double fwd, double strike, double expiry, const SabrParams& p) {
  return sabr_normal_vol<double>(fwd, strike, expiry, p.alpha, p.rho, p.nu, p.beta);
}

// Black (lognormal) implied vol under general-β SABR (Hagan B.65a). For the lognormal-quoting workflow and
// the β=1 benchmark. Uses Hagan's published backbone ζ=(ν/α)(FK)^{(1−β)/2}log(F/K) (the standard Black
// form). Requires F>0, K>0. At β=1, ATM: σ_B = α·[1 + (ρνα/4 + (2−3ρ²)ν²/24)·T].
template <class S>
inline S sabr_black_vol(const S& fwd, const S& strike, double expiry, const S& alpha, const S& rho,
                        const S& nu, const S& beta = S(0.0)) {
  using std::log;
  using std::sqrt;
  const auto pw = [](const S& x, const S& y) {
    using std::exp;
    using std::log;
    return exp(y * log(x));
  };
  const S one(1.0), two(2.0);
  const S L = log(fwd / strike);
  const S FK = fwd * strike;
  const S e = one - beta;
  const S FKhe = pw(FK, e * S(0.5));  // (FK)^{(1−β)/2}
  const S FKe = pw(FK, e);            // (FK)^{1−β}
  const S L2 = L * L, L4 = L2 * L2;
  const S den_br = one + (e * e / S(24.0)) * L2 + (e * e * e * e / S(1920.0)) * L4;
  const S zeta = (nu / alpha) * FKhe * L;
  S zx;
  if (zeta > S(-1e-7) && zeta < S(1e-7)) {
    zx = one;
  } else {
    const S xhat = log((sqrt(one - two * rho * zeta + zeta * zeta) - rho + zeta) / (one - rho));
    zx = zeta / xhat;
  }
  const S tterm = (e * e / S(24.0) * (alpha * alpha) / FKe + rho * beta * nu / S(4.0) * alpha / FKhe +
                   (two - S(3.0) * rho * rho) / S(24.0) * nu * nu) *
                  S(expiry);
  const S pref = alpha / (FKhe * den_br);
  return pref * zx * (one + tterm);
}

inline double sabr_black_vol(double fwd, double strike, double expiry, const SabrParams& p) {
  return sabr_black_vol<double>(fwd, strike, expiry, p.alpha, p.rho, p.nu, p.beta);
}

// ATM normal vol, β=0 closed form (the level the beta=0 smile pivots on). KEPT for backward compatibility
// and BYTE-IDENTICAL — valid only for β=0 (β>0 ATM depends on the forward; use the 3-arg overload below).
inline double sabr_atm_normal_vol(double expiry, const SabrParams& p) {
  return p.alpha * (1.0 + ((2.0 - 3.0 * p.rho * p.rho) / 24.0) * p.nu * p.nu * expiry);
}

// ATM normal vol for ANY β (needs the forward): the K→F limit of the smile, singularity-free.
inline double sabr_atm_normal_vol(double forward, double expiry, const SabrParams& p) {
  return sabr_normal_vol(forward, forward, expiry, p);
}

}  // namespace swaps::vol
