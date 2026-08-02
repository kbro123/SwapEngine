#pragma once
// SABR normal-vol smile (Hagan et al. 2002), the beta=0 "normal SABR" case that is standard for SOFR — vols
// are quoted normal (bp) and beta=0 handles low/negative strikes without a shift. Given (alpha, rho, nu) it
// returns the Bachelier (normal) implied vol at any strike; feed that vol into vol/bachelier.hpp to price the
// swaption/caplet at that strike. This is the smile layer the CMS static-replication (phase 3) integrates
// over, and the calibration target the model's local-vol skin later fits.
//
//   σ_N(K) = α · (ζ / x̂(ζ)) · [ 1 + ((2 − 3ρ²)/24)·ν²·T ],   ζ = (ν/α)(F − K),
//   x̂(ζ) = log( (√(1 − 2ρζ + ζ²) − ρ + ζ) / (1 − ρ) ).   ATM (K=F): σ_N = α·[1 + ((2−3ρ²)/24)ν²T].
//
// α>0 (the overall level, ≈ ATM normal vol), ρ∈(−1,1) (skew), ν≥0 (vol-of-vol / smile curvature).
// Scalar-templated so the calibration and the model fit can differentiate through it.
#include <cmath>

namespace swaps::vol {

struct SabrParams {
  double alpha = 0.0;  // level (~ ATM normal vol)
  double rho = 0.0;    // correlation / skew, in (-1, 1)
  double nu = 0.0;     // vol-of-vol, >= 0
};

// Normal (Bachelier) implied vol under beta=0 SABR. `expiry` in years.
template <class S>
inline S sabr_normal_vol(const S& fwd, const S& strike, double expiry, const S& alpha, const S& rho,
                         const S& nu) {
  using std::log;
  using std::sqrt;
  const S time_corr = S(1.0) + ((S(2.0) - S(3.0) * rho * rho) / S(24.0)) * nu * nu * S(expiry);
  const S dk = fwd - strike;
  // ATM (or nu -> 0): ζ/x̂ -> 1 smoothly; guard the removable singularity.
  if (nu <= S(0.0)) return alpha * time_corr;
  const S zeta = (nu / alpha) * dk;
  if (zeta > S(-1e-7) && zeta < S(1e-7)) return alpha * time_corr;  // ζ/x̂(ζ) -> 1 at the ATM point
  const S xhat = log((sqrt(S(1.0) - S(2.0) * rho * zeta + zeta * zeta) - rho + zeta) / (S(1.0) - rho));
  return alpha * (zeta / xhat) * time_corr;
}

inline double sabr_normal_vol(double fwd, double strike, double expiry, const SabrParams& p) {
  return sabr_normal_vol<double>(fwd, strike, expiry, p.alpha, p.rho, p.nu);
}

// The ATM normal vol in closed form (the level the smile pivots on).
inline double sabr_atm_normal_vol(double expiry, const SabrParams& p) {
  return p.alpha * (1.0 + ((2.0 - 3.0 * p.rho * p.rho) / 24.0) * p.nu * p.nu * expiry);
}

}  // namespace swaps::vol
