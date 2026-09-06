#pragma once
// SABR strip calibration — fit (alpha, rho, nu) to a strip of market normal vols at one expiry x tenor, with
// an ANALYTIC Jacobian (no bumping) and an arbitrage gate. This is Phase A §1.1 of the vol-book roadmap: the
// first building block of a real vol surface, sitting directly on the beta=0 normal SABR smile (vol/sabr.hpp).
//
// The Jacobian is free because `sabr_normal_vol` is Scalar-templated: evaluating it on `ad::Dual` (a 3-wide
// AutoDiffScalar seeded on alpha/rho/nu) yields d(vol)/d(params) in one differentiated pass — the SAME
// discipline that gives the curve calibration its analytic Jacobian. A Levenberg-Marquardt loop (box-
// constrained: alpha>0, rho in (-1,1), nu>=0) then fits the strip; `sabr_arbitrage_free` checks the calibrated
// smile has a non-negative butterfly (implied density d^2C/dK^2 >= 0) over the quoted range.
//
// Units are the model's own decimals (forward/strike absolute rates, vols normal/absolute) — the same as
// vol/sabr.hpp. The Python/Excel layer converts percent/bp at the boundary.
#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "swaps/ad/dual.hpp"
#include "swaps/vol/bachelier.hpp"
#include "swaps/vol/sabr.hpp"

namespace swaps::vol {

struct SabrCalibResult {
  SabrParams params;      // fitted (alpha, rho, nu, beta)
  double rms = 0.0;       // root-mean-square vol error over the strip (in vol units)
  int iterations = 0;
  bool converged = false;
};

// Fit SABR to (strikes, market_vols) at (forward, expiry) with β FIXED at `guess.beta` (default 0.0 → the
// legacy normal-SABR fit, BYTE-IDENTICAL to before: the β=0 path threads a constant zero-derivative β dual
// and `sabr_normal_vol` takes its verbatim β=0 branch, so residuals/Jacobian/LM trajectory are unchanged).
// This is the common desk workflow — β is a modelling choice, {α, ρ, ν} are fitted. `guess` seeds the
// search; a zero/empty guess uses alpha = the closest-to-ATM market vol, rho = 0, nu = 0.30.
// Levenberg-Marquardt with an analytic ad::Dual Jacobian.
inline SabrCalibResult sabr_calibrate(double forward, double expiry, const std::vector<double>& strikes,
                                      const std::vector<double>& market_vols, SabrParams guess = {},
                                      int max_iter = 100, double tol = 1e-13) {
  const int m = static_cast<int>(strikes.size());
  const double B = guess.beta;  // fixed backbone
  // Seed. Alpha ~ the market vol nearest the forward (the ATM level).
  int atm = 0;
  for (int i = 1; i < m; ++i)
    if (std::abs(strikes[i] - forward) < std::abs(strikes[atm] - forward)) atm = i;
  double a = guess.alpha > 0.0 ? guess.alpha : (m > 0 ? market_vols[atm] : 0.01);
  double r = guess.rho;
  double n = guess.nu > 0.0 ? guess.nu : 0.30;

  const auto rms_of = [&](double aa, double rr, double nn) {
    double s = 0.0;
    for (int i = 0; i < m; ++i) {
      const double v = sabr_normal_vol(forward, strikes[i], expiry, SabrParams{aa, rr, nn, B});
      s += (v - market_vols[i]) * (v - market_vols[i]);
    }
    return m > 0 ? std::sqrt(s / m) : 0.0;
  };

  double lambda = 1e-3;
  double cur = rms_of(a, r, n);
  SabrCalibResult out;
  int it = 0;
  for (; it < max_iter; ++it) {
    // Residuals + analytic Jacobian J (m x 3) in one AAD pass per strike. β enters as a constant dual.
    Eigen::MatrixXd J(m, 3);
    Eigen::VectorXd res(m);
    ad::Dual da, dr, dn, db;
    da.value() = a; da.derivatives() = Eigen::VectorXd::Unit(3, 0);
    dr.value() = r; dr.derivatives() = Eigen::VectorXd::Unit(3, 1);
    dn.value() = n; dn.derivatives() = Eigen::VectorXd::Unit(3, 2);
    db.value() = B; db.derivatives() = Eigen::VectorXd::Zero(3);
    for (int i = 0; i < m; ++i) {
      ad::Dual F, K;
      F.value() = forward; F.derivatives() = Eigen::VectorXd::Zero(3);
      K.value() = strikes[i]; K.derivatives() = Eigen::VectorXd::Zero(3);
      const ad::Dual v = sabr_normal_vol<ad::Dual>(F, K, expiry, da, dr, dn, db);
      res[i] = v.value() - market_vols[i];
      J.row(i) = v.derivatives().transpose();
    }
    // Marquardt step: (JtJ + lambda*diag(JtJ)) d = -Jt res.
    const Eigen::Matrix3d JtJ = J.transpose() * J;
    const Eigen::Vector3d Jtr = J.transpose() * res;
    Eigen::Matrix3d A = JtJ;
    for (int k = 0; k < 3; ++k) A(k, k) += lambda * std::max(JtJ(k, k), 1e-12);
    const Eigen::Vector3d d = A.ldlt().solve(-Jtr);

    double na = std::max(a + d[0], 1e-10);
    double nr = std::clamp(r + d[1], -0.9999, 0.9999);
    double nn = std::max(n + d[2], 1e-8);
    const double trial = rms_of(na, nr, nn);
    if (trial < cur) {  // accept, relax damping
      a = na; r = nr; n = nn;
      lambda = std::max(lambda * 0.5, 1e-12);
      const bool small_step = (cur - trial) < tol;
      cur = trial;
      if (small_step) { out.converged = true; ++it; break; }
    } else {            // reject, tighten damping
      lambda *= 4.0;
      if (lambda > 1e12) break;
    }
  }
  out.params = SabrParams{a, r, n, B};
  out.rms = cur;
  out.iterations = it;
  return out;
}

// Fit SABR to a strip FREEING β as well — a 4-parameter {α, ρ, ν, β} Levenberg-Marquardt with the analytic
// ad::Dual Jacobian (β differentiated through the general Hagan backbone). β is box-constrained to
// [1e-6, 1] (kept strictly positive so the general CEV branch — not the β=0 shortcut, whose ∂/∂β vanishes —
// always drives the step). A zero/empty `guess.beta` seeds β = 0.5. NOTE: {α, β} are near-collinear at the
// money, so a bare strip can be an ill-posed 4-way fit; prefer FIXING β (the overload above) unless the
// smile has enough wing information to pin the backbone. The fit is still exact when the market strip was
// generated by a SABR of the same β.
inline SabrCalibResult sabr_calibrate_free_beta(double forward, double expiry,
                                                const std::vector<double>& strikes,
                                                const std::vector<double>& market_vols, SabrParams guess = {},
                                                int max_iter = 200, double tol = 1e-14) {
  const int m = static_cast<int>(strikes.size());
  int atm = 0;
  for (int i = 1; i < m; ++i)
    if (std::abs(strikes[i] - forward) < std::abs(strikes[atm] - forward)) atm = i;
  double a = guess.alpha > 0.0 ? guess.alpha : (m > 0 ? market_vols[atm] : 0.01);
  double r = guess.rho;
  double n = guess.nu > 0.0 ? guess.nu : 0.30;
  double b = (guess.beta > 0.0) ? std::clamp(guess.beta, 1e-6, 1.0) : 0.5;

  const auto rms_of = [&](double aa, double rr, double nn, double bb) {
    double s = 0.0;
    for (int i = 0; i < m; ++i) {
      const double v = sabr_normal_vol(forward, strikes[i], expiry, SabrParams{aa, rr, nn, bb});
      s += (v - market_vols[i]) * (v - market_vols[i]);
    }
    return m > 0 ? std::sqrt(s / m) : 0.0;
  };

  double lambda = 1e-3;
  double cur = rms_of(a, r, n, b);
  SabrCalibResult out;
  int it = 0;
  for (; it < max_iter; ++it) {
    Eigen::MatrixXd J(m, 4);
    Eigen::VectorXd res(m);
    ad::Dual da, dr, dn, db;
    da.value() = a; da.derivatives() = Eigen::VectorXd::Unit(4, 0);
    dr.value() = r; dr.derivatives() = Eigen::VectorXd::Unit(4, 1);
    dn.value() = n; dn.derivatives() = Eigen::VectorXd::Unit(4, 2);
    db.value() = b; db.derivatives() = Eigen::VectorXd::Unit(4, 3);
    for (int i = 0; i < m; ++i) {
      ad::Dual F, K;
      F.value() = forward; F.derivatives() = Eigen::VectorXd::Zero(4);
      K.value() = strikes[i]; K.derivatives() = Eigen::VectorXd::Zero(4);
      const ad::Dual v = sabr_normal_vol<ad::Dual>(F, K, expiry, da, dr, dn, db);
      res[i] = v.value() - market_vols[i];
      J.row(i) = v.derivatives().transpose();
    }
    const Eigen::Matrix4d JtJ = J.transpose() * J;
    const Eigen::Vector4d Jtr = J.transpose() * res;
    Eigen::Matrix4d A = JtJ;
    for (int k = 0; k < 4; ++k) A(k, k) += lambda * std::max(JtJ(k, k), 1e-12);
    const Eigen::Vector4d d = A.ldlt().solve(-Jtr);

    double na = std::max(a + d[0], 1e-10);
    double nr = std::clamp(r + d[1], -0.9999, 0.9999);
    double nn = std::max(n + d[2], 1e-8);
    double nb = std::clamp(b + d[3], 1e-6, 1.0);
    const double trial = rms_of(na, nr, nn, nb);
    if (trial < cur) {
      a = na; r = nr; n = nn; b = nb;
      lambda = std::max(lambda * 0.5, 1e-12);
      const bool small_step = (cur - trial) < tol;
      cur = trial;
      if (small_step) { out.converged = true; ++it; break; }
    } else {
      lambda *= 4.0;
      if (lambda > 1e12) break;
    }
  }
  out.params = SabrParams{a, r, n, b};
  out.rms = cur;
  out.iterations = it;
  return out;
}

// The SABR smile's sensitivity to its own parameters at one strike: d(vol)/d(alpha, rho, nu) — the level,
// SKEW and vol-of-vol/CURVATURE risk of a fitted smile — in one analytic AAD pass (no bumping). Multiply each
// by the Bachelier vega for the price sensitivity to that parameter (the natural risk representation for a
// SABR-marked book: hedge/attribute by alpha/rho/nu rather than by strike-by-strike vega).
struct SabrVolGrad {
  double d_alpha = 0.0;  // d(vol)/d(alpha)  — level
  double d_rho = 0.0;    // d(vol)/d(rho)    — skew
  double d_nu = 0.0;     // d(vol)/d(nu)     — curvature (vol-of-vol)
};
inline SabrVolGrad sabr_vol_gradient(double forward, double strike, double expiry, const SabrParams& p) {
  ad::Dual a, r, n, F, K;
  a.value() = p.alpha; a.derivatives() = Eigen::VectorXd::Unit(3, 0);
  r.value() = p.rho;   r.derivatives() = Eigen::VectorXd::Unit(3, 1);
  n.value() = p.nu;    n.derivatives() = Eigen::VectorXd::Unit(3, 2);
  F.value() = forward; F.derivatives() = Eigen::VectorXd::Zero(3);
  K.value() = strike;  K.derivatives() = Eigen::VectorXd::Zero(3);
  const ad::Dual v = sabr_normal_vol<ad::Dual>(F, K, expiry, a, r, n);
  return {v.derivatives()[0], v.derivatives()[1], v.derivatives()[2]};
}

// Minimum discrete second derivative of a call-price ladder on an EVEN strike grid of step dk. The risk-neutral
// density is d^2C/dK^2, so this is the density proxy; < 0 anywhere is a butterfly arbitrage. This is the
// general detector — it takes any call ladder, so it guards a FITTED/interpolated smile (Phase A §1.2), not
// only SABR. (Normal beta=0 SABR is itself arb-robust; on it this returns ~0 in the far wing, floating-point
// noise absorbed by the arbitrage-free tolerance below.)
inline double min_price_convexity(const std::vector<double>& call, double dk) {
  double worst = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i + 1 < call.size(); ++i)
    worst = std::min(worst, (call[i + 1] - 2.0 * call[i] + call[i - 1]) / (dk * dk));
  return std::isinf(worst) ? 0.0 : worst;
}

// The smallest butterfly of the SABR smile over [lo, hi] (>= -tol means arbitrage-free): price a Bachelier
// call ladder at the SABR vols, then take its minimum convexity.
inline double sabr_min_butterfly(double forward, double expiry, const SabrParams& p, double lo, double hi,
                                 int n = 200) {
  if (!(hi > lo) || n < 2) return 0.0;
  const double dk = (hi - lo) / n;
  std::vector<double> call(n + 1);
  for (int i = 0; i <= n; ++i) {
    const double K = lo + i * dk;
    call[i] = bachelier_price<double>(forward, K, sabr_normal_vol(forward, K, expiry, p), expiry, 1.0,
                                      Payoff::Payer);
  }
  return min_price_convexity(call, dk);
}

inline bool sabr_arbitrage_free(double forward, double expiry, const SabrParams& p, double lo, double hi,
                                double tol = 1e-8) {
  return sabr_min_butterfly(forward, expiry, p, lo, hi) >= -tol;
}

}  // namespace swaps::vol
