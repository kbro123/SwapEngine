#pragma once
// calibration/bond_fit.hpp — fit a govvie DISCOUNT curve to a UNIVERSE of bonds by MINIMUM PRICING ERROR.
//
// The bond RV workhorse. Given N bonds' curve-space cashflows + their N market clean prices, solve for the
// K-knot curve forwards x that minimize  Σ_b w_b (model_clean_b(x) − market_clean_b)²  (K ≪ N: the curve is
// the smooth "fair value" surface, each bond's leftover residual is its RICHNESS / CHEAPNESS — the RV
// signal). model_clean_b is the templated curve-space bond price (pricing::bond_clean_price), so each
// residual is analytic in x and this rides the SAME least-squares LM (calibrate / aad_jacobian) as swap
// calibration — no new solver.
//
// This is a plain single-curve Problem ADAPTER: it exposes residuals<Scalar>(x) / n_knots() / n_residuals()
// exactly like CalibrationProblem, so cal::calibrate and aad_jacobian drive it UNCHANGED. There is NO new
// QuoteKind and NO hot-path change. It shares the curve-space bond kernel (pricing/bond.hpp) with
// portfolio::CompiledBondBook — the compiled/vectorized repricer and z-spread engine — so a fit here and a
// fast reprice / z-spread there can never diverge: fit with GovvieBondFit, then read per-bond z-spreads off
// the SAME curve with CompiledBondBook::z_spreads() to get the RV ladder.
//
// It is deliberately OVERDETERMINED (N ≫ K); the engine's LM handles the rectangular least squares. The same
// curve is what an AssetSwapConvention (derive/asset_swap.hpp) references by name: in the headline-spread
// case it is pinned to one benchmark's yield; in the RV case it is this minimum-pricing-error fit over the
// whole universe. Same object, same kernel — an asset swap computed against the fitted curve IS the bond's
// asset-swap richness.

#include <Eigen/Core>

#include <stdexcept>
#include <vector>

#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/parametric.hpp"
#include "swaps/pricing/bond.hpp"

namespace swaps::calibration {

namespace curve = swaps::curve;
namespace pricing = swaps::pricing;

// Which curve the universe is fitted with. Spline = a free-knot ModularCurve (flexible, many params);
// NelsonSiegel / Svensson = a PARAMETRIC term-structure model (few, time-stable params — the fair-value-curve
// RV form). The free variables x are the knot forwards (Spline) or the model parameters (parametric).
enum class CurveModel { Spline, NelsonSiegel, Svensson };

struct GovvieBondFit {
  CurveModel model = CurveModel::Spline;
  std::vector<curve::CurveModule> regions;  // Spline: the curve topology — e.g. curve::flat_hermite
  double tau1 = 2.0, tau2 = 5.0;            // NelsonSiegel/Svensson decay hyperparameters (fixed)
  std::vector<pricing::Bond> bonds;         // per-bond curve-space cashflows (build::BuiltBond::curve)
  Eigen::VectorXd market_clean;             // per-bond market clean prices (the fit targets)
  Eigen::VectorXd weight;                   // per-bond least-squares weights (empty => all 1; e.g. 1/duration)

  int n_knots() const {
    switch (model) {
      case CurveModel::NelsonSiegel: return curve::NelsonSiegel<double>::n_params();  // 3
      case CurveModel::Svensson: return curve::Svensson<double>::n_params();          // 4
      default: {
        int n = 0;
        for (const auto& r : regions) n += static_cast<int>(r.knots.size());
        return n;
      }
    }
  }
  int n_residuals() const { return static_cast<int>(bonds.size()); }

  // Target "quotes" in residual order (the market clean prices), so warm/streaming can recover
  // model = residual + market. Weight-free (the residual folds the weight in; this is the raw target).
  Eigen::VectorXd market() const { return market_clean; }

  // residual_b = w_b · (model_clean_b(x) − market_clean_b), in PRICE units (per unit notional). The free
  // variables x are knot forwards (Spline) or model parameters (parametric); either way the priced curve is
  // a Scalar `discount(t)`, so this is one templated expression. Scalar=double drives the LM value pass and
  // Scalar=ad::Dual yields the analytic Jacobian row (d(price)/d(param)).
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    if (market_clean.size() != n_residuals())
      throw std::invalid_argument("GovvieBondFit: market_clean size must match the bond count");
    const int nk = n_knots();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> p(nk);
    for (int i = 0; i < nk; ++i) p[i] = x[i];

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    const auto price_on = [&](const auto& c) {
      for (int b = 0; b < n_residuals(); ++b) {
        const double w = weight.size() ? weight[b] : 1.0;
        r[b] = Scalar(w) * (pricing::bond_clean_price<Scalar>(bonds[b], c) - Scalar(market_clean[b]));
      }
    };
    switch (model) {
      case CurveModel::NelsonSiegel: {
        curve::NelsonSiegel<Scalar> c(tau1);
        c.set_params(p);
        price_on(c);
        break;
      }
      case CurveModel::Svensson: {
        curve::Svensson<Scalar> c(tau1, tau2);
        c.set_params(p);
        price_on(c);
        break;
      }
      default: {
        auto c = curve::make_modular_curve<Scalar>(regions);
        c.set_forwards(p);
        price_on(c);
        break;
      }
    }
    return r;
  }
};

}  // namespace swaps::calibration
