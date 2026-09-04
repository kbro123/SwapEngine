#pragma once
// calibration/bond_fit.hpp — fit a govvie DISCOUNT curve to a UNIVERSE of bonds by MINIMUM PRICING ERROR.
//
// The bond RV workhorse. Given N bonds' curve-space cashflows + their N market clean prices, solve for the
// curve parameters x that minimize  Σ_b w_b (model_clean_b(x) − market_clean_b)²  (params ≪ N: the curve is
// the smooth "fair value" surface, each bond's leftover residual is its RICHNESS / CHEAPNESS — the RV
// signal). model_clean_b is the templated curve-space bond price (pricing::bond_clean_price), so each
// residual is analytic in x and this rides the SAME least-squares LM (calibrate / aad_jacobian) as swap
// calibration — no new solver, no new QuoteKind, no hot-path change.
//
// TWO fit types over the ONE curve contract (`Scalar discount(double t)` is all the pricing kernel reads):
//   * GovvieBondFit          — the free-knot SPLINE (a curve::ModularCurve from explicit regions). Local:
//                              an outlier bond stays an outlier. This is the same representation the
//                              calibrated swap curves use, so z-spreads off it (CompiledBondBook) and the
//                              asset-swap factor curve line up.
//   * ParametricBondFit<M>   — a PARAMETRIC term-structure family M (curve/parametric.hpp NelsonSiegel /
//                              Svensson, or any Scalar-templated model exposing n_params / set_params /
//                              discount). Few, time-stable, interpretable parameters — the fair-value-curve
//                              RV form. The curve TYPE is the template parameter: no model enum, no switch,
//                              and this header depends on no concrete model.
//
// Both are plain single-curve Problem ADAPTERS: they expose residuals<Scalar>(x) / n_knots() /
// n_residuals() exactly like CalibrationProblem, so cal::calibrate and aad_jacobian drive them UNCHANGED.
// Deliberately OVERDETERMINED (N ≫ params); the engine's LM handles the rectangular least squares. The
// fitted curve is what an AssetSwapConvention (derive/asset_swap.hpp) references by name — an asset swap
// computed against the fitted curve IS the bond's asset-swap richness.

#include <Eigen/Core>

#include <stdexcept>
#include <vector>

#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/bond.hpp"

namespace swaps::calibration {

namespace curve = swaps::curve;
namespace pricing = swaps::pricing;

namespace bond_fit_detail {
// residual_b = w_b · (model_clean_b − market_clean_b), in PRICE units (per unit notional), off ANY curve
// exposing Scalar discount(t) — the shared back half of both fit types (one pricing expression, not two).
template <class Scalar, class Curve>
void price_residuals(const std::vector<pricing::Bond>& bonds, const Eigen::VectorXd& market_clean,
                     const Eigen::VectorXd& weight, const Curve& c,
                     Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& r) {
  const int n = static_cast<int>(bonds.size());
  if (market_clean.size() != n)
    throw std::invalid_argument("bond fit: market_clean size must match the bond count");
  r.resize(n);
  for (int b = 0; b < n; ++b) {
    const double w = weight.size() ? weight[b] : 1.0;
    r[b] = Scalar(w) * (pricing::bond_clean_price<Scalar>(bonds[b], c) - Scalar(market_clean[b]));
  }
}
}  // namespace bond_fit_detail

// The free-knot SPLINE fit: x = the knot forwards of a ModularCurve built from `regions`.
struct GovvieBondFit {
  std::vector<curve::CurveModule> regions;  // the govvie curve topology — e.g. curve::flat_hermite
  std::vector<pricing::Bond> bonds;         // per-bond curve-space cashflows (build::BuiltBond::curve)
  Eigen::VectorXd market_clean;             // per-bond market clean prices (the fit targets)
  Eigen::VectorXd weight;                   // per-bond least-squares weights (empty => all 1; e.g. 1/duration)

  int n_knots() const {
    int n = 0;
    for (const auto& r : regions) n += static_cast<int>(r.knots.size());
    return n;
  }
  int n_residuals() const { return static_cast<int>(bonds.size()); }
  // Target "quotes" in residual order (the market clean prices), so warm/streaming can recover
  // model = residual + market. Weight-free (the residual folds the weight in; this is the raw target).
  Eigen::VectorXd market() const { return market_clean; }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    const int nk = n_knots();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> p(nk);
    for (int i = 0; i < nk; ++i) p[i] = x[i];
    auto c = curve::make_modular_curve<Scalar>(regions);
    c.set_forwards(p);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r;
    bond_fit_detail::price_residuals<Scalar>(bonds, market_clean, weight, c, r);
    return r;
  }
};

// The PARAMETRIC fit: x = the model's parameters (β's). `Model` is any Scalar-templated term-structure
// family exposing `static n_params()`, `set_params(vector)` and `Scalar discount(double)` — e.g.
// curve::NelsonSiegel (3 params, one decay) or curve::Svensson (4 params, two decays). The decays are
// fixed hyperparameters of the fit (the usual stability choice); a 3-param model reads tau1 only.
template <template <class> class Model>
struct ParametricBondFit {
  double tau1 = 2.0, tau2 = 5.0;            // decay hyperparameters (tau2 read by 4-param models only)
  std::vector<pricing::Bond> bonds;         // per-bond curve-space cashflows
  Eigen::VectorXd market_clean;             // per-bond market clean prices (the fit targets)
  Eigen::VectorXd weight;                   // per-bond least-squares weights (empty => all 1)

  int n_knots() const { return Model<double>::n_params(); }
  int n_residuals() const { return static_cast<int>(bonds.size()); }
  Eigen::VectorXd market() const { return market_clean; }

  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    const int np = n_knots();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> p(np);
    for (int i = 0; i < np; ++i) p[i] = x[i];
    Model<Scalar> c = make_model<Scalar>();
    c.set_params(p);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r;
    bond_fit_detail::price_residuals<Scalar>(bonds, market_clean, weight, c, r);
    return r;
  }

 private:
  template <class Scalar>
  Model<Scalar> make_model() const {
    if constexpr (Model<Scalar>::n_params() >= 4)
      return Model<Scalar>(tau1, tau2);
    else
      return Model<Scalar>(tau1);
  }
};

}  // namespace swaps::calibration
