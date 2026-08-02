#pragma once
// Standard-normal density and CDF, Scalar-templated (double for pricing; a reverse-AD type later for
// Greeks). CDF uses erfc for accuracy in the left tail. Part of the QuantLib-free volatility/options
// layer (include/swaps/vol/), which sits ON TOP of the calibrated curve and NEVER touches the frozen
// calculation kernel — it only consumes discount factors and forward rates the curve already provides.
#include <cmath>

namespace swaps::vol {

inline constexpr double kInvSqrt2Pi = 0.39894228040143267794;  // 1/sqrt(2*pi)
inline constexpr double kInvSqrt2   = 0.70710678118654752440;  // 1/sqrt(2)

template <class S>
inline S normal_pdf(const S& x) {
  using std::exp;
  return S(kInvSqrt2Pi) * exp(S(-0.5) * x * x);
}

template <class S>
inline S normal_cdf(const S& x) {
  using std::erfc;
  return S(0.5) * erfc(-x * S(kInvSqrt2));
}

}  // namespace swaps::vol
