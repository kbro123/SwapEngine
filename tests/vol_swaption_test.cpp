// Phase-3 gate for swaption pricing off discount factors (vol/swaption.hpp). QuantLib-free: build the
// underlying's DFs from a flat forward, check the forward swap rate + annuity against an independent hand
// computation, then the swaption identities (ATM closed form, put-call parity, SABR nu=0 == flat vol).
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/vol/swaption.hpp"

namespace v = swaps::vol;

namespace {
// Flat forward -> DF(t) = exp(-f t). 1y into 5y annual swap: start t=1, pays t=2..6, tau=1, maturity t=6.
constexpr double f = 0.03;
double df(double t) { return std::exp(-f * t); }
}  // namespace

TEST(Swaption, ForwardSwapMatchesIndependentParRate) {
  const std::vector<double> pay_t{2, 3, 4, 5, 6};
  std::vector<double> df_pay, tau(5, 1.0);
  for (double t : pay_t) df_pay.push_back(df(t));
  const v::ForwardSwap fs = v::forward_swap(df(1.0), df(6.0), df_pay, tau);

  double a_expected = 0.0;
  for (double t : pay_t) a_expected += df(t);
  const double s_expected = (df(1.0) - df(6.0)) / a_expected;
  EXPECT_NEAR(fs.annuity, a_expected, 1e-14);
  EXPECT_NEAR(fs.rate, s_expected, 1e-14);
  EXPECT_GT(fs.rate, 0.028);
  EXPECT_LT(fs.rate, 0.032);  // a flat 3% forward gives a ~3.05% annual par rate
  // Par identity: fixed PV at par == float PV.
  EXPECT_NEAR(fs.rate * fs.annuity, df(1.0) - df(6.0), 1e-14);
}

TEST(Swaption, AtmPriceAndParity) {
  const std::vector<double> pay_t{2, 3, 4, 5, 6};
  std::vector<double> df_pay, tau(5, 1.0);
  for (double t : pay_t) df_pay.push_back(df(t));
  const v::ForwardSwap fs = v::forward_swap(df(1.0), df(6.0), df_pay, tau);
  const double T = 1.0, vol = 0.0090;

  // ATM (K=S0): price = A0 · σ√T · φ(0), identical payer/receiver.
  const double atm = fs.annuity * vol * std::sqrt(T) * v::kInvSqrt2Pi;
  EXPECT_NEAR(v::swaption_price(fs.rate, fs.annuity, fs.rate, vol, T, v::Payoff::Payer), atm, 1e-13);
  EXPECT_NEAR(v::swaption_price(fs.rate, fs.annuity, fs.rate, vol, T, v::Payoff::Receiver), atm, 1e-13);

  // Put-call parity: payer - receiver = A0 (S0 - K).
  const double K = 0.035;
  const double pay = v::swaption_price(fs.rate, fs.annuity, K, vol, T, v::Payoff::Payer);
  const double rec = v::swaption_price(fs.rate, fs.annuity, K, vol, T, v::Payoff::Receiver);
  EXPECT_NEAR(pay - rec, fs.annuity * (fs.rate - K), 1e-12);
}

TEST(Swaption, SabrFlatMatchesNormal) {
  const v::ForwardSwap fs{0.030, 4.3};
  const v::SabrParams flat{0.0088, -0.3, 0.0};  // nu=0 -> flat smile at alpha
  const double T = 2.0, K = 0.025;
  EXPECT_NEAR(v::swaption_price_sabr(fs.rate, fs.annuity, K, T, flat, v::Payoff::Payer),
              v::swaption_price(fs.rate, fs.annuity, K, 0.0088, T, v::Payoff::Payer), 1e-13);
}
