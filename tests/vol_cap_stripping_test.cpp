// Gate for cap/floor pricing + caplet-vol stripping (vol/cap_stripping.hpp).
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/vol/cap_stripping.hpp"
#include "swaps/vol/swaption.hpp"

namespace v = swaps::vol;

namespace {
// A semi-annual cap schedule off a flat 3% forward: 8 caplets to 4y (fix at 0.5y..4.0y, tau 0.5).
std::vector<v::CapletLeg> flat_cap(double f = 0.03) {
  std::vector<v::CapletLeg> legs;
  for (int i = 0; i < 8; ++i) {
    const double expiry = (i + 1) * 0.5;
    legs.push_back({f, 0.5, std::exp(-f * (expiry + 0.5)), expiry});
  }
  return legs;
}
}  // namespace

TEST(CapStrip, CapletIsAOnePeriodSwaption) {
  const v::CapletLeg L{0.030, 0.5, std::exp(-0.03 * 1.0), 0.5};
  const double K = 0.032, vol = 0.0090;
  // A caplet == a swaption whose underlying is a single-period swap: forward = L.forward, annuity = tau*DF.
  EXPECT_NEAR(v::caplet_price(L, K, vol),
              v::swaption_price(L.forward, L.tau * L.df_pay, K, vol, L.expiry, v::Payoff::Payer), 1e-15);
}

TEST(CapStrip, CapIsSumOfCapletsAndFlatInvertsRoundTrip) {
  const auto legs = flat_cap();
  const double K = 0.030;
  const std::vector<double> vols(legs.size(), 0.0090);
  double sum = 0.0;
  for (std::size_t i = 0; i < legs.size(); ++i) sum += v::caplet_price(legs[i], K, vols[i]);
  EXPECT_NEAR(v::cap_price(legs, K, vols), sum, 1e-15);
  EXPECT_NEAR(v::cap_price_flat(legs, K, 0.0090), sum, 1e-15);
  // Invert the flat vol from the price.
  const double px = v::cap_price_flat(legs, K, 0.0095);
  EXPECT_NEAR(v::cap_implied_flat_vol(legs, K, px), 0.0095, 1e-10);
}

TEST(CapStrip, StripRecoversKnownPiecewiseCapletVols) {
  const auto legs = flat_cap();
  const double K = 0.030;                                   // ATM
  // Known piecewise caplet vols in 4 buckets of 2 caplets each.
  const std::vector<double> known{0.0070, 0.0070, 0.0080, 0.0080, 0.0090, 0.0090, 0.0095, 0.0095};
  const std::vector<int> cap_last{1, 3, 5, 7};
  // The market quotes each cap by its FLAT vol equivalent.
  std::vector<double> cap_flat;
  for (int e : cap_last) {
    const std::vector<v::CapletLeg> whole(legs.begin(), legs.begin() + (e + 1));
    const std::vector<double> kv(known.begin(), known.begin() + (e + 1));
    cap_flat.push_back(v::cap_implied_flat_vol(whole, K, v::cap_price(whole, K, kv)));
  }
  // Stripping those flat vols recovers the piecewise caplet vols (buckets align with the cap maturities).
  const std::vector<double> stripped = v::strip_caplet_vols(legs, K, cap_last, cap_flat);
  ASSERT_EQ(stripped.size(), known.size());
  for (std::size_t i = 0; i < known.size(); ++i) EXPECT_NEAR(stripped[i], known[i], 1e-8) << "caplet " << i;

  // And the stripped vols reprice every quoted cap at its flat vol (self-consistency).
  for (std::size_t j = 0; j < cap_last.size(); ++j) {
    const int e = cap_last[j];
    const std::vector<v::CapletLeg> whole(legs.begin(), legs.begin() + (e + 1));
    const std::vector<double> sv(stripped.begin(), stripped.begin() + (e + 1));
    EXPECT_NEAR(v::cap_price(whole, K, sv), v::cap_price_flat(whole, K, cap_flat[j]), 1e-12);
  }
}
