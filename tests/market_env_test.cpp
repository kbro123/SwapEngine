// market::Market — the composed market environment. Proves the snapshot owns a NAMED curve that is the
// real realized curve, plus FX (triangulated), currencies, quotes and "today" in one object — the market
// as of date D you build once and price against. Value-type composition; QuantLib-free.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include "swaps/build/date.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/market/market.hpp"

namespace mkt = swaps::market;
namespace cv = swaps::curve;
namespace b = swaps::build;

TEST(Market, ComposesNamedCurvesFxQuotesCurrenciesAndToday) {
  const std::vector<double> front{0.25}, back{2.0, 5.0, 10.0};
  const auto modules = cv::flat_hermite(front, back);  // Flat(1) + Hermite(3) = 4 knots
  Eigen::VectorXd x(4);
  x << 0.043, 0.041, 0.038, 0.045;

  // The same curve built directly, to prove the Market stores the REAL realized curve (not a copy/sample).
  auto direct = cv::make_modular_curve<double>(modules);
  direct.set_forwards(x);

  mkt::FxMatrix fx;
  fx.add("EUR", "USD", 1.09);
  fx.add("USD", "JPY", 150.0);

  mkt::Market m;
  m.as_of(b::Date::from_iso("2026-09-01"))
      .add_currency(mkt::Currency::of("USD"))
      .add_currency(mkt::Currency::of("EUR"))
      .set_fx(std::move(fx))
      .add_curve("USD-SOFR-DISC", modules, x)
      .add_quote("USD-10Y", mkt::Quote::bid_ask(3.83, 3.87));

  // Today.
  EXPECT_EQ(m.today().serial(), b::Date::from_iso("2026-09-01").serial());

  // Named curve lookup == the real curve (the gap: curves were addressable only by int role).
  EXPECT_TRUE(m.has_curve("USD-SOFR-DISC"));
  EXPECT_FALSE(m.has_curve("EUR-ESTR-DISC"));
  for (double t : {1.0, 5.0, 10.0, 30.0}) {
    EXPECT_NEAR(m.discount("USD-SOFR-DISC", t), direct.discount(t), 1e-15);
    EXPECT_NEAR(m.curve("USD-SOFR-DISC").forward(t), direct.forward(t), 1e-15);
  }
  EXPECT_THROW(m.curve("EUR-ESTR-DISC"), std::runtime_error);

  // FX with triangulation, from the one shared matrix.
  EXPECT_NEAR(m.fx_rate("EUR", "JPY"), 1.09 * 150.0, 1e-9);
  EXPECT_NEAR(m.fx_rate("USD", "EUR"), 1.0 / 1.09, 1e-9);

  // Quotes as first-class two-sided markets.
  EXPECT_NEAR(m.quote("USD-10Y").mid(), 3.85, 1e-12);
  EXPECT_TRUE(m.quote("USD-10Y").is_two_sided());
  EXPECT_THROW(m.quote("nope"), std::runtime_error);

  // Currencies are objects in the snapshot, not int tags.
  EXPECT_EQ(m.currencies().size(), 2u);
  EXPECT_TRUE(m.currency("USD").discount_index().is_overnight());
  EXPECT_EQ(m.currency("USD").discount_index().currency(), "USD");
  EXPECT_THROW(m.currency("JPY"), std::runtime_error);  // not added to this snapshot
}
