// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// The RV library behind the bond_universe / govvie_fit / swap_spread verbs (derive/bond_rv.hpp, derive/asset_swap.hpp;
// E7 stage 3.4). Each function is compared with the pipeline the pre-lift verb (api/rv.cpp) wrote out by hand --
// bitwise where the arithmetic is the same -- and with the scalar street kernel (itself QuantLib / rateslib pinned).
// Header-only (swaps_tests), so tools/mutate.py reaches it.
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "swaps/derive/bond_rv.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace cvd = swaps::conventions;
namespace der = swaps::derive;
namespace mkt = swaps::market;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {
b::Date d(const char* iso) { return b::Date::from_iso(iso); }
const b::Date kVd = b::Date::from_iso("2026-09-04");

b::BondId bond(const char* id, const char* issue, const char* maturity, double coupon,
               std::optional<b::Date> first_coupon = std::nullopt) {
  return b::BondId{id, "US-TREASURY", d(issue), d(maturity), coupon, first_coupon};
}
// The settlement rule restated from the DB row, independently of derive::settlement_date.
b::Date row_settle(const b::Date& vd) {
  const cvd::BondConv row = cvd::require_bond("US-TREASURY");
  return b::advance_bd(std::string(row.calendar), vd, row.settle_lag);
}
der::AssetSwapConvention row_convention() {
  const cvd::BondConv row = cvd::require_bond("US-TREASURY");
  der::AssetSwapConvention c;
  c.currency = std::string(row.currency);
  c.settle_calendar = std::string(row.calendar);
  c.settle_lag = row.settle_lag;
  return c;
}
std::vector<b::BondId> street_universe() {
  return {bond("A", "2026-08-15", "2031-08-15", 0.04), bond("B", "2026-08-15", "2036-08-15", 0.045),
          bond("WI", "2026-08-31", "2031-08-15", 0.0425, d("2027-02-15"))};  // when-issued, short first coupon
}
std::vector<b::BondId> fit_universe() {
  return {bond("B1", "2026-08-15", "2028-09-15", 0.035), bond("B2", "2026-08-15", "2030-09-15", 0.038),
          bond("B3", "2026-08-15", "2032-09-15", 0.042), bond("B4", "2026-08-15", "2036-09-15", 0.043),
          bond("B5", "2026-08-15", "2041-09-15", 0.05)};
}
Eigen::Vector3d ns_truth() { return Eigen::Vector3d(0.045, -0.015, -0.020); }
std::vector<double> ns_cleans(const std::vector<b::BondId>& u) {
  cv::NelsonSiegel<double> truth(2.0);
  truth.set_params(ns_truth());
  std::vector<double> clean;
  for (const b::BondId& bi : u)
    clean.push_back(px::bond_clean_price<double>(b::build_bond(bi, kVd, row_settle(kVd)).curve, truth));
  return clean;
}
mkt::Market clean_market(const std::vector<b::BondId>& u, const std::vector<double>& clean) {
  mkt::Market m;
  m.as_of(kVd);
  for (std::size_t i = 0; i < u.size(); ++i) m.add_quote(u[i].id, mkt::Quote::mid(clean[i]));
  return m;
}
}  // namespace

TEST(DeriveAssetSwap, AssetSwapConventionIsTheBondRowUnlessOverridden) {
  const cvd::BondConv row = cvd::require_bond("US-TREASURY");
  der::AssetSwapConvention c = der::asset_swap_convention("US-TREASURY");
  EXPECT_EQ(c.currency, std::string(row.currency));
  EXPECT_EQ(c.settle_calendar, std::string(row.calendar));
  EXPECT_EQ(c.settle_lag, row.settle_lag);
  EXPECT_EQ(der::settlement_date(c, kVd), row_settle(kVd));

  der::SettlementOverride lag_only;
  lag_only.lag = row.settle_lag + 2;
  c = der::asset_swap_convention("US-TREASURY", lag_only);
  EXPECT_EQ(c.settle_lag, row.settle_lag + 2);
  EXPECT_EQ(c.settle_calendar, std::string(row.calendar));

  der::SettlementOverride calendar_only;
  calendar_only.calendar = "NONE";
  c = der::asset_swap_convention("US-TREASURY", calendar_only);
  EXPECT_EQ(c.settle_calendar, "NONE");
  EXPECT_EQ(c.settle_lag, row.settle_lag);
  EXPECT_EQ(der::settlement_date(c, kVd), b::advance_bd("NONE", kVd, row.settle_lag));

  EXPECT_THROW(der::asset_swap_convention(""), std::invalid_argument);
  EXPECT_THROW(der::asset_swap_convention("NO-SUCH"), std::invalid_argument);
}

TEST(DeriveRv, BondUniverseEqualsThePerBondStreetKernel) {
  const std::vector<b::BondId> u = street_universe();
  const b::Date settle = row_settle(kVd);
  der::BondUniverseRequest r;
  r.value_date = kVd;
  r.convention = "US-TREASURY";
  r.bonds = u;
  r.clean = std::vector<double>{0.991, 1.012, 1.001};
  const der::BondUniverseResult res = der::bond_universe(r);
  ASSERT_EQ(res.yield.size(), 3);
  for (int i = 0; i < 3; ++i) {
    const b::BuiltBond bb = b::build_bond(u[static_cast<std::size_t>(i)], kVd, settle);
    const px::StreetAnalytics s = px::street_analytics(bb.yield, (*r.clean)[static_cast<std::size_t>(i)], std::nullopt);
    // 1e-10: the universe solves every yield in one batched Newton (price tol 1e-13), the scalar kernel one at a
    // time (1e-14); durations / convexities inherit that yield gap times their slope.
    EXPECT_NEAR(res.yield[i], s.yield, 1e-10) << u[static_cast<std::size_t>(i)].id;
    EXPECT_EQ(res.clean[i], (*r.clean)[static_cast<std::size_t>(i)]) << "a quoted clean is echoed exactly";
    EXPECT_EQ(res.accrued[i], bb.accrued);
    EXPECT_NEAR(res.modified_duration[i], s.modified_duration, 1e-8);
    EXPECT_NEAR(res.convexity[i], s.convexity, 1e-6);
  }
  // The other direction: quoted yields -> cleans.
  r.clean.reset();
  r.yield = std::vector<double>{res.yield[0], res.yield[1], res.yield[2]};
  const der::BondUniverseResult back = der::bond_universe(r);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(back.yield[i], res.yield[i]) << "a quoted yield is echoed exactly";
    EXPECT_NEAR(back.clean[i], res.clean[i], 1e-12);
  }
}

TEST(DeriveRv, BondUniverseSettlesOnTheConventionUnlessToldOtherwise) {
  der::BondUniverseRequest r;
  r.value_date = kVd;
  r.convention = "US-TREASURY";
  r.bonds = {street_universe()[0]};
  r.yield = std::vector<double>{0.04};
  const double by_rule = der::bond_universe(r).accrued[0];
  EXPECT_EQ(by_rule, b::build_bond(r.bonds[0], kVd, row_settle(kVd)).accrued);
  r.settle = d("2026-09-15");
  const double explicit_settle = der::bond_universe(r).accrued[0];
  EXPECT_EQ(explicit_settle, b::build_bond(r.bonds[0], kVd, d("2026-09-15")).accrued);
  EXPECT_GT(explicit_settle, by_rule) << "a later settlement accrues more";
}

TEST(DeriveRv, BondUniverseRejectsAmbiguousShortOrEmptyQuotes) {
  der::BondUniverseRequest r;
  r.value_date = kVd;
  r.convention = "US-TREASURY";
  r.bonds = street_universe();
  EXPECT_THROW(der::bond_universe(r), std::invalid_argument) << "neither clean nor yield";
  r.clean = std::vector<double>{0.99, 1.0, 1.0};
  r.yield = std::vector<double>{0.04, 0.04, 0.04};
  EXPECT_THROW(der::bond_universe(r), std::invalid_argument) << "both";
  r.yield.reset();
  r.clean = std::vector<double>{0.99, 1.0};
  EXPECT_THROW(der::bond_universe(r), std::invalid_argument) << "short";
  r.bonds.clear();
  r.clean = std::vector<double>{};
  EXPECT_THROW(der::bond_universe(r), std::invalid_argument) << "empty universe (was maxCoeff on an empty vector)";
}

TEST(DeriveRv, GovvieFitSplineIsTheHandAssembledPipeline) {
  const std::vector<b::BondId> u = fit_universe();
  const std::vector<double> clean = ns_cleans(u);
  der::GovvieFitRequest r;
  r.value_date = kVd;
  r.convention = "US-TREASURY";
  r.bonds = u;
  r.clean = clean;
  r.meeting = {1.0};
  r.back = {3.0, 7.0, 15.0};
  const der::GovvieFitResult res = der::govvie_fit(r);

  // The pre-lift verb, restated: make_govvie_fit, calibrate from a flat 3 %, then REBUILD every bond at the
  // settlement date and target clean + accrued for the z-spreads.
  const der::AssetSwapConvention conv = row_convention();
  const mkt::Market m = clean_market(u, clean);
  cal::GovvieBondFit fit = der::make_govvie_fit(conv, u, m, r.meeting, r.back);
  const cal::CalibrationResult hand = cal::calibrate(fit, Eigen::VectorXd::Constant(fit.n_knots(), 0.03));
  EXPECT_TRUE(res.fit.x == hand.x);
  EXPECT_TRUE(res.residuals == fit.residuals<double>(hand.x));
  std::vector<px::Bond> bonds;
  Eigen::VectorXd target(static_cast<int>(u.size()));
  for (std::size_t i = 0; i < u.size(); ++i) {
    b::BuiltBond built = b::build_bond(u[i], kVd, row_settle(kVd));
    target[static_cast<int>(i)] = clean[i] + built.accrued;
    bonds.push_back(std::move(built.curve));
  }
  const pf::CompiledBondBook book(r.meeting, r.back, bonds);
  ASSERT_TRUE(res.z_spread.has_value());
  EXPECT_TRUE(*res.z_spread == book.z_spreads(hand.x, target)) << "reusing the fit's bonds is bit-identical";
}

TEST(DeriveRv, GovvieFitParametricRecoversNelsonSiegelAndCarriesTheDecays) {
  const std::vector<b::BondId> u = fit_universe();
  const std::vector<double> clean = ns_cleans(u);
  der::GovvieFitRequest r;
  r.value_date = kVd;
  r.convention = "US-TREASURY";
  r.bonds = u;
  r.clean = clean;
  r.model = der::GovvieModel::NelsonSiegel;  // no tau1: the fit's own default, which is the truth's 2.0
  const der::GovvieFitResult ns = der::govvie_fit(r);
  ASSERT_EQ(ns.fit.x.size(), 3);
  for (int i = 0; i < 3; ++i) EXPECT_NEAR(ns.fit.x[i], ns_truth()[i], 1e-6);
  EXPECT_LT(ns.fit.rms_residual, 1e-9);
  EXPECT_FALSE(ns.z_spread.has_value());

  const der::AssetSwapConvention conv = row_convention();
  const mkt::Market m = clean_market(u, clean);
  const auto hand = [&](double tau1, double tau2) {
    auto fit = der::make_parametric_fit<cv::Svensson>(conv, u, m, tau1, tau2);
    Eigen::VectorXd seed = Eigen::VectorXd::Zero(fit.n_knots());
    seed[0] = 0.03;
    return cal::calibrate(fit, seed).x;
  };
  // Non-default decays reach the model...
  r.model = der::GovvieModel::Svensson;
  r.tau1 = 1.5;
  r.tau2 = 7.0;
  EXPECT_TRUE(der::govvie_fit(r).fit.x == hand(1.5, 7.0));
  // ... and absent decays are cal::ParametricBondFit's own, not a literal restated here.
  r.tau1.reset();
  r.tau2.reset();
  const cal::ParametricBondFit<cv::Svensson> defaults;
  EXPECT_TRUE(der::govvie_fit(r).fit.x == hand(defaults.tau1, defaults.tau2));
}

TEST(DeriveRv, GovvieFitRejectsMalformedUniverses) {
  const std::vector<b::BondId> u = fit_universe();
  der::GovvieFitRequest r;
  r.value_date = kVd;
  r.convention = "US-TREASURY";
  r.bonds = u;
  r.clean = ns_cleans(u);
  r.model = der::GovvieModel::NelsonSiegel;

  der::GovvieFitRequest bad = r;
  bad.clean.pop_back();
  EXPECT_THROW(der::govvie_fit(bad), std::invalid_argument) << "clean length";
  bad = r;
  bad.weight = {1.0, 1.0};
  EXPECT_ANY_THROW(der::govvie_fit(bad)) << "weight length (was an out-of-bounds read)";
  bad = r;
  bad.bonds[4].id = "B1";
  EXPECT_THROW(der::govvie_fit(bad), std::invalid_argument) << "duplicate id (was silently the last clean)";
  bad = r;
  bad.model = der::GovvieModel::Spline;
  EXPECT_THROW(der::govvie_fit(bad), std::invalid_argument) << "spline without back knots";
  bad = r;
  bad.bonds.clear();
  bad.clean.clear();
  EXPECT_THROW(der::govvie_fit(bad), std::invalid_argument) << "empty universe";
}

TEST(DeriveAssetSwap, SwapSpreadIsTheIndexConventionTenorSwap) {
  der::SwapSpreadRequest r;
  r.value_date = kVd;
  r.bond = bond("UST-5Y", "2026-08-15", "2031-08-15", 0.04);
  r.clean = 0.991;
  r.spread = -0.0032;
  r.index = "USD-SOFR";
  r.tenor = "5Y";
  r.swap_curve = 0;
  r.factor_curve = 1;
  const der::SwapSpreadResult res = der::swap_spread(r);

  // The pre-lift verb's chain, by hand: the index's par convention, the tenor from SPOT on its calendar, the par
  // swap, then derive_asset_swap anchored at that swap's maturity.
  const b::SwapConv sc = b::Index("USD-SOFR").par_convention().resolve();
  const b::Date mat = b::resolve("5Y", kVd, sc.calendar, sc.bdc, sc.spot_lag);
  const cal::Instrument swap = b::par_swap(kVd, sc, mat, 0, 0, 0.0);
  EXPECT_EQ(res.anchor, b::curve_time(kVd, mat));
  mkt::Market m = clean_market({r.bond}, {0.991});
  m.add_quote("quoted-spread", mkt::Quote::mid(-0.0032));
  const der::DerivedAssetSwap hand = der::derive_asset_swap(row_convention(), r.bond, swap, 1, res.anchor, m, "quoted-spread");
  EXPECT_EQ(res.derived.bond_yield, hand.bond_yield);
  EXPECT_EQ(res.derived.spread, hand.spread);
  EXPECT_EQ(res.derived.rows.pin.market, hand.rows.pin.market);
  EXPECT_EQ(res.derived.rows.asw.market, hand.rows.asw.market);
  ASSERT_EQ(res.derived.rows.asw.combination.size(), hand.rows.asw.combination.size());
  const cal::Instrument& got = res.derived.rows.asw.combination[0].instrument;
  const cal::Instrument& want = hand.rows.asw.combination[0].instrument;
  ASSERT_EQ(got.fwd.coupons.size(), want.fwd.coupons.size());
  for (std::size_t k = 0; k < got.fwd.coupons.size(); ++k) EXPECT_EQ(got.fwd.coupons[k].pay, want.fwd.coupons[k].pay);
  ASSERT_EQ(got.fixed.coupons.size(), want.fixed.coupons.size());
  for (std::size_t k = 0; k < got.fixed.coupons.size(); ++k) EXPECT_EQ(got.fixed.coupons[k].pay, want.fixed.coupons[k].pay);

  der::SwapSpreadRequest anchored = r;
  anchored.anchor = 4.2;
  EXPECT_EQ(der::swap_spread(anchored).anchor, 4.2);

  der::SwapSpreadRequest bad = r;
  bad.index.clear();
  EXPECT_THROW(der::swap_spread(bad), std::invalid_argument);
  bad = r;
  bad.tenor.clear();
  EXPECT_THROW(der::swap_spread(bad), std::invalid_argument);
  bad = r;
  bad.factor_curve = -1;
  EXPECT_THROW(der::swap_spread(bad), std::invalid_argument);
  bad = r;
  bad.clean = std::nan("");
  EXPECT_THROW(der::swap_spread(bad), std::invalid_argument) << "NaN passed the old `clean <= 0` check";
  bad = r;
  bad.bond.id = der::kSwapSpreadQuoteId;
  EXPECT_THROW(der::swap_spread(bad), std::invalid_argument);
}
