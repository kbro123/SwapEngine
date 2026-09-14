// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// SC2 (owner decision 2026-09-14): FX moves are exact per currency pair. portfolio/fx_pairs.hpp (each xccy position's
// pair and the one place it picks up its factor), derive/fx_move.hpp (a move's bumps -> one factor per pair, and the
// request-level book_fx_moves) and CompiledMultiCurveBook::set_fx_factors (the in-place rescale), each checked against
// hand-scaled books and closed-form factors. Header-only (swaps_tests), so tools/mutate.py reaches them; the verbs end
// to end are pinned by tests/scenario_fx_pairs_repro_test.cpp.
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/calibration/bundle_state.hpp"
#include "swaps/derive/fx_move.hpp"
#include "swaps/portfolio/compiled_multi.hpp"
#include "swaps/portfolio/fx_pairs.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace dv = swaps::derive;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

cal::BundleCurveSpec flat_curve(int currency) {
  cal::BundleCurveSpec s;
  s.currency = currency;
  crv::CurveModule m;
  m.scheme = crv::Scheme::Flat;
  m.knots = {1.0};
  s.regions = {m};
  return s;
}
px::FloatCoupon float_coupon(double a, double b, double spread) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  c.spread = spread;
  return c;
}
// num = foreign curve, den = domestic curve (the domestic leg forecasts and discounts on den).
pf::MultiCurveBook::Position xccy(int num, int den, double fx_spot, double notional) {
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  p.notional = notional;
  p.fwd_curve = den;
  p.disc_curve = den;
  p.float_coupons = {float_coupon(0.0, 1.0, 0.0), float_coupon(1.0, 2.0, 0.0)};
  p.mtm_coupons = {float_coupon(0.0, 1.0, 0.005), float_coupon(1.0, 2.0, 0.005)};
  p.mtm_fwd_curve = num;
  p.mtm_disc_curve = num;
  p.mtm_reset_num = num;
  p.mtm_reset_den = den;
  p.fx_spot = fx_spot;
  return p;
}
pf::MultiCurveBook::Position swap() {
  pf::MultiCurveBook::Position p;
  p.notional = 1e6;
  p.fixed_rate = 0.02;
  p.float_coupons = {float_coupon(0.0, 1.0, 0.0)};
  px::FixedCoupon f;
  f.pay = 1.0;
  f.tau = 1.0;
  p.fixed_coupons = {f};
  return p;
}

// Currency tags: 0 USD, 1 EUR, 2 GBP.
std::vector<cal::BundleCurveSpec> usd_eur_gbp() { return {flat_curve(0), flat_curve(1), flat_curve(2)}; }
const std::vector<std::string> kCodes{"USD", "EUR", "GBP"};

dv::FxMoveResolver resolver(const pf::MultiCurveBook& book, std::optional<std::string> pivot = std::nullopt) {
  dv::FxMoveResolver r(kCodes, pf::fx_pair_slots(usd_eur_gbp(), book), pivot);
  r.reserve(4);
  return r;
}
std::vector<double> factors(const dv::FxMoveResolver& r, const std::vector<dv::FxBumpRef>& bumps) {
  std::vector<double> f(static_cast<std::size_t>(r.slots().n_slots()));
  r.factors(bumps, f);
  return f;
}

}  // namespace

TEST(FxPairSlots, APositionsPairIsItsResetCurvesCurrenciesInOrder) {
  const pf::MultiCurveBook book{{swap(), xccy(1, 0, 1.10, 3e7), xccy(2, 0, 1.30, 2e7), xccy(1, 0, 1.10, -1e7),
                                 xccy(0, 1, 0.91, 1e7)}};
  const pf::FxPairSlots s = pf::fx_pair_slots(usd_eur_gbp(), book);
  EXPECT_EQ(s.slot_of, (std::vector<int>{-1, 0, 1, 0, 2})) << "a swap has no pair; a repeated pair shares its slot";
  EXPECT_EQ(s.base, (std::vector<int>{1, 2, 0}));
  EXPECT_EQ(s.quote, (std::vector<int>{0, 0, 1})) << "USDEUR is its own slot, not merged with EURUSD";
  pf::MultiCurveBook bad{{xccy(1, 0, 1.10, 3e7)}};
  bad.positions[0].mtm_reset_num = 3;
  EXPECT_THROW((void)pf::fx_pair_slots(usd_eur_gbp(), bad), std::invalid_argument);
}

TEST(FxMoveResolver, DirectInverseCrossAndRepeatedBumps) {
  // slots: EURUSD, GBPUSD, USDEUR, EURGBP
  const pf::MultiCurveBook book{
      {xccy(1, 0, 1.10, 1.0), xccy(2, 0, 1.30, 1.0), xccy(0, 1, 0.91, 1.0), xccy(1, 2, 0.85, 1.0)}};
  dv::FxMoveResolver r = resolver(book);
  const int EUR = r.intern("EUR"), USD = r.intern("USD"), GBP = r.intern("GBP");
  EXPECT_EQ(EUR, 1) << "a bundle code's id is its curve currency tag";

  const std::vector<double> f = factors(r, {{EUR, USD, 0.02}, {GBP, USD, -0.01}});
  EXPECT_EQ(f[0], 1.0 * (1.0 + 0.02)) << "a directly bumped pair: bitwise the single-pair factor";
  EXPECT_EQ(f[1], 1.0 * (1.0 + -0.01));
  EXPECT_EQ(f[2], 1.0 / (1.0 + 0.02)) << "the inverse orientation divides";
  EXPECT_EQ(f[3], (1.0 * (1.0 + 0.02)) / (1.0 * (1.0 + -0.01))) << "EURGBP = EURUSD / GBPUSD";

  const pf::MultiCurveBook eur_only{{xccy(1, 0, 1.10, 1.0)}};
  dv::FxMoveResolver r1 = resolver(eur_only);
  const int E1 = r1.intern("EUR"), U1 = r1.intern("USD");
  const std::vector<double> rep = factors(r1, {{E1, U1, 0.02}, {E1, U1, 0.02}, {U1, E1, 0.01}});
  EXPECT_EQ(rep[0], 1.0 * (1.0 + 0.02) * (1.0 + 0.02) / (1.0 + 0.01))
      << "one pair compounds in request order; a repeat named the other way divides";
}

TEST(FxMoveResolver, UntouchedCurrenciesHoldAndAPivotAnchorsThem) {
  const pf::MultiCurveBook book{{xccy(1, 0, 1.10, 1.0), xccy(1, 2, 0.85, 1.0)}};  // EURUSD, EURGBP
  // No pivot: under an EURUSD bump alone, EURGBP depends on whether GBP holds against USD or against EUR -> refused.
  dv::FxMoveResolver loose = resolver(book);
  const int EUR = loose.intern("EUR"), USD = loose.intern("USD");
  EXPECT_THROW((void)factors(loose, {{EUR, USD, 0.02}}), std::invalid_argument);
  // A bump on currencies no position uses leaves both pairs exactly where they are.
  const int AUD = loose.intern("AUD"), NZD = loose.intern("NZD");
  EXPECT_EQ(factors(loose, {{AUD, NZD, 0.03}}), (std::vector<double>{1.0, 1.0}));
  const double up = 1.0 * (1.0 + 0.02);
  // fx_pivot USD: GBP holds against USD, so EURGBP moves with EURUSD.
  dv::FxMoveResolver usd = resolver(book, "USD");
  EXPECT_EQ(factors(usd, {{usd.intern("EUR"), usd.intern("USD"), 0.02}}), (std::vector<double>{up, up}));
  // fx_pivot EUR: GBP holds against EUR, so EURGBP does not move.
  dv::FxMoveResolver eur = resolver(book, "EUR");
  EXPECT_EQ(factors(eur, {{eur.intern("EUR"), eur.intern("USD"), 0.02}}), (std::vector<double>{up, 1.0}));
}

TEST(FxMoveResolver, AnOverDeterminedTriangleIsRefused) {
  const pf::MultiCurveBook book{{xccy(1, 0, 1.10, 1.0)}};
  dv::FxMoveResolver r = resolver(book);
  const int EUR = r.intern("EUR"), USD = r.intern("USD"), GBP = r.intern("GBP");
  EXPECT_THROW((void)factors(r, {{EUR, USD, 0.02}, {GBP, USD, -0.01}, {EUR, GBP, 0.0}}), std::invalid_argument);
  EXPECT_NO_THROW((void)factors(r, {{EUR, USD, 0.02}, {USD, EUR, 0.01}})) << "one pair named both ways is not a cycle";
}

TEST(FxMoveResolver, ASlotWhoseTagHasNoCodeIsRefusedAtSetup) {
  const pf::MultiCurveBook book{{xccy(2, 0, 1.30, 1.0)}};
  EXPECT_THROW(dv::FxMoveResolver(std::vector<std::string>{"USD", "EUR"}, pf::fx_pair_slots(usd_eur_gbp(), book),
                                  std::nullopt),
               std::invalid_argument);
}

TEST(BookFxMoves, CurrencyCodesAreNeededOnlyForAnFxMoveOnAnXccyBook) {
  const std::vector<cal::BundleCurveSpec> curves = usd_eur_gbp();
  const pf::MultiCurveBook xccy_book{{xccy(1, 0, 1.10, 1.0)}};
  const pf::MultiCurveBook swap_book{{swap()}};
  const std::vector<std::vector<dv::FxBump>> fx_move{{dv::FxBump{"EUR", "USD", 0.02}}};
  const std::vector<std::vector<dv::FxBump>> no_fx{{}};
  const std::vector<std::string> no_codes;
  EXPECT_THROW((void)dv::book_fx_moves(no_codes, curves, xccy_book, fx_move, std::nullopt), std::invalid_argument)
      << "an FX move on an xccy book: the pair's orientation needs the codes";
  EXPECT_NO_THROW((void)dv::book_fx_moves(no_codes, curves, xccy_book, no_fx, std::nullopt)) << "no FX move";
  EXPECT_NO_THROW((void)dv::book_fx_moves(no_codes, curves, swap_book, fx_move, std::nullopt)) << "no xccy position";

  const std::vector<std::vector<dv::FxBump>> two_moves{{dv::FxBump{"EUR", "USD", 0.02}}, {}};
  const dv::BookFxMoves m = dv::book_fx_moves(kCodes, curves, xccy_book, two_moves, std::nullopt);
  ASSERT_EQ(m.factor.size(), 2u);
  EXPECT_EQ(m.factor[0], (std::vector<double>{1.0 * (1.0 + 0.02)}));
  EXPECT_TRUE(m.factor[1].empty()) << "a move with no bump leaves every spot alone";
}

TEST(SetXccyFx, EachPositionTakesItsSlotsFactorAndTheTemplatedAndCompiledBooksAgree) {
  cal::BundleProblem p;
  p.curves = usd_eur_gbp();
  const Eigen::Vector3d x(0.03, 0.02, 0.04);
  const pf::MultiCurveBook book{{swap(), xccy(1, 0, 1.10, 3e7), xccy(2, 0, 1.30, 2e7)}};
  const pf::FxPairSlots slots = pf::fx_pair_slots(p.curves, book);
  const std::vector<double> f{1.02, 0.99};

  pf::MultiCurveBook moved = book;
  pf::set_xccy_fx(moved, book, slots, f);
  EXPECT_EQ(moved.positions[0].fx_spot, book.positions[0].fx_spot);
  EXPECT_EQ(moved.positions[1].fx_spot, 1.10 * 1.02);
  EXPECT_EQ(moved.positions[2].fx_spot, 1.30 * 0.99) << "before SC2 both took 1.02 * 0.99";

  pf::MultiCurveBook by_hand = book;  // written out independently of set_xccy_fx
  by_hand.positions[1].fx_spot = 1.10 * 1.02;
  by_hand.positions[2].fx_spot = 1.30 * 0.99;
  EXPECT_EQ(cal::book_value_at(moved, p, x), cal::book_value_at(by_hand, p, x));

  // The compiled book rescales its rows in place: bitwise a fresh compile of the hand-scaled book, and back again.
  pf::CompiledMultiCurveBook compiled(p.curves, book);
  const double base = compiled.npv(x);
  compiled.set_fx_factors(slots, f);
  EXPECT_EQ(compiled.npv(x), pf::CompiledMultiCurveBook(p.curves, by_hand).npv(x));
  const std::vector<double> one{1.0, 1.0};
  compiled.set_fx_factors(slots, one);
  EXPECT_EQ(compiled.npv(x), base) << "factors of 1 restore the constructed rows exactly";

  // A seasoned (fallback) xccy position rides the templated half: its spot moves too.
  pf::MultiCurveBook seasoned = book;
  seasoned.positions[2].mtm_coupons[0].reset_fx = 1.25;  // a fixed first reset: not compilable
  pf::MultiCurveBook seasoned_by_hand = seasoned;
  seasoned_by_hand.positions[1].fx_spot = 1.10 * 1.02;
  seasoned_by_hand.positions[2].fx_spot = 1.30 * 0.99;
  pf::CompiledMultiCurveBook with_fallback(p.curves, seasoned);
  ASSERT_EQ(with_fallback.n_fallback(), 1);
  with_fallback.set_fx_factors(slots, f);
  EXPECT_EQ(with_fallback.npv(x), pf::CompiledMultiCurveBook(p.curves, seasoned_by_hand).npv(x));
}
