// PROPERTY / FUZZ tests over the WHOLE conventions DB surface (conventions/conventions.json ->
// conventions_data.hpp). Complements the per-feature gates:
//   - conventions_test.cpp pins a HAND-PICKED set of indices/products against QuantLib;
//   - calendar_data_test.cpp golden-checks yearly holiday COUNTS;
//   - bus252_daycount_test.cpp checks BUS/252 on a couple of KNOWN spans.
// This file instead sweeps EVERY row of kIndices / kProducts / kCalendars / kBonds and asserts the
// STRUCTURAL invariants that must hold for any DB entry — so a newly-added currency/index/product that
// forgets to wire a calendar, day count, frequency or par_product round-trip is caught without a bespoke
// per-instrument test. QuantLib-free (targets swaps_tests). Deterministic (seeded).
//
// The North Star (CLAUDE.md "generic building blocks, specifics as data"): a new instrument is a DB row,
// so the DB's internal consistency is the contract these properties defend.

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <string_view>

#include "swaps/build/day_count.hpp"   // year_frac (+ business_days_between, is_business_day via calendar.hpp)
#include "swaps/conventions_data.hpp"

namespace db = swaps::conventions;
namespace bd = swaps::build;

namespace {

// The day counts the accrual kernel (build/day_count.hpp year_frac) can actually compute. Every product/
// index leg day_count in the DB must be one of these — that ties the DB string to code that consumes it,
// not just to a spelling convention. BUS/252 is handled via the 4-arg calendar-aware overload.
bool is_calendar_free_dc(std::string_view dc) {
  return dc == "ACT/360" || dc == "ACT/365F" || dc == "ACT/ACT" || dc == "ACT/ACT.ISDA" ||
         dc == "30E/360" || dc == "30U/360";
}
// Bond day counts additionally allow ACT/ACT-ICMA (reference-period day count, priced elsewhere).
bool is_known_bond_dc(std::string_view dc) {
  return is_calendar_free_dc(dc) || dc == "ACT/ACT-ICMA" || dc == "BUS/252";
}

// A DB calendar id resolves to a real CalendarConv.
bool calendar_resolves(std::string_view id) { return db::calendar(id).has_value(); }

// Two representative in-year dates for a functional year_frac probe.
const bd::Date kD1 = bd::Date::from_iso("2025-01-15");
const bd::Date kD2 = bd::Date::from_iso("2025-07-15");

// Assert a (non-empty) leg day_count both resolves AND actually computes a positive accrual through the
// engine's own year_frac, using `cal` for the BUS/252 calendar-aware path.
void expect_leg_daycount_computes(std::string_view dc, const std::string& cal, const char* who) {
  if (dc.empty()) return;  // legs of futures / FX / basis-fixed rows legitimately carry no day count
  ASSERT_TRUE(is_calendar_free_dc(dc) || dc == "BUS/252") << who << " unknown day count " << dc;
  double yf = 0.0;
  ASSERT_NO_THROW(yf = bd::year_frac(std::string(dc), kD1, kD2, cal)) << who << " dc=" << dc;
  EXPECT_GT(yf, 0.0) << who << " dc=" << dc;  // a forward span accrues positive time
}

// A frequency/tenor token is "sane": it parses to a strictly-positive year fraction no longer than ~1Y
// (every coupon frequency in the DB is sub-annual through annual).
void expect_sane_frequency(std::string_view tok, const char* who) {
  if (tok.empty()) return;
  const double py = db::period_years(tok);
  EXPECT_GT(py, 0.0) << who << " frequency '" << tok << "' does not parse";
  EXPECT_LT(py, 1.5) << who << " frequency '" << tok << "' is implausibly long";
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// INDICES: every index round-trips to a known calendar + known day count + a par product that exists in
// the SAME currency and whose float leg points back at the index. Sweeps all of kIndices (not a curated
// subset like conventions_test.cpp's G20 row list).
// ---------------------------------------------------------------------------------------------------
TEST(ConventionsProperty, EveryIndexResolvesCalendarDayCountAndParProductRoundTrip) {
  ASSERT_GT(db::kIndices.size(), 0u);
  for (const auto& ix : db::kIndices) {
    const std::string who = std::string("index ") + std::string(ix.id);
    EXPECT_FALSE(ix.currency.empty()) << who << " has no currency";

    // calendar resolves ...
    ASSERT_TRUE(calendar_resolves(ix.calendar)) << who << " -> unknown calendar " << ix.calendar;
    // ... day count is one the engine understands ...
    EXPECT_TRUE(is_calendar_free_dc(ix.day_count) || ix.day_count == "BUS/252")
        << who << " unknown day count " << ix.day_count;
    // ... and the accrual actually computes (BUS/252 through the index's own calendar).
    expect_leg_daycount_computes(ix.day_count, std::string(ix.calendar), who.c_str());

    // par product exists, same currency, float leg names this index back.
    ASSERT_FALSE(ix.par_product.empty()) << who << " has no par product";
    const auto p = db::product(ix.par_product);
    ASSERT_TRUE(p.has_value()) << who << " -> missing par product " << ix.par_product;
    EXPECT_EQ(p->currency, ix.currency) << who << " currency mismatch vs par product";
    EXPECT_EQ(p->floating.index, ix.id) << ix.par_product << " float leg does not point back at " << ix.id;

    // index type is one of the two families the builders branch on.
    EXPECT_TRUE(ix.type == "overnight" || ix.type == "ibor") << who << " unknown type " << ix.type;
  }
}

// ---------------------------------------------------------------------------------------------------
// PRODUCTS: every product's declared calendar resolves, its bdc (when present) is one the adjuster
// accepts, and every NON-EMPTY leg references a computable day count + a sane frequency. Empty legs
// (FX forwards, futures, xccy/basis fixed side) are skipped by design — the DB uses "" to mean "no leg".
// ---------------------------------------------------------------------------------------------------
TEST(ConventionsProperty, EveryProductLegHasResolvableCalendarDayCountFrequency) {
  ASSERT_GT(db::kProducts.size(), 0u);
  for (const auto& p : db::kProducts) {
    const std::string who = std::string("product ") + std::string(p.id);
    ASSERT_TRUE(calendar_resolves(p.calendar)) << who << " -> unknown calendar " << p.calendar;

    if (!p.bdc.empty()) {
      const bool known = p.bdc == "Following" || p.bdc == "Preceding" || p.bdc == "ModifiedFollowing" ||
                         p.bdc == "ModifiedPreceding" || p.bdc == "Unadjusted";
      EXPECT_TRUE(known) << who << " unknown business-day convention " << p.bdc;
    }

    for (const auto* leg : {&p.fixed, &p.floating}) {
      expect_leg_daycount_computes(leg->day_count, std::string(p.calendar), who.c_str());
      if (!p.zero_coupon) expect_sane_frequency(leg->frequency, who.c_str());
    }
    // A zero_coupon product (BRL DI×Pre) is ONE period spot->maturity: day counts, never a frequency.
    if (p.zero_coupon) {
      EXPECT_TRUE(p.fixed.frequency.empty() && p.floating.frequency.empty()) << who << " zero_coupon carries a frequency";
      EXPECT_FALSE(p.fixed.day_count.empty() || p.floating.day_count.empty()) << who << " zero_coupon leg lacks a day count";
      continue;
    }
    // Any accruing leg carries a frequency -- EXCEPT xccy / MTM cross-currency products, which declare a
    // single shared frequency at the product level in the JSON (not represented per-leg in ProductConv),
    // so their legs legitimately have a day count but an empty per-leg frequency. Skip those.
    const bool is_xccy = std::string(p.id).find("XCCY") != std::string::npos ||
                         std::string(p.id).find("MTM") != std::string::npos;
    if (!is_xccy)
      for (const auto* leg : {&p.fixed, &p.floating})
        if (!leg->day_count.empty())
          EXPECT_FALSE(leg->frequency.empty()) << who << " leg has a day count but no frequency";
  }
}

// ---------------------------------------------------------------------------------------------------
// BONDS: each catalogued bond convention resolves to a known calendar, a computable day count and a sane
// coupon frequency (light structural sweep; pricing correctness is bond_*_test.cpp's job).
// ---------------------------------------------------------------------------------------------------
TEST(ConventionsProperty, EveryBondConventionIsStructurallyValid) {
  for (const auto& bnd : db::kBonds) {
    const std::string who = std::string("bond ") + std::string(bnd.id);
    EXPECT_FALSE(bnd.currency.empty()) << who << " has no currency";
    ASSERT_TRUE(calendar_resolves(bnd.calendar)) << who << " -> unknown calendar " << bnd.calendar;
    EXPECT_TRUE(is_known_bond_dc(bnd.day_count)) << who << " unknown day count " << bnd.day_count;
    expect_sane_frequency(bnd.frequency, who.c_str());
    EXPECT_TRUE(bnd.stub_discount == "compound" || bnd.stub_discount == "simple")
        << who << " unknown stub_discount " << bnd.stub_discount;
  }
}

// ---------------------------------------------------------------------------------------------------
// CALENDARS: the weekend mask of every calendar is well-formed (1 or 2 weekend days, none past Sunday),
// and is_business_day is STABLE across a multi-year day-by-day sweep — no exceptions, deterministic, and
// a masked weekend day is a business day ONLY when a working_day rule names it (China). This is the property behind every schedule the DB drives.
// ---------------------------------------------------------------------------------------------------
TEST(ConventionsProperty, WeekendMasksWellFormedAndBusinessDayStableAcrossYears) {
  ASSERT_GT(db::kCalendars.size(), 0u);
  const bd::Date sweep_begin = bd::Date::from_iso("2023-01-01");
  const bd::Date sweep_end = bd::Date::from_iso("2026-01-01");  // 3 full years: every weekday recurs often

  for (const auto& cal : db::kCalendars) {
    const std::string id{cal.id};
    // Mask: nonzero, within the 7-bit weekday range, and 1–2 weekend days.
    EXPECT_GT(cal.weekend_mask, 0) << id << " has an empty weekend mask";
    EXPECT_LT(cal.weekend_mask, 128) << id << " weekend mask has a bit past Sunday";
    const int wk = __builtin_popcount(static_cast<unsigned>(cal.weekend_mask));
    EXPECT_GE(wk, 1) << id;
    EXPECT_LE(wk, 2) << id;

    for (bd::Date d = sweep_begin; d < sweep_end; d = d.plus_days(1)) {
      const bool biz = bd::is_business_day(id, d);
      // deterministic: a second identical query agrees.
      EXPECT_EQ(biz, bd::is_business_day(id, d)) << id << " non-deterministic on serial " << d.serial();
      // a masked weekend weekday is never a business day — UNLESS a "working_day" rule names that exact date
      // (China's adjusted working weekends, verified day-by-day against QuantLib in calendar_ql_oracle_test).
      if ((cal.weekend_mask >> d.weekday()) & 1) {
        bool working = false;
        for (std::size_t i = cal.rule_begin; i < cal.rule_begin + cal.rule_count; ++i) {
          const auto& r = db::kHolidayRules[i];
          if (r.kind == "working_day" && r.month == int(d.month()) && r.day == int(d.day()) &&
              (r.from_year == 0 || d.year() >= r.from_year) && (r.to_year == 0 || d.year() <= r.to_year))
            working = true;
        }
        EXPECT_EQ(biz, working) << id << " reports weekend weekday " << d.weekday() << " (" << bd::iso(d)
                                << ") as " << (biz ? "a business day with no working_day rule" : "closed despite a working_day rule");
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------
// BUS/252 invariants on the Brazilian (B3/ANBIMA) calendar. bus252_daycount_test.cpp checks TWO known
// spans; here we assert the ALGEBRA of business_days_between over randomised (seeded) split points:
//   additivity over an interior date, sign-flip, non-negativity for d1<d2, and the whole-year identity
//   Σ per-month counts == the direct year count == 252 for 2025.
// ---------------------------------------------------------------------------------------------------
TEST(ConventionsProperty, Bus252BusinessDayCountAlgebraOnBrl) {
  const std::string cal = "BRL";
  ASSERT_TRUE(calendar_resolves(cal));

  // (1) Whole-year identity: Σ_{m=1..12} bdb(month_start_m, month_start_{m+1}) == bdb(Jan1, next Jan1).
  const bd::Date y0 = bd::Date::from_iso("2025-01-01");
  const bd::Date y1 = bd::Date::from_iso("2026-01-01");
  long month_sum = 0;
  for (unsigned m = 1; m <= 12; ++m) {
    const bd::Date a = bd::Date::ymd(2025, m, 1);
    const bd::Date b = (m == 12) ? y1 : bd::Date::ymd(2025, m + 1, 1);
    const long c = bd::business_days_between(cal, a, b);
    EXPECT_GE(c, 0) << "month " << m << " count is negative";
    month_sum += c;
  }
  const long year_direct = bd::business_days_between(cal, y0, y1);
  EXPECT_EQ(month_sum, year_direct) << "per-month BUS/252 counts do not telescope to the year";
  EXPECT_EQ(year_direct, 252) << "2025 must hold exactly 252 BRL business days";

  // (2) Randomised algebra: additivity over an interior split, sign-flip, non-negativity for d1<d2.
  std::mt19937 rng(0xB2A11u);
  std::uniform_int_distribution<int> off(0, 730);  // day offsets within a ~2y window from a fixed anchor
  const bd::Date anchor = bd::Date::from_iso("2024-06-03");
  for (int trial = 0; trial < 200; ++trial) {
    int o1 = off(rng), o2 = off(rng), o3 = off(rng);
    if (o1 > o2) std::swap(o1, o2);
    if (o2 > o3) std::swap(o2, o3);
    if (o1 > o2) std::swap(o1, o2);  // now o1 <= o2 <= o3
    const bd::Date a = anchor.plus_days(o1), b = anchor.plus_days(o2), c = anchor.plus_days(o3);

    const long ab = bd::business_days_between(cal, a, b);
    const long bc = bd::business_days_between(cal, b, c);
    const long ac = bd::business_days_between(cal, a, c);
    EXPECT_EQ(ac, ab + bc) << "additivity failed on trial " << trial;   // additive over the split date
    EXPECT_GE(ab, 0) << "trial " << trial;                              // non-negative for d1 <= d2
    EXPECT_GE(bc, 0) << "trial " << trial;
    EXPECT_EQ(bd::business_days_between(cal, b, a), -ab) << "sign-flip failed on trial " << trial;
    // the count never exceeds the calendar-day span (business days are a subset).
    EXPECT_LE(ac, (c - a)) << "trial " << trial;
  }
}
