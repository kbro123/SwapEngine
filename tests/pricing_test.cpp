// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
// Validates the templated pricing kernel (swaps/pricing/cashflows.hpp) fed by the QuantLib
// extractor (swaps/ql/extract.hpp) against QuantLib's OWN pricing off the same curve.
//
// This is the crux of the Phase 2 architecture: QuantLib defines the instruments and we extract
// their cashflow structure once; the kernel then reprices with plain DF math. If the kernel
// matches QuantLib coupon-for-coupon at 1e-10, we have earned the right to (a) run AAD through the
// kernel and (b) vectorise it — neither of which QuantLib's pricing allows.

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <cmath>
#include <sstream>

#include "reference_curve.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/build/observations.hpp"  // the SHIPPED observation builder, oracled below
#include "swaps/build/conventions.hpp"   // the SHIPPED DB -> SwapConv assembly, oracled below
#include "swaps/build/instruments.hpp"   // the SHIPPED instrument builders, oracled below
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;

namespace {

// QuantLib Date -> the engine's Unix-day Date, via ISO text (the two epochs differ).
swaps::build::Date eng_date(const QuantLib::Date& d) {
  std::ostringstream os;
  os << QuantLib::io::iso_date(d);
  return swaps::build::Date::from_iso(os.str());
}

::testing::AssertionResult close(double got, double want, double rel) {
  const double err = std::abs(got - want) / std::max(1.0, std::abs(want));
  return err <= rel ? ::testing::AssertionSuccess()
                    : (::testing::AssertionFailure()
                       << "got=" << got << " want=" << want << " rel=" << err << " > " << rel);
}

// Fixture: build the reference market + curve, wire our curve in as the QuantLib term structure.
struct Pricing : ::testing::Test {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  rb::Curve curve = rb::reference_curve(mk);
  ext::shared_ptr<swaps::qlx::CurveTermStructure<rb::Curve>> ts;

  void SetUp() override {
    ts = ext::make_shared<swaps::qlx::CurveTermStructure<rb::Curve>>(mk.today, mk.dc, &curve);
    ts->enableExtrapolation();
    h.linkTo(ts);
    for (auto& s : mk.swaps) s->deepUpdate();
  }
};

}  // namespace

TEST_F(Pricing, OisParRateMatchesQuantLib) {
  double worst = 0.0;
  for (const auto& swap : mk.swaps) {
    const double ql = swap->fairRate();
    const auto fl = swaps::qlx::extract_float_leg(swap->overnightLeg(), mk.today, mk.dc);
    const auto fx = swaps::qlx::extract_fixed_leg(swap->fixedLeg(), mk.today, mk.dc);
    const double ours = swaps::pricing::par_rate<double>(fl, fx, curve, curve);
    EXPECT_TRUE(close(ours, ql, swaps::tol::curve_rel))
        << " swap maturity " << swap->maturityDate();
    worst = std::max(worst, std::abs(ours - ql));
  }
  std::cout << "  [ois par] max |ours - QuantLib| = " << worst << "\n";
}

TEST_F(Pricing, CompoundedFutureRateMatchesQuantLib) {
  double worst = 0.0;
  int n = 0;
  for (const auto& f : mk.futures) {
    if (!f.quarterly) continue;  // 3M compounded only; 1M averaged handled separately
    // QuantLib future with ZERO convexity: NPV = 100*(1 - R), so R_ql = 1 - NPV/100.
    OvernightIndexFuture qlf(mk.sofr, f.start, f.end, Handle<Quote>(), RateAveraging::Compound);
    const double ql_rate = 1.0 - qlf.NPV() / 100.0;

    // 3M compounding future == ONE telescoped sub-period [start, end], denominator on the index dc.
    const auto obs = swaps::qlx::make_observation(
        {{f.start, f.end}}, 0.0, mk.sofr->dayCounter().yearFraction(f.start, f.end), mk.today, mk.dc);
    const double ours = swaps::pricing::rate<double>(obs, curve);
    EXPECT_TRUE(close(ours, ql_rate, swaps::tol::curve_rel))
        << " 3M future " << f.start << ".." << f.end;
    worst = std::max(worst, std::abs(ours - ql_rate));
    ++n;
  }
  ASSERT_EQ(n, 8);
  std::cout << "  [3m future] max |ours - QuantLib| = " << worst << "\n";
}

// THE oracle that would have caught the 365/360 error (added 2026-09-10). Every other averaging oracle
// builds its observation with a test-side reconstruction of QuantLib's averagedRate(), so the SHIPPED
// builder was never compared to QuantLib at all: its only reference was the Python compiler, which divided
// by curve time the same wrong way. Here `build::observation` itself produces the observation, and QuantLib
// prices the same contract off the same curve. Fully-forecast contracts only, since a partially fixed one
// needs a realized prefix from the fixing history rather than the 0.0 passed here.
TEST_F(Pricing, AveragedFutureFromTheShippedBuilderMatchesQuantLib) {
  double worst = 0.0;
  int n = 0;
  for (const auto& f : mk.futures) {
    if (f.quarterly || f.start <= mk.today) continue;  // 1M arithmetic-average, fully forecast
    OvernightIndexFuture qlf(mk.sofr, f.start, f.end, Handle<Quote>(), RateAveraging::Simple);
    const double ql_rate = 1.0 - qlf.NPV() / 100.0;

    // The production path: engine dates, the DB's own calendar and day count, no test-side reconstruction.
    const auto obs = swaps::build::observation(eng_date(mk.today), eng_date(f.start), eng_date(f.end),
                                               "averaged", 0.0, "ACT/360", "USD-SOFR");
    const double ours = swaps::pricing::rate<double>(obs, curve);

    EXPECT_TRUE(close(ours, ql_rate, swaps::tol::curve_rel))
        << " 1M future " << f.start << ".." << f.end << " builder=" << ours << " QuantLib=" << ql_rate;
    worst = std::max(worst, std::abs(ours - ql_rate));
    ++n;
  }
  ASSERT_GE(n, 8) << "expected the fully-forecast 1M strip";
  std::cout << "  [1m future, shipped builder] max |ours - QuantLib| = " << worst << "\n";
}

TEST_F(Pricing, AveragedFutureRateMatchesQuantLib) {
  double worst = 0.0;
  int n = 0;
  for (const auto& f : mk.futures) {
    if (f.quarterly) continue;  // 1M arithmetic-average only
    OvernightIndexFuture qlf(mk.sofr, f.start, f.end, Handle<Quote>(), RateAveraging::Simple);
    const double ql_rate = 1.0 - qlf.NPV() / 100.0;

    // 1M arithmetic-average future == one sub-period per business day (past days folded into realized).
    const auto obs = rb::avg_future_obs(mk, f);
    const double ours = swaps::pricing::rate<double>(obs, curve);
    EXPECT_TRUE(close(ours, ql_rate, swaps::tol::curve_rel))
        << " 1M future " << f.start << ".." << f.end << " (realized=" << obs.realized << ")";
    worst = std::max(worst, std::abs(ours - ql_rate));
    ++n;
  }
  ASSERT_EQ(n, 12);
  std::cout << "  [1m future] max |ours - QuantLib| = " << worst << "\n";
}

// THE SHIPPED SWAP BUILDER vs QuantLib (golden-source step 6's S1 gap, 2026-09-10). Every other oracle in
// this file prices an instrument the FIXTURE assembled: QuantLib defines the trade, the extractor lifts its
// cashflows, and the kernel reprices them. That checks the KERNEL, and it is exactly why the averaged-weight
// bug survived -- nothing compared what `build::par_swap` itself produces from the conventions DB to an
// independent number. Here the engine is given a currency, an index and a maturity, and everything else --
// spot lag, payment lag, roll convention, both frequencies, both day counts, the compounding mode -- comes
// from the DB row through the shipped path. QuantLib's OIS is then built to those SAME conventions rather
// than to MakeOIS's defaults, so a disagreement is a real disagreement and not a convention mismatch.
TEST_F(Pricing, ParSwapFromTheShippedBuilderMatchesQuantLib) {
  const swaps::build::SwapConv conv = swaps::build::swap_conv("USD", "USD-SOFR");
  ASSERT_EQ(conv.product_id, "USD-SOFR-OIS");

  double worst = 0.0;
  int n = 0;
  for (int y : {1, 2, 3, 5, 7, 10, 20, 30}) {
    const ext::shared_ptr<OvernightIndexedSwap> qls = MakeOIS(Period(y, Years), mk.sofr, 0.03)
                                                          .withDiscountingTermStructure(h)
                                                          .withSettlementDays(conv.spot_lag)
                                                          .withPaymentLag(conv.pay_lag)
                                                          .withPaymentAdjustment(ModifiedFollowing);
    const double ql_rate = qls->fairRate();

    const swaps::calibration::Instrument ins =
        swaps::build::par_swap(eng_date(mk.today), conv, eng_date(qls->maturityDate()), 0, 0, 0.0);
    const double ours = swaps::pricing::par_rate<double>(ins.fwd.coupons, ins.fixed.coupons, curve, curve);

    EXPECT_TRUE(close(ours, ql_rate, swaps::tol::curve_rel))
        << " " << y << "y OIS  builder=" << ours << " QuantLib=" << ql_rate;
    worst = std::max(worst, std::abs(ours - ql_rate));
    ++n;
  }
  ASSERT_EQ(n, 8);
  std::cout << "  [par swap, shipped builder] max |ours - QuantLib| = " << worst << "\n";

  // NEGATIVE CONTROLS -- an oracle that cannot fail is not an oracle. The E2 bug was a day count read from
  // the wrong window, so perturb exactly that, plus the fixed frequency, and demand a visible disagreement.
  const ext::shared_ptr<OvernightIndexedSwap> ref = MakeOIS(Period(10, Years), mk.sofr, 0.03)
                                                        .withDiscountingTermStructure(h)
                                                        .withSettlementDays(conv.spot_lag)
                                                        .withPaymentLag(conv.pay_lag)
                                                        .withPaymentAdjustment(ModifiedFollowing);
  const swaps::build::Date mat10 = eng_date(ref->maturityDate());
  const auto rate_with = [&](swaps::build::SwapConv c) {
    const swaps::calibration::Instrument i =
        swaps::build::par_swap(eng_date(mk.today), c, mat10, 0, 0, 0.0);
    return swaps::pricing::par_rate<double>(i.fwd.coupons, i.fixed.coupons, curve, curve);
  };
  swaps::build::SwapConv wrong_fixed_dc = conv;
  wrong_fixed_dc.fixed_dc = "ACT/365F";  // the 365/360 family the E2 bug lived in
  EXPECT_GT(std::abs(rate_with(wrong_fixed_dc) - ref->fairRate()), 1e-4)
      << "a wrong FIXED day count must move the par rate well outside tolerance";
  swaps::build::SwapConv wrong_freq = conv;
  wrong_freq.fixed_freq_tok = "6M";
  EXPECT_GT(std::abs(rate_with(wrong_freq) - ref->fairRate()), swaps::tol::curve_rel)
      << "a wrong fixed frequency must be visible too";

  // And the structural reason the E2 bug could only ever bite an AVERAGED leg, pinned rather than assumed:
  // on a COMPOUNDED leg the float day count cancels EXACTLY. The coupon is (DF(s)/DF(e) - 1) / tau_index
  // accrued over tau_pay, and a plain leg's accrual and observation windows coincide, so tau_pay / tau_index
  // = 1 whatever the day count is. Getting that day count wrong is invisible here -- and was, for a year --
  // while on an averaged leg the same mistake scaled every rate by (index dc)/(curve dc).
  swaps::build::SwapConv other_float_dc = conv;
  other_float_dc.float_dc = "ACT/365F";
  EXPECT_LT(std::abs(rate_with(other_float_dc) - ref->fairRate()), 1e-15)
      << "a compounded leg's day count cancels between the accrual factor and the index year fraction";
}
