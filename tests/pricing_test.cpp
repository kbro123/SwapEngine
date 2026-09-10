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
