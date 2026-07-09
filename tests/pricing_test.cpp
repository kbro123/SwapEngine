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

#include "reference_curve.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;

namespace {

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
  ext::shared_ptr<swaps::qlx::TwoRegionTermStructure> ts;

  void SetUp() override {
    ts = ext::make_shared<swaps::qlx::TwoRegionTermStructure>(mk.today, mk.dc, &curve);
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
    const auto sched = swaps::qlx::extract_ois_swap(*swap, mk.today, mk.dc);
    const double ours = swaps::pricing::ois_par_rate<double>(sched, curve);
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

    const auto sched = swaps::qlx::extract_compounded_future(f.start, f.end, mk.today, mk.dc);
    const double ours = swaps::pricing::compounded_future_rate<double>(sched, curve);
    EXPECT_TRUE(close(ours, ql_rate, swaps::tol::curve_rel))
        << " 3M future " << f.start << ".." << f.end;
    worst = std::max(worst, std::abs(ours - ql_rate));
    ++n;
  }
  ASSERT_EQ(n, 8);
  std::cout << "  [3m future] max |ours - QuantLib| = " << worst << "\n";
}

TEST_F(Pricing, AveragedFutureRateMatchesQuantLib) {
  double worst = 0.0;
  int n = 0;
  for (const auto& f : mk.futures) {
    if (f.quarterly) continue;  // 1M arithmetic-average only
    OvernightIndexFuture qlf(mk.sofr, f.start, f.end, Handle<Quote>(), RateAveraging::Simple);
    const double ql_rate = 1.0 - qlf.NPV() / 100.0;

    const auto sched = swaps::qlx::extract_averaged_future(mk.sofr, f.start, f.end, mk.today, mk.dc);
    const double ours = swaps::pricing::averaged_future_rate<double>(sched, curve);
    EXPECT_TRUE(close(ours, ql_rate, swaps::tol::curve_rel))
        << " 1M future " << f.start << ".." << f.end
        << " (realized_days sum=" << sched.realized_sum << ")";
    worst = std::max(worst, std::abs(ours - ql_rate));
    ++n;
  }
  ASSERT_EQ(n, 12);
  std::cout << "  [1m future] max |ours - QuantLib| = " << worst << "\n";
}
