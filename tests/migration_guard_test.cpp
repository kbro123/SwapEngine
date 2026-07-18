// Stage 4 adoption guard: the reference market is now built as GENERIC Instruments
// (reference_curve.hpp build_problem / build_square_problem). This file pins the two facts that make
// that migration byte-safe, so a future change that breaks either fails loudly:
//
//   1. The DATE PIN. The generic overnight extractor reads valueDates().front()/back() where the
//      retired legacy extractor read accrualStartDate()/accrualEndDate(). On this market they
//      coincide (no realized prefix on any forward-starting coupon), which is the ONLY reason the
//      generic swap reprices the legacy OisSwap form bit-for-bit. Assert it on every coupon.
//
//   2. BIT-EXACT residual equivalence. Build the SAME market both ways -- legacy groups
//      (extract_ois_swap / extract_compounded_future / extract_averaged_future) and the generic
//      build_problem -- and assert their residual vectors are EQUAL to the last bit at the reference
//      forwards. (This half exercises the legacy extractors and is removed when they are deleted;
//      the date pin above is the durable guard.)

#include <gtest/gtest.h>

#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <vector>

#include "reference_curve.hpp"
#include "reference_market.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/ql/extract.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace rm = swaps::refmkt;
namespace cal = swaps::calibration;

namespace {

Eigen::VectorXd reference_forwards() {
  std::vector<double> x(rm::reference_front_forwards.begin(), rm::reference_front_forwards.end());
  x.insert(x.end(), rm::reference_back_forwards.begin(), rm::reference_back_forwards.end());
  return Eigen::Map<Eigen::VectorXd>(x.data(), static_cast<int>(x.size()));
}

// The OLD build_problem, reconstructed inline from the (still-present) legacy extractors, so the
// migration can be checked residual-for-residual. Residual order avg | comp | swaps, exactly what the
// generic build_problem reproduces in its single instrument list.
cal::CalibrationProblem legacy_problem(const rb::Market& mk) {
  cal::CalibrationProblem p;
  p.meeting_times = mk.meeting_times;
  p.back_times = mk.back_times;
  for (const auto& f : mk.futures)
    if (!f.quarterly)
      p.avg_futs.push_back({swaps::qlx::extract_averaged_future(mk.sofr, f.start, f.end, mk.today, mk.dc),
                            mk.convexity(f), 1.0 - f.market_price / 100.0});
  for (const auto& f : mk.futures)
    if (f.quarterly)
      p.comp_futs.push_back(
          {swaps::qlx::extract_compounded_future(f.start, f.end, mk.today, mk.dc, mk.sofr->dayCounter()),
           mk.convexity(f), 1.0 - f.market_price / 100.0});
  for (std::size_t i = 0; i < mk.swaps.size(); ++i)
    p.swaps.push_back({swaps::qlx::extract_ois_swap(*mk.swaps[i], mk.today, mk.dc), rm::swaps[i].par_rate});
  return p;
}

}  // namespace

// ---- 1. The date pin ---------------------------------------------------------------------------
TEST(MigrationGuard, OvernightValueDatesCoincideWithAccrualDatesOnEveryCoupon) {
  RelinkableHandle<YieldTermStructure> h;
  const rb::Market mk = rb::build_market(h);

  int coupons = 0;
  for (const auto& swap : mk.swaps) {
    for (const auto& cf : swap->overnightLeg()) {
      auto c = ext::dynamic_pointer_cast<OvernightIndexedCoupon>(cf);
      ASSERT_TRUE(c) << "overnight leg cashflow is not an OvernightIndexedCoupon";
      // No realized prefix on this market => the generic extractor's forward sub-period is exactly
      // [valueDates().front(), valueDates().back()], which must equal the legacy accrual span.
      EXPECT_EQ(c->valueDates().front(), c->accrualStartDate())
          << "generic sub_start (valueDates.front) must equal legacy float_acc_start (accrualStart)";
      EXPECT_EQ(c->valueDates().back(), c->accrualEndDate())
          << "generic sub_end (valueDates.back) must equal legacy float_acc_end (accrualEnd)";
      ++coupons;
    }
  }
  EXPECT_GT(coupons, 0) << "expected at least one overnight coupon to check";
}

// ---- 2. Bit-exact residual equivalence ---------------------------------------------------------
TEST(MigrationGuard, GenericBuildProblemReproducesLegacyResidualsBitForBit) {
  RelinkableHandle<YieldTermStructure> h;
  const rb::Market mk = rb::build_market(h);

  const cal::CalibrationProblem gen = rb::build_problem(mk);
  const cal::CalibrationProblem leg = legacy_problem(mk);

  ASSERT_EQ(gen.n_residuals(), leg.n_residuals());
  ASSERT_EQ(gen.n_knots(), leg.n_knots());

  const Eigen::VectorXd x = reference_forwards();
  ASSERT_EQ(x.size(), gen.n_knots());

  const Eigen::VectorXd rg = gen.residuals<double>(x);
  const Eigen::VectorXd rl = leg.residuals<double>(x);
  ASSERT_EQ(rg.size(), rl.size());

  double max_abs = 0.0;
  for (int i = 0; i < rg.size(); ++i) {
    EXPECT_EQ(rg[i], rl[i]) << "generic vs legacy residual row " << i << " must be bit-identical";
    max_abs = std::max(max_abs, std::abs(rg[i] - rl[i]));
  }
  std::cout << "  [migration] max |generic - legacy| residual = " << max_abs << " over " << rg.size()
            << " rows\n";
  EXPECT_EQ(max_abs, 0.0);
}
