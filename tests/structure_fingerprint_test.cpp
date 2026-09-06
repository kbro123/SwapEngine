// Property test for structure_fingerprint (the warm-vs-recompile switch behind the calibrated Model's
// cached statics). The whole warm path's correctness rests on ONE invariant: the fingerprint must change
// on ANY structural edit (else a stale W-cache silently prices the wrong curve) and must NOT change on a
// pure re-quote (else the µs warm tick never fires). This enumerates both directions.

#include <gtest/gtest.h>

#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/structure_fingerprint.hpp"
#include "swaps/curve/curve_module.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cvr = swaps::curve;

namespace {

px::FloatCoupon ois_coupon(double s, double e) {
  px::FloatCoupon c;
  c.obs.sub_start = {s};
  c.obs.sub_end = {e};
  c.obs.tau_index = e - s;
  c.pay = e;
  c.tau_pay = e - s;
  return c;
}
cal::FloatLeg float_leg(double T, int fc, int disc) {
  cal::FloatLeg leg;
  leg.forecast = fc;
  leg.discount = disc;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { leg.coupons.push_back(ois_coupon(prev, t)); prev = t; }
  return leg;
}
cal::FixedLeg fixed_leg(double T, int disc) {
  cal::FixedLeg leg;
  leg.discount = disc;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { px::FixedCoupon fc; fc.pay = t; fc.tau = 1.0; leg.coupons.push_back(fc); }
  return leg;
}
cal::Instrument par_swap(double T, int curve, double market = 0.0) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd = float_leg(T, curve, curve);
  ins.fixed = fixed_leg(T, curve);
  ins.market = market;
  return ins;
}

cal::BundleProblem base_problem() {
  cal::BundleProblem p;
  const std::vector<double> meeting{0.5}, back{2.0, 5.0, 10.0};
  cal::BundleCurveSpec cs;
  cs.base = -1;
  cs.currency = 0;
  cs.regions = cvr::flat_hermite(meeting, back);
  p.curves.push_back(cs);
  p.instruments.push_back(par_swap(0.5, 0, 0.030));
  p.instruments.push_back(par_swap(2.0, 0, 0.032));
  p.instruments.push_back(par_swap(5.0, 0, 0.036));
  p.instruments.push_back(par_swap(10.0, 0, 0.040));
  return p;
}

}  // namespace

// A pure re-quote (market level or soft band) must NOT move the fingerprint — else the warm tick never fires.
TEST(StructureFingerprint, QuoteOnlyChangesAreInvisible) {
  const cal::BundleProblem b = base_problem();
  const std::uint64_t fp = cal::structure_fingerprint(b);
  {
    cal::BundleProblem p = b;
    p.instruments[1].market += 0.01;
    EXPECT_EQ(cal::structure_fingerprint(p), fp) << "a market re-quote";
  }
  {
    cal::BundleProblem p = b;
    p.instruments[1].band_lower = 0.031;
    p.instruments[1].band_upper = 0.033;
    p.instruments[1].band_decay = 0.1;
    EXPECT_EQ(cal::structure_fingerprint(p), fp) << "a soft-quote band edit";
  }
  {
    cal::BundleProblem p = b;
    for (auto& ins : p.instruments) ins.market += 0.005;  // reprice the whole strip
    EXPECT_EQ(cal::structure_fingerprint(p), fp) << "a whole-market re-quote";
  }
}

// ANY structural edit MUST move the fingerprint — else a stale W-cache silently prices the wrong curve.
TEST(StructureFingerprint, EveryStructuralMutationMovesIt) {
  const cal::BundleProblem b = base_problem();
  const std::uint64_t fp = cal::structure_fingerprint(b);
  const auto moved = [&](cal::BundleProblem p, const char* what) {
    EXPECT_NE(cal::structure_fingerprint(p), fp) << "structural edit did NOT move the fingerprint: " << what;
  };

  { auto p = b; p.curves[0].regions[1].scheme = cvr::Scheme::Linear;              moved(p, "region scheme"); }
  { auto p = b; p.curves[0].regions[1].knots[0] += 0.25;                          moved(p, "a knot time"); }
  { auto p = b; p.curves[0].regions[1].sigma += 1.0;                              moved(p, "region sigma"); }
  { auto p = b; p.curves[0].regions[1].reg_lambda += 0.5;                         moved(p, "region reg_lambda"); }
  { auto p = b; p.curves[0].currency = 1;                                         moved(p, "curve currency"); }
  { auto p = b; p.curves[0].base = 0;                                             moved(p, "curve base role"); }
  { auto p = b; p.instruments.push_back(par_swap(7.0, 0, 0.038));                 moved(p, "added an instrument"); }
  { auto p = b; p.instruments.pop_back();                                         moved(p, "removed an instrument"); }
  { auto p = b; std::swap(p.instruments[1], p.instruments[2]);                    moved(p, "reordered instruments"); }
  { auto p = b; p.instruments[2].quote = cal::QuoteKind::ParSpread;               moved(p, "quote kind"); }
  { auto p = b; p.instruments[2].fwd.forecast = 1;                                moved(p, "forecast curve role"); }
  { auto p = b; p.instruments[2].fwd.discount = 1;                               moved(p, "discount curve role"); }
  { auto p = b; p.instruments[2].fwd.coupons[0].pay += 0.02;                      moved(p, "a coupon pay time"); }
  { auto p = b; p.instruments[2].fwd.coupons[0].tau_pay += 0.02;                  moved(p, "a coupon accrual"); }
  { auto p = b; p.instruments[2].fwd.coupons[0].spread += 0.001;                  moved(p, "a float spread"); }
  { auto p = b; p.instruments[2].fwd.coupons[0].scale = 0.5;                      moved(p, "a notional scale"); }
  { auto p = b; p.instruments[2].fwd.coupons[0].obs.sub_end[0] += 0.02;           moved(p, "an observation window"); }
  { auto p = b; p.instruments[2].fixed.coupons[0].tau += 0.01;                    moved(p, "a fixed-leg accrual"); }
  { auto p = b; px::Turn tn; tn.start = 0.9; tn.end = 1.1; p.curves[0].turns.push_back(tn); moved(p, "an added turn"); }
  { auto p = b; p.instruments[2].convexity += 0.001;                              moved(p, "a futures convexity adj"); }
}
