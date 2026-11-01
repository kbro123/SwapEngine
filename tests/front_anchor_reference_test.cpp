// @consistency-test — QuantLib-LINKED SELF-CONSISTENCY test (QuantLib builds the reference fixtures; the engine's
// E5 taxonomy: T2 calibration (PRINCIPLES P15: every curve of every QuantLib-built reference fixture has a front anchor)
// front-anchor rule is checked on them, not a QuantLib number). DO NOT DELETE OR WEAKEN. See tests/ORACLE_TESTS.md.
// THE FRONT ANCHOR RULE over the QuantLib-built reference fixtures: the SOFR reference market (single curve, as a 1-curve
// bundle), the realistic basis chain / star, and every multi-currency bundle. calibration::front_anchors names, per curve,
// the row that reads its forward over one window from the front; a curve with none is a fixture defect.
#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <vector>

#include <ql/quantlib.hpp>

#include "reference_bundle.hpp"
#include "reference_curve.hpp"
#include "reference_multicurrency.hpp"
#include "swaps/calibration/compiled_bundle.hpp"  // single_curve_bundle
#include "swaps/calibration/diagnostics.hpp"

namespace cal = swaps::calibration;
namespace rb = swaps::refbuild;

namespace {
void expect_anchored(const std::string& name, const cal::BundleProblem& p) {
  for (const cal::FrontAnchor& a : cal::front_anchors(p)) {
    if (a.first_knot < 0.0) continue;
    EXPECT_GE(a.row, 0) << name << ": curve " << a.curve << " (first knot " << a.first_knot
                        << "y) has no front anchor -- no row reads its forward over one window from the front";
    std::printf("  %-26s curve %2d first knot %7.4fy  anchor row %3d  window [%.4f, %.4f]\n", name.c_str(), a.curve, a.first_knot,
                a.row, a.start, a.end);
  }
}
}  // namespace

TEST(FrontAnchorReference, TheSofrReferenceMarketIsAnchored) {
  QuantLib::RelinkableHandle<QuantLib::YieldTermStructure> h;
  const rb::Market mk = rb::build_market(h);
  expect_anchored("sofr_reference", cal::single_curve_bundle(rb::build_problem(mk)));
  expect_anchored("sofr_reference_square", cal::single_curve_bundle(rb::build_square_problem(mk)));
}

TEST(FrontAnchorReference, TheRealisticBasisBundlesAreAnchored) {
  for (const auto topo : {rb::BundleTopology::Chain, rb::BundleTopology::Star}) {
    QuantLib::RelinkableHandle<QuantLib::YieldTermStructure> h;
    rb::Market mk = rb::build_market(h);
    const rb::RealisticBundle b = rb::build_realistic_bundle(mk, h, 3, topo);
    expect_anchored(topo == rb::BundleTopology::Chain ? "realistic_chain" : "realistic_star", b.prob);
  }
}

TEST(FrontAnchorReference, TheMultiCurrencyBundlesAreAnchored) {
  expect_anchored("eur_bundle", rb::build_eur_bundle().prob);
  expect_anchored("usd_bundle", rb::build_usd_bundle().prob);
  expect_anchored("xccy_bundle", rb::build_xccy_bundle().prob);
  expect_anchored("xccy_fx_bundle", rb::build_xccy_fx_bundle().prob);
  expect_anchored("eur_curves", rb::build_eur_curves().prob);
  expect_anchored("eur_multicurrency", rb::build_eur_multicurrency().prob);
  expect_anchored("eur_streamable", rb::build_eur_streamable().prob);
  expect_anchored("full_multicurrency", rb::build_full_multicurrency().prob);
  expect_anchored("full_multicurrency_noxccy", rb::build_full_multicurrency(false).prob);
}
