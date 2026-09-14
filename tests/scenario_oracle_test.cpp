// @oracle-test — the `scenario` and `scenario_grid` run_json VERBS vs QuantLib swaps repricing off the SHOCKED curves.
// DO NOT DELETE OR WEAKEN without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// WHAT THIS PINS THAT NOTHING ELSE DID (draft 2026-09-14). The two what-if verbs were covered only by shape tests
// (scenario_verb_test.cpp: "moves ~25 bp", "delta ~ PV01*25 within 5 %") and by byte-for-byte goldens
// (scenario_golden_test.cpp), which only prove a refactor changed nothing. Nobody had checked that the NUMBERS are
// right against an independent pricer: request decode -> calibrate -> fork the fitted state by the move -> sample
// the curves -> value the book, plus the unit (bp / 1e4), the sign, the per-curve role wiring and the SC1 rules
// (`scenario` lets an explicit shift_curve key REPLACE the parallel; `scenario_grid` ADDS axes).
//
// THE BOOK IS REAL DATED TRADES, BUILT TWICE FROM THE CONVENTIONS DB:
//   ours      build::swap_conv -> build::par_swap (spot lag, roll, pay lag, both frequencies, both day counts)
//   QuantLib  OvernightIndexedSwap / VanillaSwap on QuantLib's OWN Schedule(Backward) and QuantLib's OWN index
//             objects (Estr, Euribor6M, Sofr: their day counts, calendars and fixing days are QuantLib's, not the DB's)
// The same pair of builders prices the calibration instruments, so the verb calibrates a real three-curve bundle
// (ESTR with a year-end TURN, EURIBOR-6M projected on ESTR discounting, SOFR).
//
// TWO QUANTLIB ARMS PER MOVE, so a failure says WHICH half is wrong:
//   L1 "same shocked curve"  OUR ModularCurve rebuilt at x_base + shift (added to each curve's INTERPOLATION knots
//                            by the loop below, turns untouched), handed to QuantLib through qlx::CurveTermStructure.
//                            Checks the verb's assembly: book decode, role wiring, kernel, sign, bp unit.
//   L2 "QuantLib shocks it"  QuantLib's ZeroSpreadedTermStructure(Continuous) over the BASE curve with the move's
//                            spread. Independent of our shift arithmetic: it states what "+s on every forward" means
//                            (DF'(t) = DF(t)·exp(−s·t), zero'(t) = zero(t) + s). L1 == L2 holds exactly for this bundle
//                            because a Hermite region's tangents are linear in knot DIFFERENCES and its extrapolation
//                            is flat (curve/regions.hpp), so +s on every knot is +s on the forward everywhere, and
//                            the turn δ is a separate additive overlay (pricing/curve_handle.hpp TurnedCurve).
// x_base is the verb's OWN `base.x` (read back from the response), so the calibration itself is not under test
// here -- calibration_oracle_test.cpp owns that.
//
// TOLERANCE, MEASURED (first run, 2026-09-14): worst |verb − QuantLib| NPV 2.47e-8 (scenario) / 2.62e-8 (grid) on
// 1e8 gross notional, DF 3.3e-16, zero 3.9e-15. The drafted bounds (NPV tol::analytics_rel × gross = 0.01, DF/zero
// tol::curve_rel = 1e-10) sat 4e5x / 2.5e4x above that, so they are re-locked near it: kNpvTol = 1e-6 (~40x) and
// kCurveTol = 1e-13 (~25x). Still >= 1e7 below the smallest defect the NPV check exists to catch (a one-day pay-date
// error on one coupon ≈ 40; a 365/360 day count ≈ 1e5). Measured negative-control gaps: add-rule 5.5e5, sign 1.9e6,
// turn 1035 (book) / 4e-5 (DF), grid override-rule 1.1e6, wrong role 1.7e6.
//
// SCOPE, stated rather than implied. OUT (no QuantLib counterpart, so not faked): MtM XCCY positions and every
// `bump_fx` move / `fx` grid axis (a MultiCurveBook position carries no pair; the FX factor is SC2, an owner rule);
// a TURN JUMP as a calibrated free variable (QuantLib cannot calibrate it -- here it is only READ through the
// adapter, and the rule that a shift leaves it untouched IS checked, by L2 and by negative control 3); bands.
// The book `npv` is the verb's raw sum of per-position values, each in its own discount currency (EUR + USD here,
// no FX): the oracle reproduces that sum, it does not endorse it. SPREAD curves (base >= 0): the third test pins a
// parallel move on a spread curve against QuantLib -- written while this file was drafted, it predicted the 2x
// double move this oracle found (fixed in b5f3a4d, pricing::parallel_direction).
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <boost/json.hpp>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "conventions_ql.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/api/codec.hpp"          // bundle_to_json, instrument_to_json
#include "swaps/api/scenario.hpp"       // scenario_json — the verb under test
#include "swaps/api/scenario_grid.hpp"  // scenario_grid_json — the verb under test
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "tolerances.hpp"

namespace {

namespace ql = QuantLib;
namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace bld = swaps::build;
namespace cv = swaps::curve;
namespace json = boost::json;
namespace qconv = swaps::refbuild::conv;
namespace tol = swaps::tol;

constexpr double kNpvTol = 1e-6;     // book NPV / P&L, currency units on 1e8 gross: ~40x the measured worst
constexpr double kCurveTol = 1e-13;  // sampled DF / zero: ~25x the measured worst

// Curve roles == bundle curve indices == the verbs' integer shift_curve keys.
enum Role { ESTR = 0, E6M = 1, SOFR = 2, NROLES = 3 };
const char* const kIndex[NROLES] = {"EUR-ESTR", "EUR-EURIBOR-6M", "USD-SOFR"};
const char* const kCcy[NROLES] = {"EUR", "EUR", "USD"};

bld::Date eng_date(const ql::Date& d) {
  std::ostringstream os;
  os << ql::io::iso_date(d);
  return bld::Date::from_iso(os.str());
}

// ---- one swap, built twice on the same DB conventions -----------------------------------------------------------
struct Deal {
  int fc, disc, years;
  double notional;  // > 0 pays fixed, < 0 receives (the MultiCurveBook sign convention)
  double fixed_rate;
};

struct Built {
  cal::Instrument ins;                  // ours: build::par_swap
  ql::ext::shared_ptr<ql::Swap> qls;    // QuantLib's: OvernightIndexedSwap / VanillaSwap
};

struct Fixture {
  ql::Date today{15, ql::September, 2026};  // a Tuesday; spot 2026-09-17 on TARGET and the SOFR calendar
  bld::Date vd = bld::Date::from_iso("2026-09-15");
  std::vector<ql::RelinkableHandle<ql::YieldTermStructure>> h{NROLES};
  ql::ext::shared_ptr<ql::Estr> estr;
  ql::ext::shared_ptr<ql::Euribor6M> e6m;
  ql::ext::shared_ptr<ql::Sofr> sofr;

  cal::BundleProblem prob;
  json::value bundle_json;
  json::array positions;      // the verb's book
  std::vector<Built> book;    // the same trades, both sides, same order
  double gross = 0.0;         // Σ|notional|

  // What is linked into h right now (kept alive: the adapters hold raw curve pointers).
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> live_curves;
  std::vector<ql::ext::shared_ptr<ql::YieldTermStructure>> live_ts;
};

Built build_deal(Fixture& f, const Deal& d) {
  const bld::SwapConv conv = bld::swap_conv(kCcy[d.fc], kIndex[d.fc]);
  const ql::Calendar calq = qconv::calendar(conv.calendar);
  const ql::BusinessDayConvention bdc = qconv::bdc(conv.bdc);
  const ql::Date spot = calq.advance(f.today, conv.spot_lag, ql::Days);
  const ql::Date end = spot + ql::Period(d.years, ql::Years);  // unadjusted; the schedule adjusts it (ModF)
  const ql::Schedule fixed_s(spot, end, qconv::period(conv.fixed_freq_tok), calq, bdc, bdc,
                             ql::DateGeneration::Backward, /*endOfMonth=*/false);
  const ql::Schedule float_s(spot, end, qconv::period(conv.float_freq_tok), calq, bdc, bdc,
                             ql::DateGeneration::Backward, /*endOfMonth=*/false);
  const ql::Swap::Type type = d.notional > 0.0 ? ql::Swap::Payer : ql::Swap::Receiver;
  const double nominal = std::abs(d.notional);

  Built b;
  if (d.fc == E6M) {
    // VanillaSwap has no payment lag; the DB row says 0 (EUR-EURIBOR-6M-IRS). Fixed 30E/360 annual from the DB;
    // the float coupons forecast off QuantLib's own Euribor6M (at-par coupons, see the test preamble).
    if (conv.pay_lag != 0) throw std::runtime_error("fixture: VanillaSwap cannot express payment_lag != 0");
    b.qls = ql::ext::make_shared<ql::VanillaSwap>(type, nominal, fixed_s, d.fixed_rate, qconv::day_counter(conv.fixed_dc),
                                                  float_s, f.e6m, 0.0, qconv::day_counter(conv.float_dc));
  } else {
    // One schedule for both legs: the DB rows (EUR-ESTR-OIS, USD-SOFR-OIS) are annual/annual.
    if (conv.fixed_freq_tok != conv.float_freq_tok) throw std::runtime_error("fixture: OIS legs on different frequencies");
    const ql::ext::shared_ptr<ql::OvernightIndex> idx =
        d.fc == ESTR ? ql::ext::shared_ptr<ql::OvernightIndex>(f.estr) : ql::ext::shared_ptr<ql::OvernightIndex>(f.sofr);
    b.qls = ql::ext::make_shared<ql::OvernightIndexedSwap>(type, nominal, float_s, d.fixed_rate,
                                                           qconv::day_counter(conv.fixed_dc), idx, /*spread=*/0.0,
                                                           conv.pay_lag, bdc, calq);
  }
  b.qls->setPricingEngine(ql::ext::make_shared<ql::DiscountingSwapEngine>(f.h[static_cast<std::size_t>(d.disc)]));
  // Ours: the same maturity date QuantLib's schedule ended on; everything else from the DB row.
  b.ins = bld::par_swap(f.vd, conv, eng_date(float_s.dates().back()), d.fc, d.disc, /*market=*/0.0);
  return b;
}

// A book position from our instrument's legs, through the SAME coupon JSON shapes book_from_json reads.
json::object position_json(const cal::Instrument& ins, const Deal& d) {
  const json::object ij = api::instrument_to_json(ins).as_object();
  json::object p;
  p["kind"] = "swap";
  p["notional"] = d.notional;
  p["fixed_rate"] = d.fixed_rate;
  p["fwd_curve"] = d.fc;
  p["disc_curve"] = d.disc;
  p["fixed_curve"] = d.disc;
  p["float_coupons"] = ij.at("fwd").as_object().at("coupons");
  p["fixed_coupons"] = ij.at("fixed").as_object().at("coupons");
  return p;
}

const Deal kBook[] = {
    {ESTR, ESTR, 7, +25e6, 0.0240},   // ESTR OIS payer
    {E6M, ESTR, 10, -40e6, 0.0265},   // EURIBOR-6M IRS receiver, ESTR discounting
    {E6M, ESTR, 4, +15e6, 0.0225},    // off the calibration tenors
    {SOFR, SOFR, 6, -20e6, 0.0330},   // SOFR OIS receiver (a different currency: see SCOPE)
};

// e6m_over_estr: EURIBOR-6M as a SPREAD over ESTR (only the DISABLED pin uses it).
std::unique_ptr<Fixture> make_fixture(bool e6m_over_estr = false) {
  auto f = std::make_unique<Fixture>();
  f->estr = ql::ext::make_shared<ql::Estr>(f->h[ESTR]);
  f->e6m = ql::ext::make_shared<ql::Euribor6M>(f->h[E6M]);
  f->sofr = ql::ext::make_shared<ql::Sofr>(f->h[SOFR]);

  // Calibration instruments: par OIS / IRS at 1..15y on each curve, one knot at each final fixed payment.
  std::vector<double> knots[NROLES];
  for (int c = 0; c < NROLES; ++c)
    for (int y : {1, 2, 3, 5, 7, 10, 15}) {
      Built b = build_deal(*f, Deal{c, c == E6M ? ESTR : c, y, 1.0, 0.0});
      knots[c].push_back(b.ins.fixed.coupons.back().pay);
      f->prob.instruments.push_back(std::move(b.ins));
    }
  f->prob.curves.resize(NROLES);
  for (int c = 0; c < NROLES; ++c)
    f->prob.curves[static_cast<std::size_t>(c)] =
        px::CurveStructure{.base = -1, .currency = c == SOFR ? 1 : 0, .regions = cv::flat_hermite({}, knots[c])};
  if (e6m_over_estr) f->prob.curves[E6M].base = ESTR;
  // ESTR year-end turn: Thu 2026-12-31 -> Mon 2027-01-04 (TARGET closes Jan 1), pinned by a TurnJump row.
  f->prob.curves[ESTR].turns = {px::Turn{bld::curve_time(f->vd, bld::Date::from_iso("2026-12-31")),
                                         bld::curve_time(f->vd, bld::Date::from_iso("2027-01-04"))}};
  f->prob.instruments.push_back(bld::turn_jump(ESTR, 0, 0.0));

  // A plausible generating state; the market is our model quote there (the verb then calibrates from flat).
  const double level[NROLES] = {0.0215, e6m_over_estr ? 0.0025 : 0.0240, 0.0355};
  const double tilt[NROLES] = {3e-4, e6m_over_estr ? 1e-5 : 2.5e-4, -2e-4};
  Eigen::VectorXd x_true = Eigen::VectorXd::Zero(f->prob.n_knots());
  for (int c = 0; c < NROLES; ++c)
    for (int i = 0; i < f->prob.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i)
      x_true[f->prob.offset(c) + i] = level[c] + tilt[c] * i;
  x_true[f->prob.offset(ESTR) + f->prob.curves[ESTR].n_interp_knots()] = 0.0025;  // the turn's δ
  const Eigen::VectorXd q = f->prob.residuals<double>(x_true) + f->prob.market();
  for (int i = 0; i < f->prob.n_residuals(); ++i) f->prob.instruments[static_cast<std::size_t>(i)].market = q[i];
  f->bundle_json = api::bundle_to_json(f->prob);

  for (const Deal& d : kBook) {
    Built b = build_deal(*f, d);
    f->positions.push_back(position_json(b.ins, d));
    f->gross += std::abs(d.notional);
    f->book.push_back(std::move(b));
  }
  return f;
}

// ---- linking a curve state into QuantLib ---------------------------------------------------------------------
std::vector<std::unique_ptr<cal::CurveHandle<double>>> our_curves(const Fixture& f, const Eigen::VectorXd& x) {
  return cal::build_bundle_curves<double>(f.prob.curves, [&](int c, int i) { return x[f.prob.offset(c) + i]; });
}

ql::ext::shared_ptr<ql::YieldTermStructure> adapter(const Fixture& f, const cal::CurveHandle<double>* c) {
  // Actual365Fixed from the value date == build::curve_time ((d - vd) / 365.0): QuantLib's discount(date) is our
  // discount at the same curve time.
  auto ts = ql::ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(f.today, ql::Actual365Fixed(), c);
  ts->enableExtrapolation();
  return ts;
}

// L1: OUR curves at state x.
void link_state(Fixture& f, const Eigen::VectorXd& x) {
  auto C = our_curves(f, x);
  std::vector<ql::ext::shared_ptr<ql::YieldTermStructure>> ts;
  for (int c = 0; c < NROLES; ++c) {
    ts.push_back(adapter(f, C[static_cast<std::size_t>(c)].get()));
    f.h[static_cast<std::size_t>(c)].linkTo(ts.back());  // notifies every swap built on this handle
  }
  f.live_curves = std::move(C);
  f.live_ts = std::move(ts);
}

// L2: QuantLib's zero-spreaded view of the BASE curves (shift in rate units, continuous compounding).
void link_zero_spread(Fixture& f, const Eigen::VectorXd& x_base, const std::vector<double>& shift) {
  auto C = our_curves(f, x_base);
  std::vector<ql::ext::shared_ptr<ql::YieldTermStructure>> ts;
  for (int c = 0; c < NROLES; ++c) {
    const auto base = adapter(f, C[static_cast<std::size_t>(c)].get());
    const auto spreaded = ql::ext::make_shared<ql::ZeroSpreadedTermStructure>(
        ql::Handle<ql::YieldTermStructure>(base),
        ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(shift[static_cast<std::size_t>(c)])),
        ql::Continuous, ql::NoFrequency);
    ts.push_back(base);
    ts.push_back(spreaded);
    f.h[static_cast<std::size_t>(c)].linkTo(spreaded);
  }
  f.live_curves = std::move(C);
  f.live_ts = std::move(ts);
}

double ql_book_npv(const Fixture& f) {
  double s = 0.0;
  for (const Built& b : f.book) s += b.qls->NPV();
  return s;
}

// The move's rate shift per role, from bp written out BY HAND in the test (not by the verb's resolver).
std::vector<double> shift_of(const std::vector<double>& bp) {
  std::vector<double> s;
  for (double v : bp) s.push_back(v / 1e4);
  return s;
}

// x_base + shift on each curve's interpolation knots; the turn δ's only when `turns_too` (a negative control).
Eigen::VectorXd shifted_state(const cal::BundleProblem& p, Eigen::VectorXd x, const std::vector<double>& shift,
                              bool turns_too = false) {
  for (int c = 0; c < p.n_curves(); ++c) {
    const auto& cs = p.curves[static_cast<std::size_t>(c)];
    const int n = turns_too ? cs.n_knots() : cs.n_interp_knots();
    for (int i = 0; i < n; ++i) x[p.offset(c) + i] += shift[static_cast<std::size_t>(c)];
  }
  return x;
}

// ---- reading the response ----------------------------------------------------------------------------------------
double num(const json::value& v) { return v.to_number<double>(); }
std::vector<double> nums(const json::value& v) {
  std::vector<double> out;
  for (const auto& e : v.as_array()) out.push_back(e.to_number<double>());
  return out;
}
Eigen::VectorXd vec(const json::value& v) {
  const std::vector<double> a = nums(v);
  Eigen::VectorXd x(static_cast<Eigen::Index>(a.size()));
  for (std::size_t i = 0; i < a.size(); ++i) x[static_cast<Eigen::Index>(i)] = a[i];
  return x;
}

json::object base_body(const Fixture& f) {
  json::object body;
  body["bundle"] = f.bundle_json;  // the json::value itself: object seam, no text round trip on the way in
  body["book"] = json::object{{"positions", f.positions}};
  return body;
}

// The precondition every comparison below rests on: both constructions of each trade pay on the same dates.
void expect_same_payment_dates(const Fixture& f) {
  for (std::size_t k = 0; k < f.book.size(); ++k) {
    const Built& b = f.book[k];
    ASSERT_EQ(b.ins.fixed.coupons.size(), b.qls->leg(0).size()) << "position " << k << " fixed leg";
    ASSERT_EQ(b.ins.fwd.coupons.size(), b.qls->leg(1).size()) << "position " << k << " float leg";
    for (std::size_t i = 0; i < b.ins.fixed.coupons.size(); ++i)
      EXPECT_EQ(b.ins.fixed.coupons[i].pay, bld::curve_time(f.vd, eng_date(b.qls->leg(0)[i]->date())))
          << "position " << k << " fixed coupon " << i;
    for (std::size_t i = 0; i < b.ins.fwd.coupons.size(); ++i)
      EXPECT_EQ(b.ins.fwd.coupons[i].pay, bld::curve_time(f.vd, eng_date(b.qls->leg(1)[i]->date())))
          << "position " << k << " float coupon " << i;
  }
}

}  // namespace

// ================================================================================================================
TEST(ScenarioVerbOracle, EveryMoveMatchesQuantLibOnTheShockedCurves) {
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = ql::Date(15, ql::September, 2026);
  ql::IborCoupon::Settings::instance().createAtParCoupons();  // the build default; stated, not assumed
  const std::unique_ptr<Fixture> f = make_fixture();
  expect_same_payment_dates(*f);
  const double tol_npv = kNpvTol;

  struct Move {
    json::object request;
    std::vector<double> bp;  // per role, the `scenario` rule applied BY HAND: a key replaces the parallel
  };
  const std::vector<Move> moves = {
      {json::object{{"name", "parallel"}, {"parallel_bp", 37.5}}, {37.5, 37.5, 37.5}},
      {json::object{{"name", "override"}, {"parallel_bp", 34.0}, {"shift_curve", json::object{{"0", 14.5}}}},
       {14.5, 34.0, 34.0}},
      {json::object{{"name", "euribor"}, {"shift_curve", json::object{{"1", -21.25}}}}, {0.0, -21.25, 0.0}},
      {json::object{{"name", "sofr"}, {"shift_curve", json::object{{"2", 45.5}}}}, {0.0, 0.0, 45.5}},
      {json::object{{"name", "down_estr_up"}, {"parallel_bp", -50.0}, {"shift_curve", json::object{{"0", 5.0}}}},
       {5.0, -50.0, -50.0}},
      {json::object{{"name", "noop"}}, {0.0, 0.0, 0.0}},
  };
  const std::vector<double> times = {0.05, 0.25, 0.5, 2.5, 7.0, 12.0};  // 0.25 before the turn, 0.5 after it

  json::object body = base_body(*f);
  json::array times_json;
  for (double t : times) times_json.push_back(t);
  body["sample_times"] = times_json;
  json::array scen;
  for (const Move& m : moves) scen.push_back(m.request);
  body["scenarios"] = scen;

  const json::value resp = json::parse(api::scenario_json(json::object{{"scenario", body}}));
  ASSERT_TRUE(resp.as_object().contains("scenario")) << json::serialize(resp);
  const json::object& out = resp.as_object().at("scenario").as_object();
  const Eigen::VectorXd x_base = vec(out.at("base").as_object().at("x"));
  ASSERT_EQ(x_base.size(), f->prob.n_knots());
  const json::array& rows = out.at("scenarios").as_array();
  ASSERT_EQ(rows.size(), moves.size());

  // Base: the verb's book value at its own calibrated state vs QuantLib's swaps on the same curves.
  link_state(*f, x_base);
  const double ql_base = ql_book_npv(*f);
  EXPECT_NEAR(num(out.at("base").as_object().at("npv")), ql_base, tol_npv);

  double worst_npv = 0.0, worst_df = 0.0, worst_zero = 0.0;
  std::vector<double> verb_npv(moves.size());
  for (std::size_t k = 0; k < moves.size(); ++k) {
    const json::object& row = rows[k].as_object();
    const std::vector<double> s = shift_of(moves[k].bp);
    const double npv = num(row.at("npv")), npv_delta = num(row.at("npv_delta"));
    verb_npv[k] = npv;

    // L1: the same shocked curve, ours, priced by QuantLib.
    link_state(*f, shifted_state(f->prob, x_base, s));
    const double ql_l1 = ql_book_npv(*f);
    EXPECT_NEAR(npv, ql_l1, tol_npv) << "L1 row " << k;
    EXPECT_NEAR(npv_delta, ql_l1 - ql_base, tol_npv) << "L1 row " << k;

    // L2: QuantLib applies the shock itself.
    link_zero_spread(*f, x_base, s);
    const double ql_l2 = ql_book_npv(*f);
    EXPECT_NEAR(npv, ql_l2, tol_npv) << "L2 row " << k;
    EXPECT_NEAR(npv_delta, ql_l2 - ql_base, tol_npv) << "L2 row " << k;
    worst_npv = std::max({worst_npv, std::abs(npv - ql_l1), std::abs(npv - ql_l2)});

    // The sampled shocked curves vs QuantLib's zero-spreaded ones (still linked).
    const json::array& curves = row.at("curves").as_array();
    ASSERT_EQ(curves.size(), static_cast<std::size_t>(NROLES));
    for (int c = 0; c < NROLES; ++c) {
      const json::object& co = curves[static_cast<std::size_t>(c)].as_object();
      const std::vector<double> t = nums(co.at("t")), df = nums(co.at("discount")), z = nums(co.at("zero"));
      ASSERT_EQ(t.size(), times.size());
      for (std::size_t i = 0; i < t.size(); ++i) {
        const double ql_df = f->h[static_cast<std::size_t>(c)]->discount(t[i]);
        const double ql_z = f->h[static_cast<std::size_t>(c)]->zeroRate(t[i], ql::Continuous, ql::NoFrequency, true).rate();
        EXPECT_NEAR(df[i], ql_df, kCurveTol) << "row " << k << " curve " << c << " t=" << t[i];
        EXPECT_NEAR(z[i], ql_z, kCurveTol) << "row " << k << " curve " << c << " t=" << t[i];
        worst_df = std::max(worst_df, std::abs(df[i] - ql_df));
        worst_zero = std::max(worst_zero, std::abs(z[i] - ql_z));
      }
    }
    // Liveness: every real move moves this book far outside the tolerance, so the rows compare non-trivial numbers.
    if (k + 1 < moves.size()) EXPECT_GT(std::abs(npv_delta), 1e3 * tol_npv) << "row " << k << " is dead";
  }
  std::cout << "  [scenario] worst |verb - QuantLib|: npv " << worst_npv << " (tol " << tol_npv << "), df "
            << worst_df << ", zero " << worst_zero << "\n";

  // NEGATIVE CONTROLS -- the comparison has teeth.
  // 1. SC1 against QuantLib: the override move priced with the ADD rule (ESTR +48.5 instead of +14.5) disagrees
  //    by ~34 bp of ESTR PV01 (order 1e5 here).
  link_zero_spread(*f, x_base, shift_of({48.5, 34.0, 34.0}));
  const double gap_rule = std::abs(verb_npv[1] - ql_book_npv(*f));
  EXPECT_GT(gap_rule, 1e3 * tol_npv) << "the override row must not price like the add rule";
  // 2. The sign of a move: +37.5 bp priced by QuantLib as −37.5 bp.
  link_zero_spread(*f, x_base, shift_of({-37.5, -37.5, -37.5}));
  const double gap_sign = std::abs(verb_npv[0] - ql_book_npv(*f));
  EXPECT_GT(gap_sign, 1e3 * tol_npv);
  // 3. Turns untouched: shifting the ESTR turn δ as well multiplies every ESTR DF past the turn by
  //    exp(−37.5e-4 · 4/365) ≈ 1 − 4.1e-5 -- visible in the book (~1e3) and in the sampled DF at t = 0.5, while the
  //    DF at t = 0.25 (before the window) still agrees.
  link_state(*f, shifted_state(f->prob, x_base, shift_of({37.5, 37.5, 37.5}), /*turns_too=*/true));
  const double gap_turn = std::abs(verb_npv[0] - ql_book_npv(*f));
  EXPECT_GT(gap_turn, 1e3 * tol_npv);
  const std::vector<double> df0 = nums(rows[0].as_object().at("curves").as_array()[ESTR].as_object().at("discount"));
  EXPECT_GT(std::abs(df0[2] - f->h[ESTR]->discount(times[2])), 1e3 * kCurveTol) << "t=0.5, after the turn";
  EXPECT_NEAR(df0[1], f->h[ESTR]->discount(times[1]), kCurveTol) << "t=0.25, before the turn";
  std::cout << "  [scenario] negative-control gaps: add-rule " << gap_rule << ", sign " << gap_sign << ", turn "
            << gap_turn << "\n";
}

// ================================================================================================================
TEST(ScenarioGridVerbOracle, EveryCellOfAParallelByCurveSurfaceMatchesQuantLib) {
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = ql::Date(15, ql::September, 2026);
  ql::IborCoupon::Settings::instance().createAtParCoupons();
  const std::unique_ptr<Fixture> f = make_fixture();
  expect_same_payment_dates(*f);
  const double tol_npv = kNpvTol;

  const std::vector<double> par = {-36.0, 0.0, 34.5};      // axis 0: parallel_bp
  const std::vector<double> e6m = {-31.75, 0.0, 36.75};    // axis 1: shift_curve role 1 (EURIBOR-6M)
  json::array par_json, e6m_json;
  for (double v : par) par_json.push_back(v);
  for (double v : e6m) e6m_json.push_back(v);
  json::object body = base_body(*f);
  body["axes"] = json::array{
      json::object{{"label", "parallel"}, {"kind", "parallel_bp"}, {"values", par_json}},
      json::object{{"label", "euribor"}, {"kind", "shift_curve"}, {"role", static_cast<int>(E6M)}, {"values", e6m_json}}};

  const json::value resp = json::parse(api::scenario_grid_json(json::object{{"scenario_grid", body}}));
  ASSERT_TRUE(resp.as_object().contains("scenario_grid")) << json::serialize(resp);
  const json::object& out = resp.as_object().at("scenario_grid").as_object();
  ASSERT_EQ(nums(out.at("shape")), (std::vector<double>{3.0, 3.0}));
  const Eigen::VectorXd x_base = vec(out.at("base").as_object().at("x"));
  ASSERT_EQ(x_base.size(), f->prob.n_knots());
  const json::array& npv = out.at("npv").as_array();
  const json::array& pnl = out.at("pnl").as_array();
  const auto cell = [](const json::array& a, std::size_t i, std::size_t j) { return num(a.at(i).as_array().at(j)); };

  link_state(*f, x_base);
  const double ql_base = ql_book_npv(*f);
  EXPECT_NEAR(num(out.at("base").as_object().at("npv")), ql_base, tol_npv);

  // The GRID rule applied by hand: every axis ADDS to the curves it moves.
  double worst = 0.0;
  for (std::size_t i = 0; i < par.size(); ++i)
    for (std::size_t j = 0; j < e6m.size(); ++j) {
      const std::vector<double> s = shift_of({par[i], par[i] + e6m[j], par[i]});
      link_state(*f, shifted_state(f->prob, x_base, s));
      const double ql_l1 = ql_book_npv(*f);
      link_zero_spread(*f, x_base, s);
      const double ql_l2 = ql_book_npv(*f);
      EXPECT_NEAR(cell(npv, i, j), ql_l1, tol_npv) << "L1 cell " << i << "," << j;
      EXPECT_NEAR(cell(npv, i, j), ql_l2, tol_npv) << "L2 cell " << i << "," << j;
      EXPECT_NEAR(cell(pnl, i, j), ql_l2 - ql_base, tol_npv) << "cell " << i << "," << j;
      worst = std::max({worst, std::abs(cell(npv, i, j) - ql_l1), std::abs(cell(npv, i, j) - ql_l2)});
    }
  std::cout << "  [scenario_grid] worst |verb - QuantLib| over 9 cells: " << worst << " (tol " << tol_npv << ")\n";

  // NEGATIVE CONTROLS. With a non-zero parallel, pricing the curve axis by the `scenario` OVERRIDE rule (EURIBOR
  // takes the curve value only) must disagree -- by ~36 bp of EURIBOR forecast PV01 at cell (0,0) ...
  link_zero_spread(*f, x_base, shift_of({par[0], e6m[0], par[0]}));
  const double gap_rule = std::abs(cell(npv, 0, 0) - ql_book_npv(*f));
  EXPECT_GT(gap_rule, 1e3 * tol_npv) << "a grid cell must not price like the override rule";
  // ... while on the zero-parallel row both rules coincide, so the control isolates exactly the add-vs-override rule.
  link_zero_spread(*f, x_base, shift_of({0.0, e6m[2], 0.0}));
  EXPECT_NEAR(cell(npv, 1, 2), ql_book_npv(*f), tol_npv);
  // A cell priced with the axes swapped onto the wrong curve (the curve shift on ESTR instead of EURIBOR) disagrees.
  link_zero_spread(*f, x_base, shift_of({e6m[2], 0.0, 0.0}));
  const double gap_role = std::abs(cell(npv, 1, 2) - ql_book_npv(*f));
  EXPECT_GT(gap_role, 1e3 * tol_npv);
  std::cout << "  [scenario_grid] negative-control gaps: override-rule " << gap_rule << ", wrong role " << gap_role << "\n";
}

// ================================================================================================================
// A PARALLEL move on a bundle whose projection curve is a SPREAD over its discount curve moves every curve a trade sees
// by the move ONCE, as QuantLib reads "+25 bp on every curve". Found while drafting this file: the fork gave the spread
// curve's knots the move as well, so EURIBOR-6M moved +50 bp (tests/scenario_spread_repro_test.cpp; fixed in b5f3a4d --
// a parallel move reaches outright curves' knots only, a spread curve inherits it from its base). This is how the web
// composer builds FF-over-SOFR and EURIBOR-over-ESTR bundles.
TEST(ScenarioVerbOracle, ParallelMoveOnASpreadCurveMovesItsForwardOnceLikeQuantLib) {
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = ql::Date(15, ql::September, 2026);
  ql::IborCoupon::Settings::instance().createAtParCoupons();
  const std::unique_ptr<Fixture> f = make_fixture(/*e6m_over_estr=*/true);
  const double tol_npv = kNpvTol;

  json::object body = base_body(*f);
  body["scenarios"] = json::array{json::object{{"name", "parallel"}, {"parallel_bp", 25.0}}};
  const json::value resp = json::parse(api::scenario_json(json::object{{"scenario", body}}));
  const json::object& out = resp.as_object().at("scenario").as_object();
  const Eigen::VectorXd x_base = vec(out.at("base").as_object().at("x"));

  // QuantLib: +25 bp on every curve AS A TRADE SEES IT (the composite EURIBOR curve included).
  link_zero_spread(*f, x_base, shift_of({25.0, 25.0, 25.0}));
  EXPECT_NEAR(num(out.at("scenarios").as_array()[0].as_object().at("npv")), ql_book_npv(*f), tol_npv);
}
