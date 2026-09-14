// @oracle-test — the bond RV verbs ("bond_universe", "govvie_fit", "swap_spread"; api/rv.cpp) driven end to end
// through run_json, every number compared with a QuantLib number. DO NOT DELETE OR WEAKEN without reproducing it.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// See tests/ORACLE_TESTS.md before changing anything here.
//
// WHAT THIS PINS THAT NOTHING ELSE DID (draft 2026-09-14). tests/bond_oracle_test.cpp pins the bond KERNELS on bonds
// the FIXTURE built; tests/derive_rv_test.cpp pins the RV library against its own pre-lift pipeline (T3 parity). No
// oracle named any of the three RV verbs, so the request -> conventions row -> settlement roll -> BondId build ->
// batched sweep / fit / swap-spread rows -> JSON path was unchecked against an independent number.
//
// PER VERB (tests/ORACLE_TESTS.md row + NOTES.md carry the reasoning):
//   * bond_universe — accrued, yield <-> clean, modified duration, convexity vs BondFunctions on QuantLib's own
//     FixedRateBond, settlement rolled by QuantLib on the DB row's calendar (T+1 over Veterans Day), per-bond
//     compounding chosen from the DB row (street: Compounded, SimpleThenCompounded in the final period; Treasury
//     method: SimpleThenCompounded). Seasoned, final-period and when-issued bonds.
//   * govvie_fit, parametric — the OBJECTIVES DO NOT MATCH QuantLib's FittedBondDiscountCurve (QuantLib frees the
//     decay(s) κ, defaults to 1/duration weights and a Simplex; see NOTES.md). So QuantLib is run as an EVALUATOR at
//     the ENGINE's fitted parameters (maxEvaluations = 0, guess = [β..., 1/τ...]): QuantLib's own NS / Svensson
//     discount function, its own BondHelper repricing and its own FittingCost define the residuals and the cost;
//     and the engine's β are checked to be a STATIONARY POINT of QuantLib's cost with κ held fixed.
//   * govvie_fit, spline — no QuantLib counterpart (CubicBSplinesFitting is a B-spline basis on the DISCOUNT
//     function; the engine's is flat+Hermite on knot forwards). QuantLib reprices the universe off the engine's
//     fitted curve through qlx::CurveTermStructure: residuals, and z-spreads vs ZeroSpreadedTermStructure and
//     BondFunctions::zSpread.
//   * swap_spread (HEADLINE ONLY) — the benchmark street yield vs BondFunctions::yield; the anchor vs QuantLib's 5Y
//     OIS maturity; the emitted ASW row's swap vs MakeOIS::fairRate on the same curve; and the composed rows (factor
//     pinned) quote QuantLib's fairRate − QuantLib's yield. `matched_maturity` is EXCLUDED: it still returns the
//     tenor-swap numbers (TASKS-ENGINE E7 3.4 open gap) and must not be pinned as correct.
//
// SCOPE, stated rather than implied: US-TREASURY / US-TREASURY-TSY only (the only bond rows in the DB); unadjusted
// coupon dates (build::fixed_rate_bond's choice, pinned in bond_oracle_test); single-curve swap; no ex-coupon.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>
#include <ql/termstructures/yield/fittedbonddiscountcurve.hpp>
#include <ql/termstructures/yield/nonlinearfittingmethods.hpp>

#include <boost/json.hpp>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "conventions_ql.hpp"
#include "swaps/api/bundle_api.hpp"  // run_json
#include "swaps/api/codec.hpp"       // instrument_from_json (decode the emitted swap-spread rows)
#include "swaps/api/rv.hpp"          // the verbs under test -- named so the include-closure coverage tool sees them
#include "swaps/build/conventions.hpp"
#include "swaps/build/swap_spread.hpp"  // govvie_factor_regions
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"  // instrument_model_quote
#include "swaps/conventions_data.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/parametric.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "tolerances.hpp"

namespace {

namespace ql = QuantLib;
namespace api = swaps::api;
namespace bld = swaps::build;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace cvd = swaps::conventions;
namespace json = boost::json;
namespace qconv = swaps::refbuild::conv;
namespace tol = swaps::tol;

// Tuesday. 2026-11-11 (Veterans Day) closes the US bond market and SOFR, so T+1 bond settlement is 2026-11-12 and
// a NullCalendar roll (11-11) would be wrong by a day: the DB calendar is live in every test below.
const char* const kVd = "2026-11-10";

ql::Date qd(const std::string& iso) {
  const bld::Date d = bld::Date::from_iso(iso);
  return ql::Date(ql::Day(d.day()), ql::Month(d.month()), ql::Year(d.year()));
}

::testing::AssertionResult close(double got, double want, double rel) {
  const double err = std::abs(got - want) / std::max(1.0, std::abs(want));
  return err <= rel ? ::testing::AssertionSuccess()
                    : (::testing::AssertionFailure() << "got " << got << " want " << want << " rel " << err);
}

// ---- the request side -------------------------------------------------------------------------------------------
struct BondSpec {
  const char* id;
  const char* issue;  // dated date
  const char* maturity;
  double coupon;
  const char* first_coupon;  // nullptr => seasoned; set => when-issued (short first coupon)
};

json::object bond_json(const BondSpec& b) {
  json::object o;
  o["id"] = b.id;
  o["issue"] = b.issue;
  o["maturity"] = b.maturity;
  o["coupon"] = b.coupon;
  if (b.first_coupon) o["first_coupon"] = b.first_coupon;
  return o;
}
json::array bonds_json(const std::vector<BondSpec>& u) {
  json::array a;
  for (const BondSpec& b : u) a.push_back(bond_json(b));
  return a;
}
json::array arr(const std::vector<double>& v) {
  json::array a;
  for (double d : v) a.emplace_back(d);
  return a;
}

// run_json on {verb: body}; unwraps a {verb: {...}} response envelope if present (an error stays visible either way).
json::object run_verb(const char* verb, json::object body) {
  json::object req;
  req[verb] = std::move(body);
  const json::value v = json::parse(api::run_json(req));
  const json::object& o = v.as_object();
  if (o.contains(verb) && o.at(verb).is_object()) return o.at(verb).as_object();
  return o;
}
double num(const json::object& o, const char* key, std::size_t i) {
  return o.at(key).as_array().at(i).to_number<double>();
}
double num(const json::object& o, const char* key) { return o.at(key).to_number<double>(); }

// ---- QuantLib's side of one bond --------------------------------------------------------------------------------
// FixedRateBond on the DB row's frequency, ACT/ACT ISMA on QuantLib's own Backward schedule (firstDate for a WI bond),
// coupons UNADJUSTED, settlement = QuantLib's roll of the DB row's settle_lag on the DB row's calendar (paymentCalendar
// is the settlement calendar of Bond; Unadjusted payments are unaffected by it). Nothing here reads the engine's dates.
struct QlBond {
  ql::Schedule sched;
  ql::DayCounter dc;
  ql::ext::shared_ptr<ql::FixedRateBond> bond;
  ql::Frequency freq;
};

QlBond ql_bond(const std::string& conv_id, const BondSpec& b) {
  const cvd::BondConv row = cvd::require_bond(conv_id);
  const ql::Period tenor = qconv::period(row.frequency);
  const ql::Date first = b.first_coupon ? qd(b.first_coupon) : ql::Date();
  ql::Schedule s(qd(b.issue), qd(b.maturity), tenor, ql::NullCalendar(), ql::Unadjusted, ql::Unadjusted,
                 ql::DateGeneration::Backward, /*endOfMonth=*/false, first);
  ql::DayCounter dc = ql::ActualActual(ql::ActualActual::ISMA, s);
  auto bond = ql::ext::make_shared<ql::FixedRateBond>(static_cast<ql::Natural>(row.settle_lag), 100.0, s,
                                                      std::vector<ql::Rate>{b.coupon}, dc, ql::Unadjusted, 100.0,
                                                      qd(b.issue), qconv::calendar(row.calendar));
  return {s, dc, bond, tenor.frequency()};
}

// The DB row's yield convention in QuantLib's Compounding enum (bond_oracle_test.cpp pins this mapping):
//   stub_discount "simple"                      -> SimpleThenCompounded (Treasury method, every stub simple)
//   final_period_simple && one coupon remaining -> SimpleThenCompounded (street, final period)
//   otherwise                                   -> Compounded
ql::Compounding row_compounding(const std::string& conv_id, const QlBond& q, const ql::Date& settle) {
  const cvd::BondConv row = cvd::require_bond(conv_id);
  if (row.stub_discount == "simple") return ql::SimpleThenCompounded;
  if (row.final_period_simple) {
    int coupons_left = 0;
    for (const auto& cf : q.bond->cashflows())
      if (cf->date() > settle && ql::ext::dynamic_pointer_cast<ql::Coupon>(cf)) ++coupons_left;
    if (coupons_left == 1) return ql::SimpleThenCompounded;
  }
  return ql::Compounded;
}

// Modified duration / convexity at the ENGINE's yield (no yield-solve noise enters). Under Compounded QuantLib's
// analytic derivatives are the derivatives of its own price. Under SimpleThenCompounded they are NOT for a multi-flow
// bond (bond_oracle_test.cpp TreasuryMethodMatchesSimpleThenCompounded measured 1.4e-4), so there the reference is a
// central difference of QuantLib's own dirty price, with that test's step and tolerances.
void expect_risk_matches_ql(const QlBond& q, double y, ql::Compounding comp, const ql::Date& settle, double md_eng,
                            double cx_eng, const std::string& what) {
  if (comp == ql::Compounded) {
    const double md_ql = ql::BondFunctions::duration(*q.bond, y, q.dc, comp, q.freq, ql::Duration::Modified, settle);
    const double cx_ql = ql::BondFunctions::convexity(*q.bond, y, q.dc, comp, q.freq, settle);
    EXPECT_TRUE(close(md_eng, md_ql, tol::curve_rel)) << "mod-dur " << what;
    EXPECT_TRUE(close(cx_eng, cx_ql, tol::curve_rel)) << "convexity " << what;
  } else {
    const double h = 1e-5;  // bond_oracle_test.cpp's step
    const auto P = [&](double yy) { return ql::BondFunctions::dirtyPrice(*q.bond, yy, q.dc, comp, q.freq, settle); };
    const double p0 = P(y), pu = P(y + h), pd = P(y - h);
    EXPECT_TRUE(close(md_eng, -(pu - pd) / (2.0 * h) / p0, 1e-8)) << "mod-dur (FD of QuantLib price) " << what;
    EXPECT_TRUE(close(cx_eng, (pu - 2.0 * p0 + pd) / (h * h) / p0, 1e-5)) << "convexity (FD of QuantLib price) " << what;
  }
}

// A street universe that exercises every bond path the verb has: a bond settling 3 days before a coupon, a deep
// discount long bond, a mid-period bond, a bond in its FINAL coupon period, and a WHEN-ISSUED bond (short first coupon).
std::vector<BondSpec> street_universe() {
  return {{"A", "2021-11-15", "2031-11-15", 0.0425, nullptr},
          {"LONG", "2016-08-15", "2046-08-15", 0.025, nullptr},
          {"MID", "2025-05-15", "2035-05-15", 0.04375, nullptr},
          {"FINAL", "2024-02-15", "2027-02-15", 0.04, nullptr},
          {"WI", "2026-10-15", "2031-08-15", 0.04125, "2027-02-15"}};
}

// ================================================================================================================
// bond_universe
// ================================================================================================================
TEST(BondRvVerbOracle, BondUniverseStreetCleanQuotesMatchBondFunctions) {
  const std::string conv = "US-TREASURY";
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = qd(kVd);

  const std::vector<BondSpec> u = street_universe();
  const std::vector<double> clean = {1.0012, 0.7431, 1.0147, 0.99985, 0.9968};
  json::object body;
  body["value_date"] = kVd;
  body["convention"] = conv;
  body["bonds"] = bonds_json(u);
  body["clean"] = arr(clean);
  const json::object out = run_verb("bond_universe", body);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_EQ(out.at("n").to_number<int>(), static_cast<int>(u.size()));

  bool saw_final_period = false;
  for (std::size_t i = 0; i < u.size(); ++i) {
    const QlBond q = ql_bond(conv, u[i]);
    const ql::Date settle = q.bond->settlementDate();
    ASSERT_EQ(settle, ql::Date(12, ql::November, 2026)) << "fixture premise: T+1 over Veterans Day";
    const ql::Compounding comp = row_compounding(conv, q, settle);
    saw_final_period = saw_final_period || (comp == ql::SimpleThenCompounded);
    const double y = num(out, "yield", i);

    EXPECT_TRUE(close(num(out, "accrued", i) * 100.0, q.bond->accruedAmount(settle), tol::curve_rel)) << u[i].id;
    // yield: forward (QuantLib's clean at the verb's yield reproduces the quote -- no solver in the reference) ...
    EXPECT_TRUE(close(ql::BondFunctions::cleanPrice(*q.bond, y, q.dc, comp, q.freq, settle), clean[i] * 100.0,
                      tol::curve_rel))
        << "clean at verb yield " << u[i].id;
    // ... and inverse (QuantLib's own yield solve, accuracy far below the tolerance).
    const double y_ql = ql::BondFunctions::yield(*q.bond, ql::Bond::Price(clean[i] * 100.0, ql::Bond::Price::Clean),
                                                 q.dc, comp, q.freq, settle, /*accuracy=*/1e-14);
    EXPECT_TRUE(close(y, y_ql, tol::curve_rel)) << "yield " << u[i].id;
    expect_risk_matches_ql(q, y, comp, settle, num(out, "modified_duration", i), num(out, "convexity", i), u[i].id);
  }
  ASSERT_TRUE(saw_final_period) << "fixture premise: FINAL is in its last coupon period";

  // Negative controls -- the comparison has teeth:
  //   the settlement CALENDAR is live (a NullCalendar T+1 accrues one day less: ~0.0115 per 100) ...
  const QlBond a = ql_bond(conv, u[0]);
  EXPECT_GT(std::abs(num(out, "accrued", 0) * 100.0 - a.bond->accruedAmount(qd(kVd) + 1)), 1e-4);
  //   ... and the FINAL-PERIOD rule is live (plain Compounded misprices the final-period bond by ~5e-3 per 100).
  const QlBond f = ql_bond(conv, u[3]);
  const ql::Date fs = f.bond->settlementDate();
  EXPECT_GT(std::abs(ql::BondFunctions::cleanPrice(*f.bond, num(out, "yield", 3), f.dc, ql::Compounded, f.freq, fs) -
                     clean[3] * 100.0),
            1e-4);
}

TEST(BondRvVerbOracle, BondUniverseTreasuryMethodYieldQuotesWithExplicitSettleMatchBondFunctions) {
  const std::string conv = "US-TREASURY-TSY";
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = qd(kVd);

  const std::vector<BondSpec> u = street_universe();
  const std::vector<double> yield = {0.0421, 0.0467, 0.0413, 0.0391, 0.04185};
  const std::string settle_iso = "2026-11-16";  // explicit override (a Monday), not the convention's T+1
  json::object body;
  body["value_date"] = kVd;
  body["convention"] = conv;
  body["settle"] = settle_iso;
  body["bonds"] = bonds_json(u);
  body["yield"] = arr(yield);
  const json::object out = run_verb("bond_universe", body);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_EQ(out.at("n").to_number<int>(), static_cast<int>(u.size()));

  const ql::Date settle = qd(settle_iso);
  for (std::size_t i = 0; i < u.size(); ++i) {
    const QlBond q = ql_bond(conv, u[i]);
    const ql::Compounding comp = row_compounding(conv, q, settle);
    ASSERT_EQ(comp, ql::SimpleThenCompounded) << "the Treasury method discounts every stub simple";
    EXPECT_TRUE(close(num(out, "accrued", i) * 100.0, q.bond->accruedAmount(settle), tol::curve_rel)) << u[i].id;
    EXPECT_TRUE(close(num(out, "clean", i) * 100.0,
                      ql::BondFunctions::cleanPrice(*q.bond, yield[i], q.dc, comp, q.freq, settle), tol::curve_rel))
        << "clean from yield " << u[i].id;
    expect_risk_matches_ql(q, yield[i], comp, settle, num(out, "modified_duration", i), num(out, "convexity", i),
                           u[i].id);
  }

  // Negative controls: the explicit settle is live (the convention's T+1 accrues 4 days less) and the TREASURY-METHOD
  // stub is live (the street Compounded price of a seasoned bond differs by ~0.7 bp of price).
  const QlBond m = ql_bond(conv, u[2]);
  EXPECT_GT(std::abs(num(out, "accrued", 2) * 100.0 - m.bond->accruedAmount(m.bond->settlementDate())), 1e-3);
  EXPECT_GT(std::abs(num(out, "clean", 2) * 100.0 -
                     ql::BondFunctions::cleanPrice(*m.bond, yield[2], m.dc, ql::Compounded, m.freq, settle)),
            1e-4);
}

// ================================================================================================================
// govvie_fit
// ================================================================================================================
// A seasoned universe with deliberately NOISY clean prices (no curve in either model family reprices it exactly),
// so the fit has a non-zero residual vector and first-order optimality is a real statement.
std::vector<BondSpec> fit_universe() {
  return {{"F28", "2018-08-15", "2028-08-15", 0.035, nullptr},   {"F29", "2019-11-15", "2029-11-15", 0.04125, nullptr},
          {"F30", "2020-05-15", "2030-05-15", 0.0375, nullptr},  {"F32", "2022-02-15", "2032-02-15", 0.045, nullptr},
          {"F34", "2024-08-15", "2034-08-15", 0.03875, nullptr}, {"F36", "2026-05-15", "2036-05-15", 0.0425, nullptr},
          {"F41", "2021-11-15", "2041-11-15", 0.0375, nullptr},  {"F46", "2016-08-15", "2046-08-15", 0.0425, nullptr}};
}
const std::vector<double> kFitClean = {0.9912, 1.0031, 0.9875, 1.0104, 0.9718, 0.9893, 0.9420, 0.9605};
const std::vector<double> kFitWeight = {1.0, 0.9, 1.1, 1.0, 0.8, 1.2, 0.7, 1.0};  // explicit: QuantLib gets the same

struct QlFitEval {
  ql::Array solution;
  double cost = 0.0;              // QuantLib's FittingCost: Σ (w_b · (implied_b − quote_b))², prices per 100
  std::vector<double> implied;    // QuantLib's BondHelper::impliedQuote per bond (clean, per 100)
  ql::ext::shared_ptr<ql::FittedBondDiscountCurve> curve;
};

// QuantLib's FittedBondDiscountCurve as an EVALUATOR at `params` (fittedbonddiscountcurve.cpp: maxEvaluations == 0 =>
// solution = guess, cost = FittingCost::value(guess), no optimisation). Weights are passed explicitly, so QuantLib's
// 1/duration default weighting is not used.
QlFitEval ql_fit_at(bool svensson, const ql::Array& params) {
  const std::string conv = "US-TREASURY";
  const std::vector<BondSpec> u = fit_universe();
  std::vector<ql::ext::shared_ptr<ql::BondHelper>> helpers;
  for (std::size_t b = 0; b < u.size(); ++b) {
    const QlBond q = ql_bond(conv, u[b]);
    helpers.push_back(ql::ext::make_shared<ql::BondHelper>(
        ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(kFitClean[b] * 100.0)), q.bond));
  }
  ql::Array w(u.size());
  for (std::size_t b = 0; b < u.size(); ++b) w[b] = kFitWeight[b];
  const ql::Size kEvaluateOnly = 0;
  QlFitEval e;
  if (svensson)
    e.curve = ql::ext::make_shared<ql::FittedBondDiscountCurve>(qd(kVd), helpers, ql::Actual365Fixed(),
                                                                ql::SvenssonFitting(w), 1.0e-10, kEvaluateOnly, params);
  else
    e.curve = ql::ext::make_shared<ql::FittedBondDiscountCurve>(qd(kVd), helpers, ql::Actual365Fixed(),
                                                                ql::NelsonSiegelFitting(w), 1.0e-10, kEvaluateOnly,
                                                                params);
  const ql::FittedBondDiscountCurve::FittingMethod& fr = e.curve->fitResults();
  e.solution = fr.solution();
  e.cost = fr.minimumCostValue();
  for (const auto& h : helpers) e.implied.push_back(h->impliedQuote());
  return e;
}

struct ParametricCase {
  const char* model;
  bool svensson;
  double tau1, tau2;  // fixed decay hyperparameters, given explicitly so no default is restated
};

TEST(BondRvVerbOracle, GovvieFitParametricMatchesFittedBondDiscountCurveAtTheEngineParameters) {
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = qd(kVd);
  const std::vector<BondSpec> u = fit_universe();
  const std::size_t n = u.size();

  for (const ParametricCase& pc : {ParametricCase{"nelson_siegel", false, 1.7, 0.0},
                                   ParametricCase{"svensson", true, 1.7, 6.0}}) {
    SCOPED_TRACE(pc.model);
    json::object body;
    body["value_date"] = kVd;
    body["convention"] = "US-TREASURY";
    body["bonds"] = bonds_json(u);
    body["clean"] = arr(kFitClean);
    body["weight"] = arr(kFitWeight);
    body["model"] = pc.model;
    body["tau1"] = pc.tau1;
    if (pc.svensson) body["tau2"] = pc.tau2;
    const json::object out = run_verb("govvie_fit", body);
    ASSERT_FALSE(out.contains("error")) << json::serialize(out);
    ASSERT_EQ(out.at("n").to_number<int>(), static_cast<int>(n));

    // The engine's betas, and QuantLib's parameter vector at them. Same model family, re-parameterised:
    //   engine   y(t) = β0 + β1·L(τ1) + β2·(L(τ1) − e^{−t/τ1}) [+ β3·(L(τ2) − e^{−t/τ2})],  L(τ) = (1 − e^{−t/τ})/(t/τ)
    //   QuantLib r(t) = c0 + (c1 + c2)·(1 − e^{−κt})/(κt) − c2·e^{−κt} [+ c3·((1 − e^{−κ1 t})/(κ1 t) − e^{−κ1 t})]
    // => c_i = β_i, κ = 1/τ1, κ1 = 1/τ2 (nonlinearfittingmethods.cpp NelsonSiegelFitting/SvenssonFitting).
    const json::array& xj = out.at("x").as_array();
    const std::size_t nb = pc.svensson ? 4u : 3u;
    ASSERT_EQ(xj.size(), nb);
    ql::Array params(nb + (pc.svensson ? 2u : 1u));
    Eigen::VectorXd beta(static_cast<int>(nb));
    for (std::size_t i = 0; i < nb; ++i) params[i] = beta[static_cast<int>(i)] = xj.at(i).to_number<double>();
    params[nb] = 1.0 / pc.tau1;
    if (pc.svensson) params[nb + 1] = 1.0 / pc.tau2;

    const QlFitEval e = ql_fit_at(pc.svensson, params);
    ASSERT_EQ(e.solution.size(), params.size());
    for (std::size_t i = 0; i < params.size(); ++i)
      ASSERT_EQ(e.solution[i], params[i]) << "QuantLib must EVALUATE at the engine's parameters, not re-fit";

    // (1) The model discount function: the engine's curve/parametric.hpp vs QuantLib's formula at the same parameters.
    //     QuantLib adds QL_EPSILON to κ and t in the loading denominator: a relative DF effect < 1e-13 for t >= 0.25.
    const ql::FittedBondDiscountCurve::FittingMethod& fr = e.curve->fitResults();
    for (double t : {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0}) {
      double df_eng;
      if (pc.svensson) {
        cv::Svensson<double> m(pc.tau1, pc.tau2);
        m.set_params(beta);
        df_eng = m.discount(t);
      } else {
        cv::NelsonSiegel<double> m(pc.tau1);
        m.set_params(beta);
        df_eng = m.discount(t);
      }
      EXPECT_TRUE(close(df_eng, fr.discount(e.solution, t), tol::curve_rel)) << "DF t=" << t;
    }

    // (2) The RV ladder: residual_b = w_b · (model clean − market clean), per unit notional. QuantLib's model clean is
    //     BondHelper::impliedQuote -- its own FixedRateBond repriced by DiscountingBondEngine off the fitted curve.
    double sum_abs_r = 0.0, sum_sq_r = 0.0;
    std::vector<double> r_ql(n);
    for (std::size_t b = 0; b < n; ++b) {
      r_ql[b] = kFitWeight[b] * (e.implied[b] / 100.0 - kFitClean[b]);
      const double r_eng = num(out, "residuals", b);
      EXPECT_TRUE(close(r_eng, r_ql[b], tol::curve_rel)) << "residual " << u[b].id;
      sum_abs_r += std::abs(r_eng);
      sum_sq_r += r_eng * r_eng;
    }

    // (3) QuantLib's objective value at the engine's parameters. QuantLib prices per 100, so C_ql = 1e4 · Σ r². Bound
    //     DERIVED from (2): each r agrees to δ = curve_rel, so |ΔC| <= 1e4 · Σ (2|r|δ + δ²).
    const double d = tol::curve_rel;
    EXPECT_NEAR(e.cost, 1e4 * sum_sq_r, 1e4 * (2.0 * sum_abs_r * d + static_cast<double>(n) * d * d));

    // (4) The engine's betas are a STATIONARY POINT of QuantLib's objective with κ held (the engine's problem is
    //     QuantLib's cost restricted to fixed decays -- same residual definition, same weights, price units scale only).
    //     First-order optimality JᵀWr = 0 is checked as a cosine: |Σ_b J_bi r_b| <= c · ||J_i|| · ||r||, J by a central
    //     difference of QuantLib's implied quotes. c = 1e-6: FD round-off on a per-100 quote is ~1e-12/h = 1e-6 against
    //     |J| ~ 1e2..1e3 (~1e-9 relative); a converged Gauss-Newton on this near-linear problem is far below 1e-6. A
    //     failure here is a finding about the engine's LM stopping rule, NOT a reason to raise c. (Unmeasured.)
    const double h = 1e-6;
    double r_norm2 = 0.0;
    for (std::size_t b = 0; b < n; ++b) r_norm2 += (100.0 * r_ql[b]) * (100.0 * r_ql[b]);
    ASSERT_GT(r_norm2, 1e-6) << "fixture premise: the universe is noisy, the optimum has a non-zero residual";
    for (std::size_t i = 0; i < nb; ++i) {
      ql::Array up = params, dn = params;
      up[i] += h;
      dn[i] -= h;
      const QlFitEval eu = ql_fit_at(pc.svensson, up), ed = ql_fit_at(pc.svensson, dn);
      double jr = 0.0, jj = 0.0;
      for (std::size_t b = 0; b < n; ++b) {
        const double J = kFitWeight[b] * (eu.implied[b] - ed.implied[b]) / (2.0 * h);
        jr += J * 100.0 * r_ql[b];
        jj += J * J;
      }
      EXPECT_LE(std::abs(jr), 1e-6 * std::sqrt(jj * r_norm2)) << "QuantLib-cost gradient along beta_" << i;
    }

    // Negative controls: the decay mapping is live (κ·1.01 moves QuantLib's quotes far outside tolerance) and the
    // weights reached the verb (an unweighted residual disagrees with the engine's).
    ql::Array bumped = params;
    bumped[nb] *= 1.01;
    const QlFitEval ek = ql_fit_at(pc.svensson, bumped);
    double worst_k = 0.0, worst_w = 0.0;
    for (std::size_t b = 0; b < n; ++b) {
      worst_k = std::max(worst_k, std::abs(ek.implied[b] - e.implied[b]));
      worst_w = std::max(worst_w, std::abs(num(out, "residuals", b) - (e.implied[b] / 100.0 - kFitClean[b])));
    }
    EXPECT_GT(worst_k, 1e-6);
    EXPECT_GT(worst_w, 1e-6);
  }
}

TEST(BondRvVerbOracle, GovvieFitSplineResidualsAndZSpreadsMatchQuantLibOnTheFittedCurve) {
  const std::string conv = "US-TREASURY";
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = qd(kVd);
  const std::vector<BondSpec> u = fit_universe();
  const std::size_t n = u.size();
  const std::vector<double> meeting = {1.0}, back = {3.0, 7.0, 15.0, 25.0};

  json::object body;
  body["value_date"] = kVd;
  body["convention"] = conv;
  body["bonds"] = bonds_json(u);
  body["clean"] = arr(kFitClean);
  body["weight"] = arr(kFitWeight);
  body["model"] = "spline";
  body["meeting"] = arr(meeting);
  body["back"] = arr(back);
  const json::object out = run_verb("govvie_fit", body);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_EQ(out.at("n").to_number<int>(), static_cast<int>(n));
  ASSERT_TRUE(out.contains("z_spread"));

  // The fitted curve, rebuilt from the verb's own state x on the same topology, handed to QuantLib. Curve time is
  // ACT/365F from the value date on both sides (build::curve_time == Actual365Fixed from qd(kVd)).
  const json::array& xj = out.at("x").as_array();
  Eigen::VectorXd x(static_cast<int>(xj.size()));
  for (std::size_t i = 0; i < xj.size(); ++i) x[static_cast<int>(i)] = xj.at(i).to_number<double>();
  cv::ModularCurve<double> fitted = cv::make_modular_curve<double>(cv::flat_hermite(meeting, back));
  ASSERT_EQ(x.size(), static_cast<int>(meeting.size() + back.size()));
  fitted.set_forwards(x);
  const auto ts = ql::ext::make_shared<swaps::qlx::CurveTermStructure<cv::ModularCurve<double>>>(
      qd(kVd), ql::Actual365Fixed(), &fitted);
  const ql::Handle<ql::YieldTermStructure> disc(ts);

  double worst_z0 = 0.0;
  for (std::size_t b = 0; b < n; ++b) {
    const QlBond q = ql_bond(conv, u[b]);
    q.bond->setPricingEngine(ql::ext::make_shared<ql::DiscountingBondEngine>(disc));
    const ql::Date settle = q.bond->settlementDate();
    ASSERT_EQ(settle, ql::Date(12, ql::November, 2026));

    // Residual: w_b · (QuantLib clean off the fitted curve − market clean).
    EXPECT_TRUE(close(num(out, "residuals", b), kFitWeight[b] * (q.bond->cleanPrice() / 100.0 - kFitClean[b]),
                      tol::curve_rel))
        << "residual " << u[b].id;

    // z-spread (continuous, curve time) to the MARKET dirty price = clean + QuantLib's accrued.
    const double z = num(out, "z_spread", b);
    const double target_dirty = kFitClean[b] * 100.0 + q.bond->accruedAmount(settle);
    //   forward: QuantLib's bond off base ⊕ z (ZeroSpreadedTermStructure) reprices to the market dirty ...
    const ql::Handle<ql::YieldTermStructure> spreaded(ql::ext::make_shared<ql::ZeroSpreadedTermStructure>(
        disc, ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(z)), ql::Continuous, ql::Annual,
        ql::Actual365Fixed()));
    const QlBond qs = ql_bond(conv, u[b]);
    qs.bond->setPricingEngine(ql::ext::make_shared<ql::DiscountingBondEngine>(spreaded));
    EXPECT_TRUE(close(qs.bond->dirtyPrice(), target_dirty, tol::curve_rel)) << "dirty at verb z " << u[b].id;
    //   ... and inverse: QuantLib's own z-spread solve (decimal spread; 1e-10 = 1e-6 bp).
    const double z_ql =
        ql::BondFunctions::zSpread(*q.bond, ql::Bond::Price(kFitClean[b] * 100.0, ql::Bond::Price::Clean), ts,
                                   ql::Actual365Fixed(), ql::Continuous, ql::Annual, settle, /*accuracy=*/1e-14);
    EXPECT_NEAR(z, z_ql, tol::curve_rel) << "z-spread " << u[b].id;
    worst_z0 = std::max(worst_z0, std::abs(q.bond->dirtyPrice() - target_dirty));
  }
  // Negative control: the z-spreads carry information (at z = 0 some bond misses its market dirty by > 1e-4 per 100).
  EXPECT_GT(worst_z0, 1e-4);
}

// ================================================================================================================
// swap_spread (HEADLINE)
// ================================================================================================================
// A curve handle exposed with a plain discount(t), so qlx::CurveTermStructure can wrap the bundle's swap curve.
struct HandleCurve {
  const cal::CurveHandle<double>* h = nullptr;
  double discount(double t) const { return h->discount(t); }
};

TEST(BondRvVerbOracle, SwapSpreadHeadlineYieldAnchorAndTenorSwapMatchQuantLib) {
  const std::string conv = "US-TREASURY";
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = qd(kVd);

  const BondSpec bench{"UST-5Y", "2026-08-15", "2031-08-15", 0.04, nullptr};
  const double clean = 0.9917, spread = -0.0035;
  json::object body;
  body["value_date"] = kVd;
  body["convention"] = conv;
  body["bond"] = bond_json(bench);
  body["clean"] = clean;
  body["spread"] = spread;
  body["index"] = "USD-SOFR";
  body["tenor"] = "5Y";
  body["swap_curve"] = 0;
  body["factor_curve"] = 1;
  body["spread_type"] = "headline";  // matched_maturity is a known open gap (TASKS-ENGINE E7 3.4): NOT oracled
  const json::object out = run_verb("swap_spread", body);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);

  // (1) The benchmark's street yield vs BondFunctions on QuantLib's bond, settled by QuantLib (T+1 over Veterans Day).
  const QlBond q = ql_bond(conv, bench);
  const ql::Date settle = q.bond->settlementDate();
  ASSERT_EQ(settle, ql::Date(12, ql::November, 2026));
  const ql::Compounding comp = row_compounding(conv, q, settle);
  const double y_eng = num(out, "bond_yield");
  const double y_ql = ql::BondFunctions::yield(*q.bond, ql::Bond::Price(clean * 100.0, ql::Bond::Price::Clean), q.dc,
                                               comp, q.freq, settle, /*accuracy=*/1e-14);
  EXPECT_TRUE(close(y_eng, y_ql, tol::curve_rel));
  EXPECT_TRUE(close(ql::BondFunctions::cleanPrice(*q.bond, y_eng, q.dc, comp, q.freq, settle), clean * 100.0,
                    tol::curve_rel));

  // (2) The bundle the rows compose into: curve 0 = a swap curve (upward forwards), curve 1 = the govvie factor at the
  //     verb's anchor. Its swap curve goes to QuantLib through the adapter; QuantLib's SOFR OIS is built to the DB
  //     product's conventions (as Pricing.ParSwapFromTheShippedBuilderMatchesQuantLib does), NOT from the verb's dates.
  const double anchor = num(out, "anchor");
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({0.25, 0.5}, {1.0, 2.0, 3.0, 5.0, 7.0, 10.0})});
  p.curves.push_back({.base = -1, .regions = bld::govvie_factor_regions(anchor)});
  Eigen::VectorXd x(p.n_knots());
  for (int i = 0; i < x.size(); ++i) x[i] = 0.0360 + 0.0008 * i;
  const auto state_at = [&](double f) {
    Eigen::VectorXd xs = x;
    xs[p.offset(1)] = f;
    return xs;
  };
  const Eigen::VectorXd x0 = state_at(y_eng);
  const auto C0 = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x0[p.offset(c) + i]; });
  HandleCurve swap_curve{C0[0].get()};
  const ql::Handle<ql::YieldTermStructure> disc(ql::ext::make_shared<swaps::qlx::CurveTermStructure<HandleCurve>>(
      qd(kVd), ql::Actual365Fixed(), &swap_curve));

  const bld::SwapConv sc = bld::swap_conv("USD", "USD-SOFR");
  ASSERT_EQ(sc.product_id, "USD-SOFR-OIS");
  const auto sofr = ql::ext::make_shared<ql::Sofr>(disc);
  const ql::ext::shared_ptr<ql::OvernightIndexedSwap> ois = ql::MakeOIS(ql::Period(5, ql::Years), sofr, 0.03)
                                                                .withDiscountingTermStructure(disc)
                                                                .withSettlementDays(sc.spot_lag)
                                                                .withPaymentLag(sc.pay_lag)
                                                                .withPaymentAdjustment(qconv::bdc(sc.bdc));
  const double fair_ql = ois->fairRate();

  // (3) The anchor is the matched tenor swap's maturity in curve time: QuantLib's 5Y-from-spot OIS maturity.
  EXPECT_NEAR(anchor, ql::Actual365Fixed().yearFraction(qd(kVd), ois->maturityDate()), tol::literal);

  // (4) The emitted rows. The ASW row is Portfolio{+1·swap, −1·Rate(factor)}; its swap is QuantLib's OIS.
  const cal::Instrument pin = api::instrument_from_json(out.at("rows").at("pin"));
  const cal::Instrument asw = api::instrument_from_json(out.at("rows").at("asw"));
  ASSERT_EQ(asw.quote, cal::QuoteKind::Portfolio);
  ASSERT_EQ(asw.combination.size(), 2u);
  EXPECT_EQ(asw.combination[0].weight, 1.0);
  EXPECT_EQ(asw.combination[1].weight, -1.0);
  const auto model_at = [&](const cal::Instrument& ins, double f) {
    const Eigen::VectorXd xs = state_at(f);
    const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return xs[p.offset(c) + i]; });
    const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[static_cast<std::size_t>(i)]; };
    return cal::instrument_model_quote<double>(ins, curve_of);
  };
  EXPECT_TRUE(close(model_at(asw.combination[0].instrument, y_eng), fair_ql, tol::curve_rel)) << "tenor swap par rate";
  EXPECT_TRUE(close(pin.market, y_ql, tol::curve_rel)) << "pin target = QuantLib's benchmark yield";
  EXPECT_EQ(asw.market, spread);

  // (5) The composed headline: solve the factor so the PIN reprices (test-side secant on the pin's own model quote,
  //     no rate-transform formula assumed), then the ASW row must quote QuantLib's fairRate − QuantLib's yield.
  const auto g = [&](double f) { return model_at(pin, f) - pin.market; };
  double f0 = pin.market, f1 = pin.market + 1e-4, g0 = g(f0), g1 = g(f1);
  for (int it = 0; it < 50 && std::abs(g1) > 1e-16 && g1 != g0; ++it) {
    const double f2 = f1 - g1 * (f1 - f0) / (g1 - g0);
    f0 = f1;
    g0 = g1;
    f1 = f2;
    g1 = g(f1);
  }
  ASSERT_LT(std::abs(g1), 1e-14) << "factor pin did not solve";
  EXPECT_TRUE(close(model_at(asw, f1), fair_ql - y_ql, tol::curve_rel)) << "headline swap spread quote";

  // Negative controls: the SPOT LAG is live (a 5Y OIS from today matures ≥ 2 days earlier: anchor moves by ≥ 5e-3) and
  // the settlement CALENDAR is live (a NullCalendar T+1 settle moves the yield by ~2e-5).
  const ql::ext::shared_ptr<ql::OvernightIndexedSwap> no_spot =
      ql::MakeOIS(ql::Period(5, ql::Years), sofr, 0.03).withDiscountingTermStructure(disc).withSettlementDays(0);
  EXPECT_GT(std::abs(anchor - ql::Actual365Fixed().yearFraction(qd(kVd), no_spot->maturityDate())), 1e-3);
  const double y_null = ql::BondFunctions::yield(*q.bond, ql::Bond::Price(clean * 100.0, ql::Bond::Price::Clean), q.dc,
                                                 comp, q.freq, qd(kVd) + 1, 1e-14);
  EXPECT_GT(std::abs(y_eng - y_null), 1e-7);
}

}  // namespace
