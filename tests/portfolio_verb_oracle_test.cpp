// @oracle-test — the portfolio / portfolio_risk / risk / transform run_json VERBS, on a book of TYPED TRADES, vs
// QuantLib OvernightIndexedSwap / VanillaSwap priced off the verb's own calibrated curve. DO NOT DELETE OR WEAKEN.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// See tests/ORACLE_TESTS.md before changing anything here.
//
// WHAT THIS PINS THAT NOTHING ELSE DID (draft 2026-09-14). tools/oracle_coverage.py: `portfolio` was named by NO
// test, and no oracle named `portfolio`, `portfolio_risk`, `risk` or `transform`. api_test.cpp /
// portfolio_reprice_test.cpp check the book path against ITSELF (payer + receiver nets to 0, compiled == templated,
// AAD == FD of the same kernel) — a shared error cancels. Here every request goes through swaps::api::run_json:
//   request -> book_from_json (typed `trades`, `value_date`, `curve_roles`, `csa` / `discount_index`)
//           -> trade::Trade::vanilla_swap -> Trade::to_position (conventions DB row of the trade's OWN index,
//              float_leg_from / fixed_coupons_from rolled from the booked effective date, signed notional)
//           -> BundleSession::price_portfolio / price_portfolio_risk / risk_operator / transform_matrix
//           -> JSON out
// and QuantLib builds each trade from scratch: its own Schedule (Forward roll), its own coupons, its own payment-lag
// arithmetic, its own Payer/Receiver sign, its own DiscountingSwapEngine on the curve the CSA says.
//
// WHAT IS SHARED, AND WHAT IS NOT (so a pass means something):
//   SHARED  the discount factors — the verb calibrates the bundle and returns `x`; the test rebuilds OUR ModularCurve
//           from that x (bundle_from_json of the SAME json value the verb decoded, cal::build_bundle_curves) and hands
//           it to QuantLib through qlx::CurveTermStructure with Actual365Fixed from the value date, which is
//           bit-for-bit build::curve_time ((d - vd)/365). And the conventions DB STRINGS (calendar id, bdc, day
//           counts, frequencies, payment lag) — read here from the raw generated rows (swaps::conventions via
//           conventions_ql.hpp), NOT through build::swap_conv, so the engine's DB->SwapConv mapping is under test.
//   OURS    everything between: schedule generation, date adjustment, pay dates, accruals, coupon forecasting,
//           direction sign, notional, CSA -> discount role, index -> forecast role, aggregation, PV01 AD, the risk
//           operator, the cross-bundle Jacobian.
//   The CSA -> discount-curve mapping on the QuantLib side is a stated MARKET FACT (USD cash CSA discounts on SOFR,
//   EUR on ESTR), not the DB's currencies[].discount_index.
//
// UNITS AND SIGNS (verified in code, not assumed):
//   * NPV: currency units at the booked notional, discounted to the value date. Trade "pay":"fixed" => payer of
//     fixed => signed_notional() = +notional and value = N·(float − fixed) (include/swaps/trade/trade.hpp
//     signed_notional, include/swaps/portfolio/portfolio.hpp position_value). QuantLib Swap::Payer => payer_[0] = −1
//     on the fixed leg, +1 on the float leg (ql/instruments/fixedvsfloatingswap.cpp:73-79). So pay "fixed" <->
//     Swap::Payer and pay "float" <-> Swap::Receiver.
//   * PV01 (`portfolio.pv01`): the NPV change for +1bp on EVERY curve's forward ONCE (api/bundle_api.cpp
//     price_portfolio along pricing::parallel_direction): +1bp on each OUTRIGHT curve's interpolation forwards, which a
//     SPREAD curve (EUR6M = ESTR + spread) inherits from its base. NOT QuantLib's BPS (an annuity), NOT a zero-rate
//     shift. Until b5f3a4d it moved every state entry, so EUR6M moved 2bp -- this oracle's first draft reproduced that
//     as its reference; the rule is now written out below (parallel_bump), not taken from the engine. The QuantLib
//     number is the central difference of QuantLib's NPV under that bump.
//   * curve_grad = ∂NPV/∂x (length n_knots); risk_operator M = dx/dq (n_knots × n_res, market-scaled by
//     residual_market_scale, which is 1 for every hard ParRate row); ladder = Mᵀ·curve_grad = ∂NPV/∂q_i in currency
//     per unit (decimal) quote; transform T = ∂q_src/∂q_this (nr_src × nr_this, dimensionless).
//
// TOLERANCES (tests/tolerances.hpp; each looser one says why):
//   * NPV            tol::curve_rel (1e-10) PER UNIT NOTIONAL: same DFs, same algebra; only summation order and
//                    QuantLib's telescoped compounding differ (~1e-15 expected).
//   * par quotes     tol::reprice (1e-9): the verb's curve must reprice the bundle's QuantLib twins (hard, square).
//   * PV01, curve_grad  tol::jacobian_rel (1e-6) relative: exact AD vs a QuantLib central difference at kFdBump.
//                    Predicted FD noise ~3e-8 relative (roundoff ~1e-16·N·#flows / 2·1e-7), truncation ~1e-10.
//   * M, ladder, transform  kInverseRel (1e-5): QuantLib has no AD, so its Jacobian is a central difference
//                    (~1e-9 relative) and inverting it amplifies that by cond(J) (printed; expected 1e2..1e3).
//                    E5: re-lock at 100× the measured worst after the first green run.
//
// SCOPE — EXCLUDED, stated rather than implied:
//   * Seasoned trades (effective < value_date): the stateless run_json seam has no fixings input, so the verb must
//     REFUSE them. Pinned below as a control (error, not a price) — there is nothing to compare.
//   * Xccy MtM positions (MultiCurveBook::Kind::Xccy, raw `positions` path): no QuantLib core counterpart; the typed
//     `trades` path cannot book one (Trade::vanilla_swap only).
//   * Averaged float legs, RFR lookback / lockout / observation shift, float spreads, stepped fixed rates,
//     principal exchanges: not expressible through the typed `trades` JSON (book_from_json reads none of them).
//   * Stubs: every trade here is a whole number of coupon steps. (Weekend maturities ARE covered: T6 and T7 are booked
//     to their unadjusted weekend anniversaries, which QuantLib's Schedule rolls by the convention. Before d52038c the
//     engine accrued to the raw weekend date -- WK1 -- and a QuantLib build with an Unadjusted termination is kept
//     below as the control that the weekend is live.)
//   * Regularised risk (`regularize` present): M becomes a penalised pseudo-inverse whose penalty is an engine
//     modelling choice with no QuantLib counterpart. Only reg OFF is compared.
//   * vol_cube: out of scope.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <boost/json.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "conventions_ql.hpp"
#include "swaps/api/bundle_api.hpp"          // run_json; bundle_to_json / bundle_from_json (api/codec.hpp)
#include "swaps/build/conventions.hpp"       // swap_conv — CALIBRATION bundle only; the book side reads raw DB rows
#include "swaps/build/instruments.hpp"       // par_swap (the calibration bundle)
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"     // residual_market_scale
#include "swaps/conventions_data.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/trade/csa.hpp"               // named so the include-closure coverage tool sees the CSA rule
#include "swaps/trade/trade.hpp"             // named so the include-closure coverage tool sees Trade::to_position
#include "tolerances.hpp"

namespace {

namespace ql = QuantLib;
namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace bld = swaps::build;
namespace cvd = swaps::conventions;
namespace json = boost::json;
namespace qconv = swaps::refbuild::conv;
namespace tol = swaps::tol;

// Wednesday. Spot (T+2 on USD-SOFR and on TARGET) is Friday 2026-07-10. The same value date as
// calibration_oracle_test.cpp, whose bundle construction (par_swap vs MakeOIS/MakeVanillaSwap) is proven at 5e-13.
const char* const kValueDate = "2026-07-08";

constexpr double kFdBump = 1e-7;       // knot forwards are O(1e-2); keeps ~8 digits of every central difference
constexpr double kInverseRel = 3e-6;  // measured 2026-09-14: ladder 3.1e-8, M 2.1e-9, transform 1.2e-8 (cond 475); ~100x the worst
constexpr double kControlGap = 1e-5;   // per unit notional: a control must miss by >= 1e5 × tol::curve_rel

ql::Date qd(const std::string& iso) {
  const bld::Date d = bld::Date::from_iso(iso);
  return ql::Date(ql::Day(d.day()), ql::Month(d.month()), ql::Year(d.year()));
}

bld::Date eng_date(const ql::Date& d) {
  std::ostringstream os;
  os << ql::io::iso_date(d);
  return bld::Date::from_iso(os.str());
}

// Curve roles of the bundle (== the book's curve_roles binding). EUR6M is a SPREAD over ESTR.
enum Role : int { SOFR = 0, ESTR = 1, EUR6M = 2, NROLES = 3 };

using SwapPtr = ql::ext::shared_ptr<ql::FixedVsFloatingSwap>;

// ---- the world: the bundle the verb receives + QuantLib's view of the verb's curve -------------------------------
struct World {
  ql::Date today;
  std::vector<ql::RelinkableHandle<ql::YieldTermStructure>> h =
      std::vector<ql::RelinkableHandle<ql::YieldTermStructure>>(NROLES);
  ql::ext::shared_ptr<ql::OvernightIndex> sofr, estr;
  ql::ext::shared_ptr<ql::IborIndex> eur6m;

  cal::BundleProblem prob;     // as BUILT (our DB builders)
  json::value bundle;          // bundle_to_json(prob): the exact json value every request carries
  cal::BundleProblem decoded;  // bundle_from_json(bundle): the verb's own decode; the mirror's curves come from THIS
  std::vector<SwapPtr> twins;  // QuantLib twin of prob.instruments[i], same order
  std::vector<SwapPtr> observed;  // every QuantLib swap alive in the test, deep-updated on every relink

  std::vector<std::unique_ptr<cal::CurveHandle<double>>> curves;
  std::vector<ql::ext::shared_ptr<ql::YieldTermStructure>> ts;

  World() = default;
  World(const World&) = delete;
  World& operator=(const World&) = delete;

  // Rebuild OUR curves at state x (the verb's decode, the verb's state) and relink every QuantLib handle to them.
  void relink(const Eigen::VectorXd& x) {
    std::vector<std::unique_ptr<cal::CurveHandle<double>>> next =
        cal::build_bundle_curves<double>(decoded.curves, [&](int c, int i) { return x[decoded.offset(c) + i]; });
    std::vector<ql::ext::shared_ptr<ql::YieldTermStructure>> next_ts;
    for (std::size_t c = 0; c < next.size(); ++c) {
      next_ts.push_back(ql::ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
          today, ql::Actual365Fixed(), next[c].get()));
      h[c].linkTo(next_ts.back());
    }
    curves = std::move(next);  // the old curves die only after no handle points at them
    ts = std::move(next_ts);
    for (const SwapPtr& s : observed) s->deepUpdate();
  }
};

const std::vector<const char*>& curve_tenors() {
  static const std::vector<const char*> t{"1Y", "2Y", "3Y", "5Y", "10Y", "30Y"};
  return t;
}

// One OIS block: our par_swap from the DB, and QuantLib's OIS for the same trade on the same conventions (the
// calibration_oracle_test.cpp construction). `knots` (optional) collects the last fixed pay time as a curve knot.
void add_ois(World& w, cal::BundleProblem& into, std::vector<SwapPtr>& twins, const bld::SwapConv& conv,
             const ql::ext::shared_ptr<ql::OvernightIndex>& idx, Role role, const std::vector<const char*>& tenors,
             std::vector<double>* knots) {
  for (const char* t : tenors) {
    const ql::ext::shared_ptr<ql::OvernightIndexedSwap> s =
        ql::MakeOIS(ql::PeriodParser::parse(t), idx, 0.03)
            .withDiscountingTermStructure(w.h[role])
            .withSettlementDays(static_cast<ql::Natural>(conv.spot_lag))
            .withPaymentLag(conv.pay_lag)
            .withPaymentAdjustment(ql::ModifiedFollowing);
    into.instruments.push_back(bld::par_swap(eng_date(w.today), conv, eng_date(s->maturityDate()), role, role, 0.03));
    if (knots) knots->push_back(into.instruments.back().fixed.coupons.back().pay);
    twins.push_back(s);
    w.observed.push_back(s);
  }
}

// One IRS block: fixed vs EURIBOR-6M (forecast `fc_index`), discounted on ESTR.
void add_irs(World& w, cal::BundleProblem& into, std::vector<SwapPtr>& twins, const bld::SwapConv& conv,
             const ql::ext::shared_ptr<ql::IborIndex>& fc_index, const std::vector<const char*>& tenors,
             std::vector<double>* knots) {
  for (const char* t : tenors) {
    const ql::ext::shared_ptr<ql::VanillaSwap> s =
        ql::MakeVanillaSwap(ql::PeriodParser::parse(t), fc_index, 0.03)
            .withDiscountingTermStructure(w.h[ESTR])
            .withSettlementDays(static_cast<ql::Natural>(conv.spot_lag))
            .withFixedLegTenor(qconv::period(conv.fixed_freq_tok))
            .withFixedLegDayCount(qconv::day_counter(conv.fixed_dc))
            .withAtParCoupons(true);
    into.instruments.push_back(bld::par_swap(eng_date(w.today), conv, eng_date(s->maturityDate()), EUR6M, ESTR, 0.03));
    if (knots) knots->push_back(into.instruments.back().fixed.coupons.back().pay);
    twins.push_back(s);
    w.observed.push_back(s);
  }
}

// The bundle: SOFR outright (USD), ESTR outright (EUR), EURIBOR-6M as a spread over ESTR. Square and hard (one knot
// per instrument), so reg-off M is exactly J⁻¹. Its market is our model quote at a tilted x_true; the verb then
// calibrates it from flat_x0 — the test never uses x_true again, only the verb's returned x.
std::unique_ptr<World> make_world() {
  auto w = std::make_unique<World>();
  w->today = qd(kValueDate);
  ql::Settings::instance().evaluationDate() = w->today;
  ql::IborCoupon::Settings::instance().createAtParCoupons();  // stated, not assumed (VanillaSwap below also passes it)

  w->sofr = ql::ext::make_shared<ql::Sofr>(w->h[SOFR]);
  w->estr = ql::ext::make_shared<ql::Estr>(w->h[ESTR]);
  // EURIBOR-6M from its DB row (tenor, fixing lag, calendar, day count), end-of-month off (the engine rolls none).
  const cvd::IndexConv e6 = qconv::index("EUR-EURIBOR-6M");
  w->eur6m = ql::ext::make_shared<ql::IborIndex>(
      "EUR-EURIBOR-6M-DB", qconv::period(e6.tenor), static_cast<ql::Natural>(e6.fixing_lag), ql::EURCurrency(),
      qconv::calendar(e6.calendar), qconv::bdc(qconv::product(e6.par_product).bdc), /*endOfMonth=*/false,
      qconv::day_counter(e6.day_count), w->h[EUR6M]);

  std::array<std::vector<double>, NROLES> k;
  add_ois(*w, w->prob, w->twins, bld::swap_conv("USD", "USD-SOFR"), w->sofr, SOFR, curve_tenors(), &k[SOFR]);
  add_ois(*w, w->prob, w->twins, bld::swap_conv("EUR", "EUR-ESTR"), w->estr, ESTR, curve_tenors(), &k[ESTR]);
  add_irs(*w, w->prob, w->twins, bld::swap_conv("EUR", "EUR-EURIBOR-6M"), w->eur6m, curve_tenors(), &k[EUR6M]);

  w->prob.curves.resize(NROLES);
  w->prob.curves[SOFR] = px::CurveStructure{.base = -1, .currency = 0, .regions = swaps::curve::flat_hermite({}, k[SOFR])};
  w->prob.curves[ESTR] = px::CurveStructure{.base = -1, .currency = 1, .regions = swaps::curve::flat_hermite({}, k[ESTR])};
  w->prob.curves[EUR6M] =
      px::CurveStructure{.base = ESTR, .currency = 1, .regions = swaps::curve::flat_hermite({}, k[EUR6M])};

  const double level[NROLES] = {0.030, 0.025, 0.0015};
  const double tilt[NROLES] = {4e-4, 3e-4, 1e-5};
  Eigen::VectorXd x_true = Eigen::VectorXd::Zero(w->prob.n_knots());
  for (int c = 0; c < NROLES; ++c)
    for (int i = 0; i < w->prob.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i)
      x_true[w->prob.offset(c) + i] = level[c] + tilt[c] * i;
  {
    const auto C = cal::build_bundle_curves<double>(
        w->prob.curves, [&](int c, int i) { return x_true[w->prob.offset(c) + i]; });
    const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[static_cast<std::size_t>(i)]; };
    for (auto& ins : w->prob.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  }
  w->bundle = api::bundle_to_json(w->prob);
  w->decoded = api::bundle_from_json(w->bundle);
  return w;
}

// ---- the book ---------------------------------------------------------------------------------------------------
struct TradeCase {
  const char* id;
  const char* index;  // DB index id; curve_roles binds it to a bundle role
  const char* ccy;    // the CSA collateral currency
  const char* pay;    // "fixed" = payer of fixed, "float" = receiver of fixed
  double notional;
  double fixed_rate;  // deliberately OFF par (asserted), so a sign or a leg error is large
  const char* effective;
  const char* maturity;
};

// Every effective date is a business day on its calendar, and every maturity is a whole number of coupon steps after
// it (no stub). T1-T4 mature on business days. Interior anniversaries DO hit weekends (2027-07-10 Sat, 2031-07-12 Sat),
// so the modified-following interior roll is live. T6 and T7 are booked to a WEEKEND maturity -- the unadjusted
// anniversary, as a trader books it -- so the termination roll is live too: Sunday 2033-07-10 -> Monday 07-11 on
// US-GOVT-SOFR, Saturday 2031-01-11 -> Monday 01-13 on TARGET. Both are well off par (1.5%) so a one- or two-day
// accrual error is far outside the tolerance. T2, T4 and T7 are FORWARD-starting (T4's and T7's first EURIBOR
// fixings, 2027-01-08 and 2027-01-07, are after the value date, so no "today's fixing" is involved on either side).
const TradeCase kSofr10yPayer{"T1", "USD-SOFR", "USD", "fixed", 10.0e6, 0.0325, "2026-07-10", "2036-07-10"};
const TradeCase kSofrFwd5yReceiver{"T2", "USD-SOFR", "USD", "float", 25.0e6, 0.0290, "2027-07-12", "2032-07-12"};
const TradeCase kEstr5yPayer{"T3", "EUR-ESTR", "EUR", "fixed", 15.0e6, 0.0240, "2026-07-10", "2031-07-10"};
const TradeCase kEuriborFwd10yReceiver{"T4", "EUR-EURIBOR-6M", "EUR", "float", 20.0e6, 0.0310, "2027-01-12",
                                       "2037-01-12"};
const TradeCase kSofr7ySundayPayer{"T6", "USD-SOFR", "USD", "fixed", 12.0e6, 0.0450, "2026-07-10", "2033-07-10"};
const TradeCase kEuriborFwd4ySaturdayReceiver{"T7", "EUR-EURIBOR-6M", "EUR", "float", 8.0e6, 0.0150, "2027-01-11",
                                              "2031-01-11"};

const std::vector<TradeCase>& book_cases() {
  // Appended, never inserted: the controls below address T1 as [0] and T4 as [3].
  static const std::vector<TradeCase> b{kSofr10yPayer,         kSofrFwd5yReceiver, kEstr5yPayer,
                                        kEuriborFwd10yReceiver, kSofr7ySundayPayer, kEuriborFwd4ySaturdayReceiver};
  return b;
}

// Market facts, stated independently of the DB: a USD cash CSA discounts on SOFR, a EUR one on ESTR.
Role csa_discount_role(const TradeCase& t) { return std::string_view(t.ccy) == "USD" ? SOFR : ESTR; }
ql::Swap::Type ql_type(const TradeCase& t) {
  return std::string_view(t.pay) == "fixed" ? ql::Swap::Payer : ql::Swap::Receiver;
}

json::object trade_json(const TradeCase& t) {
  json::object o;
  o["id"] = t.id;
  o["index"] = t.index;
  o["currency"] = t.ccy;
  o["pay"] = t.pay;
  o["notional"] = t.notional;
  o["fixed_rate"] = t.fixed_rate;
  o["effective"] = t.effective;
  o["maturity"] = t.maturity;
  json::object csa;
  csa["collateral_currency"] = t.ccy;
  o["csa"] = std::move(csa);
  return o;
}

json::object book_json(const json::array& trades) {
  json::object roles;
  roles["USD-SOFR"] = static_cast<int>(SOFR);
  roles["EUR-ESTR"] = static_cast<int>(ESTR);
  roles["EUR-EURIBOR-6M"] = static_cast<int>(EUR6M);
  json::object b;
  b["value_date"] = kValueDate;
  b["curve_roles"] = std::move(roles);
  b["trades"] = trades;
  return b;
}

json::array all_trades_json() {
  json::array a;
  for (const TradeCase& t : book_cases()) a.push_back(trade_json(t));
  return a;
}

// The verb: the SAME json value the mirror decoded (object seam, no text round trip on the way in).
json::object run_request(const World& w, json::object req) {
  req["bundle"] = w.bundle;
  const json::value v = json::parse(api::run_json(req));
  return v.as_object();
}

Eigen::VectorXd vec(const json::value& v) {
  const json::array& a = v.as_array();
  Eigen::VectorXd out(static_cast<Eigen::Index>(a.size()));
  for (std::size_t i = 0; i < a.size(); ++i) out[static_cast<Eigen::Index>(i)] = a[i].to_number<double>();
  return out;
}

Eigen::MatrixXd mat(const json::value& v) {
  const json::array& rows = v.as_array();
  const std::size_t m = rows.size(), n = m ? rows[0].as_array().size() : 0;
  Eigen::MatrixXd out(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j)
      out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = rows[i].as_array().at(j).to_number<double>();
  return out;
}

// ---- QuantLib's side of a booked trade: built from the RAW DB row of the trade's index, dates and all -------------
SwapPtr ql_trade(World& w, const TradeCase& t, ql::Swap::Type type, Role discount,
                 const std::string& fixed_dc_override = "", bool raw_termination = false) {
  const cvd::IndexConv ix = qconv::index(t.index);
  const cvd::ProductConv p = qconv::product(ix.par_product);
  const ql::Calendar calendar = qconv::calendar(p.calendar);
  const ql::BusinessDayConvention bdc = qconv::bdc(p.bdc);
  const ql::DayCounter fixed_dc =
      qconv::day_counter(fixed_dc_override.empty() ? p.fixed.day_count : std::string_view(fixed_dc_override));
  const ql::Date eff = qd(t.effective), mat_date = qd(t.maturity);
  const auto schedule = [&](std::string_view freq) {
    return ql::Schedule(eff, mat_date, qconv::period(freq), calendar, bdc, raw_termination ? ql::Unadjusted : bdc,
                        ql::DateGeneration::Forward, /*endOfMonth=*/false);
  };

  SwapPtr s;
  if (ix.type == std::string_view("overnight")) {
    // QuantLib's single-schedule OIS: valid only because the DB row pays both legs on one frequency.
    if (p.fixed.frequency != p.floating.frequency)
      throw std::runtime_error(std::string("oracle premise: OIS product ") + std::string(p.id) +
                               " has different fixed/float frequencies");
    const ql::ext::shared_ptr<ql::OvernightIndex> on = std::string_view(t.index) == "USD-SOFR" ? w.sofr : w.estr;
    s = ql::ext::make_shared<ql::OvernightIndexedSwap>(type, t.notional, schedule(p.fixed.frequency), t.fixed_rate,
                                                       fixed_dc, on, /*spread=*/0.0, /*paymentLag=*/p.payment_lag,
                                                       ql::ModifiedFollowing, calendar);
  } else {
    // VanillaSwap has no payment lag: valid only because the IRS row has none.
    if (p.payment_lag != 0)
      throw std::runtime_error(std::string("oracle premise: IRS product ") + std::string(p.id) + " has a payment lag");
    s = ql::ext::make_shared<ql::VanillaSwap>(type, t.notional, schedule(p.fixed.frequency), t.fixed_rate, fixed_dc,
                                              schedule(p.floating.frequency), w.eur6m, /*spread=*/0.0,
                                              qconv::day_counter(p.floating.day_count), bdc,
                                              /*useIndexedCoupons=*/false);  // par coupons: accrual-date forecast
  }
  s->setPricingEngine(ql::ext::make_shared<ql::DiscountingSwapEngine>(w.h[discount]));
  w.observed.push_back(s);
  return s;
}

double ql_npv(World& w, const std::vector<SwapPtr>& book, const Eigen::VectorXd& x) {
  w.relink(x);
  double v = 0.0;
  for (const SwapPtr& s : book) v += s->NPV();
  return v;
}

// +h on every OUTRIGHT curve's interpolation forwards: every curve's forward moves by h exactly once (a spread curve
// inherits it from its base; there are no turns in this bundle). Written out HERE, deliberately not taken from
// pricing::parallel_direction, so the definition is checked rather than borrowed.
Eigen::VectorXd parallel_bump(const cal::BundleProblem& p, Eigen::VectorXd x, double h) {
  for (int c = 0; c < p.n_curves(); ++c)
    if (p.curves[static_cast<std::size_t>(c)].base < 0)
      for (int i = 0; i < p.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i) x[p.offset(c) + i] += h;
  return x;
}

// 1e-4 · d(NPV)/d(+1 on every curve's forward once), by QuantLib central difference — the engine's PV01 definition.
double ql_pv01(World& w, const std::vector<SwapPtr>& book, const Eigen::VectorXd& x) {
  const Eigen::VectorXd up = parallel_bump(w.decoded, x, kFdBump), dn = parallel_bump(w.decoded, x, -kFdBump);
  const double d = (ql_npv(w, book, up) - ql_npv(w, book, dn)) / (2.0 * kFdBump);
  w.relink(x);
  return 1.0e-4 * d;
}

Eigen::VectorXd ql_gradient(World& w, const std::vector<SwapPtr>& book, const Eigen::VectorXd& x) {
  Eigen::VectorXd g(x.size());
  for (Eigen::Index j = 0; j < x.size(); ++j) {
    Eigen::VectorXd up = x, dn = x;
    up[j] += kFdBump;
    dn[j] -= kFdBump;
    g[j] = (ql_npv(w, book, up) - ql_npv(w, book, dn)) / (2.0 * kFdBump);
  }
  w.relink(x);
  return g;
}

Eigen::VectorXd ql_fair_rates(World& w, const std::vector<SwapPtr>& quotes, const Eigen::VectorXd& x) {
  w.relink(x);
  Eigen::VectorXd r(static_cast<Eigen::Index>(quotes.size()));
  for (std::size_t i = 0; i < quotes.size(); ++i) r[static_cast<Eigen::Index>(i)] = quotes[i]->fairRate();
  return r;
}

// d(QuantLib fair rate)/d(x_j): rows = instruments, cols = knots.
Eigen::MatrixXd ql_quote_jacobian(World& w, const std::vector<SwapPtr>& quotes, const Eigen::VectorXd& x) {
  Eigen::MatrixXd J(static_cast<Eigen::Index>(quotes.size()), x.size());
  for (Eigen::Index j = 0; j < x.size(); ++j) {
    Eigen::VectorXd up = x, dn = x;
    up[j] += kFdBump;
    dn[j] -= kFdBump;
    J.col(j) = (ql_fair_rates(w, quotes, up) - ql_fair_rates(w, quotes, dn)) / (2.0 * kFdBump);
  }
  w.relink(x);
  return J;
}

double max_abs(const Eigen::MatrixXd& m) { return m.size() ? m.cwiseAbs().maxCoeff() : 0.0; }

}  // namespace

// ================================================================================================================
// portfolio — per trade: NPV and PV01 of every typed trade vs QuantLib, with the sign of each direction pinned.
// ================================================================================================================
TEST(PortfolioVerbOracle, EachTypedTradeNpvAndPv01MatchQuantLib) {
  ql::SavedSettings saved;
  const std::unique_ptr<World> w = make_world();

  double worst_npv = 0.0, worst_pv01 = 0.0;
  bool premise_checked = false;
  for (const TradeCase& t : book_cases()) {
    json::object req;
    req["portfolio"] = book_json(json::array{trade_json(t)});
    const json::object out = run_request(*w, std::move(req));
    ASSERT_FALSE(out.contains("error")) << t.id << ": " << json::serialize(out);
    ASSERT_TRUE(out.at("calibration").at("converged").as_bool()) << t.id;
    const json::object& pf = out.at("portfolio").as_object();
    ASSERT_EQ(pf.at("n").to_number<int>(), 1) << t.id;
    const Eigen::VectorXd x = vec(out.at("x"));
    ASSERT_EQ(x.size(), w->decoded.n_knots());

    // Premise (once): the curve the book is priced on reprices the bundle in QuantLib's own terms — the verb's
    // calibration, checked at the verb's x rather than at the generator.
    if (!premise_checked) {
      const Eigen::VectorXd q = ql_fair_rates(*w, w->twins, x);
      for (std::size_t i = 0; i < w->twins.size(); ++i)
        EXPECT_NEAR(q[static_cast<Eigen::Index>(i)], w->decoded.instruments[i].market, tol::reprice) << "bundle row " << i;
      premise_checked = true;
    }

    const SwapPtr q = ql_trade(*w, t, ql_type(t), csa_discount_role(t));
    const double npv_ql = ql_npv(*w, {q}, x);
    const double pv01_ql = ql_pv01(*w, {q}, x);
    const double npv = pf.at("npv").to_number<double>();
    const double pv01 = pf.at("pv01").to_number<double>();

    EXPECT_NEAR(npv, npv_ql, tol::curve_rel * t.notional) << t.id;
    EXPECT_NEAR(pv01, pv01_ql, tol::jacobian_rel * std::abs(pv01_ql)) << t.id;

    // The fixture has teeth: every trade is well off par (>= 1bp·annuity-ish of notional) ...
    EXPECT_GT(std::abs(npv_ql), 1e-4 * t.notional) << t.id << " is too close to par for a sign check to bite";
    // ... and both arms agree on WHICH WAY rates hurt: payer of fixed gains when rates rise, receiver loses.
    const double dir = ql_type(t) == ql::Swap::Payer ? 1.0 : -1.0;
    EXPECT_GT(dir * pv01_ql, 0.0) << t.id << " QuantLib PV01 sign";
    EXPECT_GT(dir * pv01, 0.0) << t.id << " engine PV01 sign";

    worst_npv = std::max(worst_npv, std::abs(npv - npv_ql) / t.notional);
    worst_pv01 = std::max(worst_pv01, std::abs(pv01 - pv01_ql) / std::abs(pv01_ql));
  }
  std::cout << "  [portfolio verb] worst |NPV - QuantLib|/notional = " << worst_npv
            << ", worst PV01 relative = " << worst_pv01 << "\n";
}

// ================================================================================================================
// portfolio — the whole book, par feedback, and negative controls (direction, discount role, day count, seasoned,
// weekend maturity).
// ================================================================================================================
TEST(PortfolioVerbOracle, WholeBookParFeedbackAndNegativeControls) {
  ql::SavedSettings saved;
  const std::unique_ptr<World> w = make_world();

  json::object req;
  req["portfolio"] = book_json(all_trades_json());
  const json::object out = run_request(*w, std::move(req));
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_TRUE(out.at("calibration").at("converged").as_bool());
  const json::object& pf = out.at("portfolio").as_object();
  const Eigen::VectorXd x = vec(out.at("x"));
  ASSERT_EQ(pf.at("n").to_number<int>(), static_cast<int>(book_cases().size()));

  std::vector<SwapPtr> qbook;
  double total_notional = 0.0;
  for (const TradeCase& t : book_cases()) {
    qbook.push_back(ql_trade(*w, t, ql_type(t), csa_discount_role(t)));
    total_notional += t.notional;
  }
  const double npv = pf.at("npv").to_number<double>();
  const double npv_ql = ql_npv(*w, qbook, x);
  const double pv01_ql = ql_pv01(*w, qbook, x);
  EXPECT_NEAR(npv, npv_ql, tol::curve_rel * total_notional) << "book NPV (mixed USD+EUR, in raw units: no FX)";
  EXPECT_NEAR(pf.at("pv01").to_number<double>(), pv01_ql, tol::jacobian_rel * std::abs(pv01_ql));

  // (1) PAR FEEDBACK. QuantLib's fair rate of T1 at the verb's curve, booked back through the verb, is worth 0.
  w->relink(x);
  TradeCase at_par = kSofr10yPayer;
  at_par.fixed_rate = qbook[0]->fairRate();
  json::object req_par;
  req_par["portfolio"] = book_json(json::array{trade_json(at_par)});
  const json::object out_par = run_request(*w, std::move(req_par));
  ASSERT_FALSE(out_par.contains("error")) << json::serialize(out_par);
  // Same bundle, same seam: the calibration must land on the same x (else the par check tests two curves).
  ASSERT_LE(max_abs(vec(out_par.at("x")) - x), 1e-14);
  EXPECT_NEAR(out_par.at("portfolio").at("npv").to_number<double>(), 0.0, tol::curve_rel * at_par.notional)
      << "QuantLib's par rate must be the verb's par rate";

  // (2) WRONG DIRECTION must disagree: the engine book vs QuantLib with T1 flipped to Receiver.
  std::vector<SwapPtr> flipped = qbook;
  flipped[0] = ql_trade(*w, kSofr10yPayer, ql::Swap::Receiver, SOFR);
  EXPECT_GT(std::abs(npv - ql_npv(*w, flipped, x)), kControlGap * kSofr10yPayer.notional)
      << "a pay/receive swap on T1 must be visible";

  // (3) DISCOUNT ROLE, two-sided. T4 booked with an explicit discount_index = its own forecast curve (no CSA):
  //     it must MATCH QuantLib discounting on EUR6M and MISS QuantLib discounting on ESTR.
  json::object t4_own = trade_json(kEuriborFwd10yReceiver);
  t4_own.erase("csa");
  t4_own["discount_index"] = "EUR-EURIBOR-6M";
  json::object req_disc;
  req_disc["portfolio"] = book_json(json::array{t4_own});
  const json::object out_disc = run_request(*w, std::move(req_disc));
  ASSERT_FALSE(out_disc.contains("error")) << json::serialize(out_disc);
  ASSERT_LE(max_abs(vec(out_disc.at("x")) - x), 1e-14);
  const double npv_disc = out_disc.at("portfolio").at("npv").to_number<double>();
  const SwapPtr q4_on_6m = ql_trade(*w, kEuriborFwd10yReceiver, ql::Swap::Receiver, EUR6M);
  EXPECT_NEAR(npv_disc, ql_npv(*w, {q4_on_6m}, x), tol::curve_rel * kEuriborFwd10yReceiver.notional)
      << "discount_index reaches the discount role";
  EXPECT_GT(std::abs(npv_disc - ql_npv(*w, {qbook[3]}, x)), kControlGap * kEuriborFwd10yReceiver.notional)
      << "discounting is live: EUR6M vs ESTR discounting must disagree";

  // (4) The DB's fixed day count is live in the ORACLE: 30E/360 -> ACT/360 on T4 must break the book match.
  std::vector<SwapPtr> wrong_dc = qbook;
  wrong_dc[3] = ql_trade(*w, kEuriborFwd10yReceiver, ql::Swap::Receiver, ESTR, "ACT/360");
  EXPECT_GT(std::abs(npv - ql_npv(*w, wrong_dc, x)), kControlGap * kEuriborFwd10yReceiver.notional)
      << "a wrong fixed day count must be visible";

  // (5) SEASONED trade (effective before the value date): the stateless seam carries no fixings, so the verb must
  //     refuse — never price the accruing coupon as if it had no realized part.
  const TradeCase seasoned{"T5", "USD-SOFR", "USD", "fixed", 10.0e6, 0.0300, "2025-07-10", "2030-07-10"};
  json::object req_s;
  req_s["portfolio"] = book_json(json::array{trade_json(seasoned)});
  const json::object out_s = run_request(*w, std::move(req_s));
  ASSERT_TRUE(out_s.contains("error")) << json::serialize(out_s);
  EXPECT_NE(std::string(out_s.at("error").as_string().c_str()).find("fixing"), std::string::npos)
      << json::serialize(out_s);

  // (6) WEEKEND MATURITY is live: T6 (Sunday) and T7 (Saturday) built by QuantLib with an UNADJUSTED termination --
  //     accruing to the raw weekend date, as the engine did before d52038c -- must miss the book. The premise that the
  //     booked dates really are weekends is asserted, not assumed.
  double weekend_gap = 1e300;
  for (const std::size_t i : {std::size_t{4}, std::size_t{5}}) {
    const TradeCase& t = book_cases()[i];
    ASSERT_GE(bld::Date::from_iso(t.maturity).weekday(), 5) << t.id << " fixture premise: a weekend maturity";
    std::vector<SwapPtr> raw_end = qbook;
    raw_end[i] = ql_trade(*w, t, ql_type(t), csa_discount_role(t), "", /*raw_termination=*/true);
    const double gap = std::abs(npv - ql_npv(*w, raw_end, x)) / t.notional;
    EXPECT_GT(gap, kControlGap) << t.id << ": accruing to the raw weekend maturity must be visible";
    weekend_gap = std::min(weekend_gap, gap);
  }

  std::cout << "  [portfolio verb, book] |NPV - QuantLib| = " << std::abs(npv - npv_ql) << " on "
            << total_notional << " notional; weekend-maturity control misses by >= " << weekend_gap
            << " per unit notional\n";
}

// ================================================================================================================
// portfolio_risk + risk — curve gradient, risk operator and delta ladder vs QuantLib's Jacobian (reg OFF).
// ================================================================================================================
TEST(PortfolioRiskVerbOracle, CurveGradientRiskOperatorAndLadderMatchQuantLib) {
  ql::SavedSettings saved;
  const std::unique_ptr<World> w = make_world();
  const int nk = w->decoded.n_knots(), nr = w->decoded.n_residuals();
  ASSERT_EQ(nk, nr) << "the comparison needs a square, hard bundle (M = J^-1)";
  const Eigen::VectorXd scale = cal::residual_market_scale(w->decoded.instruments);
  ASSERT_EQ(max_abs((scale.array() - 1.0).matrix()), 0.0) << "every row is a hard ParRate: no market scaling in M";

  json::object req;
  req["portfolio_risk"] = book_json(all_trades_json());
  req["risk"] = true;  // no `regularize`: RegSpec off
  const json::object out = run_request(*w, std::move(req));
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_TRUE(out.at("calibration").at("converged").as_bool());
  const json::object& pr = out.at("portfolio_risk").as_object();
  const Eigen::VectorXd x = vec(out.at("x"));
  const Eigen::VectorXd grad = vec(pr.at("curve_grad"));
  const Eigen::VectorXd ladder = vec(pr.at("ladder"));
  const Eigen::MatrixXd M = mat(out.at("risk_operator"));
  ASSERT_EQ(grad.size(), nk);
  ASSERT_EQ(ladder.size(), nr);
  ASSERT_EQ(M.rows(), nk);
  ASSERT_EQ(M.cols(), nr);
  ASSERT_EQ(pr.at("n").to_number<int>(), static_cast<int>(book_cases().size()));

  std::vector<SwapPtr> qbook;
  double total_notional = 0.0;
  for (const TradeCase& t : book_cases()) {
    qbook.push_back(ql_trade(*w, t, ql_type(t), csa_discount_role(t)));
    total_notional += t.notional;
  }
  EXPECT_NEAR(pr.at("npv").to_number<double>(), ql_npv(*w, qbook, x), tol::curve_rel * total_notional);

  // curve_grad = dNPV/dx vs QuantLib's central difference, knot by knot.
  const Eigen::VectorXd grad_ql = ql_gradient(*w, qbook, x);
  const double grad_scale = max_abs(grad_ql);
  ASSERT_GT(grad_scale, 0.0);
  EXPECT_LE(max_abs(grad - grad_ql), tol::jacobian_rel * grad_scale);
  for (int j = 0; j < nk; ++j)
    EXPECT_NEAR(grad[j], grad_ql[j], tol::jacobian_rel * grad_scale) << "knot " << j;

  // risk operator M = dx/dq = J^-1 with J = d(QuantLib fair rate)/dx.
  const Eigen::MatrixXd J_ql = ql_quote_jacobian(*w, w->twins, x);
  const Eigen::FullPivLU<Eigen::MatrixXd> lu(J_ql);
  ASSERT_EQ(lu.rank(), nk) << "QuantLib's calibration Jacobian must be invertible";
  const Eigen::MatrixXd M_ql = lu.inverse();
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_ql);
  const double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
  EXPECT_LE(max_abs(M - M_ql), kInverseRel * max_abs(M_ql)) << "cond(J) = " << cond;

  // ladder = Mᵀ·grad = dNPV/dq_i.
  const Eigen::VectorXd ladder_ql = M_ql.transpose() * grad_ql;
  const double ladder_scale = max_abs(ladder_ql);
  ASSERT_GT(ladder_scale, 0.0);
  EXPECT_LE(max_abs(ladder - ladder_ql), kInverseRel * ladder_scale);
  for (int i = 0; i < nr; ++i)
    EXPECT_NEAR(ladder[i], ladder_ql[i], kInverseRel * ladder_scale) << "calibration instrument " << i;

  // Negative control: the ladder sees the discount role. T4 discounted on EUR6M in QuantLib must move it by far
  // more than the tolerance (the discount sensitivity lands on the EUR6M spread quotes).
  std::vector<SwapPtr> wrong = qbook;
  wrong[3] = ql_trade(*w, kEuriborFwd10yReceiver, ql::Swap::Receiver, EUR6M);
  const Eigen::VectorXd ladder_wrong = M_ql.transpose() * ql_gradient(*w, wrong, x);
  EXPECT_GT(max_abs(ladder - ladder_wrong), 100.0 * kInverseRel * ladder_scale)
      << "the ladder comparison must be able to see a wrong discount curve";

  std::cout << "  [portfolio_risk verb] cond(J) = " << cond << ", grad rel = " << max_abs(grad - grad_ql) / grad_scale
            << ", M rel = " << max_abs(M - M_ql) / max_abs(M_ql)
            << ", ladder rel = " << max_abs(ladder - ladder_ql) / ladder_scale << "\n";
}

// ================================================================================================================
// transform — T = d(source quote)/d(this quote) vs QuantLib J_src · J_this^-1 (reg OFF).
// ================================================================================================================
TEST(TransformVerbOracle, CrossBundleTransformMatchesQuantLib) {
  ql::SavedSettings saved;
  const std::unique_ptr<World> w = make_world();
  const int nr = w->decoded.n_residuals();

  // Source bundle: the SAME curve set (same_curve_set: currency + outright/spread per curve, same order), quoting
  // tenors this bundle does not. Its markets are placeholders: cross_jacobian differentiates model quotes only.
  cal::BundleProblem src;
  src.curves = w->prob.curves;
  std::vector<SwapPtr> src_twins;
  add_ois(*w, src, src_twins, bld::swap_conv("USD", "USD-SOFR"), w->sofr, SOFR, {"4Y", "7Y", "15Y"}, nullptr);
  add_ois(*w, src, src_twins, bld::swap_conv("EUR", "EUR-ESTR"), w->estr, ESTR, {"7Y", "20Y"}, nullptr);
  add_irs(*w, src, src_twins, bld::swap_conv("EUR", "EUR-EURIBOR-6M"), w->eur6m, {"4Y", "12Y"}, nullptr);
  const int ns = src.n_residuals();
  constexpr int kUsdRows = 3;       // src rows 0..2 are SOFR
  constexpr int kUsdCols = 6;       // this bundle's rows 0..5 are SOFR
  constexpr int kEur12yRow = 6;     // src row 6 is the EURIBOR-6M 12Y

  json::object tr;
  tr["source_bundle"] = api::bundle_to_json(src);
  json::object req;
  req["transform"] = std::move(tr);  // no `regularize`: RegSpec off
  const json::object out = run_request(*w, std::move(req));
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_TRUE(out.at("calibration").at("converged").as_bool());
  const Eigen::VectorXd x = vec(out.at("x"));
  const Eigen::MatrixXd T = mat(out.at("transform"));
  ASSERT_EQ(T.rows(), ns);
  ASSERT_EQ(T.cols(), nr);

  const Eigen::MatrixXd J_this = ql_quote_jacobian(*w, w->twins, x);
  const Eigen::FullPivLU<Eigen::MatrixXd> lu(J_this);
  ASSERT_EQ(lu.rank(), nr);
  const Eigen::MatrixXd M_ql = lu.inverse();
  const Eigen::MatrixXd J_src = ql_quote_jacobian(*w, src_twins, x);
  const Eigen::MatrixXd T_ql = J_src * M_ql;
  const double t_scale = std::max(1.0, max_abs(T_ql));
  EXPECT_LE(max_abs(T - T_ql), kInverseRel * t_scale);
  for (int i = 0; i < ns; ++i)
    for (int j = 0; j < nr; ++j) EXPECT_NEAR(T(i, j), T_ql(i, j), kInverseRel * t_scale) << "src " << i << " this " << j;

  // Role wiring pinned in both arms: a USD source quote does not move with any EUR calibration quote.
  EXPECT_LE(max_abs(T.block(0, kUsdCols, kUsdRows, nr - kUsdCols)), kInverseRel);
  EXPECT_LE(max_abs(T_ql.block(0, kUsdCols, kUsdRows, nr - kUsdCols)), kInverseRel);

  // Negative control: the EURIBOR-6M 12Y source twin forecasting off ESTR instead of EUR6M (a forecast-role error)
  // must produce a visibly different transform row.
  cal::BundleProblem scratch;
  std::vector<SwapPtr> wrong_twin;
  add_irs(*w, scratch, wrong_twin, bld::swap_conv("EUR", "EUR-EURIBOR-6M"), w->eur6m->clone(w->h[ESTR]), {"12Y"},
          nullptr);
  const Eigen::MatrixXd row_wrong = ql_quote_jacobian(*w, wrong_twin, x) * M_ql;
  EXPECT_GT(max_abs(T.row(kEur12yRow) - row_wrong.row(0)), 1e3 * kInverseRel)
      << "a wrong forecast role on the source instrument must be visible";

  std::cout << "  [transform verb] |T - QuantLib|max = " << max_abs(T - T_ql) << " over " << ns << "x" << nr << "\n";
}
