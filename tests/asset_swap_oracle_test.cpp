// @oracle-test — the asset_swap run_json VERB (api/bond.cpp) vs QuantLib::AssetSwap on IDENTICAL discount
// factors and IDENTICAL schedules. DO NOT DELETE OR WEAKEN without reproducing the QuantLib comparison.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// See tests/ORACLE_TESTS.md before changing anything here.
//
// WHAT THIS PINS THAT NOTHING ELSE DID (draft 2026-09-13, ahead of E7 stage 3 lifting the verb into the library).
// tests/bond_asset_swap_oracle.cpp checks the KERNEL (build/par_asset_swap.hpp) on a float schedule the FIXTURE
// built. Nothing checked the VERB: request -> bond convention lookup -> the float leg it rolls BY HAND from the
// bond currency's default swap product -> curve sampling -> purchase-price rule -> SoA output. This drives the
// real verb through swaps::api::run_json and builds every QuantLib object in-test.
//
// WHY THE COMPARISON IS EXACT (so any difference is a definition or schedule choice, never the curve):
//   * SAME DFs. The test decodes the SAME bundle JSON value the verb receives (object seam: no text round
//     trip), calibrates it with the verb's recipe (smoothing_preset(Light), flat_x0), and hands that session's
//     curve to QuantLib through qlx::CurveTermStructure with Actual365Fixed from the value date. That is
//     bit-for-bit build::curve_time ((d - value_date)/365.0), so QuantLib's discount(date) and the verb's
//     sampled DF are the same call on the same x.
//   * INDEPENDENT schedules. The float schedule is QuantLib's own Schedule(Backward, endOfMonth=false) on the
//     conventions the DB names for the bond currency's default swap product (build::swap_conv ->
//     conventions_ql.hpp) — not the verb's dates. The bond is QuantLib's FixedRateBond on the DB row's frequency.
//   * The par-floater identity holds EXACTLY in QuantLib: par coupons, 0 fixing days, index day count == payment
//     day count, pay date == accrual end, forwarding curve == discount curve => the float leg telescopes to
//     N·(DF(settle) − DF(T)), which is what the verb's annuity-only formula assumes.
//   * settle = value_date + 1. QuantLib's DiscountingSwapEngine DROPS flows ON the evaluation date
//     (includeReferenceDateEvents = false), which would silently lose the par swap's upfront at settle.
//     (That is the likely reason bond_asset_swap_oracle.cpp needs 5e-5: its settle == today.)
//
// SCOPE, stated rather than implied: single-curve (projection == discount); no payment lag (the verb ignores the
// DB's payment_lag); bond coupons unadjusted (build::fixed_rate_bond's choice, pinned in bond_oracle_test).
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <boost/json.hpp>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "conventions_ql.hpp"
#include "swaps/api/bond.hpp"  // asset_swap_json — named so the include-closure coverage tool sees the verb
#include "swaps/api/bundle_api.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/calibration/residual_engine.hpp"
#include "swaps/conventions_data.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "tolerances.hpp"

namespace {

namespace ql = QuantLib;
namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace bld = swaps::build;
namespace json = boost::json;
namespace qconv = swaps::refbuild::conv;
namespace tol = swaps::tol;

ql::Date qd(const std::string& iso) {
  const bld::Date d = bld::Date::from_iso(iso);
  return ql::Date(ql::Day(d.day()), ql::Month(d.month()), ql::Year(d.year()));
}

// ---- the bundle: one outright curve, self-consistent OIS-style market on curve time -------------------------
px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}
cal::Instrument par_ois(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.forecast = 0;
  ins.fwd.discount = 0;
  ins.fixed.discount = 0;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) {
    ins.fwd.coupons.push_back(ois_coupon(prev, t));
    px::FixedCoupon x;
    x.pay = t;
    x.tau = t - prev;
    ins.fixed.coupons.push_back(x);
    prev = t;
  }
  return ins;
}
cal::Instrument front_rate(double a, double b) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  return ins;
}
// Upward-sloping forwards (3.00% -> 3.96%) so a wrong pay date or accrual moves the annuity visibly. The market
// is generated from these forwards; the verb (and the mirror below) then calibrate it with Light smoothing —
// the answer need not equal the generator, only be the SAME in both arms.
json::value make_bundle_json() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = swaps::curve::flat_hermite({0.25, 0.5}, {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0})});
  p.instruments.push_back(front_rate(0.0, 0.25));
  p.instruments.push_back(front_rate(0.25, 0.5));
  for (double T : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0}) p.instruments.push_back(par_ois(T));
  Eigen::VectorXd x(p.n_knots());
  for (int i = 0; i < x.size(); ++i) x[i] = 0.0300 + 0.0012 * i;
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  return api::bundle_to_json(p);
}

// ---- the mirror: the verb's calibration, exposed to QuantLib --------------------------------------------------
struct SessionCurve {
  const api::BundleSession* s = nullptr;
  int curve = 0;
  double discount(double t) const { return s->sample({t}).at(static_cast<std::size_t>(curve)).discount.at(0); }
};

struct Mirror {
  json::value bundle;
  std::unique_ptr<api::BundleSession> sess;
  SessionCurve curve;
  ql::Handle<ql::YieldTermStructure> disc;

  explicit Mirror(const std::string& value_date) : bundle(make_bundle_json()) {
    // EXACTLY api/bond.cpp's recipe: bundle_from_json -> BundleSession -> Light preset -> flat_x0.
    sess = std::make_unique<api::BundleSession>(api::bundle_from_json(bundle));
    const api::RegSpec reg =
        cal::smoothing_preset(cal::Smoothing::Light, static_cast<int>(sess->problem().curves.size()));
    sess->calibrate(api::flat_x0(sess->problem()), reg);
    curve = SessionCurve{sess.get(), /*curve=*/0};
    disc = ql::Handle<ql::YieldTermStructure>(ql::ext::make_shared<swaps::qlx::CurveTermStructure<SessionCurve>>(
        qd(value_date), ql::Actual365Fixed(), &curve));
  }
  Mirror(const Mirror&) = delete;
  Mirror& operator=(const Mirror&) = delete;
};

// ---- QuantLib's side of one bond ---------------------------------------------------------------------------
struct BondCase {
  const char* issue;
  const char* maturity;
  double coupon;
};

class QlPackage {
 public:
  QlPackage(const Mirror& m, const std::string& conv_id, const BondCase& bc, const std::string& settle)
      : disc_(m.disc) {
    const swaps::conventions::BondConv bconv = swaps::conventions::require_bond(conv_id);
    // Bond: the DB row's frequency, ACT/ACT ICMA on the unadjusted maturity-anchored grid, coupons paid
    // UNADJUSTED (build::fixed_rate_bond's choice). settlementDays on a NullCalendar == settle − value date.
    ql::Schedule bsched(qd(bc.issue), qd(bc.maturity), qconv::period(bconv.frequency), ql::NullCalendar(),
                        ql::Unadjusted, ql::Unadjusted, ql::DateGeneration::Backward, false);
    const ql::Natural sdays = static_cast<ql::Natural>(qd(settle) - disc_->referenceDate());
    bond_ = ql::ext::make_shared<ql::FixedRateBond>(sdays, 100.0, bsched, std::vector<ql::Rate>{bc.coupon},
                                                    ql::ActualActual(ql::ActualActual::ISMA, bsched),
                                                    ql::Unadjusted, 100.0, qd(bc.issue));
    bond_->setPricingEngine(ql::ext::make_shared<ql::DiscountingBondEngine>(disc_));

    // Float leg: the bond currency's DEFAULT swap product, read from the DB exactly as the verb reads it.
    const bld::SwapConv sc = bld::swap_conv(std::string(bconv.currency), "");
    cal_ = qconv::calendar(sc.calendar);
    dc_ = qconv::day_counter(sc.float_dc);
    const ql::BusinessDayConvention bdc = qconv::bdc(sc.bdc);
    const ql::Period tenor = qconv::period(sc.float_freq_tok);
    // 0 fixing days, forwarding == discounting; the currency object is a label only (it prices nothing).
    idx_ = ql::ext::make_shared<ql::IborIndex>("AswVerbOracleFloat", tenor, /*fixingDays=*/0, ql::USDCurrency(),
                                               cal_, bdc, /*endOfMonth=*/false, dc_, disc_);
    fsched_ = ql::Schedule(qd(settle), qd(bc.maturity), tenor, cal_, bdc, bdc, ql::DateGeneration::Backward,
                           /*endOfMonth=*/false);
  }

  const ql::ext::shared_ptr<ql::FixedRateBond>& bond() const { return bond_; }
  const ql::Schedule& float_schedule() const { return fsched_; }
  const ql::DayCounter& float_dc() const { return dc_; }

  // QuantLib's fair spread. parSwap=true: par asset swap (upfront = dirty − 100 at settle, float notional 100).
  // parSwap=false: market-value / proceeds asset swap (float notional scaled by the dirty price).
  double fair_spread(double clean_per_100, bool par_swap) const {
    ql::AssetSwap asw(/*payBondCoupon=*/true, bond_, clean_per_100, idx_, /*spread=*/0.0, fsched_, dc_, par_swap);
    asw.setPricingEngine(ql::ext::make_shared<ql::DiscountingSwapEngine>(disc_));
    return asw.fairSpread();
  }

  // Float annuity per unit notional, forward to settle: Σ τ·DF / DF(settle), from QuantLib's floating-leg BPS on a
  // par swap (BPS = 1bp · N · Σ τ·DF / DF(npv date), npv date = the curve reference date).
  double annuity(const ql::DayCounter& pay_dc) const {
    ql::AssetSwap asw(true, bond_, 100.0, idx_, 0.0, fsched_, pay_dc, /*parSwap=*/true);
    asw.setPricingEngine(ql::ext::make_shared<ql::DiscountingSwapEngine>(disc_));
    const double basis_point = 1.0e-4;  // QuantLib's BPS unit, not a tolerance
    const double notional = bond_->notional(fsched_.startDate());
    return std::abs(asw.floatingLegBPS()) / (basis_point * notional) * disc_->discount(disc_->referenceDate()) /
           disc_->discount(fsched_.startDate());
  }

 private:
  ql::Handle<ql::YieldTermStructure> disc_;
  ql::ext::shared_ptr<ql::FixedRateBond> bond_;
  ql::ext::shared_ptr<ql::IborIndex> idx_;
  ql::Schedule fsched_;
  ql::Calendar cal_;
  ql::DayCounter dc_;
};

// ---- the verb ----------------------------------------------------------------------------------------------
json::object bond_row(const std::string& conv, const BondCase& bc, const std::string& settle) {
  json::object r;
  r["convention"] = conv;
  r["issue"] = bc.issue;
  r["settle"] = settle;
  r["maturity"] = bc.maturity;
  r["coupon"] = bc.coupon;
  return r;
}

json::object run_verb(const json::value& bundle, const std::string& value_date, const json::array& bonds) {
  json::object body;
  body["value_date"] = value_date;
  body["bundle"] = bundle;  // the same json::value the mirror decoded — an exact copy, no text
  body["bonds"] = bonds;
  json::object req;
  req["asset_swap"] = std::move(body);
  const json::value v = json::parse(api::run_json(req));
  return v.as_object();
}

double num(const json::object& o, const char* key, std::size_t i) {
  return o.at(key).as_array().at(i).to_number<double>();
}

// Tolerance: both arms evaluate the same algebra on the same DFs; the only noise is float rounding in QuantLib's
// telescoping float leg (~20 flows at notional 100) and one JSON serialize->parse (<= 1 ULP). Expected agreement
// ~1e-14; tol::curve_rel (1e-10, = 1e-6 bp on a decimal spread) leaves four orders of headroom and is still
// ~25x below the leap-day schedule defect the second test is built to expose.

TEST(AssetSwapVerbOracle, SpreadAnnuityAndCurvePricesMatchQuantLib) {
  const std::string vd = "2026-09-15", settle = "2026-09-16";  // T+1: DB bonds[US-TREASURY].settle_lag
  const std::string conv = "US-TREASURY";
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = qd(vd);
  ql::IborCoupon::Settings::instance().createAtParCoupons();  // the build default; stated, not assumed

  const Mirror m(vd);
  // Feb-15 maturity: annual SOFR-OIS roll lands on Presidents' Day in 2027/2031/2032 (ModF -> Tuesday), so the
  // DB calendar and bdc are live; no month-end clamp anywhere (that case is the next test).
  const BondCase bc{"2023-02-15", "2033-02-15", 0.035};
  const QlPackage q(m, conv, bc, settle);
  ASSERT_EQ(q.bond()->settlementDate(), qd(settle));

  const double clean_mkt = 0.9725;
  const double accrued_ql = q.bond()->accruedAmount(qd(settle)) / 100.0;
  const double dirty_mkt = clean_mkt + accrued_ql;

  json::array bonds;
  bonds.push_back(bond_row(conv, bc, settle));  // [0] no price: the verb's "par-par" default (clean = 1)
  json::object with_clean = bond_row(conv, bc, settle);
  with_clean["clean"] = clean_mkt;
  bonds.push_back(with_clean);  // [1] clean given
  json::object with_dirty = bond_row(conv, bc, settle);
  with_dirty["dirty"] = dirty_mkt;
  bonds.push_back(with_dirty);  // [2] dirty given (same package as [1])

  const json::object out = run_verb(m.bundle, vd, bonds);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_EQ(out.at("n").to_number<int>(), 3);

  // Bond side: accrued and the curve prices (independent of the price mode).
  const double ann_ql = q.annuity(q.float_dc());
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(num(out, "accrued", i), accrued_ql, tol::curve_rel) << "row " << i;
    EXPECT_NEAR(num(out, "dirty_curve", i), q.bond()->dirtyPrice() / 100.0, tol::curve_rel) << "row " << i;
    EXPECT_NEAR(num(out, "clean_curve", i), q.bond()->cleanPrice() / 100.0, tol::curve_rel) << "row " << i;
    EXPECT_NEAR(num(out, "annuity", i), ann_ql, tol::curve_rel * ann_ql) << "row " << i;
  }

  // Spreads. The verb's formula is QuantLib's PAR asset swap at the purchase price it resolves.
  const double par_at_100_ql = q.fair_spread(100.0, /*par_swap=*/true);
  const double par_at_mkt_ql = q.fair_spread(clean_mkt * 100.0, /*par_swap=*/true);
  const double proceeds_ql = q.fair_spread(clean_mkt * 100.0, /*par_swap=*/false);
  EXPECT_NEAR(num(out, "asw_spread", 0), par_at_100_ql, tol::curve_rel);
  EXPECT_NEAR(num(out, "asw_spread", 1), par_at_mkt_ql, tol::curve_rel);
  EXPECT_NEAR(num(out, "asw_spread", 2), par_at_mkt_ql, tol::curve_rel);

  // PROCEEDS. The verb's comment calls [1] "the proceeds spread"; it is not. QuantLib's market-value (proceeds)
  // spread is the par spread divided by the dirty purchase price — pinned here as an identity, plus a control
  // that the verb's number is NOT the proceeds number (they differ by the factor 1/dirty ≈ 2.6 % here).
  EXPECT_NEAR(num(out, "asw_spread", 1) / dirty_mkt, proceeds_ql, tol::curve_rel);
  EXPECT_GT(std::abs(num(out, "asw_spread", 1) - proceeds_ql), 1e-6)
      << "the verb now returns the proceeds spread: update the identity above and the verb's docs together";

  // Negative controls — the comparison has teeth:
  //   the price mode is live ((1 − 0.9725)/annuity ≈ 47 bp between rows 0 and 1) ...
  EXPECT_GT(std::abs(par_at_mkt_ql - par_at_100_ql), 1e-4);
  //   ... and the DB float day count is live (ACT/365F would move the annuity by 365/360 − 1 ≈ 1.4 %).
  EXPECT_GT(std::abs(q.annuity(ql::Actual365Fixed()) / ann_ql - 1.0), 1e-3);
}

// PREDICTED TO FAIL against api/bond.cpp as of 2026-09-13 (annuity by ~2.6e-7 relative, spreads by ~2.5e-9 and
// ~3.5e-9 absolute), which is what it is for. The verb rolls the float schedule by CHAINING
// `d = d.plus_months(-step)` from the previous date, so a day clamped once stays clamped: from a 2036-02-29
// maturity it yields 2035-02-28, ..., 2029-02-28, 2028-02-28, while QuantLib (and ISDA, and
// build/bond.hpp coupon_dates_backward, whose comment warns about exactly this) offsets every date from the
// anchor and lands on 2028-02-29 — a business day, so the boundary really moves. The same defect drops month-end
// roll days (Aug-31 -> Feb-28 -> Aug-28) for any 3M/6M float product. Fix: build the leg with
// schedule.hpp swap_periods_between (StubSide::Front, roll_dom = maturity day) instead of the hand loop.
// If the pin must land before the fix, rename to DISABLED_… (still counted by check_oracle_tests.sh).
TEST(AssetSwapVerbOracle, LeapDayMaturityFloatRollMatchesQuantLib) {
  const std::string vd = "2026-09-15", settle = "2026-09-16";
  const std::string conv = "US-TREASURY";
  ql::SavedSettings saved;
  ql::Settings::instance().evaluationDate() = qd(vd);
  ql::IborCoupon::Settings::instance().createAtParCoupons();

  const Mirror m(vd);
  // 2036-02-29 is a Friday (QuantLib's AssetSwap requires the adjusted float end == adjusted bond maturity).
  const BondCase bc{"2026-02-28", "2036-02-29", 0.045};
  const QlPackage q(m, conv, bc, settle);

  // The fixture's premise, in QuantLib alone: its schedule keeps the leap day.
  const std::vector<ql::Date>& fd = q.float_schedule().dates();
  ASSERT_NE(std::find(fd.begin(), fd.end(), ql::Date(29, ql::February, 2028)), fd.end());

  const double clean_mkt = 0.97;
  json::array bonds;
  bonds.push_back(bond_row(conv, bc, settle));
  json::object with_clean = bond_row(conv, bc, settle);
  with_clean["clean"] = clean_mkt;
  bonds.push_back(with_clean);

  const json::object out = run_verb(m.bundle, vd, bonds);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_EQ(out.at("n").to_number<int>(), 2);

  const double ann_ql = q.annuity(q.float_dc());
  // Bond side is unaffected by the float roll — these pass today.
  EXPECT_NEAR(num(out, "dirty_curve", 0), q.bond()->dirtyPrice() / 100.0, tol::curve_rel);
  EXPECT_NEAR(num(out, "accrued", 0), q.bond()->accruedAmount(qd(settle)) / 100.0, tol::curve_rel);
  // Float side — these fail until the roll is fixed.
  EXPECT_NEAR(num(out, "annuity", 0), ann_ql, tol::curve_rel * ann_ql);
  EXPECT_NEAR(num(out, "asw_spread", 0), q.fair_spread(100.0, true), tol::curve_rel);
  EXPECT_NEAR(num(out, "asw_spread", 1), q.fair_spread(clean_mkt * 100.0, true), tol::curve_rel);
}

}  // namespace
