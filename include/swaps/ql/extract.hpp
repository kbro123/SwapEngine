#pragma once
// QuantLib -> plain-data cashflow schedules.
//
// This is the ONLY layer that touches QuantLib instrument internals. It runs once at setup: it
// reads the exact coupon accrual/payment dates and accrual factors that QuantLib itself would use,
// converts them to curve-time year fractions, and hands them to the templated kernel
// (swaps/pricing/cashflows.hpp). Calendars, schedules and day counts are QuantLib's job here — we
// never reimplement them (CLAUDE.md §1).

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/overnightindexedcoupon.hpp>
#include <ql/indexes/iborindex.hpp>
#include <ql/indexes/indexmanager.hpp>
#include <ql/instruments/overnightindexedswap.hpp>
#include <ql/optional.hpp>
#include <ql/settings.hpp>
#include <ql/time/daycounter.hpp>

#include <utility>
#include <vector>

#include "swaps/pricing/cashflows.hpp"

namespace swaps::qlx {

// Convert a QuantLib swap into the plain-data OIS schedule. `dc`/`ref` define curve time and MUST
// match the curve the schedule will later be priced against.
inline pricing::OisSwap extract_ois_swap(const QuantLib::OvernightIndexedSwap& swap,
                                         const QuantLib::Date& ref,
                                         const QuantLib::DayCounter& dc) {
  using namespace QuantLib;
  pricing::OisSwap out;
  auto t = [&](const Date& d) { return dc.yearFraction(ref, d); };

  for (const auto& cf : swap.overnightLeg()) {
    auto c = ext::dynamic_pointer_cast<OvernightIndexedCoupon>(cf);
    QL_REQUIRE(c, "overnight leg cashflow is not an OvernightIndexedCoupon");
    out.float_acc_start.push_back(t(c->accrualStartDate()));
    out.float_acc_end.push_back(t(c->accrualEndDate()));
    out.float_pay.push_back(t(c->date()));
  }
  for (const auto& cf : swap.fixedLeg()) {
    auto c = ext::dynamic_pointer_cast<FixedRateCoupon>(cf);
    QL_REQUIRE(c, "fixed leg cashflow is not a FixedRateCoupon");
    out.fixed_pay.push_back(t(c->date()));
    out.fixed_accrual.push_back(c->accrualPeriod());
  }
  return out;
}

// 3M compounded future: accrual start/end and the ACT/360 accrual QuantLib uses for the reference
// rate. (start, end are the IMM value/maturity dates.)
inline pricing::CompoundedFuture extract_compounded_future(const QuantLib::Date& start,
                                                           const QuantLib::Date& end,
                                                           const QuantLib::Date& ref,
                                                           const QuantLib::DayCounter& curveDc) {
  using namespace QuantLib;
  pricing::CompoundedFuture out;
  out.start = curveDc.yearFraction(ref, start);
  out.end = curveDc.yearFraction(ref, end);
  out.accrual = Actual360().yearFraction(start, end);  // SOFR compounding day count
  return out;
}

// 1M arithmetic-average future. Mirrors OvernightIndexFuture::averagedRate() day by day: past days
// use the index history (fixed, zero derivative), future days become curve-priced sub-periods, and
// the denominator is the single span year fraction on the index day count.
inline pricing::AveragedFuture extract_averaged_future(
    const QuantLib::ext::shared_ptr<QuantLib::OvernightIndex>& index, const QuantLib::Date& valueDate,
    const QuantLib::Date& maturityDate, const QuantLib::Date& ref,
    const QuantLib::DayCounter& curveDc) {
  using namespace QuantLib;
  pricing::AveragedFuture out;
  const Date today = Settings::instance().evaluationDate();
  const Calendar cal = index->fixingCalendar();
  const DayCounter idc = index->dayCounter();  // ACT/360 for SOFR
  out.period_yf = idc.yearFraction(valueDate, maturityDate);

  const TimeSeries<Real>& history = IndexManager::instance().getHistory(index->name());
  for (Date d1 = valueDate; d1 < maturityDate;) {
    const Date d2 = cal.advance(d1, 1, Days);
    if (d1 < today) {
      const Real f = history[d1];
      QL_REQUIRE(f != Null<Real>(), "missing SOFR fixing on " << d1);
      out.realized_sum += f * idc.yearFraction(d1, d2);
    } else {
      out.sub_start.push_back(curveDc.yearFraction(ref, d1));
      out.sub_end.push_back(curveDc.yearFraction(ref, d2));
    }
    d1 = d2;
  }
  return out;
}

// =================================================================================================
// GENERIC EXTRACTORS (docs/generic-instrument-pipeline.md §5)
// =================================================================================================
// QuantLib -> the ONE generic cashflow model (pricing::FloatCoupon / RateObservation / FixedCoupon).
//
// DISPATCH IS ON QUANTLIB COUPON TYPE ONLY -- never on index identity. No index name, currency,
// calendar or tenor appears below (or anywhere in include/): "SOFR 3M future", "FF averaging future"
// and "Euribor 3M swap" are not engine concepts, they are RateObservation DATA a test builds
// (CLAUDE.md §1, design §1/§6). Everything here reads the generic QuantLib coupon interface.
//
// THE DAY-COUNT SEPARATION (the correctness rule that makes 30/360-fixed vs ACT/360-float work):
//   * every TIME (sub_start, sub_end, pay) is a year fraction on the CURVE's day counter, measured
//     from the curve reference date -- `curveDc.yearFraction(ref, d)`;
//   * every ACCRUAL (tau_pay, tau_index) comes from the INSTRUMENT's / INDEX's OWN day counter, read
//     back out of the QuantLib coupon (`accrualPeriod()`, `dt()`, `spanningTime()`).
// The two are independent and must never be conflated.
//
// GEARING folds exactly into the observation: QuantLib's coupon rate is `gearing*rate + spread`, and
//   gearing * (Σ w_k(DF/DF − 1) + realized)/tau  ==  (Σ (g·w_k)(DF/DF − 1) + g·realized)/tau,
// so a gearing != 1 scales the weights and `realized`. At gearing == 1 we leave `weight` EMPTY, which
// is what keeps the standard shape on the compiled engine's fused unit-weight fast path
// (`BundleFloatBatch::sub_is_identity` / `cpn_is_plain`) and bit-exact vs the legacy kernel.

namespace detail {

// Curve time: year fraction on the CURVE day counter from the curve reference date.
struct CurveTime {
  QuantLib::Date ref;
  QuantLib::DayCounter dc;
  double operator()(const QuantLib::Date& d) const { return dc.yearFraction(ref, d); }
};

// The already-fixed prefix of a multi-fixing overnight coupon, exactly as QuantLib's own pricers
// walk it (OvernightIndexedCouponPricer::averageRate and
// ArithmeticAveragedOvernightIndexedCouponPricer::swapletRate share this loop): fixings strictly
// before today MUST be present; today is a border case -- use the fixing if published, else forecast.
// Returns the number of leading sub-periods consumed; `compound` accumulates Π(1+f·dt) and `average`
// accumulates Σ f·dt (the caller uses whichever its averaging convention needs).
inline QuantLib::Size overnight_fixed_prefix(const QuantLib::OvernightIndexedCoupon& c,
                                             QuantLib::Real& compound, QuantLib::Real& average) {
  using namespace QuantLib;
  const auto& fixingDates = c.fixingDates();
  const auto& dt = c.dt();
  const Size n = dt.size();
  const Date today = Settings::instance().evaluationDate();
  const TimeSeries<Real>& history = IndexManager::instance().getHistory(c.index()->name());

  compound = 1.0;
  average = 0.0;
  Size i = 0;
  // Strictly past: the fixing must exist.
  while (i < n && fixingDates[i] < today) {
    const Rate f = history[fixingDates[i]];
    QL_REQUIRE(f != Null<Real>(),
               "missing " << c.index()->name() << " fixing for " << fixingDates[i]);
    compound *= (1.0 + f * dt[i]);
    average += f * dt[i];
    ++i;
  }
  // Today: fixed if published, otherwise fall through and forecast.
  if (i < n && fixingDates[i] == today) {
    const Rate f = history[fixingDates[i]];
    if (f != Null<Real>()) {
      compound *= (1.0 + f * dt[i]);
      average += f * dt[i];
      ++i;
    }
  }
  return i;
}

}  // namespace detail

// ---- Overnight-indexed coupon (any overnight index, any currency) ------------------------------
// Dispatches on `averagingMethod()` -- the coupon's own declaration of its convention.
//
// COMPOUNDED (RateAveraging::Compound). QuantLib's pricer forms
//   compoundFactor = P · DF(v_i)/DF(v_n),   rate = (compoundFactor − 1) / accrualPeriod()
// with P the already-fixed product. Daily compounding telescopes, so the whole forward part is ONE
// sub-period [v_i, v_n] -- and the multiplicative past folds into the generic ADDITIVE form exactly
// via the weight vector:
//   P·X − 1  ==  P·(X − 1) + (P − 1)   =>   weight = P, realized = P − 1.
// With no past fixings P == 1.0 identically, so weight is left empty and realized is exactly 0.0:
// the legacy `ois_float_coupon_pv` shape, bit for bit.
//
// AVERAGED (RateAveraging::Simple). Arithmetic averaging does NOT telescope, so we keep one
// sub-period per business day. QuantLib's forecast term is `index->fixing(fixingDates[j]) * dt[j]`,
// and since dt[j] is the index day count over exactly the fixing's own [v_j, v_{j+1}], the day counts
// cancel: the term IS `DF(v_j)/DF(v_{j+1}) − 1`. Past days fold into `realized` as Σ f·dt.
// NB this is QuantLib's exact (byApprox = false) arithmetic path. A coupon built with
// telescopicValueDates = true selects QuantLib's Takada LOG-approximation instead, which is a
// different model and will not agree -- build averaged coupons non-telescopic.
//
// tau_index is `accrualPeriod()` because that is the denominator BOTH QuantLib pricers divide by. It
// is the period year fraction on the coupon's own day counter, which for a standard overnight coupon
// is the index's day counter (design §5).
inline pricing::RateObservation extract_overnight_obs(const QuantLib::OvernightIndexedCoupon& c,
                                                      const QuantLib::Date& ref,
                                                      const QuantLib::DayCounter& curveDc) {
  using namespace QuantLib;
  const detail::CurveTime t{ref, curveDc};
  const auto& valueDates = c.valueDates();
  const Size n = c.dt().size();
  const Real g = c.gearing();

  Real compound = 1.0, average = 0.0;
  const Size i = detail::overnight_fixed_prefix(c, compound, average);

  pricing::RateObservation o;
  o.tau_index = c.accrualPeriod();
  QL_REQUIRE(o.tau_index > 0.0, "overnight coupon has non-positive accrual period");

  if (c.averagingMethod() == RateAveraging::Compound) {
    o.realized = g * (compound - 1.0);
    if (i < n) {  // forward part left: ONE telescoped sub-period
      o.sub_start.push_back(t(valueDates[i]));
      o.sub_end.push_back(t(valueDates[n]));
      const Real w = g * compound;
      if (w != 1.0) o.weight.push_back(w);  // empty weight == all-ones (the fast path)
    }
  } else {  // RateAveraging::Simple -- per-business-day sub-periods
    o.realized = g * average;
    for (Size j = i; j < n; ++j) {
      o.sub_start.push_back(t(valueDates[j]));
      o.sub_end.push_back(t(valueDates[j + 1]));
    }
    if (g != 1.0) o.weight.assign(n - i, g);
  }
  return o;
}

// ---- IBOR coupon (any tenor, any currency) ----------------------------------------------------
// ONE sub-period over the deposit period underlying the fixing. QuantLib's own forecast is
//   IborIndex::forecastFixing(d1, d2, t) = (DF(d1)/DF(d2) − 1) / t
// with d1 = fixingValueDate(), d2 = fixingEndDate(), t = spanningTime() (see IborCoupon::indexFixing).
// fixingEndDate() is NOT fixingMaturityDate() under QuantLib's default par-coupon approximation --
// reading the coupon's own cached dates is what keeps par-coupon vs indexed-coupon correct here
// instead of us re-deriving the period. tau_index = spanningTime() is on the INDEX's day counter,
// while tau_pay = accrualPeriod() is on the COUPON's -- generally different, which is the whole point.
//
// REQUIRES A PRICER: fixingValueDate()/fixingEndDate()/spanningTime() are computed by the coupon's
// IborCouponPricer. Legs built through QuantLib's own IborLeg / VanillaSwap already have one.
inline pricing::RateObservation extract_ibor_obs(const QuantLib::IborCoupon& c,
                                                 const QuantLib::Date& ref,
                                                 const QuantLib::DayCounter& curveDc) {
  using namespace QuantLib;
  const detail::CurveTime t{ref, curveDc};
  const Real g = c.gearing();

  QL_REQUIRE(!c.isInArrears(),
             "in-arrears IBOR coupon needs a timing-adjustment MODEL, which does not belong in the "
             "engine (design §3/§6)");

  pricing::RateObservation o;
  o.tau_index = c.spanningTime();
  QL_REQUIRE(o.tau_index > 0.0, "IBOR coupon has non-positive index spanning time");

  // Mirrors IborCoupon::indexFixing(): past => history (required), today => history if published
  // else forecast, future => forecast.
  const Date today = Settings::instance().evaluationDate();
  const Date fd = c.fixingDate();
  ext::optional<Rate> fixed;
  if (fd < today || (fd == today && Settings::instance().enforcesTodaysHistoricFixings())) {
    const Rate f = c.index()->pastFixing(fd);
    QL_REQUIRE(f != Null<Real>(), "missing " << c.index()->name() << " fixing for " << fd);
    fixed = f;
  } else if (fd == today) {
    try {
      const Rate f = c.index()->pastFixing(fd);
      if (f != Null<Real>()) fixed = f;
    } catch (Error&) {
      // no fixing: fall through and forecast
    }
  }

  if (fixed) {  // already fixed: a pure constant, no sub-periods (rate == realized/tau_index == g*f)
    o.realized = g * (*fixed) * o.tau_index;
  } else {
    o.sub_start.push_back(t(c.fixingValueDate()));
    o.sub_end.push_back(t(c.fixingEndDate()));
    if (g != 1.0) o.weight.push_back(g);
  }
  return o;
}

// ---- Generic coupon dispatch ------------------------------------------------------------------
// The ONLY type switch in the pipeline. Add a coupon TYPE here; never an index.
inline pricing::FloatCoupon extract_float_coupon(
    const QuantLib::ext::shared_ptr<QuantLib::CashFlow>& cf, const QuantLib::Date& ref,
    const QuantLib::DayCounter& curveDc) {
  using namespace QuantLib;
  pricing::FloatCoupon out;
  const FloatingRateCoupon* base = nullptr;

  if (auto on = ext::dynamic_pointer_cast<OvernightIndexedCoupon>(cf)) {
    out.obs = extract_overnight_obs(*on, ref, curveDc);
    base = on.get();
  } else if (auto ib = ext::dynamic_pointer_cast<IborCoupon>(cf)) {
    out.obs = extract_ibor_obs(*ib, ref, curveDc);
    base = ib.get();
  } else {
    QL_FAIL("unsupported floating coupon type (not an OvernightIndexedCoupon or an IborCoupon)");
  }

  out.pay = curveDc.yearFraction(ref, base->date());
  out.tau_pay = base->accrualPeriod();  // the COUPON's own day count, not the index's
  out.spread = base->spread();          // gearing is already folded into obs (see header note)
  return out;
}

// A whole floating leg. Coupon order is the leg's order and is preserved (residual/W-cache order).
inline std::vector<pricing::FloatCoupon> extract_float_leg(const QuantLib::Leg& leg,
                                                           const QuantLib::Date& ref,
                                                           const QuantLib::DayCounter& curveDc) {
  std::vector<pricing::FloatCoupon> out;
  out.reserve(leg.size());
  for (const auto& cf : leg) out.push_back(extract_float_coupon(cf, ref, curveDc));
  return out;
}

// ---- Fixed leg --------------------------------------------------------------------------------
// The RATE is not stored: it belongs to the instrument/quote (a par-rate annuity and a fixed leg at
// a contract rate are the same data). `tau` is the coupon's own day count (30/360, ACT/365F, ...).
inline std::vector<pricing::FixedCoupon> extract_fixed_leg(const QuantLib::Leg& leg,
                                                           const QuantLib::Date& ref,
                                                           const QuantLib::DayCounter& curveDc) {
  using namespace QuantLib;
  std::vector<pricing::FixedCoupon> out;
  out.reserve(leg.size());
  for (const auto& cf : leg) {
    auto c = ext::dynamic_pointer_cast<FixedRateCoupon>(cf);
    QL_REQUIRE(c, "fixed leg cashflow is not a FixedRateCoupon");
    out.push_back(pricing::FixedCoupon{curveDc.yearFraction(ref, c->date()), c->accrualPeriod()});
  }
  return out;
}

// ---- Futures ----------------------------------------------------------------------------------
// A future is NOT an engine concept: it is a RateObservation plus a convexity NUMBER the caller
// supplies (the convexity MODEL -- Hull-White, Ho-Lee -- lives in tests, design §3/§6). So the
// caller hands us the sub-period date brackets, the realized part and the denominator; we only do
// the date -> curve-time conversion. One sub-period == a compounding / IBOR future; one per business
// day == an averaging future; the difference is the DATA, not a type.
//
// `tau_index` and `realized` are accruals/rates on the INDEX's own day count -- computed by the
// caller from the index it chose, never guessed here.
inline pricing::RateObservation make_observation(
    const std::vector<std::pair<QuantLib::Date, QuantLib::Date>>& sub_periods, double realized,
    double tau_index, const QuantLib::Date& ref, const QuantLib::DayCounter& curveDc,
    const std::vector<double>& weights = {}) {
  using namespace QuantLib;
  QL_REQUIRE(tau_index > 0.0, "observation has non-positive tau_index");
  QL_REQUIRE(weights.empty() || weights.size() == sub_periods.size(),
             "weight vector size must match the sub-period count (empty means all-ones)");
  pricing::RateObservation o;
  o.realized = realized;
  o.tau_index = tau_index;
  o.weight = weights;
  o.sub_start.reserve(sub_periods.size());
  o.sub_end.reserve(sub_periods.size());
  for (const auto& [s, e] : sub_periods) {
    QL_REQUIRE(s < e, "sub-period is not strictly increasing: " << s << " -> " << e);
    o.sub_start.push_back(curveDc.yearFraction(ref, s));
    o.sub_end.push_back(curveDc.yearFraction(ref, e));
  }
  return o;
}

}  // namespace swaps::qlx
