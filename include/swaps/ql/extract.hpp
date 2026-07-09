#pragma once
// QuantLib -> plain-data cashflow schedules.
//
// This is the ONLY layer that touches QuantLib instrument internals. It runs once at setup: it
// reads the exact coupon accrual/payment dates and accrual factors that QuantLib itself would use,
// converts them to curve-time year fractions, and hands them to the templated kernel
// (swaps/pricing/cashflows.hpp). Calendars, schedules and day counts are QuantLib's job here — we
// never reimplement them (CLAUDE.md §1).

#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/overnightindexedcoupon.hpp>
#include <ql/indexes/iborindex.hpp>
#include <ql/indexes/indexmanager.hpp>
#include <ql/instruments/overnightindexedswap.hpp>
#include <ql/settings.hpp>
#include <ql/time/daycounter.hpp>

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

}  // namespace swaps::qlx
