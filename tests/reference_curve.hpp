#pragma once
// Single builder for the reference market's curve + QuantLib instruments, shared by the golden
// generator (tools/gen_golden.cpp) and the pricing/calibration tests. Centralising this is
// deliberate: the earlier bug where a test kept its own stale copy of the knot forwards is exactly
// what this prevents. reference_market.hpp holds the DATA; this holds the QuantLib CONSTRUCTION.

#include <ql/quantlib.hpp>

#include <algorithm>
#include <utility>
#include <vector>

#include "reference_market.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/ql/extract.hpp"

namespace swaps::refbuild {

namespace rm = swaps::refmkt;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
using Curve = swaps::curve::CalibrationCurve<double>;

inline QuantLib::Date to_ql(const rm::Ymd& d) {
  return QuantLib::Date(d.d, static_cast<QuantLib::Month>(d.m), d.y);
}

// Mirrors QuantLib's anonymous-namespace getValidSofrStart/getValidSofrEnd.
inline QuantLib::Date sofr_start(QuantLib::Month m, QuantLib::Year y, QuantLib::Frequency f) {
  using namespace QuantLib;
  return f == Monthly ? UnitedStates(UnitedStates::GovernmentBond).adjust(Date(1, m, y))
                      : Date::nthWeekday(3, Wednesday, m, y);
}
inline QuantLib::Date sofr_end(QuantLib::Month m, QuantLib::Year y, QuantLib::Frequency f) {
  using namespace QuantLib;
  if (f == Monthly) {
    Calendar cal = UnitedStates(UnitedStates::GovernmentBond);
    return cal.advance(cal.endOfMonth(Date(1, m, y)), 1 * Days);
  }
  Date d = sofr_start(m, y, f) + Period(f);
  return Date::nthWeekday(3, Wednesday, d.month(), d.year());
}

struct Future {
  QuantLib::Date start, end;
  bool quarterly;
  double market_price;
};

struct Market {
  QuantLib::Date today;
  QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
  QuantLib::ext::shared_ptr<QuantLib::Sofr> sofr;
  std::vector<double> meeting_times;
  std::vector<double> back_times;                       // 3M end dates (>last mtg) U swap maturities
  std::vector<QuantLib::ext::shared_ptr<QuantLib::OvernightIndexedSwap>> swaps;
  std::vector<Future> futures;                          // 12 x 1M then 8 x 3M
  double convexity(const Future& f) const {
    return rm::hull_white_convexity(f.market_price, dc.yearFraction(today, f.start),
                                    dc.yearFraction(today, f.end));
  }
};

// Builds instruments off `h`; the caller then links `h` to the curve wrapper and deepUpdates.
inline Market build_market(QuantLib::RelinkableHandle<QuantLib::YieldTermStructure>& h) {
  using namespace QuantLib;
  Market mk;
  mk.today = to_ql(rm::evaluation_date);
  Settings::instance().evaluationDate() = mk.today;
  mk.sofr = ext::make_shared<Sofr>(h);

  // Seed elapsed SOFR fixings so the current-month 1M contract can price.
  for (Date d = mk.today - 20; d <= mk.today; ++d)
    if (mk.sofr->isValidFixingDate(d)) mk.sofr->addFixing(d, 0.0430);

  auto t = [&](const Date& d) { return mk.dc.yearFraction(mk.today, d); };

  // Placeholder curve so MakeOIS/schedules can build; relinked by the caller afterwards.
  h.linkTo(ext::make_shared<FlatForward>(mk.today, 0.03, mk.dc, Continuous));

  for (const auto& m : rm::meeting_dates) mk.meeting_times.push_back(t(to_ql(m)));

  std::vector<Date> backDates;
  for (const auto& s : rm::swaps) {
    ext::shared_ptr<OvernightIndexedSwap> p =
        MakeOIS(Period(s.tenor_years, Years), mk.sofr, 0.03).withDiscountingTermStructure(h);
    mk.swaps.push_back(p);
    backDates.push_back(p->maturityDate());
  }

  const Date lastMeeting = to_ql(rm::meeting_dates.back());
  for (const auto& q : rm::futures_1m) {
    const Month m = static_cast<Month>(q.ref_month);
    mk.futures.push_back({sofr_start(m, q.ref_year, Monthly), sofr_end(m, q.ref_year, Monthly),
                          false, q.price});
  }
  for (const auto& q : rm::futures_3m) {
    const Month m = static_cast<Month>(q.ref_month);
    const Date s = sofr_start(m, q.ref_year, Quarterly), e = sofr_end(m, q.ref_year, Quarterly);
    mk.futures.push_back({s, e, true, q.price});
    if (e > lastMeeting) backDates.push_back(e);
  }

  std::sort(backDates.begin(), backDates.end());
  backDates.erase(std::unique(backDates.begin(), backDates.end()), backDates.end());
  for (const Date& d : backDates) mk.back_times.push_back(t(d));

  QL_REQUIRE(static_cast<int>(mk.back_times.size()) == rm::n_back_knots,
             "back-knot count " << mk.back_times.size() << " != reference_back_forwards ("
                                << rm::n_back_knots << "); the calendar moved.");
  return mk;
}

// ---- Generic-instrument construction (design §3 adoption) --------------------------------------
// The reference market is built as generic `Instrument`s -- ONE pipeline for swaps and both futures
// flavours (design §3/§5). No engine call site names an index; the index/market knowledge (the SOFR
// calendar walk for the 1M averaging future) lives HERE in the test builder (CLAUDE.md §1).

// A reference OIS swap as a generic ParRate instrument: both QuantLib legs -> the generic coupon
// model via the coupon-type-dispatch extractors. Replaces extract_ois_swap. On this market the
// overnight coupons' valueDates() endpoints coincide with their accrual dates (asserted in the
// migration guard test), so this reprices the legacy OisSwap form bit-for-bit.
inline cal::Instrument swap_instrument(const Market& mk, std::size_t i) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd.coupons = swaps::qlx::extract_float_leg(mk.swaps[i]->overnightLeg(), mk.today, mk.dc);
  ins.fixed.coupons = swaps::qlx::extract_fixed_leg(mk.swaps[i]->fixedLeg(), mk.today, mk.dc);
  ins.market = rm::swaps[i].par_rate;
  return ins;
}

// The 1M arithmetic-average future's observation. The per-business-day walk (past days fold into
// `realized`, future days become sub-periods) is MARKET construction -- the calendar and history come
// from the index object, not hard-coded -- so it belongs in this test builder, not the engine. It
// mirrors the retired extract_averaged_future exactly; make_observation does the curve-time mapping.
inline px::RateObservation avg_future_obs(const Market& mk, const Future& f) {
  using namespace QuantLib;
  const Calendar fcal = mk.sofr->fixingCalendar();
  const DayCounter idc = mk.sofr->dayCounter();  // the index's own accrual day count
  const TimeSeries<Real>& history = IndexManager::instance().getHistory(mk.sofr->name());
  // Mirror QuantLib 1.35 OvernightIndexFuture::averagedRate() EXACTLY (it was refined in 1.35 vs 1.34):
  //   * the fixing for accrual day [d1,d2) is observed at fixingDate = adjust(d1, Preceding) -- i.e. the
  //     rate is accrued from d1 even when its fixing date is earlier (matters when d1 is a holiday);
  //   * the last day's accrual is capped at min(d2, maturity) (d2 can overshoot if maturity is a holiday);
  //   * a forecast day contributes forward(fixingDate,d2,Simple)*accr = (DF(fixingDate)/DF(d2)-1) * accr/
  //     yf(fixingDate,d2), which our engine reproduces as a [fixingDate,d2] sub-period with that weight
  //     (weight == 1.0 for the common interior day, so the empty-weight fast path and the fully-forecast
  //     futures stay bit-identical to before).
  std::vector<std::pair<Date, Date>> subs;
  std::vector<double> weights;
  double realized = 0.0;
  Date fixingDate = fcal.adjust(f.start, Preceding);
  for (Date d1 = f.start; d1 < f.end;) {
    const Date d2 = fcal.advance(d1, 1, Days);
    const Date d2cap = std::min(d2, f.end);
    const double accr = idc.yearFraction(d1, d2cap);
    Real fx = history[fixingDate];
    const bool past = fixingDate < mk.today || (fixingDate == mk.today && fx != Null<Real>());
    if (past) {
      QL_REQUIRE(fx != Null<Real>(), "missing " << mk.sofr->name() << " fixing on " << fixingDate);
      realized += fx * accr;
    } else {
      subs.emplace_back(fixingDate, d2);
      weights.push_back(accr / idc.yearFraction(fixingDate, d2));
    }
    fixingDate = d1 = d2;
  }
  // Drop an all-ones weight vector so standard (fully-forecast, business-day) futures keep the empty-weight
  // fast path and remain bit-for-bit identical.
  bool all_one = true;
  for (double w : weights) if (w != 1.0) { all_one = false; break; }
  if (all_one) weights.clear();
  return swaps::qlx::make_observation(subs, realized, idc.yearFraction(f.start, f.end), mk.today, mk.dc,
                                      weights);
}

// A future as a generic Rate instrument. Quarterly (3M compounding IMM) => ONE sub-period [start,end]
// (daily compounding telescopes); monthly (1M averaging) => one sub-period per business day. Convexity
// is the caller's Hull-White NUMBER (design §3) -- the square setup passes 0.
inline cal::Instrument future_instrument(const Market& mk, const Future& f, double convexity) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  ins.convexity = convexity;
  ins.market = 1.0 - f.market_price / 100.0;
  if (f.quarterly)
    ins.obs = swaps::qlx::make_observation({{f.start, f.end}}, 0.0,
                                           mk.sofr->dayCounter().yearFraction(f.start, f.end),
                                           mk.today, mk.dc);
  else
    ins.obs = avg_future_obs(mk, f);
  return ins;
}

// Build the calibration problem as generic Instruments. Residual order is preserved as
// avg | comp | swaps (CLAUDE.md §2): the 1M averaging futures first, then the 3M compounding futures,
// then the swaps, all in the ONE instrument list (the legacy avg_futs/comp_futs/swaps groups stay
// empty, so the overall residual sequence is byte-identical to the legacy build).
inline swaps::calibration::CalibrationProblem build_problem(const Market& mk) {
  swaps::calibration::CalibrationProblem p;
  p.meeting_times = mk.meeting_times;
  p.back_times = mk.back_times;
  for (const auto& f : mk.futures)
    if (!f.quarterly) p.instruments.push_back(future_instrument(mk, f, mk.convexity(f)));
  for (const auto& f : mk.futures)
    if (f.quarterly) p.instruments.push_back(future_instrument(mk, f, mk.convexity(f)));
  for (std::size_t i = 0; i < mk.swaps.size(); ++i) p.instruments.push_back(swap_instrument(mk, i));
  return p;
}

// The reference (arbitrary-but-fixed) curve.
inline Curve reference_curve(const Market& mk) {
  Curve c = swaps::curve::make_calibration_curve<double>(mk.meeting_times, mk.back_times);
  std::vector<double> x(rm::reference_front_forwards.begin(), rm::reference_front_forwards.end());
  x.insert(x.end(), rm::reference_back_forwards.begin(), rm::reference_back_forwards.end());
  c.set_forwards(x);
  return c;
}

// ---- Square setup: 6x1M + 8x3M futures + 9 swaps = 23 instruments, 23 knots at the instrument
// pillar dates, zero convexity. Shared by the curve-build and risk benchmarks/tests so this exact
// problem lives in ONE place. `build_square_ql` fills `quotes` with the SimpleQuotes so a caller can
// bump them and force a QuantLib re-bootstrap.
inline swaps::calibration::CalibrationProblem build_square_problem(const Market& mk) {
  using namespace QuantLib;
  swaps::calibration::CalibrationProblem p;
  auto t = [&](const Date& d) { return mk.dc.yearFraction(mk.today, d); };
  std::vector<double> front, back;
  // Residual order avg | comp | swaps, all as generic Rate/ParRate instruments, zero convexity to
  // match QuantLib's SofrFutureRateHelper (GlobalBootstrap square problem).
  for (int i = 0; i < 6; ++i) {
    const auto& q = rm::futures_1m[i];
    const Date s = sofr_start(Month(q.ref_month), q.ref_year, Monthly),
               e = sofr_end(Month(q.ref_month), q.ref_year, Monthly);
    p.instruments.push_back(future_instrument(mk, Future{s, e, false, q.price}, 0.0));
    front.push_back(t(e));
  }
  for (int i = 0; i < 8; ++i) {
    const auto& q = rm::futures_3m[i];
    const Date s = sofr_start(Month(q.ref_month), q.ref_year, Quarterly),
               e = sofr_end(Month(q.ref_month), q.ref_year, Quarterly);
    p.instruments.push_back(future_instrument(mk, Future{s, e, true, q.price}, 0.0));
    back.push_back(t(e));
  }
  for (std::size_t i = 0; i < rm::swaps.size(); ++i) {
    p.instruments.push_back(swap_instrument(mk, i));
    back.push_back(t(mk.swaps[i]->maturityDate()));
  }
  std::sort(front.begin(), front.end());
  std::sort(back.begin(), back.end());
  p.meeting_times = front;
  p.back_times = back;
  return p;
}

inline std::vector<QuantLib::ext::shared_ptr<QuantLib::RateHelper>> build_square_ql(
    const Market& mk, std::vector<QuantLib::ext::shared_ptr<QuantLib::SimpleQuote>>& quotes) {
  using namespace QuantLib;
  std::vector<ext::shared_ptr<RateHelper>> helpers;
  auto fut = [&](const rm::FutureQuote& q, Frequency f) {
    auto sq = ext::make_shared<SimpleQuote>(q.price);
    quotes.push_back(sq);
    helpers.push_back(
        ext::make_shared<SofrFutureRateHelper>(Handle<Quote>(sq), Month(q.ref_month), q.ref_year, f));
  };
  for (int i = 0; i < 6; ++i) fut(rm::futures_1m[i], Monthly);
  for (int i = 0; i < 8; ++i) fut(rm::futures_3m[i], Quarterly);
  for (const auto& s : rm::swaps) {
    auto sq = ext::make_shared<SimpleQuote>(s.par_rate);
    quotes.push_back(sq);
    helpers.push_back(
        ext::make_shared<OISRateHelper>(2, Period(s.tenor_years, Years), Handle<Quote>(sq), mk.sofr));
  }
  return helpers;
}

}  // namespace swaps::refbuild
