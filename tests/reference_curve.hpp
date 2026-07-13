#pragma once
// Single builder for the reference market's curve + QuantLib instruments, shared by the golden
// generator (tools/gen_golden.cpp) and the pricing/calibration tests. Centralising this is
// deliberate: the earlier bug where a test kept its own stale copy of the knot forwards is exactly
// what this prevents. reference_market.hpp holds the DATA; this holds the QuantLib CONSTRUCTION.

#include <ql/quantlib.hpp>

#include <algorithm>
#include <vector>

#include "reference_market.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/ql/extract.hpp"

namespace swaps::refbuild {

namespace rm = swaps::refmkt;
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

// Build the calibration problem: extract every instrument's schedule from QuantLib, attach its
// market quote (rate units: futures target = 1 - price/100) and Hull-White convexity.
inline swaps::calibration::CalibrationProblem build_problem(const Market& mk) {
  swaps::calibration::CalibrationProblem p;
  p.meeting_times = mk.meeting_times;
  p.back_times = mk.back_times;

  for (std::size_t i = 0; i < mk.swaps.size(); ++i)
    p.swaps.push_back({swaps::qlx::extract_ois_swap(*mk.swaps[i], mk.today, mk.dc),
                       rm::swaps[i].par_rate});

  for (const auto& f : mk.futures) {
    const double conv = mk.convexity(f);
    const double market_rate = 1.0 - f.market_price / 100.0;
    if (f.quarterly)
      p.comp_futs.push_back(
          {swaps::qlx::extract_compounded_future(f.start, f.end, mk.today, mk.dc), conv,
           market_rate});
    else
      p.avg_futs.push_back(
          {swaps::qlx::extract_averaged_future(mk.sofr, f.start, f.end, mk.today, mk.dc), conv,
           market_rate});
  }
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
  for (int i = 0; i < 6; ++i) {
    const auto& q = rm::futures_1m[i];
    const Date s = sofr_start(Month(q.ref_month), q.ref_year, Monthly),
               e = sofr_end(Month(q.ref_month), q.ref_year, Monthly);
    p.avg_futs.push_back(
        {swaps::qlx::extract_averaged_future(mk.sofr, s, e, mk.today, mk.dc), 0.0, 1.0 - q.price / 100.0});
    front.push_back(t(e));
  }
  for (int i = 0; i < 8; ++i) {
    const auto& q = rm::futures_3m[i];
    const Date s = sofr_start(Month(q.ref_month), q.ref_year, Quarterly),
               e = sofr_end(Month(q.ref_month), q.ref_year, Quarterly);
    p.comp_futs.push_back(
        {swaps::qlx::extract_compounded_future(s, e, mk.today, mk.dc), 0.0, 1.0 - q.price / 100.0});
    back.push_back(t(e));
  }
  for (std::size_t i = 0; i < rm::swaps.size(); ++i) {
    p.swaps.push_back({swaps::qlx::extract_ois_swap(*mk.swaps[i], mk.today, mk.dc), rm::swaps[i].par_rate});
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
