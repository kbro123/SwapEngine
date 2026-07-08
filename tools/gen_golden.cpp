// gen_golden — produce committed reference data for the correctness gate.
//
// Three kinds of file, deliberately kept distinct. Do not blur them:
//
//   1. ORACLE (QuantLib is authoritative; our engine must reproduce it to rel<=1e-10)
//        golden/interp_flat.csv    QuantLib BackwardFlat interpolation: value(t) and primitive(t)
//        golden/interp_cubic.csv   QuantLib natural-cubic interpolation: value(t) and primitive(t)
//      These pin down our two interpolators AND their analytic integrals independently of any
//      curve or instrument machinery. We use the interpolation objects directly, never
//      YieldTermStructure::forwardRate(t,t,...), which is a finite difference.
//
//   2. HYBRID ORACLE (QuantLib prices instruments off OUR discount factors, via ql_adapter.hpp)
//        golden/instrument_quotes.csv
//      QuantLib owns the instrument conventions (SOFR arithmetic-average vs compounded accrual,
//      IMM schedules, day counts); we own the curve. Disagreement => our pricing bug.
//
//   3. REGRESSION LOCK (our own output, committed so refactors cannot silently change it)
//        golden/composite_curve.csv
//
// This program does NOT calibrate. The calibration is over-determined (CLAUDE.md §2) and
// QuantLib's GlobalBootstrap throws on a non-zero residual. The knot forwards below are
// arbitrary but fixed; changing them invalidates every file above.

#include <ql/quantlib.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "reference_market.hpp"
#include "ql_adapter.hpp"
#include "swaps/curve/two_region_forward_curve.hpp"

using namespace QuantLib;
namespace rm = swaps::refmkt;
using Curve = swaps::curve::TwoRegionForwardCurve<double>;

namespace {

constexpr int kPrec = 17;

Date to_ql(const rm::Ymd& d) { return Date(d.d, static_cast<Month>(d.m), d.y); }

// Mirrors QuantLib's anonymous-namespace getValidSofrStart/getValidSofrEnd so our accrual dates,
// and hence the convexity we compute from them, match the library exactly.
Date sofr_start(Month m, Year y, Frequency f) {
  return f == Monthly ? UnitedStates(UnitedStates::GovernmentBond).adjust(Date(1, m, y))
                      : Date::nthWeekday(3, Wednesday, m, y);
}
Date sofr_end(Month m, Year y, Frequency f) {
  if (f == Monthly) {
    Calendar cal = UnitedStates(UnitedStates::GovernmentBond);
    return cal.advance(cal.endOfMonth(Date(1, m, y)), 1 * Days);
  }
  Date d = sofr_start(m, y, f) + Period(f);
  return Date::nthWeekday(3, Wednesday, d.month(), d.year());
}

std::string iso(const Date& d) {
  std::ostringstream os;
  os << d.year() << '-' << std::setw(2) << std::setfill('0') << static_cast<int>(d.month()) << '-'
     << std::setw(2) << std::setfill('0') << d.dayOfMonth();
  return os.str();
}

// ARBITRARY BUT FIXED. 6 front (one flat forward per meeting segment) + 11 back knot forwards.
const std::vector<double> kFront{0.0428, 0.0415, 0.0400, 0.0385, 0.0372, 0.0360};
const std::vector<double> kBack{0.0350, 0.0345, 0.0350, 0.0355, 0.0365, 0.0380,
                                0.0388, 0.0395, 0.0398, 0.0392, 0.0385};

}  // namespace

int main(int argc, char** argv) {
  const std::string out = argc > 1 ? argv[1] : "tests/golden";

  const Date today = to_ql(rm::evaluation_date);
  Settings::instance().evaluationDate() = today;
  const DayCounter dc = Actual365Fixed();
  auto T = [&](const Date& d) { return dc.yearFraction(today, d); };

  RelinkableHandle<YieldTermStructure> h;
  auto sofr = ext::make_shared<Sofr>(h);

  // ---- Pass 1: swap maturities (schedule only; independent of the curve) --------------------
  h.linkTo(ext::make_shared<FlatForward>(today, 0.03, dc, Continuous));
  std::vector<ext::shared_ptr<OvernightIndexedSwap>> oisSwaps;
  std::vector<double> backTimes;
  for (const auto& s : rm::swaps) {
    ext::shared_ptr<OvernightIndexedSwap> p =
        MakeOIS(Period(s.tenor_years, Years), sofr, 0.03).withDiscountingTermStructure(h);
    oisSwaps.push_back(p);
    backTimes.push_back(T(p->maturityDate()));
  }

  std::vector<double> meetingTimes;
  for (const auto& m : rm::meeting_dates) meetingTimes.push_back(T(to_ql(m)));

  // ---- Our composite curve ------------------------------------------------------------------
  Curve curve(meetingTimes, backTimes);
  std::vector<double> x = kFront;
  x.insert(x.end(), kBack.begin(), kBack.end());
  curve.set_forwards(x);

  const double Tjoin = curve.join_time(), Tmax = curve.max_time();
  std::cout << "evaluation date : " << iso(today) << "\n"
            << "front knots     : " << curve.n_front() << " (join at t=" << Tjoin << ")\n"
            << "back knots      : " << curve.n_back() << " (out to t=" << Tmax << ")\n"
            << "free forwards   : " << curve.n_knots() << "\n"
            << "instruments     : " << rm::n_instruments << "  (OVER-determined)\n\n";

  // ---- 1a. ORACLE: BackwardFlat interpolation over the front knots ---------------------------
  // Abscissae [0, m_1..m_6]; BackwardFlat puts the value on (x_{i-1}, x_i] at y_i, so y_0 (the
  // value exactly at t=0) is set to f_1.
  {
    std::vector<Real> xs{0.0}, ys{kFront.front()};
    for (std::size_t i = 0; i < meetingTimes.size(); ++i) {
      xs.push_back(meetingTimes[i]);
      ys.push_back(kFront[i]);
    }
    Interpolation flat = BackwardFlat().interpolate(xs.begin(), xs.end(), ys.begin());
    flat.enableExtrapolation();

    std::ofstream f(out + "/interp_flat.csv");
    f << std::setprecision(kPrec) << "# t,value,primitive   (QuantLib BackwardFlat; ORACLE)\n";
    for (int i = 0; i <= 200; ++i) {
      const double t = Tjoin * i / 200.0;
      f << t << ',' << flat(t) << ',' << flat.primitive(t) << '\n';
    }
    std::cout << "  wrote " << out << "/interp_flat.csv\n";
  }

  // ---- 1b. ORACLE: natural cubic interpolation over the spline abscissae ---------------------
  // Spline knots are (T, f_last) then the back knots — the join value is pinned, not free.
  {
    std::vector<Real> xs{Tjoin}, ys{kFront.back()};
    for (std::size_t i = 0; i < backTimes.size(); ++i) {
      xs.push_back(backTimes[i]);
      ys.push_back(kBack[i]);
    }
    const Cubic natural(CubicInterpolation::Spline, /*monotonic*/ false,
                        CubicInterpolation::SecondDerivative, 0.0,
                        CubicInterpolation::SecondDerivative, 0.0);
    Interpolation cub = natural.interpolate(xs.begin(), xs.end(), ys.begin());
    cub.enableExtrapolation();

    std::ofstream f(out + "/interp_cubic.csv");
    f << std::setprecision(kPrec)
      << "# t,value,primitive_from_join   (QuantLib natural cubic; ORACLE)\n";
    const Real p0 = cub.primitive(Tjoin);
    for (int i = 0; i <= 400; ++i) {
      const double t = Tjoin + (Tmax - Tjoin) * i / 400.0;
      f << t << ',' << cub(t) << ',' << (cub.primitive(t) - p0) << '\n';
    }
    std::cout << "  wrote " << out << "/interp_cubic.csv\n";
  }

  // ---- 2. HYBRID ORACLE: QuantLib prices instruments off OUR curve ---------------------------
  auto ts = ext::make_shared<swaps::qlx::TwoRegionTermStructure>(today, dc, &curve);
  ts->enableExtrapolation();
  h.linkTo(ts);
  {
    std::ofstream f(out + "/instrument_quotes.csv");
    f << std::setprecision(kPrec)
      << "# kind,label,start,end,convexity,implied_quote   (QuantLib pricing off OUR curve)\n";

    auto emit_future = [&](const rm::FutureQuote& q) {
      const Frequency freq = q.quarterly ? Quarterly : Monthly;
      const Month mon = static_cast<Month>(q.ref_month);
      const Date s = sofr_start(mon, q.ref_year, freq), e = sofr_end(mon, q.ref_year, freq);
      const double conv = rm::convexity_adjustment(T(s), T(e));
      OvernightIndexFuture fut(sofr, s, e, Handle<Quote>(ext::make_shared<SimpleQuote>(conv)),
                               q.quarterly ? RateAveraging::Compound : RateAveraging::Simple);
      f << (q.quarterly ? "future_3m," : "future_1m,") << q.ref_year << '-' << q.ref_month << ','
        << iso(s) << ',' << iso(e) << ',' << conv << ',' << fut.NPV() << '\n';
    };
    for (const auto& q : rm::futures_1m) emit_future(q);
    for (const auto& q : rm::futures_3m) emit_future(q);

    for (std::size_t i = 0; i < oisSwaps.size(); ++i) {
      oisSwaps[i]->deepUpdate();  // the handle was relinked after construction
      f << "par_swap," << rm::swaps[i].tenor_years << "y," << iso(oisSwaps[i]->startDate()) << ','
        << iso(oisSwaps[i]->maturityDate()) << ",0," << oisSwaps[i]->fairRate() << '\n';
    }
    std::cout << "  wrote " << out << "/instrument_quotes.csv\n";
  }

  // ---- 3. REGRESSION LOCK: our composite curve ----------------------------------------------
  {
    std::ofstream f(out + "/composite_curve.csv");
    f << std::setprecision(kPrec);
    f << "# knot,kind,time,forward\n";
    for (int i = 0; i < curve.n_front(); ++i)
      f << "knot,front," << meetingTimes[i] << ',' << kFront[i] << '\n';
    for (int i = 0; i < curve.n_back(); ++i)
      f << "knot,back," << backTimes[i] << ',' << kBack[i] << '\n';

    f << "# grid,time,forward,integral,discount\n";
    for (int i = 0; i <= 600; ++i) {
      const double t = Tmax * i / 600.0;
      f << "grid," << t << ',' << curve.forward(t) << ',' << curve.integral(t) << ','
        << curve.discount(t) << '\n';
    }
    std::cout << "  wrote " << out << "/composite_curve.csv\n";
  }

  std::cout << "\ngolden data written to " << out << "\n";
  return 0;
}
