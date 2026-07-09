#pragma once
// The reference market: the single, committed definition of the calibration problem.
// Deliberately free of QuantLib types so both the oracle generator (tools/gen_golden.cpp)
// and the engine's own tests consume exactly the same inputs.
//
// Curve structure (CLAUDE.md §2), mirroring a real SOFR build:
//   front knots = 6 FOMC meeting dates                 -> forwards flat BETWEEN meetings
//   back  knots = { 3M-futures END dates > last meeting }
//                 U { par-swap maturity dates }         -> C2 spline on forwards
//
// Instrument strip is SEQUENTIAL, not overlapping:
//   - 12 x 1M SOFR futures (arithmetic average), starting with the CURRENT month's contract,
//     covering the first year. The first contract's accrual begins before the evaluation date,
//     so gen_golden seeds the elapsed SOFR fixings.
//   - 8 x 3M SOFR futures (compounded, IMM), the first starting on/after the last 1M contract's
//     end date (2027-07-01) so the strips do NOT overlap, continuing until a contract's end date
//     passes the 3y swap maturity. There is a deliberate ~2.5-month gap between the strips; the
//     spline interpolates through it.
//   - 9 x par OIS swaps, 4y..30y (the futures cover everything shorter).
//
// The futures OVER-SPECIFY the knots on purpose, so the least-squares fit has a NON-ZERO residual
// at the optimum. Never assert exact repricing. 29 instruments vs 23 knots.

#include <array>
#include <cmath>

namespace swaps::refmkt {

struct Ymd {
  int y, m, d;
};

// Fixed evaluation date. Never read the system clock — golden data must be reproducible.
inline constexpr Ymd evaluation_date{2026, 7, 8};

// ---- Front end: FOMC meeting *effective* dates (the day the new policy rate applies) ----
// The instantaneous forward is flat on (m[k-1], m[k]]; it jumps only here.
// Note the last meeting (2027-03-18) is the day after the 3rd-Wednesday IMM date (2027-03-17),
// which is why the Dec-2026 3M contract's end date falls just *inside* the front region.
inline constexpr std::array<Ymd, 6> meeting_dates{{
    {2026, 7, 30},
    {2026, 9, 17},
    {2026, 11, 5},
    {2026, 12, 17},
    {2027, 1, 28},
    {2027, 3, 18},
}};

// ---- Futures ----
// price = 100 * (1 - (forward_rate + convexity_adjustment))   [QuantLib's convention:
//         OvernightIndexFuture does  R = convexityAdjustment() + rate();  NPV = 100*(1-R)]
struct FutureQuote {
  int ref_year;
  int ref_month;  // 1..12; for quarterly this is the IMM month (Mar/Jun/Sep/Dec)
  bool quarterly;
  double price;
};

// 12 consecutive monthly contracts, starting with the current month (Jul-2026). The Jul contract
// straddles the evaluation date; gen_golden seeds the elapsed fixings so QuantLib can price it.
// Prices trace an easing policy path (rising price = falling rate).
inline constexpr std::array<FutureQuote, 12> futures_1m{{
    {2026, 7, false, 95.68},  {2026, 8, false, 95.75},  {2026, 9, false, 95.85},
    {2026, 10, false, 95.95}, {2026, 11, false, 96.05}, {2026, 12, false, 96.15},
    {2027, 1, false, 96.25},  {2027, 2, false, 96.33},  {2027, 3, false, 96.40},
    {2027, 4, false, 96.46},  {2027, 5, false, 96.50},  {2027, 6, false, 96.53},
}};

// 8 IMM quarters, Sep-2027..Jun-2029, non-overlapping with the 1M strip (first start >= the last
// 1M end date, 2027-07-01) and running past the 3y swap end. Every end date is a back knot.
inline constexpr std::array<FutureQuote, 8> futures_3m{{
    {2027, 9, true, 96.55},  {2027, 12, true, 96.58}, {2028, 3, true, 96.60},
    {2028, 6, true, 96.62},  {2028, 9, true, 96.63},  {2028, 12, true, 96.63},
    {2029, 3, true, 96.62},  {2029, 6, true, 96.60},
}};

// ---- Back end: par OIS swaps from 4y (the 3M strip covers everything shorter) ----
struct SwapQuote {
  int tenor_years;
  double par_rate;  // decimal, e.g. 0.0355 = 3.55%
};

inline constexpr std::array<SwapQuote, 9> swaps{{
    {4, 0.0352},  {5, 0.0355},  {7, 0.0362},  {10, 0.0372}, {12, 0.0378},
    {15, 0.0385}, {20, 0.0390}, {25, 0.0388}, {30, 0.0383},
}};

// ---- Futures convexity: Hull-White (volatility AND mean reversion) ----
// Ho-Lee (0.5*sigma^2*t1*t2) is the a -> 0 limit of this and grows like t^2 without bound.
// With the strip now reaching ~3.2y the mean-reversion term matters, so we use Hull-White.
//
// This is the classic Eurodollar (term-rate) futures bias. Applying it to SOFR
// averaged/compounded overnight futures is a first-order approximation, standard in practice.
// Transcribed from QuantLib's HullWhite::convexityBias; tests/convexity_test.cpp checks the two
// agree, so the engine itself never has to link QuantLib.
inline constexpr double convexity_sigma = 0.0075;          // 75 bp/yr normal vol
inline constexpr double convexity_mean_reversion = 0.03;   // 3% mean reversion

inline double hull_white_convexity(double futures_price, double t1, double t2) {
  const double a = convexity_mean_reversion, s = convexity_sigma;
  const double dt = t2 - t1;
  const double B = (1.0 - std::exp(-a * dt)) / a;
  const double Bt = (1.0 - std::exp(-a * t1)) / a;
  const double half_sigma_sq = 0.5 * s * s;
  const double lambda = half_sigma_sq * (1.0 - std::exp(-2.0 * a * t1)) / a * B * B;
  const double phi = half_sigma_sq * B * Bt * Bt;
  const double z = lambda + phi;
  const double future_rate = (100.0 - futures_price) / 100.0;
  return (1.0 - std::exp(-z)) * (future_rate + 1.0 / dt);
}

// ---- Reference knot forwards (ARBITRARY BUT FIXED; not a calibrated solution) ----
// Changing either array invalidates every committed golden file.
inline constexpr std::array<double, 6> reference_front_forwards{
    0.0428, 0.0415, 0.0400, 0.0385, 0.0372, 0.0360};

// 8 knots from 3M-futures end dates (t ~ 1.44 .. 3.20), then 9 from swap maturities (4y..30y).
inline constexpr std::array<double, 17> reference_back_forwards{
    0.0345, 0.0342, 0.0340, 0.0339, 0.0338, 0.0338, 0.0339, 0.0341,   // futures region
    0.0345, 0.0352, 0.0362, 0.0375, 0.0382, 0.0390, 0.0396, 0.0392, 0.0385};  // swap region

// ---- Problem dimensions ----
// The number of back knots is determined by the CALENDAR (how many 3M end dates fall after the
// last meeting date), so gen_golden asserts it at runtime rather than here.
inline constexpr int n_front_knots = static_cast<int>(meeting_dates.size());          // 6
inline constexpr int n_back_knots = static_cast<int>(reference_back_forwards.size()); // 17
inline constexpr int n_knots = n_front_knots + n_back_knots;                          // 23
inline constexpr int n_instruments =
    static_cast<int>(futures_1m.size() + futures_3m.size() + swaps.size());           // 29

static_assert(n_instruments > n_knots,
              "The calibration must be OVER-determined: more instruments than knot forwards. "
              "If this ever becomes square, the 'never assert exact repricing' rule silently "
              "stops being meaningful.");
static_assert(reference_front_forwards.size() == meeting_dates.size());

}  // namespace swaps::refmkt
