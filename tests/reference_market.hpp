#pragma once
// The reference market: the single, committed definition of the calibration problem.
// Deliberately free of QuantLib types so both the oracle generator (tools/gen_golden.cpp)
// and the engine's own tests consume exactly the same inputs.
//
// Curve structure (CLAUDE.md §2):
//   knots = union(6 FOMC meeting dates, 11 par-swap maturity dates)  -> 17 free forwards
//   front (<= last meeting date): forwards flat BETWEEN meeting dates; no instrument
//                                 matures on a meeting date
//   back  (>  last meeting date): C2-smooth forwards; knots ARE the swap maturities
//
// Instruments: 1M SOFR futures (arithmetic average) + 3M SOFR futures (compounded, IMM)
//              + par OIS swaps.  The futures OVER-SPECIFY the 6 front knots on purpose,
//              so the least-squares fit has a NON-ZERO residual at the optimum.
//              Never assert exact repricing.

#include <array>

namespace swaps::refmkt {

struct Ymd {
  int y, m, d;
};

// Fixed evaluation date. Never read the system clock — golden data must be reproducible.
inline constexpr Ymd evaluation_date{2026, 7, 8};

// ---- Front end: FOMC meeting *effective* dates (the day the new policy rate applies) ----
// The instantaneous forward is flat on (m[k-1], m[k]]; it jumps only here.
inline constexpr std::array<Ymd, 6> meeting_dates{{
    {2026, 7, 30},
    {2026, 9, 17},
    {2026, 11, 5},
    {2026, 12, 17},
    {2027, 1, 28},
    {2027, 3, 18},
}};

// ---- Futures (front end, deliberately over-provided) ----
// price = 100 * (1 - (forward_rate + convexity_adjustment))   [QuantLib's convention]
// `quarterly=false` -> 1M contract, arithmetic average of daily SOFR over the calendar month.
// `quarterly=true`  -> 3M IMM contract, compounded daily SOFR over the IMM period.
struct FutureQuote {
  int ref_year;
  int ref_month;      // 1..12; for quarterly this is the IMM month (Mar/Jun/Sep/Dec)
  bool quarterly;
  double price;
};

// 1M contracts start at the first of their reference month, so we begin at Aug-2026
// (a Jul-2026 contract would already have started before the 8-Jul evaluation date and
// would require historical fixings).
inline constexpr std::array<FutureQuote, 11> futures_1m{{
    {2026, 8, false, 95.72},
    {2026, 9, false, 95.75},
    {2026, 10, false, 95.83},
    {2026, 11, false, 95.90},
    {2026, 12, false, 96.01},
    {2027, 1, false, 96.10},
    {2027, 2, false, 96.20},
    {2027, 3, false, 96.28},
    {2027, 4, false, 96.37},
    {2027, 5, false, 96.44},
    {2027, 6, false, 96.50},
}};

inline constexpr std::array<FutureQuote, 3> futures_3m{{
    {2026, 9, true, 95.78},
    {2026, 12, true, 96.03},
    {2027, 3, true, 96.30},
}};

// ---- Back end: par OIS swaps. Their maturities ARE the back-end knots. ----
struct SwapQuote {
  int tenor_years;
  double par_rate;  // decimal, e.g. 0.0355 = 3.55%
};

inline constexpr std::array<SwapQuote, 11> swaps{{
    {2, 0.0355},  {3, 0.0350},  {4, 0.0352},  {5, 0.0355},
    {7, 0.0362},  {10, 0.0372}, {12, 0.0378}, {15, 0.0385},
    {20, 0.0390}, {25, 0.0388}, {30, 0.0383},
}};

// ---- Futures convexity (modelled, not ignored) ----
// Ho-Lee / normal-model convexity: futures_rate - forward_rate = 0.5 * sigma^2 * t1 * t2,
// with t1 = accrual start, t2 = accrual end (year fractions from the evaluation date).
// A single documented sigma keeps the golden data deterministic and auditable.
inline constexpr double convexity_sigma = 0.0075;  // 75 bp/yr normal vol

constexpr double convexity_adjustment(double t1, double t2) {
  return 0.5 * convexity_sigma * convexity_sigma * t1 * t2;
}

// ---- Problem dimensions (sanity, used by tests) ----
inline constexpr int n_front_knots = static_cast<int>(meeting_dates.size());   // 6
inline constexpr int n_back_knots = static_cast<int>(swaps.size());            // 11
inline constexpr int n_knots = n_front_knots + n_back_knots;                   // 17
inline constexpr int n_instruments =
    static_cast<int>(futures_1m.size() + futures_3m.size() + swaps.size());    // 25

static_assert(n_instruments > n_knots,
              "The calibration must be OVER-determined: more instruments than knot forwards. "
              "If this ever becomes square, the 'never assert exact repricing' rule silently "
              "stops being meaningful.");

}  // namespace swaps::refmkt
