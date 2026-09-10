#pragma once
// CdsInstrument — ONE credit calibration row (a par CDS spread), priced against a survival curve by
// `model_quote<Scalar>`. A residual row, so it lives beside the other residual rows in calibration/, NOT in
// build/ (E6.4, 2026-09-10: holding it in build/ made the calibration problem include the build layer and
// closed a cycle against the declared `build --> calibration` arrow). The BUILDER `make_cds` — which needs the
// day-count DB — stays in build/credit_instruments.hpp.
//
// The par CDS row, kept verbatim from where the type used to live (the builder stays in build/):
// ORIGINAL NOTE — a par CDS (credit default swap) calibration instrument, plus the builder
// that assembles its premium schedule and protection-integration grid. Additive, QuantLib-free, header-only
// (mirrors build/inflation_instruments.hpp for the inflation swaps).
//
// CONVENTIONS chosen here (all times are curve-time year fractions, ACT/365F from the value date — the same
// convention every builder in build/instruments.hpp uses):
//
//   PREMIUM LEG — the protection buyer pays spread s on a QUARTERLY schedule (default freq = 4). The
//     coupon accrual is ACT/360; on the ACT/365F curve-time axis that day count is applied as
//         τᵢ = (tᵢ − tᵢ₋₁) · 365/360.
//     The leg's value per unit spread is the RISKY ANNUITY (a.k.a. RPV01):
//         A(s=1) = Σᵢ DF(tᵢ) · Q(tᵢ) · τᵢ,
//     i.e. each coupon is discounted (DF, nominal) AND weighted by survival to its payment date Q(tᵢ).
//     (No accrual-on-default term — premium is modelled as paid only if the name survives the period end,
//     the standard "no accrued" risky annuity; documented here so the discretisation is explicit.)
//
//   PROTECTION LEG — pays (1−R) at the moment of default. Recovery R defaults to 0.40. The exact value is
//         P = (1−R) · ∫₀ᵀ DF(u) · (−dQ(u)),
//     discretised on a fine uniform grid 0 = u₀ < u₁ < … < u_M = T with the DEFAULT-PAID-AT-PERIOD-END
//     convention (the protection cash-flow of each subinterval is discounted at the interval's RIGHT end):
//         P ≈ (1−R) · Σⱼ DF(uⱼ) · ( Q(uⱼ₋₁) − Q(uⱼ) ).
//     Q(uⱼ₋₁) − Q(uⱼ) is the default probability inside the subinterval; refining the grid (prot_steps
//     per premium period) makes the piecewise-flat-DF approximation of the integral arbitrarily accurate.
//
//   PAR SPREAD — the breakeven quote is the spread that zeroes the CDS PV:
//         s* = P / A(s=1) = protectionPV / riskyAnnuity.
//     Note s* scales with (1−R): a HIGHER recovery ⇒ a SMALLER protection leg ⇒ a LOWER par spread.
//
// The nominal DISCOUNT FACTORS DF(·) enter ONLY as BUILD-TIME WEIGHTS (doubles), captured when the
// instrument is built — exactly like the nominal weights in the YoY inflation swap. The free, calibrated
// object is the hazard/survival curve, so an AAD sweep differentiates only the Q(·) terms.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "swaps/curve/hazard.hpp"

namespace swaps::calibration {

struct CdsInstrument {
  double recovery = -1.0;  // R: recovery rate (REQUIRED; credit.cds_products[].recovery_default is the DB default)
  double maturity = 0.0;   // T (curve years)
  double market = 0.0;     // quoted par spread (decimal; 150 bp = 0.0150)

  // PREMIUM leg (build-time): pay[i] = coupon payment time, tau[i] = ACT/360 accrual fraction,
  // df_prem[i] = nominal DF(pay[i]). Risky annuity = Σ df_prem[i]·Q(pay[i])·tau[i].
  std::vector<double> pay, tau, df_prem;

  // PROTECTION integration grid (build-time): the j-th subinterval is [gl[j], gr[j]] with
  // df_prot[j] = nominal DF(gr[j]) (default paid at the subinterval's right end).
  std::vector<double> gl, gr, df_prot;

  // Risky annuity A(s=1) = Σ DF(tᵢ)·Q(tᵢ)·τᵢ. Seeds from the first curve-dependent term so the AAD
  // derivatives propagate (Q(pay[0]) carries them); the DF·τ products are build-time constants.
  template <class Scalar, class Curve>
  Scalar risky_annuity(const Curve& c) const {
    Scalar a = Scalar(df_prem[0] * tau[0]) * c.survival(pay[0]);
    for (std::size_t i = 1; i < pay.size(); ++i)
      a += Scalar(df_prem[i] * tau[i]) * c.survival(pay[i]);
    return a;
  }

  // Protection PV = (1−R)·Σⱼ DF(uⱼ)·( Q(uⱼ₋₁) − Q(uⱼ) ), default paid at the subinterval's right end.
  template <class Scalar, class Curve>
  Scalar protection_pv(const Curve& c) const {
    Scalar p = Scalar(df_prot[0]) * (c.survival(gl[0]) - c.survival(gr[0]));
    for (std::size_t j = 1; j < gr.size(); ++j)
      p += Scalar(df_prot[j]) * (c.survival(gl[j]) - c.survival(gr[j]));
    return Scalar(1.0 - recovery) * p;
  }

  // Par spread s* = protectionPV / riskyAnnuity.
  template <class Scalar, class Curve>
  Scalar model_quote(const Curve& c) const {
    return protection_pv<Scalar>(c) / risky_annuity<Scalar>(c);
  }

  template <class Scalar, class Curve>
  Scalar residual(const Curve& c) const {
    return model_quote<Scalar>(c) - market;
  }
};

// CDS builder. `maturity`/`spread` in curve years / decimal; `recovery` = R; `df` is any callable
// double(double) returning the nominal discount factor DF(t); `premium_freq` coupons per year (default 4 =
// quarterly); `prot_steps` protection-grid subintervals PER premium period (default 4). The discount
// factors are sampled ONCE here and stored as constants — the free object is the hazard curve.
// `accrual_ratio` converts the ACT/365F curve-time period length to the premium leg's day-count accrual

}  // namespace swaps::calibration
