#pragma once
// CMS convexity — the FIRST-ORDER linear-TSR (terminal-swap-rate) adjustment, normal model.
//
// A CMS coupon pays the swap rate S(T) at a payment date T_p, but S is a martingale under the ANNUITY
// measure, not the T_p-forward measure — so its expectation under the payment measure is the forward swap
// rate PLUS a convexity adjustment (Jensen: the rate is worth more where discounting is lower). Hagan's
// linear terminal-swap-rate model makes the annuity-mapping G(S)=P(T,T_p)/A(T) affine in S; the resulting
// adjustment, in the normal (Bachelier) approximation, is
//
//     CMS = S0 + theta * sigma_N^2 * T,
//
// where sigma_N is the ATM normal vol, T the expiry, and `theta` the linear-TSR slope — a mean-reversion /
// payment-delay handle a desk calibrates (theta=0 => no convexity => CMS=forward; theta>0 => CMS>forward).
//
// This is the fast, robust linear-TSR path. FULL Hagan static replication over the SABR smile (integrating a
// strip of swaptions against G''(K)) is a labelled follow-up — it refines the smile dependence but reduces to
// this at first order. Scalar-templated for calibration/AAD; the caller supplies DF(T_p) for discounting.
#include "swaps/vol/bachelier.hpp"

namespace swaps::vol {

struct CmsForward {
  double forward = 0.0;     // S0 — the plain forward swap rate
  double convexity = 0.0;   // theta * sigma_N^2 * T
  double rate = 0.0;        // S0 + convexity — the CMS-adjusted (payment-measure) forward
};

// The convexity-adjusted CMS forward rate (linear TSR, normal). `atm_normal_vol` = sigma_N, `expiry` = T.
template <class S>
inline CmsForward cms_forward(const S& s0, const S& atm_normal_vol, double expiry, const S& theta) {
  const S ca = theta * atm_normal_vol * atm_normal_vol * S(expiry);
  return {double(s0), double(ca), double(s0 + ca)};
}

// A CMS caplet (cap) / floorlet (floor): the payment-measure expectation E[(S(T) - K)^+] (cap) priced by
// Bachelier on the CONVEXITY-ADJUSTED forward, with the strike's normal vol. Returns the UNDISCOUNTED
// expectation per unit accrual — multiply by tau * DF(T_p) for the PV. `atm_normal_vol` sets the convexity
// shift, `strike_vol` the vol at the strike (from the SABR smile).
template <class S>
inline S cms_optionlet(const S& s0, const S& atm_normal_vol, double expiry, const S& theta, const S& strike,
                       const S& strike_vol, bool cap) {
  const S fwd = s0 + theta * atm_normal_vol * atm_normal_vol * S(expiry);
  // Annuity = 1 here (undiscounted, per unit accrual). Payer = cap (call on the rate), Receiver = floor.
  return bachelier_price<S>(fwd, strike, strike_vol, expiry, S(1.0), cap ? Payoff::Payer : Payoff::Receiver);
}

}  // namespace swaps::vol
