#pragma once
// Non-deliverable FX forwards (NDF) and non-deliverable swaps (NDS) — the engine's LINEAR (no-vol) FX
// forward layer. An NDF is a cash-settled outright: at maturity T no currencies change hands; instead the
// two parties settle, IN A CONVERTIBLE settlement currency (e.g. USD), the difference between a contracted
// forward outright K and the observed fixing S_fix, on a notional expressed in the NON-DELIVERABLE currency
// (e.g. BRL, KRW, INR). Because it is a forward on the FX rate with no optionality, the value is a pure
// forward/discount calculation — there is no vol input (contrast vol/fx_black.hpp's Garman-Kohlhagen
// vanilla, which this file may borrow gk_forward from for the covered-interest-parity outright).
//
// QUOTE CONVENTION (documented once, used everywhere here):
//   * spot S and strike K are quoted as UNITS OF THE SETTLEMENT CURRENCY PER 1 UNIT OF THE ND CURRENCY
//     (settle/ND). E.g. for a USD-settled BRL NDF, S is USD per 1 BRL. (This is the reciprocal of the
//     common market screen quote BRL-per-USD; pass 1/screen if your screen is ND-per-settle.)
//   * notional is in ND-CURRENCY units.
//   * r_settle is the settlement-currency (numeraire) continuously-compounded rate; r_nd the ND-currency rate.
//   * direction: +1 = BUY the ND currency forward (long ND / gains when S_fix rises above K), −1 = SELL.
//   * Settlement amount at T, per 1 unit of ND notional, in settlement ccy = direction·(S_fix − K).
//
// Under the settlement-currency risk-neutral measure the fixing's forward is the covered-interest-parity
// outright  F = S·e^{(r_settle − r_nd)T}, so the present value discounts the expected settlement:
//
//   PV = direction · notional · e^{−r_settle T} · (F − K)
//      = direction · notional · ( S·e^{−r_nd T} − K·e^{−r_settle T} )        [numeraire-clean form]
//
// The second form makes the Greeks fall out exactly (no vol, so no d1/d2):
//   dPV/dS        = direction · notional · e^{−r_nd T}              (= direction·notional·DF_nd)
//   dPV/dr_settle = direction · notional · K · T · e^{−r_settle T}
//   dPV/dr_nd     = −direction · notional · S · T · e^{−r_nd T}
//
// A NON-DELIVERABLE SWAP (NDS) is a strip of NDF fixings against ONE fixed rate K; its fair K reprices the
// whole strip to zero (a DF- and notional-weighted average of the per-period fair forwards).
//
// Scalar-templated so `double` prices and a reverse-AD type differentiates the same closed forms.
// QuantLib-free; header-only.
#include <cmath>
#include <cstddef>
#include <vector>

#include "swaps/vol/fx_black.hpp"  // reuses vol::gk_forward for the covered-interest-parity outright

namespace swaps::pricing {

// direction encoded as +1 (buy ND fwd) / −1 (sell). A convenience alias mirroring vol::CallPut's spirit.
enum class NdfSide { Buy, Sell };
inline double ndf_sign(NdfSide s) { return s == NdfSide::Buy ? 1.0 : -1.0; }

// Covered-interest-parity fair outright: F = S·e^{(r_settle − r_nd)T} (settlement ccy per 1 ND unit).
// r_settle plays the "domestic"/numeraire role, r_nd the "foreign"/asset role — so this is gk_forward.
template <class S>
inline S ndf_fair_forward(const S& spot, double T, const S& r_settle, const S& r_nd) {
  return swaps::vol::gk_forward<S>(spot, T, r_settle, r_nd);
}

// PV of an NDF, in SETTLEMENT-CURRENCY units, for a notional in ND-currency units. `direction` is a signed
// scalar (+1 buy / −1 sell); use ndf_sign(NdfSide) to build it. A trade struck at ndf_fair_forward has PV 0.
template <class S>
inline S ndf_pv(const S& spot, const S& strike, double T, const S& r_settle, const S& r_nd,
                const S& notional, const S& direction) {
  using std::exp;
  const S df_settle = exp(-r_settle * S(T));
  const S df_nd = exp(-r_nd * S(T));
  // direction·notional·( S·DF_nd − K·DF_settle ) — the numeraire-clean form (== dir·N·DF_settle·(F−K)).
  return direction * notional * (spot * df_nd - strike * df_settle);
}

// Per-point NDF Greeks (+ pv) in one pass. All are LINEAR closed forms (no vol), IDENTICAL to the analytics
// documented in the file header. delta_spot is ∂PV/∂S; dpv_dr_settle, dpv_dr_nd the two rate sensitivities.
template <class S>
struct NdfGreeks {
  S pv, fair_forward, delta_spot, dpv_dr_settle, dpv_dr_nd;
};

template <class S>
inline NdfGreeks<S> ndf_greeks(const S& spot, const S& strike, double T, const S& r_settle, const S& r_nd,
                               const S& notional, const S& direction) {
  using std::exp;
  const S TT(T);
  const S df_settle = exp(-r_settle * TT);
  const S df_nd = exp(-r_nd * TT);
  const S dn = direction * notional;
  NdfGreeks<S> g;
  g.fair_forward = spot * exp((r_settle - r_nd) * TT);
  g.pv = dn * (spot * df_nd - strike * df_settle);
  g.delta_spot = dn * df_nd;
  g.dpv_dr_settle = dn * strike * TT * df_settle;
  g.dpv_dr_nd = -dn * spot * TT * df_nd;
  return g;
}

// NDS fair fixed rate: the single K* that reprices a strip of NDF fixings (at maturities T_i on notionals
// N_i) to zero PV. From Σ_i dir·N_i·DF_settle(T_i)·(F_i − K) = 0:
//   K* = Σ_i N_i·DF_settle(T_i)·F_i / Σ_i N_i·DF_settle(T_i),
// a DF- and notional-weighted average of the per-period fair forwards (direction cancels, so it is
// direction-independent). `notionals` may be empty/short → treated as 1 per period.
template <class S>
inline S nds_fair_rate(const S& spot, const S& r_settle, const S& r_nd, const std::vector<double>& maturities,
                       const std::vector<S>& notionals) {
  using std::exp;
  S num(0.0), den(0.0);
  for (std::size_t i = 0; i < maturities.size(); ++i) {
    const double T = maturities[i];
    const S N = i < notionals.size() ? notionals[i] : S(1.0);
    const S df_settle = exp(-r_settle * S(T));
    const S fwd = ndf_fair_forward<S>(spot, T, r_settle, r_nd);
    const S w = N * df_settle;
    num += w * fwd;
    den += w;
  }
  return den > S(0.0) ? num / den : S(0.0);
}

// PV of a whole NDS strip against a fixed rate K (settlement-ccy units), same sign/discount conventions as
// ndf_pv. Reprices to ~0 when K == nds_fair_rate(...). Handy for the gate and for pricing an off-market NDS.
template <class S>
inline S nds_pv(const S& spot, const S& strike, const S& r_settle, const S& r_nd,
                const std::vector<double>& maturities, const std::vector<S>& notionals, const S& direction) {
  S pv(0.0);
  for (std::size_t i = 0; i < maturities.size(); ++i) {
    const S N = i < notionals.size() ? notionals[i] : S(1.0);
    pv += ndf_pv<S>(spot, strike, maturities[i], r_settle, r_nd, N, direction);
  }
  return pv;
}

}  // namespace swaps::pricing
