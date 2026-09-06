#pragma once
// FX-options seam declaration. The stateless "fx_option" (alias "fx_vol") run_json verb prices a book of
// vanilla FX options with the Garman-Kohlhagen lognormal model (vol/fx_black.hpp) off a supplied market
// {spot, r_dom, r_for} and a vol source — a flat vol, an interbank delta-quoted smile {ATM, RR, BF}, or
// explicit (strike, vol) knots (vol/fx_vol_surface.hpp). No bundle, no curve. See api/fx_option.cpp for the
// request/response schema.
#include <string>

namespace swaps::api {

// request = {"fx_option"|"fx_vol": {pair?, spot, r_dom?, r_for?, expiry,
//              vol? | surface:{atm,rr25,bf25,rr10?,bf10?} | surface:{strikes:[...],vols:[...]},
//              options:[{strike?|delta?, call?|put?, notional?, price?}]}}
//   -> {forward, expiry, options:[{strike, vol, forward, price, delta, gamma, vega, theta, rho_dom, rho_for,
//       notional_price, implied_vol?}]}. Units are decimals (rates/vols absolute, strikes/spot in the FX
//   quote). Greeks are per unit foreign notional; notional_price scales by the option's notional.
std::string fx_option_json(const std::string& request);

}  // namespace swaps::api
