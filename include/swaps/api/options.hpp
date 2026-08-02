#pragma once
// Options seam declarations. The "swaption" run_json verb prices European swaptions off a calibrated curve
// (Bachelier normal model, optional SABR smile). See api/options.cpp for the request/response schema.
#include <string>

namespace swaps::api {

// request = {"swaption": {value_date, bundle, currency?, index?, curve?, trades:[{expiry,tenor,strike,
// payer?, normal_vol?|sabr?}]}} -> {"trades":[{forward,annuity,price,normal_vol,atm_normal_vol,vega,delta,...}]}.
std::string swaption_json(const std::string& request);

}  // namespace swaps::api
