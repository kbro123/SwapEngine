#pragma once
#include <boost/json.hpp>
// Options seam declarations. The "swaption" run_json verb prices European swaptions off a calibrated curve
// (Bachelier normal model, optional SABR smile). See api/options.cpp for the request/response schema.
#include <string>

namespace swaps::api {

// request = {"swaption": {value_date, bundle, currency?, index?, curve?, trades:[{expiry,tenor,strike,
// payer?, normal_vol?|sabr?}]}} -> {"trades":[{forward,annuity,price,normal_vol,atm_normal_vol,vega,delta,...}]}.
std::string swaption_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string swaption_json(const std::string& request);

// Stateless SABR strip calibration (no bundle). request = {"sabr_calibrate": {forward, expiry, strikes:[...],
// market_vols:[...], guess?:{alpha,rho,nu}, arb_lo?, arb_hi?}} -> {alpha, rho, nu, rms, iterations, converged,
// arbitrage_free}. Units are the model's decimals (forward/strikes absolute rates, vols normal/absolute).
std::string sabr_calibrate_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string sabr_calibrate_json(const std::string& request);

}  // namespace swaps::api
