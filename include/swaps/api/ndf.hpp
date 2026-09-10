#pragma once
#include <boost/json.hpp>
// NDF/NDS seam declaration. The stateless "ndf" run_json verb prices a book of non-deliverable FX forwards
// (and, via a fixed-rate strip, non-deliverable swaps) with the LINEAR forward/discount kernel
// (pricing/ndf.hpp) off a supplied market {spot, r_settle, r_nd}. No bundle, no calibrated curve, no vol —
// NDFs are linear. QuantLib-free. See api/ndf.cpp for the request/response schema.
#include <string>

namespace swaps::api {

// request = {"ndf": {pair?, spot, r_settle?, r_nd?,
//              trades:[{maturity, strike?, notional?, direction?:"buy"|"sell"}],
//              nds?:{maturities:[...], notionals?:[...], fixed_rate?, direction?}}}
//   -> {forward, trades:[{fair_forward, pv, delta_spot, dpv_dr_settle, dpv_dr_nd, notional_settlement}],
//       nds?:{fair_rate, pv}}. Units are decimals; spot/strike are settlement-ccy per 1 ND-ccy unit,
//   notional is ND-ccy, PV/Greeks are settlement-ccy. A trade with no strike prices at the fair forward
//   (PV ≈ 0).
std::string ndf_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string ndf_json(const std::string& request);

}  // namespace swaps::api
