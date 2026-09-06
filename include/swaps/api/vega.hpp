#pragma once
// Vega-ladder seam declaration. The "vega" run_json verb computes a swaption book's aggregate sensitivity to a
// vol surface's PARAMETERS (per-cell normal vol, or per-cell SABR {alpha,rho,nu}) off a CALIBRATED curve — the
// vol analogue of the rates delta ladder. See api/vega.cpp for the request/response schema and vol/vega_ladder.hpp
// for the computation.
#include <string>

namespace swaps::api {

// request = {"vega": {value_date, bundle, currency?, index?, curve?,
//   cells:[{expiry, tenor, normal_vol? | sabr?{alpha,rho,nu}}],
//   swaptions:[{expiry, tenor, strike? | moneyness_bp?, payer?, notional?}]}}
// -> {"vega": {n_cells, n_swaptions, book_value, cell_forward, cell_annuity, cell_expiry_years, cell_has_sabr,
//              d_normal_vol, d_alpha, d_rho, d_nu}}. Each swaption maps to the cell with its (expiry, tenor);
// the ladder buckets d(book value)/d(vol parameter) per cell (normal-vol -> d_normal_vol; SABR -> d_alpha/rho/nu).
std::string vega_json(const std::string& request);

}  // namespace swaps::api
