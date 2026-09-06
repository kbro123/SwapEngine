#pragma once
// Bond-future seam declaration. The stateless "bond_future" run_json verb does the deliverable-basket /
// cheapest-to-deliver (CTD) analytics for a bond future (curve-free, no bundle): the exchange conversion
// factor, gross & net basis, implied repo rate and invoice price for every bond in a delivery basket, plus
// the CTD (max implied repo). See api/bond_future.cpp for the schema. QuantLib-free; reuses the bond kernel
// (pricing/bond.hpp) for accrued and pricing/bond_future.hpp for the CTD math.
#include <string>

namespace swaps::api {

// request = {"bond_future": {
//     value_date,                          // curve reference / "today"
//     first_delivery,                      // first delivery day of the contract month (drives CF rounding)
//     delivery?,                           // delivery/settlement date for basis (default = first_delivery)
//     futures_price,                       // the futures quote, per unit face (0.98 = 98-00)
//     repo,                                // funding rate for net basis, ACT/360 (0.053 = 5.3%)
//     notional_coupon?=0.06,               // CME/Eurex 6% notional
//     round_months?=3,                     // maturity rounding: 3 = bond/10y quarters, 1 = 2/3/5y note
//     basket:[{
//        id?,                              // echoed back; used to name the CTD
//        convention?="US-TREASURY",        // conventions.json bond id (freq + stub-discount rule)
//        settle?,                          // cash settlement (default = value_date)
//        issue,                            // SEASONED: dated date;  OR  dated,first_coupon for when-issued
//        maturity, coupon,                 // absolute coupon rate (0.045 = 4.5%)
//        freq?,                            // overrides the convention frequency
//        clean }]}}                        // observed clean price, per unit face
// -> flat SoA {conversion_factor, gross_basis, net_basis, implied_repo, invoice_price, n,
//             ctd_index, ctd_id}. Basis fields are in price points (per unit face); implied_repo/net_basis
//             use ACT/360 to the delivery date. The CTD is the max-implied-repo deliverable.
std::string bond_future_json(const std::string& request);

}  // namespace swaps::api
