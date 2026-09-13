#pragma once
#include <boost/json.hpp>
// Bond-future seam declaration. The stateless "bond_future" run_json verb does the deliverable-basket /
// cheapest-to-deliver (CTD) analytics for a bond future (curve-free, no bundle): the exchange conversion
// factor, gross & net basis, implied repo rate and invoice price for every bond in a delivery basket, plus
// the CTD (max implied repo). See api/bond_future.cpp for the schema. QuantLib-free; reuses the bond kernel
// (pricing/bond.hpp) for accrued and pricing/bond_future.hpp for the CTD math.
#include <string>

namespace swaps::api {

// request = {"bond_future": {
//     contract,                            // REQUIRED bond_futures[] row id (e.g. CME-TY): CF notional coupon,
//                                          //   maturity rounding, repo day count, deliverable convention, and the
//                                          //   decimals the exchange rounds the conversion factor to
//     value_date,                          // curve reference / "today"
//     first_delivery,                      // any day in the delivery month; the CF term counts from its 1st
//     delivery?,                           // delivery/settlement date for basis (default = first_delivery)
//     futures_price,                       // REQUIRED futures quote, per unit face (0.98 = 98-00)
//     repo,                                // REQUIRED funding rate for net basis, on the contract's repo day count
//     notional_coupon?, round_months?,     // override the contract row
//     basket:[{
//        id?,                              // echoed back; used to name the CTD
//        convention?,                      // conventions.json bond id; default = the contract's deliverable one
//        settle?,                          // cash settlement (default = value_date)
//        issue,                            // SEASONED: dated date;  OR  dated,first_coupon for when-issued
//        maturity, coupon,                 // absolute coupon rate (0.045 = 4.5%)
//        freq?,                            // overrides the convention frequency
//        clean }]}}                        // observed clean price, per unit face
// -> flat SoA {conversion_factor, gross_basis, net_basis, implied_repo, invoice_price, n,
//             ctd_index, ctd_id}. conversion_factor is the EXCHANGE factor (rounded to the contract's decimals);
//             invoice and basis use it. Basis fields are in price points (per unit face). The CTD is the
//             max-implied-repo deliverable. The computation is build::analyze_delivery_basket.
std::string bond_future_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string bond_future_json(const std::string& request);

}  // namespace swaps::api
