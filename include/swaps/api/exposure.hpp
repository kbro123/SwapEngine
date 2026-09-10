#pragma once
#include <boost/json.hpp>
// Exposure seam declaration. The "exposure" run_json verb computes an EPE/ENE/PFE counterparty-exposure
// profile off a calibrated SOFR curve, using the MC-exposure kernel (portfolio/compiled.hpp
// CompiledPortfolio::npv_grid — ~1.9M full-book reprices/sec) + swaps/xva/exposure.hpp aggregation. See
// api/exposure.cpp for the schema. Demo scope: a single classic (Flat+Hermite) self-discounting curve.
#include <string>

namespace swaps::api {

// request = {"exposure": {bundle, book?, value_date?, curve_roles?, netting_sets?, n_paths?, n_nodes?,
//                         horizon_years?, sigma?, kappa?, pfe_q?}} (at least one of book / netting_sets).
//
// Legacy whole-book form ("book", unchanged): the whole book is ONE netting set ->
//   {node_time[], epe[], ene[], pfe[], mtm, wall_us, n_paths, n_nodes, n}.
//
// Netting-set form ("netting_sets", may accompany "book"): trades grouped under a CSA (trade::NettingSet);
// the CSA's collateral currency DECIDES each set's discount role (its OIS, resolved through "curve_roles"
// exactly as book_from_json — an unmapped index fails loudly), and the profile is computed PER SET (netting
// aggregates within a set, never across sets):
//   "value_date": ISO, "curve_roles": {index_id: bundle_role},
//   "netting_sets": [{"id": "CP-1", "csa": {"collateral_currency": "USD"},
//                     "trades": [{id, notional, pay, fixed_rate, index, effective, maturity}, ...]}]
// -> response gains "netting_sets": [{id, node_time[], epe[], ene[], pfe[], mtm, n}, ...].
//
// The state grid is a Gaussian curve-state proxy x(t,path)=x_cal + OU(sigma,kappa;t)·Z shared by every set
// (illustrative — a calibrated LGM/HW1F is a later step). Single-curve swap-kind positions only (xccy /
// custom-region / turn'd curves rejected — CompiledPortfolio is single self-discounting Flat+Hermite).
std::string exposure_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string exposure_json(const std::string& request);

}  // namespace swaps::api
