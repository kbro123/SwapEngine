#pragma once
// Exposure seam declaration. The "exposure" run_json verb computes an EPE/ENE/PFE counterparty-exposure
// profile for a swap book off a calibrated SOFR curve, using the MC-exposure kernel (portfolio/compiled.hpp
// CompiledPortfolio::npv_grid — ~1.9M full-book reprices/sec) + swaps/xva/exposure.hpp aggregation. See
// api/exposure.cpp for the schema. Demo scope: a single classic (Flat+Hermite) self-discounting curve.
#include <string>

namespace swaps::api {

// request = {"exposure": {value_date, bundle, book, n_paths?, n_nodes?, horizon_years?, sigma?, pfe_q?}} ->
// {node_time[], epe[], ene[], pfe[], mtm, wall_us, n_paths, n_nodes, n}. The state grid is a Gaussian
// curve-state proxy x(t,path)=x_cal + sigma·sqrt(t)·Z (illustrative — a calibrated LGM/HW1F is a later step);
// the whole book is one netting set. Single-curve swap-kind positions only (xccy / region / turn'd curves
// rejected — CompiledPortfolio is single self-discounting Flat+Hermite).
std::string exposure_json(const std::string& request);

}  // namespace swaps::api
