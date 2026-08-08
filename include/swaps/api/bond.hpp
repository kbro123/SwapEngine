#pragma once
// Bond seam declaration. The stateless "bonds" run_json verb does street/yield-space bond math (curve-free,
// no bundle): clean/dirty/accrued, yield-to-maturity, and modified/Macaulay duration + convexity for a list
// of fixed-rate bonds. See api/bond.cpp for the schema. Curve-space z-spread/PV off a calibrated curve is a
// follow-on verb on BundleSession.
#include <string>

namespace swaps::api {

// request = {"bonds": {value_date?, bonds:[{issue, settle, maturity, coupon, freq?=2, clean?|yield?}]}} ->
// flat SoA {clean, dirty, accrued, ytm, modified_duration, macaulay_duration, convexity, n}. Each bond gives
// EXACTLY ONE of clean|yield (the other is solved). Units are the model's decimals: coupon/yield are absolute
// rates (0.045 = 4.5%), prices are per unit notional (1.0 = par). Regular coupon periods only (odd first/last
// stubs throw for now — the kernel is ready, the builder is the follow-up).
std::string bonds_json(const std::string& request);

}  // namespace swaps::api
