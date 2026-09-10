#pragma once
#include <boost/json.hpp>
// Scenario-GRID seam (api/scenario_grid.cpp): the many-scenario twin of the `scenario` verb. Where
// `scenario` forks the calibrated market once per shock in a flat LIST, `scenario_grid` sweeps a whole
// MATRIX of shocks — the outer product of one or two shock AXES — and returns a P&L SURFACE for a book.
// It is the "do what QuantLib is too slow to do interactively" primitive: the base bundle is calibrated
// ONCE, and every grid cell is a µs market fork (market::Scenario) repriced through the cached compiled
// reprice twin (portfolio::CompiledMultiCurveBook, the reprice_bound kernel) — so an N×M grid costs one
// calibration + N·M compiled repricings, not N·M calibrations.
//
//   scenario_grid  {"scenario_grid": {
//       "bundle": {...BundleProblem...},                (required) the curve bundle to anchor
//       "x0": [...],                                    (optional) calibration start; else flat guess
//       "regularize": {lambda, curves, tension, sigma}, (optional) same as the `bundle`/`scenario` verb
//       "book": {...MultiCurveBook...},                 (optional; the P&L surface needs it)
//       "sample_times": [...],                          (optional) grid the BASE curves are sampled on
//       "axes": [ axis0, axis1 ]                        1 or 2 axes; axis0 = rows, axis1 = cols
//     }}
//
//   An axis is  { "label": "USD rates (bp)",
//                 "kind": "parallel_bp" | "shift_curve" | "fx",
//                 "role": <int>,                        (shift_curve only) integer curve role to shift
//                 "base": "EUR", "quote": "USD",        (fx only) the pair to bump
//                 "values": [ ... ] }                   the axis sweep (bp for rate axes; rel for fx)
//
//   A cell (i, j) applies axis0.values[i] AND axis1.values[j] together — a parallel_bp axis shifts every
//   curve, a shift_curve axis shifts only its role, an fx axis scales every xccy position's FX reset. The
//   two axes compose (their curve shifts add; their fx factors multiply), exactly as two `scenario`
//   moves would. With one axis, n1 = 1 and the surface is a single column.
//
//   -> {"scenario_grid": {
//         n_curves, n_knots,
//         "axes": [{label, kind, values, role?}, ...],  (echo, in row/col order)
//         "shape": [n0, n1],
//         "base": {x, npv?, n?, curves?},
//         "npv":  [[...], ...],     (n0 × n1) book NPV per cell        — present iff `book`
//         "pnl":  [[...], ...],     (n0 × n1) npv − base_npv per cell  — the P&L surface
//         "grid_us": <double>       engine wall time of the whole cell sweep (µs)
//       }}
//
// The diagonal invariant: a single-axis parallel_bp grid cell i reproduces `scenario`'s npv_delta for the
// same parallel_bp to rounding (both fork the SAME anchor with the SAME arithmetic; only the reprice path
// differs — CompiledMultiCurveBook::npv == MultiCurveBook::value to ~1e-15 relative). QuantLib-free;
// additive; a new verb off the gated hot path.

#include <string>

namespace swaps::api {

std::string scenario_grid_json(const boost::json::object& request);  // parse-once entry (E6.3)
std::string scenario_grid_json(const std::string& request);

}  // namespace swaps::api
