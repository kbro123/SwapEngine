#pragma once
// derive/scenario_grid.hpp — a P&L SURFACE over the outer product of one or two shock axes (E7 stage 5.3): the
// `scenario_grid` verb's whole computation as library code over any calibration::CalibrationSession.
//
// Calibrate the base ONCE; for every cell, fork the fitted state by the cell's shifts and reprice the book through its
// compiled twin (portfolio::XccyFxScaledBooks: one compiled book per distinct FX factor, so a grid's fx column reuses
// one book across every rate cell). An N x M grid costs one calibration plus N x M compiled repricings.
//
// The GRID RULE: every axis ADDS its shift (bp / 1e4) to the curves it moves, and an fx axis compounds its factor.
// So a shift_curve axis on top of a parallel axis moves that curve by both -- where `scenario` lets an explicit key
// REPLACE the parallel (SC1, an owner decision; tests/scenario_golden_test.cpp pins both as they stand). An fx axis
// names a pair but reaches every xccy position (a position carries none: SC2).
//
// Bitwise with the verb it replaces (the grid goldens): bp / 1e4 accumulated axis by axis, the fork through
// calibration::shift_interp_forwards, the compiled book keyed by factor to 1e-12.

#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_state.hpp"
#include "swaps/calibration/diagnostics.hpp"  // CalibrationSession, seed_or_flat
#include "swaps/calibration/regularize.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/portfolio/xccy_fx_scaled.hpp"

namespace swaps::derive {

enum class ShockAxisKind { ParallelBp, ShiftCurve, Fx };

struct ShockAxis {
  ShockAxisKind kind = ShockAxisKind::ParallelBp;
  std::string label;
  std::optional<int> role;     // ShiftCurve only, and required there
  std::string base, quote;     // Fx only, and required there
  std::vector<double> values;  // bp for the rate axes, a relative bump for fx
};

inline void check_shock_axis(const ShockAxis& ax, int n_curves) {
  if (ax.kind == ShockAxisKind::ShiftCurve) {
    if (!ax.role) throw std::invalid_argument("scenario_grid: a shift_curve axis needs a 'role'");
    if (*ax.role < 0 || *ax.role >= n_curves)
      throw std::invalid_argument("scenario_grid: axis shift_curve role " + std::to_string(*ax.role) +
                                  " is out of range for this bundle");
  }
  if (ax.kind == ShockAxisKind::Fx && (ax.base.empty() || ax.quote.empty()))
    throw std::invalid_argument("scenario_grid: an fx axis needs a 'base' and a 'quote' currency");
  if (ax.values.empty())
    throw std::invalid_argument("scenario_grid: axis '" + ax.label + "' has an empty 'values' array");
}

// A parallel shift of `bp` moves EVERY curve's forward once: only OUTRIGHT curves' knots take it, and a spread curve
// inherits it from its base (fixed 2026-09-14: giving spread knots the parallel as well moved a spread curve twice --
// tests/scenario_spread_repro_test.cpp). var's reval moves use it too.
inline void add_parallel_shock(const std::vector<pricing::CurveStructure>& curves, double bp,
                               std::vector<double>& curve_delta) {
  for (std::size_t c = 0; c < curves.size(); ++c)
    if (curves[c].base < 0) curve_delta[c] += bp / 1e4;
}

// One axis's contribution to a cell, added onto what the other axis already put there.
inline void add_axis_shock(const ShockAxis& ax, double value, const std::vector<pricing::CurveStructure>& curves,
                           std::vector<double>& curve_delta, double& fx_factor) {
  switch (ax.kind) {
    case ShockAxisKind::ParallelBp:
      add_parallel_shock(curves, value, curve_delta);
      break;
    case ShockAxisKind::ShiftCurve:
      curve_delta[static_cast<std::size_t>(*ax.role)] += value / 1e4;
      break;
    case ShockAxisKind::Fx:
      fx_factor *= (1.0 + value);
      break;
  }
}

struct ScenarioGridRequest {
  calibration::BundleProblem bundle;
  std::optional<Eigen::VectorXd> x0;              // absent => the flat seed
  calibration::RegSpec reg;
  std::vector<double> sample_times;               // base curves only; empty => none
  std::optional<portfolio::MultiCurveBook> book;  // absent => no surface
  std::vector<ShockAxis> axes;                    // 1 or 2
};

struct ScenarioGridResult {
  int n_curves = 0, n_knots = 0;
  std::vector<ShockAxis> axes;  // the request's, echoed by the codec
  int n0 = 0, n1 = 0;           // cells: n0 x n1 (n1 = 1 for a single axis)
  Eigen::VectorXd x_base;
  std::vector<calibration::CurveSample> base_curves;
  bool has_book = false;
  double base_npv = 0.0;
  int n_positions = 0;
  std::vector<std::vector<double>> npv, pnl;  // [i][j] for axis-0 value i, axis-1 value j (when a book was given)
  double grid_us = 0.0;                       // wall-clock of the cell sweep
  int n_cells = 0;
};

template <calibration::CalibrationSession Session>
ScenarioGridResult scenario_grid(ScenarioGridRequest r) {
  if (r.bundle.n_curves() == 0) throw std::invalid_argument("scenario_grid: bundle has no curves");
  if (r.axes.empty() || r.axes.size() > 2) throw std::invalid_argument("scenario_grid: 'axes' must hold 1 or 2 axes");
  for (const ShockAxis& ax : r.axes) check_shock_axis(ax, r.bundle.n_curves());

  Session sess(std::move(r.bundle));
  const calibration::BundleProblem& P = sess.problem();
  sess.calibrate(calibration::seed_or_flat(P, r.x0, "scenario_grid"), r.reg);

  ScenarioGridResult out;
  out.n_curves = P.n_curves();
  out.n_knots = P.n_knots();
  out.axes = std::move(r.axes);
  out.n0 = static_cast<int>(out.axes[0].values.size());
  out.n1 = out.axes.size() == 2 ? static_cast<int>(out.axes[1].values.size()) : 1;
  out.x_base = sess.x();  // the anchor every cell forks from; never mutated
  if (!r.sample_times.empty()) out.base_curves = calibration::sample_bundle_curves(P, out.x_base, r.sample_times);
  out.has_book = r.book.has_value();
  if (!out.has_book) return out;

  portfolio::XccyFxScaledBooks books(P.curves, std::move(*r.book));
  out.n_positions = static_cast<int>(books.book().positions.size());
  out.base_npv = books.at(1.0).npv(out.x_base);
  out.npv.reserve(static_cast<std::size_t>(out.n0));
  out.pnl.reserve(static_cast<std::size_t>(out.n0));
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < out.n0; ++i) {
    std::vector<double> npv_row, pnl_row;
    for (int j = 0; j < out.n1; ++j) {
      std::vector<double> curve_delta(static_cast<std::size_t>(P.n_curves()), 0.0);
      double fx_factor = 1.0;
      add_axis_shock(out.axes[0], out.axes[0].values[static_cast<std::size_t>(i)], P.curves, curve_delta, fx_factor);
      if (out.axes.size() == 2)
        add_axis_shock(out.axes[1], out.axes[1].values[static_cast<std::size_t>(j)], P.curves, curve_delta, fx_factor);
      const double npv = books.at(fx_factor).npv(calibration::shift_interp_forwards(P, out.x_base, curve_delta));
      npv_row.push_back(npv);
      pnl_row.push_back(npv - out.base_npv);
    }
    out.npv.push_back(std::move(npv_row));
    out.pnl.push_back(std::move(pnl_row));
  }
  out.grid_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  out.n_cells = out.n0 * out.n1;
  return out;
}

}  // namespace swaps::derive
