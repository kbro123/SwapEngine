#pragma once
// derive/scenario_grid.hpp — a P&L SURFACE over the outer product of one or two shock axes (E7 stage 5.3): the
// `scenario_grid` verb's whole computation as library code over any calibration::CalibrationSession.
//
// Calibrate the base ONCE; for every cell, fork the fitted state by the cell's shifts and reprice the book through its
// compiled twin, built ONCE (an fx cell sets the xccy rows' spots per pair in place:
// CompiledMultiCurveBook::set_fx_factors). An N x M grid costs one calibration plus N x M compiled repricings.
//
// The GRID RULE: every axis ADDS its shift (bp / 1e4) to the curves it moves, and a cell's fx axes are one FX move.
// So a shift_curve axis on top of a parallel axis moves that curve by both -- the rule `scenario` and `var` share
// (derive/scenario.hpp add_parallel_shock / add_curve_shock; SC1, owner decision 2026-09-14). An fx axis
// bumps its pair, and each xccy position moves with its own pair (SC2, derive/fx_move.hpp).
//
// Bitwise with the verb it replaces (the grid goldens): bp / 1e4 accumulated axis by axis, the fork through
// calibration::shift_interp_forwards, a single-pair fx factor 1.0 * (1 + v) on notional * fx_spot as a fresh compile.

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
#include "swaps/derive/scenario.hpp"  // add_parallel_shock, add_curve_shock: the one shock-combining rule
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/portfolio/compiled_multi.hpp"

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

// One axis's contribution to a cell, added onto what the other axis already put there.
inline void add_axis_shock(const ShockAxis& ax, double value, const std::vector<pricing::CurveStructure>& curves,
                           std::vector<double>& curve_delta) {
  switch (ax.kind) {
    case ShockAxisKind::ParallelBp:
      add_parallel_shock(curves, value, curve_delta);
      break;
    case ShockAxisKind::ShiftCurve:
      add_curve_shock(*ax.role, value, curve_delta);
      break;
    case ShockAxisKind::Fx:
      break;  // an fx axis moves no curve: a cell's FX move is resolved per pair (book_fx_moves)
  }
}

struct ScenarioGridRequest {
  calibration::BundleProblem bundle;
  std::optional<Eigen::VectorXd> x0;              // absent => the flat seed
  calibration::RegSpec reg;
  std::vector<double> sample_times;               // base curves only; empty => none
  std::optional<portfolio::MultiCurveBook> book;  // absent => no surface
  std::vector<ShockAxis> axes;                    // 1 or 2
  std::optional<std::string> fx_pivot;            // the currency unbumped currencies hold against (SC2)
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
  // Every cell's FX move, per currency pair, before calibrating (SC2): a cell's fx axes bump their pairs together.
  BookFxMoves fx;
  if (r.book) {
    const int n0 = static_cast<int>(r.axes[0].values.size());
    const int n1 = r.axes.size() == 2 ? static_cast<int>(r.axes[1].values.size()) : 1;
    std::vector<std::vector<FxBump>> cells(static_cast<std::size_t>(n0 * n1));
    for (int i = 0; i < n0; ++i)
      for (int j = 0; j < n1; ++j)
        for (std::size_t a = 0; a < r.axes.size(); ++a)
          if (r.axes[a].kind == ShockAxisKind::Fx)
            cells[static_cast<std::size_t>(i * n1 + j)].push_back(
                {r.axes[a].base, r.axes[a].quote, r.axes[a].values[static_cast<std::size_t>(a == 0 ? i : j)]});
    fx = book_fx_moves(r.bundle.currency_codes, r.bundle.curves, *r.book, cells, r.fx_pivot);
  }

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

  portfolio::CompiledMultiCurveBook book(P.curves, *r.book);
  out.n_positions = static_cast<int>(r.book->positions.size());
  out.base_npv = book.npv(out.x_base);
  out.npv.reserve(static_cast<std::size_t>(out.n0));
  out.pnl.reserve(static_cast<std::size_t>(out.n0));
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < out.n0; ++i) {
    std::vector<double> npv_row, pnl_row;
    for (int j = 0; j < out.n1; ++j) {
      std::vector<double> curve_delta(static_cast<std::size_t>(P.n_curves()), 0.0);
      add_axis_shock(out.axes[0], out.axes[0].values[static_cast<std::size_t>(i)], P.curves, curve_delta);
      if (out.axes.size() == 2)
        add_axis_shock(out.axes[1], out.axes[1].values[static_cast<std::size_t>(j)], P.curves, curve_delta);
      const std::vector<double>& f = fx.factor[static_cast<std::size_t>(i * out.n1 + j)];
      if (!f.empty()) book.set_fx_factors(fx.slots, f);
      const double npv = book.npv(calibration::shift_interp_forwards(P, out.x_base, curve_delta));
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
