#pragma once
// derive/scenario.hpp — "fork over the market" what-ifs on a calibrated bundle (E7 stage 5.2): the `scenario`
// verb's whole computation as library code over any session type meeting calibration::CalibrationSession (the verb
// instantiates it with api::BundleSession; this header never includes api).
//
// Calibrate the base ONCE; then for each move copy the fitted state, add the move's per-curve rate shift to every
// curve's interpolation forwards (turn jumps untouched), sample the curves and value the book there. The base state
// is never mutated, so N moves cost one calibration plus N forks.
//
// A MOVE (ScenarioMove) in the `scenario` rule, market::Scenario's: an explicit shift_curve key REPLACES the parallel
// for its curve, the parallel applies to every other curve. (var and scenario_grid ADD the two -- SC1, an owner
// decision, tests/scenario_golden_test.cpp pins both as they stand.) FX bumps compound IN REQUEST ORDER into one
// factor on every xccy position's fx_spot (portfolio/xccy_fx_scaled.hpp); a book position carries no pair, so a
// multi-pair move multiplies (SC2, an owner decision).
//
// Bitwise with the verb it replaces: shifts are bp / 1e4 (not bp * 1e-4, which differs in the last bit for many
// sizes), the book is revalued through the templated handles, and the scaled book is used only when the factor is
// not exactly 1.

#include <cstddef>
#include <map>
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
#include "swaps/market/scenario.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/portfolio/xccy_fx_scaled.hpp"

namespace swaps::derive {

struct FxBump {
  std::string base, quote;
  double rel = 0.0;  // rate' = rate * (1 + rel)
};

struct ScenarioMove {
  std::string name;
  std::optional<double> parallel_bp;       // every curve without an explicit key
  std::map<int, double> shift_curve_bp;    // curve role -> bp; replaces the parallel for that curve
  std::vector<FxBump> fx;                  // compounded in order
};

// A move resolved against a bundle: the rate-space shift per curve and the compounded FX factor.
struct ResolvedMove {
  std::vector<double> curve_delta;
  double fx_factor = 1.0;
};

inline ResolvedMove resolve_scenario_move(const ScenarioMove& m, int n_curves) {
  market::Scenario scn;
  if (m.parallel_bp) scn = market::Scenario::parallel(*m.parallel_bp);
  for (const auto& [role, bp] : m.shift_curve_bp) {
    if (role < 0 || role >= n_curves)
      throw std::invalid_argument("scenario: shift_curve role " + std::to_string(role) +
                                  " is out of range for this bundle");
    scn.shift_curve(std::to_string(role), bp);
  }
  ResolvedMove r;
  r.curve_delta.resize(static_cast<std::size_t>(n_curves));
  for (int c = 0; c < n_curves; ++c) r.curve_delta[static_cast<std::size_t>(c)] = scn.curve_shift(std::to_string(c));
  for (const FxBump& b : m.fx) r.fx_factor *= (1.0 + b.rel);
  return r;
}

struct ScenarioRequest {
  calibration::BundleProblem bundle;
  std::optional<Eigen::VectorXd> x0;              // absent => the flat seed
  calibration::RegSpec reg;
  std::vector<double> sample_times;               // empty => no curve samples
  std::optional<portfolio::MultiCurveBook> book;  // absent => no valuation
  std::vector<ScenarioMove> scenarios;
};

struct ScenarioRow {
  std::size_t move = 0;                // index into ScenarioResult::moves
  std::vector<calibration::CurveSample> curves;
  double npv = 0.0, npv_delta = 0.0;   // when the request has a book
};

struct ScenarioResult {
  int n_curves = 0, n_knots = 0;
  Eigen::VectorXd x_base;
  std::vector<calibration::CurveSample> base_curves;
  bool has_book = false;
  double base_npv = 0.0;
  int n_positions = 0;
  std::vector<ScenarioMove> moves;  // the request's, echoed by the codec
  std::vector<ScenarioRow> rows;    // one per move, in order
};

template <calibration::CalibrationSession Session>
ScenarioResult scenarios(ScenarioRequest r) {
  if (r.bundle.n_curves() == 0) throw std::invalid_argument("scenario: bundle has no curves");
  Session sess(std::move(r.bundle));
  const calibration::BundleProblem& P = sess.problem();
  sess.calibrate(calibration::seed_or_flat(P, r.x0, "scenario"), r.reg);

  ScenarioResult out;
  out.n_curves = P.n_curves();
  out.n_knots = P.n_knots();
  out.x_base = sess.x();  // the anchor every move forks from; never mutated
  if (!r.sample_times.empty()) out.base_curves = calibration::sample_bundle_curves(P, out.x_base, r.sample_times);
  out.has_book = r.book.has_value();
  if (out.has_book) {
    out.base_npv = calibration::book_value_at(*r.book, P, out.x_base);
    out.n_positions = static_cast<int>(r.book->positions.size());
  }

  out.moves = std::move(r.scenarios);
  out.rows.reserve(out.moves.size());
  for (std::size_t k = 0; k < out.moves.size(); ++k) {
    const ResolvedMove rm = resolve_scenario_move(out.moves[k], P.n_curves());
    const Eigen::VectorXd xs = calibration::shift_interp_forwards(P, out.x_base, rm.curve_delta);
    ScenarioRow row;
    row.move = k;
    if (!r.sample_times.empty()) row.curves = calibration::sample_bundle_curves(P, xs, r.sample_times);
    if (out.has_book) {
      row.npv = rm.fx_factor != 1.0 ? calibration::book_value_at(portfolio::xccy_fx_scaled(*r.book, rm.fx_factor), P, xs)
                                    : calibration::book_value_at(*r.book, P, xs);
      row.npv_delta = row.npv - out.base_npv;
    }
    out.rows.push_back(std::move(row));
  }
  return out;
}

}  // namespace swaps::derive
