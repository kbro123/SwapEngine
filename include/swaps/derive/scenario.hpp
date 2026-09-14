#pragma once
// derive/scenario.hpp — "fork over the market" what-ifs on a calibrated bundle (E7 stage 5.2): the `scenario`
// verb's whole computation as library code over any session type meeting calibration::CalibrationSession (the verb
// instantiates it with api::BundleSession; this header never includes api).
//
// Calibrate the base ONCE; then for each move copy the fitted state, add the move's per-curve rate shift to every
// curve's interpolation forwards (turn jumps untouched), sample the curves and value the book there. The base state
// is never mutated, so N moves cost one calibration plus N forks.
//
// A MOVE (ScenarioMove): shocks ADD. The parallel moves every curve's forward once (add_parallel_shock) and an
// explicit shift_curve key adds its bp onto whatever its curve already carries (add_curve_shock) -- the one rule
// scenario, scenario_grid and var share (SC1, owner decision 2026-09-14; before it a key REPLACED the parallel
// here). FX bumps are EXACT PER CURRENCY PAIR (SC2, owner decision 2026-09-14; derive/fx_move.hpp): each xccy
// position's spot moves by its own pair's factor, resolved for every move before calibrating.
//
// Bitwise with the verb it replaces: shifts are bp / 1e4 (not bp * 1e-4, which differs in the last bit for many
// sizes), the book is revalued through the templated handles, and a move without FX bumps values the book itself.

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
#include "swaps/derive/fx_move.hpp"  // FxBump, book_fx_moves: FX per currency pair (SC2)
#include "swaps/portfolio/fx_pairs.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace swaps::derive {

struct ScenarioMove {
  std::string name;
  std::optional<double> parallel_bp;       // every curve's forward, once
  std::map<int, double> shift_curve_bp;    // curve role -> bp; ADDS onto the parallel
  std::vector<FxBump> fx;                  // exact per currency pair (derive/fx_move.hpp)
};

// A move's RATE part resolved against a bundle: the rate-space shift per curve (its FX part: book_fx_moves).
struct ResolvedMove {
  std::vector<double> curve_delta;
};

// A parallel shift moves EVERY curve's forward once: an OUTRIGHT curve takes it on its knots and a SPREAD curve inherits it
// from its base (giving the spread's knots the parallel as well moved it twice -- tests/scenario_spread_repro_test.cpp,
// fixed 2026-09-14). An explicit key moves exactly the curve it names (on a spread curve: the spread).
inline void add_parallel_shock(const std::vector<pricing::CurveStructure>& curves, double bp,
                               std::vector<double>& curve_delta) {
  for (std::size_t c = 0; c < curves.size(); ++c)
    if (curves[c].base < 0) curve_delta[c] += bp / 1e4;
}

// An explicit per-curve shock: `bp` ADDED onto whatever the curve already carries (SC1).
inline void add_curve_shock(int role, double bp, std::vector<double>& curve_delta) {
  curve_delta[static_cast<std::size_t>(role)] += bp / 1e4;
}

// `verb` names the request in the messages (scenario, var).
inline ResolvedMove resolve_scenario_move(const ScenarioMove& m, const std::vector<pricing::CurveStructure>& curves,
                                         const char* verb = "scenario") {
  const int n_curves = static_cast<int>(curves.size());
  ResolvedMove r;
  r.curve_delta.assign(curves.size(), 0.0);
  if (m.parallel_bp) add_parallel_shock(curves, *m.parallel_bp, r.curve_delta);
  for (const auto& [role, bp] : m.shift_curve_bp) {
    if (role < 0 || role >= n_curves)
      throw std::invalid_argument(std::string(verb) + ": shift_curve role " + std::to_string(role) +
                                  " is out of range for this bundle");
    add_curve_shock(role, bp, r.curve_delta);
  }
  return r;
}

struct ScenarioRequest {
  calibration::BundleProblem bundle;
  std::optional<Eigen::VectorXd> x0;              // absent => the flat seed
  calibration::RegSpec reg;
  std::vector<double> sample_times;               // empty => no curve samples
  std::optional<portfolio::MultiCurveBook> book;  // absent => no valuation
  std::vector<ScenarioMove> scenarios;
  std::optional<std::string> fx_pivot;            // the currency unbumped currencies hold against (SC2)
};

struct ScenarioRow {
  std::size_t move = 0;                // index into ScenarioResult::moves
  std::vector<calibration::CurveSample> curves;
  double npv = 0.0, npv_delta = 0.0;   // when the request has a book
};

struct ScenarioResult {
  int n_curves = 0, n_knots = 0;
  calibration::CalibrationResult calibration;  // the base's solve, as the session reported it (SC3)
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
  // Every move's FX, per currency pair, before calibrating: a missing code or an undetermined pair costs no solve.
  BookFxMoves fx;
  std::optional<portfolio::MultiCurveBook> moved;  // one copy for the FX moves: only its xccy spots are overwritten
  if (r.book) {
    std::vector<std::vector<FxBump>> bumps;
    bumps.reserve(r.scenarios.size());
    for (const ScenarioMove& m : r.scenarios) bumps.push_back(m.fx);
    fx = book_fx_moves(r.bundle.currency_codes, r.bundle.curves, *r.book, bumps, r.fx_pivot);
    for (const std::vector<double>& f : fx.factor)
      if (!f.empty()) {
        moved = *r.book;
        break;
      }
  }
  Session sess(std::move(r.bundle));
  const calibration::BundleProblem& P = sess.problem();
  ScenarioResult out;
  out.calibration = sess.calibrate(calibration::seed_or_flat(P, r.x0, "scenario"), r.reg);
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
    const ResolvedMove rm = resolve_scenario_move(out.moves[k], P.curves);
    const Eigen::VectorXd xs = calibration::shift_interp_forwards(P, out.x_base, rm.curve_delta);
    ScenarioRow row;
    row.move = k;
    if (!r.sample_times.empty()) row.curves = calibration::sample_bundle_curves(P, xs, r.sample_times);
    if (out.has_book) {
      const std::vector<double>& f = fx.factor[k];
      if (f.empty()) {
        row.npv = calibration::book_value_at(*r.book, P, xs);
      } else {
        portfolio::set_xccy_fx(*moved, *r.book, fx.slots, f);
        row.npv = calibration::book_value_at(*moved, P, xs);
      }
      row.npv_delta = row.npv - out.base_npv;
    }
    out.rows.push_back(std::move(row));
  }
  return out;
}

}  // namespace swaps::derive
