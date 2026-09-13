// Scenario-GRID verb (declared in include/swaps/api/scenario_grid.hpp): sweep a MATRIX of market shocks
// and return a P&L surface for a book. This is the many-scenario twin of api/scenario.cpp — same "fork
// over the market" model (market::Scenario), same one-time calibration of the anchor — but the shocks form
// the outer product of one or two AXES, and every cell is repriced through the CACHED compiled reprice twin
// (portfolio::CompiledMultiCurveBook, the reprice_bound kernel) rather than the templated one-shot path. So
// an N×M grid pays one calibration + N·M compiled repricings (each a matvec + a vectorized exp), the
// "interactive many-scenario" story the engine's speed is built for.
//
// Reuses, does NOT reinvent: bundle_from_json/book_from_json/flat_x0/BundleSession/RegSpec (bundle_api.hpp),
// market::Scenario's shift arithmetic (scenario.hpp), and CompiledMultiCurveBook::npv for the reprice. The
// per-cell fork mirrors api/scenario.cpp exactly (curve shift added to each curve's interp forwards; an FX
// bump compounded into a scale factor on every xccy position's fx_spot), so a single-axis parallel grid
// cell reproduces `scenario`'s npv_delta to rounding.

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/scenario_grid.hpp"

#include "swaps/api/bundle_api.hpp"              // bundle/book_from_json, flat_x0, BundleSession, RegSpec
#include "swaps/api/json_util.hpp"
#include "swaps/calibration/bundle_problem.hpp"  // build_bundle_curves / CurveHandle (base-curve sampling)
#include "swaps/portfolio/compiled_multi.hpp"    // CompiledMultiCurveBook — the reprice_bound kernel reused

namespace swaps::api {

namespace json = boost::json;
namespace cal = swaps::calibration;
namespace pf = swaps::portfolio;

namespace {


// One shock axis: a kind + its sweep of values. A parallel_bp axis shifts every curve; a shift_curve axis
// shifts only its integer role; an fx axis scales a pair. `values` are bp for the rate axes, rel for fx.
enum class AxisKind { ParallelBp, ShiftCurve, Fx };
struct Axis {
  AxisKind kind = AxisKind::ParallelBp;
  std::string label;
  int role = 0;                 // ShiftCurve only
  std::string base, quote;      // Fx only
  std::vector<double> values;
};

Axis parse_axis(const json::object& a, int n_curves) {
  Axis ax;
  ax.label = js(a, "label");
  const std::string kind = js(a, "kind", "parallel_bp");
  if (kind == "parallel_bp") {
    ax.kind = AxisKind::ParallelBp;
  } else if (kind == "shift_curve") {
    ax.kind = AxisKind::ShiftCurve;
    ax.role = static_cast<int>(jd(a, "role", 0.0));
    if (ax.role < 0 || ax.role >= n_curves)
      throw std::invalid_argument("scenario_grid: axis shift_curve role " + std::to_string(ax.role) +
                                  " is out of range for this bundle");
  } else if (kind == "fx") {
    ax.kind = AxisKind::Fx;
    ax.base = js(a, "base");
    ax.quote = js(a, "quote");
    if (ax.base.empty() || ax.quote.empty())
      throw std::invalid_argument("scenario_grid: an fx axis needs a 'base' and a 'quote' currency");
  } else {
    throw std::invalid_argument("scenario_grid: axis kind '" + kind +
                                "' is not one of parallel_bp / shift_curve / fx");
  }
  ax.values = darr(a, "values");
  if (ax.values.empty()) throw std::invalid_argument("scenario_grid: axis '" + ax.label +
                                                     "' has an empty 'values' array");
  return ax;
}

// Accumulate one axis's contribution to a cell: add its per-curve rate shift (bp/1e4) into `curve_delta`
// and compound its fx factor into `fx_factor`. Mirrors the arithmetic api/scenario.cpp applies per shock.
void apply_axis(const Axis& ax, double value, std::vector<double>& curve_delta, double& fx_factor) {
  switch (ax.kind) {
    case AxisKind::ParallelBp:
      for (double& d : curve_delta) d += value / 1e4;
      break;
    case AxisKind::ShiftCurve:
      curve_delta[static_cast<std::size_t>(ax.role)] += value / 1e4;
      break;
    case AxisKind::Fx:
      fx_factor *= (1.0 + value);
      break;
  }
}

// Sample every base curve at x on `times` (the same read BundleSession::sample does) — echoed for the base
// only, so the grid stays cheap (cells report NPV, not full curve samples).
json::array sample_curves_at(const cal::BundleProblem& P, const Eigen::VectorXd& x,
                             const std::vector<double>& times) {
  std::vector<int> off(P.n_curves());
  for (int c = 0; c < P.n_curves(); ++c) off[c] = P.offset(c);
  const auto C = cal::build_bundle_curves<double>(P.curves, [&](int c, int i) { return x[off[c] + i]; });
  json::array arr;
  for (int c = 0; c < P.n_curves(); ++c) {
    json::object co;
    co["currency"] = P.curves[c].currency;
    std::vector<double> disc, zero, fwd;
    for (double t : times) {
      disc.push_back(C[c]->discount(t));
      fwd.push_back(C[c]->forward(t));
      zero.push_back(t > 1e-12 ? C[c]->integral(t) / t : C[c]->forward(0.0));
    }
    co["t"] = vecf(times);
    co["discount"] = vecf(disc);
    co["zero"] = vecf(zero);
    co["forward"] = vecf(fwd);
    arr.push_back(std::move(co));
  }
  return arr;
}

}  // namespace

std::string scenario_grid_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o = (top.contains("scenario_grid") && top.at("scenario_grid").is_object())
                              ? top.at("scenario_grid").as_object()
                              : top;

  if (!o.contains("bundle")) throw std::invalid_argument("scenario_grid: missing 'bundle' object");
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  if (prob.n_curves() == 0) throw std::invalid_argument("scenario_grid: bundle has no curves");

  BundleSession sess(std::move(prob));
  const cal::BundleProblem& P = sess.problem();

  // ---- axes (1 or 2) --------------------------------------------------------------------------------
  if (!o.contains("axes") || !o.at("axes").is_array())
    throw std::invalid_argument("scenario_grid: missing 'axes' array (1 or 2 shock axes)");
  const json::array& axarr = o.at("axes").as_array();
  if (axarr.empty() || axarr.size() > 2)
    throw std::invalid_argument("scenario_grid: 'axes' must hold 1 or 2 axes");
  std::vector<Axis> axes;
  for (const auto& a : axarr) axes.push_back(parse_axis(a.as_object(), P.n_curves()));
  const int n0 = static_cast<int>(axes[0].values.size());
  const int n1 = axes.size() == 2 ? static_cast<int>(axes[1].values.size()) : 1;

  // ---- calibrate the BASE once (the anchor every cell forks from) ----------------------------------
  Eigen::VectorXd x0;
  if (o.contains("x0")) {
    const std::vector<double> xv = darr(o, "x0");
    if (static_cast<int>(xv.size()) != P.n_knots())
      throw std::invalid_argument("scenario_grid: x0 length does not match the bundle's knot count");
    x0 = Eigen::Map<const Eigen::VectorXd>(xv.data(), static_cast<Eigen::Index>(xv.size()));
  } else {
    x0 = flat_x0(P);
  }
  const RegSpec reg = reg_from_json(o);  // the one decoder (swaps/api/codec.hpp)
  sess.calibrate(x0, reg);
  const Eigen::VectorXd x_base = sess.x();  // the anchor: never mutated (each cell copies it)

  const std::vector<double> times = darr(o, "sample_times");
  const bool has_book = o.contains("book");
  pf::MultiCurveBook book;
  if (has_book) book = book_from_json(o.at("book"));

  // The compiled reprice twin (reprice_bound kernel) for the UNSCALED book, built once. An FX axis needs a
  // book whose xccy fx_spots are scaled by the cell's compounded factor, so those are built lazily and
  // cached by factor (few distinct fx values, reused across every rate cell in that fx column/row).
  std::map<long long, std::unique_ptr<pf::CompiledMultiCurveBook>> cbook_by_factor;
  const auto factor_key = [](double f) { return static_cast<long long>(std::llround(f * 1e12)); };
  auto cbook_for = [&](double factor) -> const pf::CompiledMultiCurveBook& {
    const long long key = factor_key(factor);
    auto it = cbook_by_factor.find(key);
    if (it != cbook_by_factor.end()) return *it->second;
    if (factor == 1.0) {
      auto cb = std::make_unique<pf::CompiledMultiCurveBook>(P.curves, book);
      return *(cbook_by_factor[key] = std::move(cb));
    }
    pf::MultiCurveBook sbook = book;  // scale the FX-reset notional of every xccy leg
    for (auto& p : sbook.positions)
      if (p.kind == pf::MultiCurveBook::Kind::Xccy) p.fx_spot *= factor;
    auto cb = std::make_unique<pf::CompiledMultiCurveBook>(P.curves, sbook);
    return *(cbook_by_factor[key] = std::move(cb));
  };
  const double base_npv = has_book ? cbook_for(1.0).npv(x_base) : 0.0;

  // ---- the cell sweep: fork the anchor per cell, reprice through the cached compiled twin ------------
  json::array npv_rows, pnl_rows;
  double grid_us = 0.0;
  if (has_book) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n0; ++i) {
      json::array npv_row, pnl_row;
      for (int j = 0; j < n1; ++j) {
        std::vector<double> curve_delta(P.n_curves(), 0.0);
        double fx_factor = 1.0;
        apply_axis(axes[0], axes[0].values[i], curve_delta, fx_factor);
        if (axes.size() == 2) apply_axis(axes[1], axes[1].values[j], curve_delta, fx_factor);

        // Fork the anchor: add each curve's shift to its interpolation forwards (turn δ overlays untouched).
        Eigen::VectorXd xs = x_base;
        for (int c = 0; c < P.n_curves(); ++c) {
          const double d = curve_delta[static_cast<std::size_t>(c)];
          if (d == 0.0) continue;
          const int oc = P.offset(c);
          const int ni = P.curves[c].n_interp_knots();
          xs.segment(oc, ni).array() += d;
        }
        const double npv = cbook_for(fx_factor).npv(xs);
        npv_row.push_back(npv);
        pnl_row.push_back(npv - base_npv);
      }
      npv_rows.push_back(std::move(npv_row));
      pnl_rows.push_back(std::move(pnl_row));
    }
    grid_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  }

  // ---- response -------------------------------------------------------------------------------------
  json::object out;
  out["n_curves"] = P.n_curves();
  out["n_knots"] = P.n_knots();
  {
    json::array ax_out;
    for (const auto& ax : axes) {
      json::object a;
      a["label"] = ax.label;
      a["kind"] = ax.kind == AxisKind::ParallelBp ? "parallel_bp"
                  : ax.kind == AxisKind::ShiftCurve ? "shift_curve"
                                                    : "fx";
      if (ax.kind == AxisKind::ShiftCurve) a["role"] = ax.role;
      if (ax.kind == AxisKind::Fx) {
        a["base"] = ax.base;
        a["quote"] = ax.quote;
      }
      a["values"] = vecf(ax.values);
      ax_out.push_back(std::move(a));
    }
    out["axes"] = std::move(ax_out);
  }
  out["shape"] = json::array{n0, n1};
  {
    json::object b;
    b["x"] = vecf(x_base);
    if (!times.empty()) b["curves"] = sample_curves_at(P, x_base, times);
    if (has_book) {
      b["npv"] = base_npv;
      b["n"] = static_cast<int>(book.positions.size());
    }
    out["base"] = std::move(b);
  }
  if (has_book) {
    out["npv"] = std::move(npv_rows);
    out["pnl"] = std::move(pnl_rows);
    out["grid_us"] = grid_us;
    out["n_cells"] = n0 * n1;
  }

  json::object resp;
  resp["scenario_grid"] = std::move(out);
  return json::serialize(resp);
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string scenario_grid_json(const std::string& request) { return scenario_grid_json(json::parse(request).as_object()); }
}  // namespace swaps::api
