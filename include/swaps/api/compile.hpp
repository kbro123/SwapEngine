#pragma once
// The bundle COMPILER: a composer `spec` (curves + generic instrument rows + interpolation regions) ->
// a resolved `cal::BundleProblem` plus the streaming config (per-instrument start markets + drift + a
// display grid). This is the C++ analog of server/compile.py's compile_spec — the single generic
// translation from the high-level spec to the engine's object graph, built on the QuantLib-free
// construction object model (include/swaps/build/*). It NEVER touches the calculation engine; it only
// aggregates the structs the engine consumes.
//
// It lives in the api/ SEAM (not include/swaps/build/) because a spec is inherently JSON-shaped and the
// assembly reuses this layer's bundle_to_json / instrument_to_json emitters; the build/ headers stay
// pure struct builders (the object model). compile_spec parses a Boost.JSON spec; compile_to_json emits
// the SAME document server/compile.py returns (so the web, the parity test and the Excel seam all agree).
//
// Only <boost/json/fwd.hpp> leaks into this header (matching bundle_api.hpp), so a consumer that never
// touches JSON pays nothing.

#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json/fwd.hpp>

#include "swaps/api/bundle_api.hpp"  // RegSpec (compile_reg_spec)
#include "swaps/calibration/bundle_problem.hpp"

namespace swaps::api {

namespace cal = swaps::calibration;

// Raised on a structurally invalid spec (mirrors server/compile.py's CompileError).
struct CompileError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Per-curve summary for the UI (mirrors compile_spec's `per_curve` entries).
struct PerCurveInfo {
  std::string name, currency, type, policies, base_name;
  bool has_base = false;
  double fwd_tenor = 0.0;
  int n_front = 0, n_back = 0, n_knots = 0, n_quotes = 0, n_turns = 0;
};

// One resolved instrument window, for the UI's read-only date columns (compile_spec's `resolved`).
struct ResolvedRow {
  std::string curve_id, ins_id, start_date, end_date;  // *_date empty => null
  double t_start = 0.0, t_end = 0.0;
};

// The full result of compiling a spec — the numeric core is {bundle, starts, drifts}; the rest is the
// streaming/display metadata compile_spec returns.
struct CompileResult {
  cal::BundleProblem bundle;
  std::vector<double> starts, drifts, grid;
  std::vector<std::string> curve_names, order_ids, currency_codes;
  std::vector<PerCurveInfo> per_curve;
  std::vector<ResolvedRow> resolved;
  std::vector<std::string> warnings;
  int n_knots = 0, n_residuals = 0;
  bool under_determined = false;
  bool has_bands = false;  // any instrument carries a soft bid/offer band (floors smoothing to light)
  std::string value_date, smoothness = "light", reg_op;  // reg_op empty => null
  bool has_reg_op = false;
  double tension_sigma = 0.0;
};

// Compile a composer spec (Boost.JSON) into the resolved bundle + streaming config. `today_iso` supplies
// the value date when the spec omits one (compile_spec defaults to date.today()); pass "" to require the
// spec to carry `value_date`. Throws CompileError on bad structure.
CompileResult compile_spec(const boost::json::value& spec, const std::string& today_iso = "");

// The smoothing the spec ASKS for, as an engine RegSpec (E3-D4, 2026-09-10; ported from server/compile.py
// reg_spec so every host -- web, Excel C-ABI, run_json compile+sample -- calibrates the same curve).
// Default: the continuous tension-energy operator, "light" 0.02 / "strong" 0.2 / "off" 0; `reg_op ==
// "second_difference"` selects the discrete operator with its ~25x heavier scale 0.5 / 5.0. An
// under-determined OR banded spec floors "off" to "light" (a penalty is what makes those well-posed). The
// RegSpec spans every curve; `sigma` is the spec's tension_sigma. lambda == 0 => RegSpec::on() is false.
RegSpec compile_reg_spec(const CompileResult& r);

// Serialize a CompileResult to the SAME JSON document server/compile.py's compile_spec returns.
boost::json::value compile_to_json(const CompileResult& r);

// Convenience one-shot for the language/CLI seam: parse a spec JSON string, compile, return the result
// document as a string. Throws CompileError / std::exception on bad input.
std::string compile_json(const std::string& spec_json, const std::string& today_iso = "");

}  // namespace swaps::api
