// Scenario / stress verb (declared in include/swaps/api/scenario.hpp): wire market::Scenario into
// production as the Tier-2 "what-if" primitive. Mirror the rv.cpp verb shape — parse a JSON doc,
// calibrate a bundle once, fork the fitted state per scenario, serialize.
//
// The pattern is "fork over the market" (market::Scenario): calibrate/anchor the base bundle ONE time,
// then for each scenario copy the fitted knot-forward vector and add the shock to that copy — the parent
// (base) state is never touched between scenarios. A per-curve shift is +bp on the shocked curve's
// forwards (added to that curve's interpolation knot forwards, which shifts forward(t) everywhere since
// the region schemes reproduce constants); a global `parallel_bp` shifts every un-keyed curve; an FX bump
// scales the pair. The book NPV under a shock reuses portfolio::MultiCurveBook pricing verbatim
// (build_bundle_curves at the shocked x, then book.value<double>) — no new pricing path is invented.
//
// CURVE NAMING. A BundleProblem addresses curves by INTEGER role, and a BundleCurveSpec carries no string
// name (only an engine-blind integer `currency` tag). So `shift_curve` keys are integer role indices (as
// JSON string keys, e.g. "0", "1") matching the bundle's curve order — the robust primary the task calls
// for. A non-integer key is rejected with a clear message rather than silently mapped.

#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/scenario.hpp"

#include "swaps/api/bundle_api.hpp"        // bundle_from_json / book_from_json / flat_x0 / BundleSession / RegSpec
#include "swaps/api/json_util.hpp"
#include "swaps/calibration/bundle_problem.hpp"  // build_bundle_curves / CurveHandle
#include "swaps/market/scenario.hpp"       // market::Scenario — the shock model wired in here

namespace swaps::api {

namespace json = boost::json;
namespace cal = swaps::calibration;
namespace mkt = swaps::market;
namespace pf = swaps::portfolio;

namespace {


// Parse a shift_curve key as an integer curve role. A bundle has no string curve names, so a non-integer
// key is an error (documented on the header) rather than a silent no-op.
int role_from_key(const std::string& key) {
  try {
    std::size_t pos = 0;
    const int r = std::stoi(key, &pos);
    if (pos != key.size()) throw std::invalid_argument("trailing");
    return r;
  } catch (const std::exception&) {
    throw std::invalid_argument(
        "scenario: shift_curve key '" + key +
        "' is not an integer curve role (curves in a bundle are addressed by integer index, not name)");
  }
}

// Sample every bundle curve at an arbitrary state x on `times` (zero/forward/discount) — the same read
// BundleSession::sample() does, but at a forked (shocked) x rather than the session's calibrated x.
json::array sample_curves_at(const cal::BundleProblem& P, const Eigen::VectorXd& x,
                             const std::vector<double>& times) {
  std::vector<int> off(P.n_curves());
  for (int c = 0; c < P.n_curves(); ++c) off[c] = P.offset(c);
  const auto C = cal::build_bundle_curves<double>(
      P.curves, [&](int c, int i) { return x[off[c] + i]; });
  json::array arr;
  for (int c = 0; c < P.n_curves(); ++c) {
    json::object co;
    co["currency"] = P.curves[c].currency;
    std::vector<double> disc, zero, fwd;
    disc.reserve(times.size());
    zero.reserve(times.size());
    fwd.reserve(times.size());
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

// Book NPV at an arbitrary state x — reuses MultiCurveBook::value (the SAME kernel price_portfolio uses),
// building the double curve handles off x exactly as BundleSession::price_portfolio does.
double book_npv_at(const pf::MultiCurveBook& book, const cal::BundleProblem& P, const Eigen::VectorXd& x) {
  if (book.positions.empty()) return 0.0;
  std::vector<int> off(P.n_curves());
  for (int c = 0; c < P.n_curves(); ++c) off[c] = P.offset(c);
  const auto C = cal::build_bundle_curves<double>(
      P.curves, [&](int c, int i) { return x[off[c] + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  return book.value<double>(curve_of);
}

}  // namespace

std::string scenario_json(const json::object& request) {
  const json::object& top = request;
  // Tolerant of being called as either the wrapped {"scenario": {...}} envelope (the run_json seam) or a
  // bare scenario object.
  const json::object& o =
      (top.contains("scenario") && top.at("scenario").is_object()) ? top.at("scenario").as_object() : top;

  if (!o.contains("bundle")) throw std::invalid_argument("scenario: missing 'bundle' object");
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  if (prob.n_curves() == 0) throw std::invalid_argument("scenario: bundle has no curves");

  BundleSession sess(std::move(prob));
  const cal::BundleProblem& P = sess.problem();

  // ---- calibrate the BASE exactly once (the anchor every scenario forks from) ----------------------
  Eigen::VectorXd x0;
  if (o.contains("x0")) {
    const std::vector<double> xv = darr(o, "x0");
    if (static_cast<int>(xv.size()) != P.n_knots())
      throw std::invalid_argument("scenario: x0 length does not match the bundle's knot count");
    x0 = Eigen::Map<const Eigen::VectorXd>(xv.data(), static_cast<Eigen::Index>(xv.size()));
  } else {
    x0 = flat_x0(P);
  }
  RegSpec reg;
  if (o.contains("regularize") && o.at("regularize").is_object()) {
    const auto& r = o.at("regularize").as_object();
    reg.lambda = jd(r, "lambda", 0.0);
    if (r.contains("curves") && r.at("curves").is_array())
      for (const auto& e : r.at("curves").as_array()) reg.curves.push_back(static_cast<int>(e.to_number<long long>()));
    reg.tension = r.contains("tension") && r.at("tension").as_bool();
    reg.sigma = jd(r, "sigma", 0.0);
  }
  sess.calibrate(x0, reg);
  const Eigen::VectorXd x_base = sess.x();  // the anchor: NEVER mutated below (each scenario copies it)

  const std::vector<double> times = darr(o, "sample_times");
  const bool has_book = o.contains("book");
  pf::MultiCurveBook book;
  if (has_book) book = book_from_json(o.at("book"));
  const double base_npv = has_book ? book_npv_at(book, P, x_base) : 0.0;

  json::object out;
  out["n_curves"] = P.n_curves();
  out["n_knots"] = P.n_knots();
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

  // ---- per-scenario fork over the anchor ----------------------------------------------------------
  json::array scen_out;
  if (o.contains("scenarios") && o.at("scenarios").is_array())
    for (const auto& e : o.at("scenarios").as_array()) {
      const auto& so = e.as_object();

      // Build the declarative market::Scenario (the shock model wired in). A global parallel constructs a
      // Scenario with a global default; explicit shift_curve entries override it per curve. Curves are keyed
      // by their integer role rendered as a string, so Scenario::curve_shift / shocked_forwards drive it.
      mkt::Scenario scn;
      bool has_par = false;
      double parallel_bp = 0.0;
      if (so.contains("parallel_bp")) {
        parallel_bp = so.at("parallel_bp").to_number<double>();
        has_par = true;
        scn = mkt::Scenario::parallel(parallel_bp);
      }
      std::map<int, double> curve_bp;
      if (so.contains("shift_curve") && so.at("shift_curve").is_object())
        for (const auto& kv : so.at("shift_curve").as_object()) {
          const int role = role_from_key(std::string(kv.key()));
          if (role < 0 || role >= P.n_curves())
            throw std::invalid_argument("scenario: shift_curve role " + std::to_string(role) +
                                        " is out of range for this bundle");
          const double bp = kv.value().to_number<double>();
          curve_bp[role] = bp;
          scn.shift_curve(std::to_string(role), bp);
        }

      // FX bumps: recorded on the Scenario, and (for the book) compounded into a single scale factor
      // applied to every xccy position's fx_spot. A MultiCurveBook position carries no currency identity,
      // so pairs cannot be matched to positions; for a single-pair xccy book this factor is exact, and
      // multiple bumps compound. Documented, simple, and it never touches a non-xccy position.
      std::vector<std::tuple<std::string, std::string, double>> fxb;
      double fx_factor = 1.0;
      if (so.contains("bump_fx") && so.at("bump_fx").is_array())
        for (const auto& fe : so.at("bump_fx").as_array()) {
          const auto& fo = fe.as_object();
          const std::string base = js(fo, "base"), quote = js(fo, "quote");
          const double rel = jd(fo, "rel", 0.0);
          scn.bump_fx(base, quote, rel);
          fxb.emplace_back(base, quote, rel);
          fx_factor *= (1.0 + rel);
        }

      // Fork the anchor: copy x_base and add each curve's shift (explicit else global) to its interp
      // forwards. Turn δ overlays (the state entries after the interp knots) are left alone — a shift is a
      // level move on the forward curve, not a change to a localized turn jump.
      Eigen::VectorXd xs = x_base;
      for (int c = 0; c < P.n_curves(); ++c) {
        const std::string name = std::to_string(c);
        const double d = scn.curve_shift(name);  // rate space: explicit shift_curve, else global, else 0
        if (d == 0.0) continue;
        const int oc = P.offset(c);
        const int ni = P.curves[c].n_interp_knots();
        const Eigen::VectorXd xi = xs.segment(oc, ni);          // this curve's interp forwards
        xs.segment(oc, ni) = scn.shocked_forwards(name, xi);    // += d on every interp forward
      }

      json::object row;
      row["name"] = js(so, "name", "");
      if (!times.empty()) row["curves"] = sample_curves_at(P, xs, times);
      if (has_book) {
        double npv;
        if (fx_factor != 1.0) {
          pf::MultiCurveBook sbook = book;  // shocked copy: scale the FX-reset notional of xccy legs
          for (auto& p : sbook.positions)
            if (p.kind == pf::MultiCurveBook::Kind::Xccy) p.fx_spot *= fx_factor;
          npv = book_npv_at(sbook, P, xs);
        } else {
          npv = book_npv_at(book, P, xs);
        }
        row["npv"] = npv;
        row["npv_delta"] = npv - base_npv;
      }
      // Echo the effective shock (so a caller can confirm what was applied).
      if (has_par) row["parallel_bp"] = parallel_bp;
      if (!curve_bp.empty()) {
        json::object sb;
        for (const auto& kv : curve_bp) sb[std::to_string(kv.first)] = kv.second;
        row["shift_bp"] = std::move(sb);
      }
      if (!fxb.empty()) {
        json::array fa;
        for (const auto& f : fxb) {
          json::object fo;
          fo["base"] = std::get<0>(f);
          fo["quote"] = std::get<1>(f);
          fo["rel"] = std::get<2>(f);
          fa.push_back(std::move(fo));
        }
        row["fx"] = std::move(fa);
      }
      scen_out.push_back(std::move(row));
    }
  out["scenarios"] = std::move(scen_out);

  json::object resp;
  resp["scenario"] = std::move(out);
  return json::serialize(resp);
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string scenario_json(const std::string& request) { return scenario_json(json::parse(request).as_object()); }
}  // namespace swaps::api
