// Exposure seam: the stateless "exposure" run_json verb — an EPE/ENE/PFE counterparty-exposure profile for a
// swap book off a CALIBRATED SOFR curve. Calibrates the bundle (like swaption_json), converts the book to the
// native single-curve Portfolio, simulates a Gaussian curve-state grid x(t,path)=x_cal+sigma·sqrt(t)·Z, and
// runs the MC-exposure kernel (CompiledPortfolio::npv_grid, ~1.9M full-book reprices/sec) + the EPE/ENE/PFE
// aggregation (swaps/xva/exposure.hpp). The verb decodes and serialises; the behaviour is derive/exposure.hpp
// (2026-11-02, review item 2.9). QuantLib-free.
//
// NETTING SETS: the legacy form ("book") treats the whole book as ONE netting set. The request may instead
// (or additionally) carry "netting_sets": typed trades grouped under a CSA (trade::NettingSet), where the
// CSA's collateral currency DECIDES the discount role (collateral OIS via trade::CSA) and the exposure
// profile is computed PER SET — netting aggregates within a set, never across sets. Same npv_grid kernel,
// same simulated state grid; only the AGGREGATION BOUNDARY moves. Schema (top-level or under "exposure"):
//   "value_date": "2026-09-04",                          required by netting_sets (trade materialization)
//   "curve_roles": {"USD-SOFR": 0},                      index id -> bundle curve index (as book_from_json)
//   "netting_sets": [{"id": "CP-1", "csa": {"collateral_currency": "USD"},
//                     "trades": [{id, notional, pay, fixed_rate, index, effective, maturity}, ...]}]
// Response gains "netting_sets": [{id, node_time, epe, ene, pfe, mtm, n}, ...]; the legacy whole-book
// fields are present (and unchanged) exactly when "book" is present.
//
// DEMO SCOPE: a single classic (Flat+Hermite) self-discounting curve; xccy / custom-region / turn'd curves
// and non-swap positions are rejected/skipped (CompiledPortfolio is single-curve). The Gaussian curve-state
// proxy is illustrative — a calibrated LGM/HW1F is a later step; the kernel is model-agnostic (pure
// f(exp(-W·x))).
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/exposure.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/derive/exposure.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace cal = swaps::calibration;
namespace der = swaps::derive;
namespace pf = swaps::portfolio;

namespace {
json::object profile_json(const der::ExposureProfileResult& r) {
  json::object so;
  so["node_time"] = vecf(r.profile.node_time);
  so["epe"] = vecf(r.profile.epe);
  so["ene"] = vecf(r.profile.ene);
  so["pfe"] = vecf(r.profile.pfe);
  so["mtm"] = r.mtm;
  return so;
}
}  // namespace

std::string exposure_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o =
      top.contains("exposure") && top.at("exposure").is_object() ? top.at("exposure").as_object() : top;

  der::ExposureParams prm;
  prm.n_paths = static_cast<int>(jd(o, "n_paths", prm.n_paths));
  prm.n_nodes = static_cast<int>(jd(o, "n_nodes", prm.n_nodes));
  prm.horizon = jd(o, "horizon_years", prm.horizon);
  prm.sigma = jd(o, "sigma", prm.sigma);
  prm.kappa = jd(o, "kappa", prm.kappa);
  prm.pfe_q = jd(o, "pfe_q", prm.pfe_q);
  if (!o.contains("bundle")) throw std::invalid_argument("exposure: missing 'bundle'");
  const bool has_book = o.contains("book");
  const bool has_sets = o.contains("netting_sets") && o.at("netting_sets").is_array();
  if (!has_book && !has_sets) throw std::invalid_argument("exposure: missing 'book'");
  if (has_sets && o.at("netting_sets").as_array().empty() && !has_book)
    throw std::invalid_argument("exposure: 'netting_sets' is empty");

  // Calibrate the bundle -> the discount curve we reprice against (same recipe as swaption_json).
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  BundleSession sess(std::move(prob));
  sess.calibrate(flat_x0(sess.problem()));  // smoothing is the bundle's region data (regions[].reg_lambda)

  // Legacy whole-book path (behaviour unchanged): the whole book is one netting set. Netting-set path: each set =
  // trade::NettingSet (trades under one CSA), the CSA's collateral-currency OIS DECIDING the discount role.
  std::optional<pf::MultiCurveBook> legacy_book;
  if (has_book) legacy_book = book_from_json(o.at("book"));
  const std::vector<der::ExposureSet> sets = netting_sets_from_json(o);

  const der::ExposureResult res = der::exposure(sess.problem().curves, sess.x(), prm, legacy_book, sets);

  json::object out;
  if (res.book) out = profile_json(*res.book);
  json::array sets_out;
  for (const auto& [id, r] : res.sets) {
    json::object so = profile_json(r);
    so["id"] = id;
    so["n"] = r.n;
    sets_out.push_back(std::move(so));
  }
  out["wall_us"] = res.wall_us;
  out["n_paths"] = res.n_paths;
  out["n_nodes"] = res.n_nodes;
  if (res.book) out["n"] = res.book->n;
  if (has_sets) out["netting_sets"] = std::move(sets_out);
  return json::serialize(json::value(std::move(out)));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string exposure_json(const std::string& request) { return exposure_json(json::parse(request).as_object()); }
}  // namespace swaps::api
