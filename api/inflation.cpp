// Inflation seam: the stateless "inflation" run_json verb. Calibrates a breakeven-inflation curve to a
// strip of ZCIS/YoY quotes through the standard LM+AAD path (calibration::InflationProblem ->
// AadResidualEngine), then projects the price index I(t) and the annual zero-coupon breakevens off the
// calibrated curve. Mirrors bonds_json / asset_swap_json: a builder fills the plain inflation structs and
// the templated kernel does the math. QuantLib-free; units are model decimals (0.025 = 2.5%).
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/inflation.hpp"
#include "swaps/build/inflation_instruments.hpp"
#include "swaps/calibration/inflation_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/inflation.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace cal = swaps::calibration;

namespace {
double jd(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
std::string js(const json::object& o, const char* k, const std::string& d = "") {
  if (!o.contains(k) || o.at(k).is_null() || !o.at(k).is_string()) return d;
  return std::string(o.at(k).as_string().c_str());
}
json::array vecf(const std::vector<double>& v) {
  json::array a;
  a.reserve(v.size());
  for (double x : v) a.push_back(x);
  return a;
}
std::vector<double> arrf(const json::object& o, const char* k) {
  std::vector<double> v;
  if (o.contains(k) && o.at(k).is_array())
    for (const auto& e : o.at(k).as_array()) v.push_back(e.to_number<double>());
  return v;
}
}  // namespace

std::string inflation_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("inflation") && top.at("inflation").is_object()
                              ? top.at("inflation").as_object()
                              : top;
  if (!o.contains("instruments") || !o.at("instruments").is_array())
    throw std::invalid_argument("inflation: missing 'instruments' array");

  if (!o.contains("base")) throw std::invalid_argument("inflation: missing 'base' (the index base level; instrument data, not a default)");
  const double base = jd(o, "base", 0.0);
  const double nominal_zero = jd(o, "nominal_zero", 0.0);

  cal::InflationProblem prob;
  prob.base = base;
  const std::vector<double> seas = arrf(o, "seasonality");
  if (seas.size() == 12) prob.seasonality = curve::Seasonality(seas);

  // Build the instruments and collect their maturities (the default breakeven-curve knots).
  std::vector<double> maturities;
  for (const auto& iv : o.at("instruments").as_array()) {
    const json::object& io = iv.as_object();
    const std::string type = js(io, "type", "zcis");
    const double rate = jd(io, "rate", 0.0);
    if (type == "zcis") {
      const double mat = jd(io, "maturity", 0.0);
      prob.instruments.push_back(b::inflation_zcis(mat, rate));
      maturities.push_back(mat);
    } else if (type == "yoy") {
      const double nz = jd(io, "nominal_zero", nominal_zero);
      const std::vector<double> ends = arrf(io, "ends");
      if (!ends.empty()) {
        prob.instruments.push_back(b::inflation_yoy(ends, rate, nz));
        maturities.push_back(ends.back());
      } else {
        const int years = static_cast<int>(jd(io, "years", 0.0) + 0.5);
        if (years <= 0) throw std::invalid_argument("inflation: yoy needs 'years' or 'ends'");
        prob.instruments.push_back(b::inflation_yoy_annual(years, rate, nz));
        maturities.push_back(static_cast<double>(years));
      }
    } else {
      throw std::invalid_argument("inflation: instrument 'type' must be 'zcis' or 'yoy'");
    }
  }
  if (prob.instruments.empty()) throw std::invalid_argument("inflation: no instruments");

  // Breakeven-curve back knots: explicit `knot_times`, else the sorted-unique instrument maturities.
  std::vector<double> knots = arrf(o, "knot_times");
  if (knots.empty()) {
    knots = maturities;
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
  }
  if (knots.empty()) throw std::invalid_argument("inflation: could not determine knot_times");
  prob.back_times = knots;

  // Flat seed: the continuous forward implied by the first quote, f0 = ln(1+rate0) (>= a tiny floor).
  double seed_f = std::log(1.0 + std::max(prob.instruments.front().market, -0.5));
  if (!std::isfinite(seed_f)) seed_f = 0.02;
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(prob.n_knots(), seed_f);

  const cal::CalibrationResult res = cal::calibrate(prob, x0);

  // Project off the calibrated breakeven curve (double).
  auto bei = curve::make_modular_curve<double>(curve::flat_hermite(prob.meeting_times, prob.back_times));
  bei.set_forwards(res.x);
  const curve::Seasonality* seas_ptr = prob.seasonality.active ? &prob.seasonality : nullptr;
  const curve::InflationIndexCurve<double> infl{base, &bei, seas_ptr};

  std::vector<double> out_t = arrf(o, "output_times");
  if (out_t.empty()) out_t = knots;

  std::vector<double> index_v, growth_v, fwd_be, zc_be;
  for (double t : out_t) {
    index_v.push_back(infl.index(t));
    growth_v.push_back(infl.growth(t));
    fwd_be.push_back(bei.forward(t));
    zc_be.push_back(t > 0.0 ? infl.zc_breakeven(t) : 0.0);
  }
  std::vector<double> fitted;
  for (const auto& ins : prob.instruments) fitted.push_back(ins.model_quote<double>(infl));

  std::vector<double> fwd_knots(res.x.data(), res.x.data() + res.x.size());

  json::object out;
  out["knot_times"] = vecf(knots);
  out["forwards"] = vecf(fwd_knots);
  out["times"] = vecf(out_t);
  out["index"] = vecf(index_v);
  out["growth"] = vecf(growth_v);
  out["forward_breakeven"] = vecf(fwd_be);
  out["zc_breakeven"] = vecf(zc_be);
  out["breakevens"] = vecf(fitted);
  out["stationarity"] = res.stationarity;
  out["rms_residual"] = res.rms_residual;
  out["rank_deficiency"] = res.rank_deficiency;
  out["iterations"] = res.iterations;
  out["n"] = static_cast<int>(prob.instruments.size());
  return json::serialize(json::value(std::move(out)));
}

}  // namespace swaps::api
