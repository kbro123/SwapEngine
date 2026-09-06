// Credit seam: the stateless "credit" run_json verb. Calibrates a hazard-rate (survival) curve to a strip
// of par CDS spreads through the standard LM+AAD path (calibration::CreditProblem -> AadResidualEngine),
// then projects the survival probability Q(t), the forward hazard h(t) and the default density off the
// calibrated curve. Mirrors inflation_json / bonds_json exactly: a builder fills the plain CDS structs and
// the templated kernel does the math. QuantLib-free; units are model decimals (150 bp spread = 0.0150).
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/credit.hpp"
#include "swaps/build/credit_instruments.hpp"
#include "swaps/calibration/credit_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/hazard.hpp"

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

// A nominal discount-factor function DF(t). Either a flat continuous zero (DF=exp(-z·t)) or a log-linear
// interpolation of an explicit {discount_times, dfs} node set (flat-forward extrapolation beyond the ends).
struct DiscountFn {
  double z = 0.0;                    // flat zero (used when `t` is empty)
  std::vector<double> t, ln_df;      // node times and ln(DF) (log-linear => piecewise-constant forward)
  double operator()(double u) const {
    if (t.empty()) return std::exp(-z * u);
    if (u <= t.front()) {  // flat forward before the first node
      const double f0 = t.front() > 0.0 ? -ln_df.front() / t.front() : 0.0;
      return std::exp(-f0 * u);
    }
    if (u >= t.back()) {  // flat forward after the last node
      const std::size_t n = t.size();
      double f = 0.0;
      if (n >= 2 && t[n - 1] > t[n - 2])
        f = -(ln_df[n - 1] - ln_df[n - 2]) / (t[n - 1] - t[n - 2]);
      return std::exp(ln_df.back() - f * (u - t.back()));
    }
    const std::size_t hi = std::upper_bound(t.begin(), t.end(), u) - t.begin();
    const std::size_t lo = hi - 1;
    const double w = (u - t[lo]) / (t[hi] - t[lo]);
    return std::exp(ln_df[lo] + w * (ln_df[hi] - ln_df[lo]));
  }
};
}  // namespace

std::string credit_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("credit") && top.at("credit").is_object()
                              ? top.at("credit").as_object()
                              : top;
  if (!o.contains("instruments") || !o.at("instruments").is_array())
    throw std::invalid_argument("credit: missing 'instruments' array");

  const double recovery = jd(o, "recovery", 0.40);
  const int premium_freq = static_cast<int>(jd(o, "premium_freq", 4.0) + 0.5);
  const int prot_steps = static_cast<int>(jd(o, "prot_steps", 4.0) + 0.5);

  // Nominal discount curve: explicit {discount_times, dfs} node set, else a flat continuous zero.
  DiscountFn df;
  df.z = jd(o, "discount_zero", 0.0);
  const std::vector<double> dtimes = arrf(o, "discount_times");
  const std::vector<double> dfs = arrf(o, "dfs");
  if (!dtimes.empty() && dtimes.size() == dfs.size()) {
    df.t = dtimes;
    df.ln_df.reserve(dfs.size());
    for (double v : dfs) {
      if (!(v > 0.0)) throw std::invalid_argument("credit: dfs must be positive");
      df.ln_df.push_back(std::log(v));
    }
  } else if (!dtimes.empty()) {
    throw std::invalid_argument("credit: 'discount_times' and 'dfs' must have equal length");
  }

  // Build the CDS instruments and collect their maturities (the default hazard-curve knots).
  cal::CreditProblem prob;
  std::vector<double> maturities;
  for (const auto& iv : o.at("instruments").as_array()) {
    const json::object& io = iv.as_object();
    const std::string type = js(io, "type", "cds");
    if (type != "cds") throw std::invalid_argument("credit: instrument 'type' must be 'cds'");
    const double mat = jd(io, "maturity", 0.0);
    const double spread = jd(io, "spread", 0.0);
    const double rec = jd(io, "recovery", recovery);
    prob.instruments.push_back(b::make_cds(mat, spread, rec, df, premium_freq, prot_steps));
    maturities.push_back(mat);
  }
  if (prob.instruments.empty()) throw std::invalid_argument("credit: no instruments");

  // Hazard-curve back knots: explicit `knot_times`, else the sorted-unique instrument maturities.
  std::vector<double> knots = arrf(o, "knot_times");
  if (knots.empty()) {
    knots = maturities;
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
  }
  if (knots.empty()) throw std::invalid_argument("credit: could not determine knot_times");
  prob.back_times = knots;

  // Flat seed: the credit-triangle hazard implied by the first quote, h0 ≈ s0/(1−R) (>= a tiny floor).
  double seed_h = prob.instruments.front().market / std::max(1.0 - recovery, 1e-6);
  if (!std::isfinite(seed_h) || seed_h <= 0.0) seed_h = 0.01;
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(prob.n_knots(), seed_h);

  const cal::CalibrationResult res = cal::calibrate(prob, x0);

  // Project off the calibrated hazard curve (double).
  auto hz = curve::make_modular_curve<double>(curve::flat_hermite(prob.meeting_times, prob.back_times));
  hz.set_forwards(res.x);
  const curve::SurvivalCurve<double> surv{&hz};

  std::vector<double> out_t = arrf(o, "output_times");
  if (out_t.empty()) out_t = knots;

  std::vector<double> survival_v, hazard_v, density_v;
  for (double t : out_t) {
    survival_v.push_back(surv.survival(t));
    hazard_v.push_back(surv.hazard(t));
    density_v.push_back(surv.default_density(t));
  }
  std::vector<double> fitted;
  for (const auto& ins : prob.instruments) fitted.push_back(ins.model_quote<double>(surv));

  std::vector<double> hazard_knots(res.x.data(), res.x.data() + res.x.size());

  json::object out;
  out["knot_times"] = vecf(knots);
  out["hazards"] = vecf(hazard_knots);
  out["times"] = vecf(out_t);
  out["survival"] = vecf(survival_v);
  out["hazard_curve"] = vecf(hazard_v);
  out["default_density"] = vecf(density_v);
  out["par_spreads"] = vecf(fitted);
  out["recovery"] = recovery;
  out["stationarity"] = res.stationarity;
  out["rms_residual"] = res.rms_residual;
  out["rank_deficiency"] = res.rank_deficiency;
  out["iterations"] = res.iterations;
  out["n"] = static_cast<int>(prob.instruments.size());
  return json::serialize(json::value(std::move(out)));
}

}  // namespace swaps::api
