// Options seam: a run_json "swaption" verb that prices European swaptions off a CALIBRATED SOFR curve.
// Request: {"swaption": {value_date, bundle, currency?, index?, curve?, trades:[{expiry, tenor, strike,
// payer?, normal_vol? , sabr?{alpha,rho,nu}}]}}. It calibrates the bundle, and for each trade builds the
// underlying forward-starting swap schedule (annual fixed), samples the curve's discount factors, forms the
// forward swap rate + annuity, and prices with the Bachelier normal model (SABR vol at the strike if given).
// Reuses BundleSession + the build/ date-schedule layer + the vol/ analytics. QuantLib-free.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/conventions_data.hpp"
#include "swaps/vol/swaption.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace v = swaps::vol;

namespace {
double jd(const json::object& o, const char* k, double d) {
  return o.contains(k) && !o.at(k).is_null() ? o.at(k).to_number<double>() : d;
}
std::string js(const json::object& o, const char* k, const char* d = "") {
  if (!o.contains(k) || o.at(k).is_null() || !o.at(k).is_string()) return d;
  return std::string(o.at(k).as_string().c_str());
}
bool jb(const json::object& o, const char* k, bool d) {
  return o.contains(k) && o.at(k).is_bool() ? o.at(k).as_bool() : d;
}
}  // namespace

std::string swaption_json(const std::string& request) {
  const json::value req = json::parse(request);
  const json::object& top = req.as_object();
  const json::object& o = top.contains("swaption") ? top.at("swaption").as_object() : top;

  const std::string vd_iso = js(o, "value_date");
  if (vd_iso.empty()) throw std::invalid_argument("swaption: missing 'value_date'");
  if (!o.contains("bundle")) throw std::invalid_argument("swaption: missing 'bundle'");
  const b::Date vd = b::Date::from_iso(vd_iso);
  const std::string currency = js(o, "currency", "USD");
  const std::string index = js(o, "index", "USD-SOFR");
  const int curve = static_cast<int>(jd(o, "curve", 0.0));
  const b::SwapConv conv = b::swap_conv(currency, 1.0, index);

  // Calibrate the bundle to its markets -> the curve we price off.
  cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
  BundleSession sess(std::move(prob));
  RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  for (int c = 0; c < static_cast<int>(sess.problem().curves.size()); ++c) reg.curves.push_back(c);
  sess.calibrate(flat_x0(sess.problem()), reg);

  json::array out_trades;
  if (o.contains("trades") && o.at("trades").is_array()) {
    for (const auto& te : o.at("trades").as_array()) {
      const json::object& t = te.as_object();
      const std::string expiry = js(t, "expiry"), tenor = js(t, "tenor");
      const double strike = jd(t, "strike", 0.0);
      const bool payer = jb(t, "payer", true);

      // Underlying forward-starting swap: option expires at `expiry`, swap starts spot after expiry, annual
      // fixed leg to `tenor`.
      const b::Date expiry_date = b::resolve(expiry, vd);
      const b::Date swap_start = b::spot_date(expiry_date, conv.calendar, conv.spot_lag);
      const int n = std::max(1, static_cast<int>(std::lround(conventions::period_years(tenor))));
      std::vector<double> pay_time, tau;
      b::Date prev = swap_start;
      for (int i = 1; i <= n; ++i) {
        const b::Date pay = b::adjust(conv.calendar, swap_start.plus_months(12 * i), conv.bdc);
        tau.push_back(b::year_frac(conv.fixed_dc, prev, pay));
        pay_time.push_back(b::curve_time(vd, pay));
        prev = pay;
      }
      const double t_start = b::curve_time(vd, swap_start);
      const double t_expiry = b::curve_time(vd, expiry_date);

      // Sample the calibrated curve's discount factors at the swap start + every pay date.
      std::vector<double> times{t_start};
      times.insert(times.end(), pay_time.begin(), pay_time.end());
      const std::vector<CurveSample> cs = sess.sample(times);
      const std::vector<double>& df = cs.at(static_cast<std::size_t>(curve)).discount;
      const double df_start = df.front();
      std::vector<double> df_pay(df.begin() + 1, df.end());
      const double df_end = df_pay.back();

      const v::ForwardSwap fs = v::forward_swap(df_start, df_end, df_pay, tau);
      const v::Payoff cp = payer ? v::Payoff::Payer : v::Payoff::Receiver;

      double vol, atm_vol;
      if (t.contains("sabr") && t.at("sabr").is_object()) {
        const auto& s = t.at("sabr").as_object();
        const v::SabrParams sp{jd(s, "alpha", 0.0), jd(s, "rho", 0.0), jd(s, "nu", 0.0)};
        vol = v::sabr_normal_vol(fs.rate, strike, t_expiry, sp);
        atm_vol = v::sabr_normal_vol(fs.rate, fs.rate, t_expiry, sp);
      } else {
        vol = atm_vol = jd(t, "normal_vol", 0.0);
      }
      const double price = v::swaption_price(fs.rate, fs.annuity, strike, vol, t_expiry, cp);

      json::object r;
      r["expiry"] = expiry;
      r["tenor"] = tenor;
      r["strike"] = strike;
      r["payer"] = payer;
      r["expiry_years"] = t_expiry;
      r["forward"] = fs.rate;
      r["annuity"] = fs.annuity;
      r["normal_vol"] = vol;
      r["atm_normal_vol"] = atm_vol;
      r["price"] = price;
      r["vega"] = v::bachelier_vega<double>(fs.rate, strike, vol, t_expiry, fs.annuity);
      r["delta"] = v::bachelier_delta<double>(fs.rate, strike, vol, t_expiry, fs.annuity, cp);
      out_trades.push_back(std::move(r));
    }
  }
  json::object out;
  out["trades"] = std::move(out_trades);
  return json::serialize(json::value(std::move(out)));
}

}  // namespace swaps::api
