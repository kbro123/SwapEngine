// Implementation of the bundle API facade (include/swaps/api/bundle_api.hpp).
//
// Boost.JSON is compiled ONCE here via <boost/json/src.hpp> so no other TU pays for it. Everything else
// is the thin glue between the JSON object graph and the engine's generic instrument model.

#include "swaps/api/bundle_api.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

#include <boost/json/src.hpp>  // header-only Boost.JSON, compiled in this TU only

#include "swaps/api/compile.hpp"                   // compile_spec / compile_to_json (the 'compile' verb)
#include "swaps/api/generate_risk.hpp"             // generate_risk_json (the 'generate_risk' verb)
#include "swaps/api/options.hpp"                   // swaption_json (the 'swaption' verb)
#include "swaps/api/bond.hpp"                       // bonds_json (the 'bonds' verb)
#include "swaps/api/exposure.hpp"                   // exposure_json (the 'exposure' verb)
#include "swaps/calibration/compiled_bundle.hpp"  // CompiledBundleResidual::model_rates (streaming anchor)
#include <type_traits>
#include "swaps/calibration/jacobian.hpp"         // aad_jacobian (risk operator)
#include "swaps/calibration/regularize.hpp"       // smoothed(), second_difference_operator()
#include "swaps/calibration/structure_fingerprint.hpp"  // structure_fingerprint (warm-vs-recompile switch)

namespace swaps::api {

namespace json = boost::json;
namespace px = swaps::pricing;
namespace curve = swaps::curve;
namespace ad = swaps::ad;  // Dual (forward-AAD scalar) for the PV01 pass in price_portfolio

// =================================================================================================
// JSON accessors -- tolerant: a missing key returns the supplied default (so a minimal document is
// valid and every field defaults to the engine struct's own default).
// =================================================================================================
namespace {

double get_d(const json::object& o, const char* k, double d) {
  return o.contains(k) ? o.at(k).to_number<double>() : d;
}
int get_i(const json::object& o, const char* k, int d) {
  return o.contains(k) ? static_cast<int>(o.at(k).to_number<long long>()) : d;
}
bool get_b(const json::object& o, const char* k, bool d) {
  return o.contains(k) ? o.at(k).as_bool() : d;
}
std::string get_s(const json::object& o, const char* k, const char* d) {
  if (!o.contains(k)) return d;
  const auto& s = o.at(k).as_string();
  return std::string(s.begin(), s.end());
}
std::vector<double> get_da(const json::object& o, const char* k) {
  std::vector<double> out;
  if (o.contains(k) && o.at(k).is_array())
    for (const auto& e : o.at(k).as_array()) out.push_back(e.to_number<double>());
  return out;
}
std::vector<int> get_ia(const json::object& o, const char* k) {
  std::vector<int> out;
  if (o.contains(k) && o.at(k).is_array())
    for (const auto& e : o.at(k).as_array()) out.push_back(static_cast<int>(e.to_number<long long>()));
  return out;
}
json::array da(const std::vector<double>& v) {
  json::array a;
  a.reserve(v.size());
  for (double x : v) a.push_back(x);
  return a;
}
json::array da(const Eigen::VectorXd& v) {
  json::array a;
  a.reserve(v.size());
  for (int i = 0; i < v.size(); ++i) a.push_back(v[i]);
  return a;
}

const char* quote_to_str(cal::QuoteKind q) {
  switch (q) {
    case cal::QuoteKind::ParRate: return "ParRate";
    case cal::QuoteKind::ParSpread: return "ParSpread";
    case cal::QuoteKind::Rate: return "Rate";
    case cal::QuoteKind::FxForward: return "FxForward";
    case cal::QuoteKind::XccyMtmBasis: return "XccyMtmBasis";
    case cal::QuoteKind::Portfolio: return "Portfolio";
    case cal::QuoteKind::TurnJump: return "TurnJump";
  }
  return "ParRate";
}
cal::QuoteKind quote_from_str(const std::string& s) {
  if (s == "ParSpread") return cal::QuoteKind::ParSpread;
  if (s == "Rate") return cal::QuoteKind::Rate;
  if (s == "FxForward") return cal::QuoteKind::FxForward;
  if (s == "XccyMtmBasis") return cal::QuoteKind::XccyMtmBasis;
  if (s == "Portfolio") return cal::QuoteKind::Portfolio;
  if (s == "TurnJump") return cal::QuoteKind::TurnJump;
  if (s == "ParRate") return cal::QuoteKind::ParRate;
  throw std::invalid_argument("unknown quote kind: " + s);
}

// ---- per-object parse -------------------------------------------------------------------------
px::RateObservation obs_from(const json::object& o) {
  px::RateObservation r;
  r.sub_start = get_da(o, "sub_start");
  r.sub_end = get_da(o, "sub_end");
  r.weight = get_da(o, "weight");
  r.realized = get_d(o, "realized", 0.0);
  r.tau_index = get_d(o, "tau_index", 0.0);
  r.fixing_step = get_d(o, "fixing_step", 0.0);
  r.fixing_step3 = get_d(o, "fixing_step3", 0.0);
  r.compounded = get_b(o, "compounded", false);
  r.realized_factor = get_d(o, "realized_factor", 1.0);
  // E2: an optional fixing SCHEDULE (index + per-day dates/accruals) — when present the engine RESOLVES
  // realized + forecast subs from the pricing context's fixing table instead of reading a baked `realized`.
  r.fixing_index = get_s(o, "fixing_index", "");
  if (o.contains("fixing_schedule") && o.at("fixing_schedule").is_array()) {
    for (const auto& e : o.at("fixing_schedule").as_array()) {
      const auto& d = e.as_object();
      px::FixingDay fd;
      fd.fixing_date = static_cast<int>(get_d(d, "fixing_date", 0.0));
      fd.accrual = get_d(d, "accrual", 0.0);
      fd.t_start = get_d(d, "t_start", 0.0);
      fd.t_end = get_d(d, "t_end", 0.0);
      fd.weight = get_d(d, "weight", 1.0);
      r.fixing_schedule.push_back(fd);
    }
  }
  return r;
}
px::FloatCoupon fcpn_from(const json::object& o) {
  px::FloatCoupon c;
  if (o.contains("obs")) c.obs = obs_from(o.at("obs").as_object());
  c.pay = get_d(o, "pay", 0.0);
  c.tau_pay = get_d(o, "tau_pay", 0.0);
  c.spread = get_d(o, "spread", 0.0);
  c.scale = get_d(o, "scale", 1.0);
  c.reset_time = get_d(o, "reset_time", -1.0);
  return c;
}
px::FixedCoupon xcpn_from(const json::object& o) {
  px::FixedCoupon c;
  c.pay = get_d(o, "pay", 0.0);
  c.tau = get_d(o, "tau", 0.0);
  c.scale = get_d(o, "scale", 1.0);
  return c;
}
cal::FloatLeg fleg_from(const json::object& o) {
  cal::FloatLeg L;
  if (o.contains("coupons"))
    for (const auto& e : o.at("coupons").as_array()) L.coupons.push_back(fcpn_from(e.as_object()));
  L.forecast = get_i(o, "forecast", 0);
  L.discount = get_i(o, "discount", 0);
  L.reset_num = get_i(o, "reset_num", -1);
  L.reset_den = get_i(o, "reset_den", -1);
  L.fx_spot = get_d(o, "fx_spot", 1.0);
  return L;
}
cal::FixedLeg xleg_from(const json::object& o) {
  cal::FixedLeg L;
  if (o.contains("coupons"))
    for (const auto& e : o.at("coupons").as_array()) L.coupons.push_back(xcpn_from(e.as_object()));
  L.discount = get_i(o, "discount", 0);
  return L;
}
curve::Scheme scheme_from_str(const std::string& s) {
  if (s == "Flat") return curve::Scheme::Flat;
  if (s == "Linear") return curve::Scheme::Linear;
  if (s == "NaturalCubic") return curve::Scheme::NaturalCubic;
  if (s == "Hermite") return curve::Scheme::Hermite;
  if (s == "MonotoneCubic") return curve::Scheme::MonotoneCubic;
  if (s == "BSpline") return curve::Scheme::BSpline;
  if (s == "Tension") return curve::Scheme::Tension;
  throw std::invalid_argument("unknown interpolation scheme: " + s);
}
const char* scheme_to_str(curve::Scheme s) {
  switch (s) {
    case curve::Scheme::Flat: return "Flat";
    case curve::Scheme::Linear: return "Linear";
    case curve::Scheme::NaturalCubic: return "NaturalCubic";
    case curve::Scheme::Hermite: return "Hermite";
    case curve::Scheme::MonotoneCubic: return "MonotoneCubic";
    case curve::Scheme::BSpline: return "BSpline";
    case curve::Scheme::Tension: return "Tension";
  }
  return "Hermite";
}

cal::BundleCurveSpec spec_from(const json::object& o) {
  cal::BundleCurveSpec s;
  const std::vector<double> legacy_meeting = get_da(o, "meeting");  // legacy Flat+Hermite layout (pre-regions)
  const std::vector<double> legacy_back = get_da(o, "back");
  s.base = get_i(o, "base", -1);
  s.currency = get_i(o, "currency", 0);
  if (o.contains("regions") && o.at("regions").is_array())
    for (const auto& e : o.at("regions").as_array()) {
      const auto& ro = e.as_object();
      curve::CurveModule m;
      m.scheme = scheme_from_str(get_s(ro, "scheme", "Hermite"));
      m.knots = get_da(ro, "knots");
      m.sigma = get_d(ro, "sigma", 0.0);  // tension hyperparameter (Scheme::Tension only); else ignored
      m.reg_lambda = get_d(ro, "reg_lambda", -1.0);  // per-region smoothing weight (Phase 1); <0 = inherit
      m.reg_sigma = get_d(ro, "reg_sigma", -1.0);    // per-region tension-energy σ (Phase 2); <0 = inherit
      s.regions.push_back(std::move(m));
    }
  // Calibration TURNS (docs/turns-calibration.md, Mode 2): an OPTIONAL array of overlay windows. Absent
  // -> empty, byte-identical to a turn-free curve. Each δ appends one free state var after the interp knots.
  if (o.contains("turns") && o.at("turns").is_array())
    for (const auto& e : o.at("turns").as_array()) {
      const auto& to = e.as_object();
      px::Turn t;
      t.start = get_d(to, "start", 0.0);
      t.end = get_d(to, "end", 0.0);
      s.turns.push_back(t);
    }
  // Backward-compat: a legacy bundle that named the Flat(meeting)+Hermite(back) layout instead of
  // `regions` is normalised to the equivalent two-region layout (identical to the old modules()).
  if (s.regions.empty() && !(legacy_meeting.empty() && legacy_back.empty()))
    s.regions = curve::flat_hermite(legacy_meeting, legacy_back);
  return s;
}

// ---- per-object serialize ----------------------------------------------------------------------
json::object obs_to(const px::RateObservation& r) {
  json::object o;
  o["sub_start"] = da(r.sub_start);
  o["sub_end"] = da(r.sub_end);
  o["weight"] = da(r.weight);
  o["realized"] = r.realized;
  o["tau_index"] = r.tau_index;
  o["fixing_step"] = r.fixing_step;
  o["fixing_step3"] = r.fixing_step3;
  o["compounded"] = r.compounded;
  o["realized_factor"] = r.realized_factor;
  if (!r.fixing_schedule.empty()) {
    o["fixing_index"] = r.fixing_index;
    json::array sch;
    for (const px::FixingDay& d : r.fixing_schedule) {
      json::object jd;
      jd["fixing_date"] = d.fixing_date;
      jd["accrual"] = d.accrual;
      jd["t_start"] = d.t_start;
      jd["t_end"] = d.t_end;
      jd["weight"] = d.weight;
      sch.push_back(std::move(jd));
    }
    o["fixing_schedule"] = std::move(sch);
  }
  return o;
}
json::object fcpn_to(const px::FloatCoupon& c) {
  json::object o;
  o["obs"] = obs_to(c.obs);
  o["pay"] = c.pay;
  o["tau_pay"] = c.tau_pay;
  o["spread"] = c.spread;
  o["scale"] = c.scale;
  o["reset_time"] = c.reset_time;
  return o;
}
json::object xcpn_to(const px::FixedCoupon& c) {
  json::object o;
  o["pay"] = c.pay;
  o["tau"] = c.tau;
  o["scale"] = c.scale;
  return o;
}
json::object fleg_to(const cal::FloatLeg& L) {
  json::object o;
  json::array cs;
  for (const auto& c : L.coupons) cs.push_back(fcpn_to(c));
  o["coupons"] = cs;
  o["forecast"] = L.forecast;
  o["discount"] = L.discount;
  o["reset_num"] = L.reset_num;
  o["reset_den"] = L.reset_den;
  o["fx_spot"] = L.fx_spot;
  return o;
}
json::object xleg_to(const cal::FixedLeg& L) {
  json::object o;
  json::array cs;
  for (const auto& c : L.coupons) cs.push_back(xcpn_to(c));
  o["coupons"] = cs;
  o["discount"] = L.discount;
  return o;
}
json::object spec_to(const cal::BundleCurveSpec& s) {
  json::object o;
  o["base"] = s.base;
  o["currency"] = s.currency;
  // Canonical curve representation on the wire: ALWAYS `regions`. modules() collapses the legacy Flat-front
  // + Hermite-back `meeting`/`back` layout into the identical region list, so a meeting/back-authored curve
  // and a region-authored curve serialize byte-identically — one representation, not two. (bundle_from_json
  // still READS meeting/back, so bundles persisted before this change still load.)
  json::array rs;
  for (const auto& m : s.modules()) {
    json::object mo;
    mo["scheme"] = scheme_to_str(m.scheme);
    mo["knots"] = da(m.knots);
    if (m.scheme == curve::Scheme::Tension) mo["sigma"] = m.sigma;
    if (m.reg_lambda >= 0.0) mo["reg_lambda"] = m.reg_lambda;  // emit only when set (default stays byte-identical)
    if (m.reg_sigma >= 0.0) mo["reg_sigma"] = m.reg_sigma;
    rs.push_back(std::move(mo));
  }
  o["regions"] = std::move(rs);
  // Emit turns ONLY when present, so a turn-free curve serializes byte-identically (mirrors `regions`).
  if (!s.turns.empty()) {
    json::array ts;
    for (const auto& t : s.turns) {
      json::object to;
      to["start"] = t.start;
      to["end"] = t.end;
      ts.push_back(to);
    }
    o["turns"] = ts;
  }
  return o;
}

std::string err(const std::string& msg) {
  json::object o;
  o["error"] = msg;
  return json::serialize(json::value(std::move(o)));
}

}  // namespace

// =================================================================================================
// Public (de)serializers
// =================================================================================================
cal::Instrument instrument_from_json(const json::value& v) {
  const auto& o = v.as_object();
  cal::Instrument ins;
  ins.quote = quote_from_str(get_s(o, "quote", "ParRate"));
  if (o.contains("fwd")) ins.fwd = fleg_from(o.at("fwd").as_object());
  if (o.contains("bench")) ins.bench = fleg_from(o.at("bench").as_object());
  if (o.contains("fixed")) ins.fixed = xleg_from(o.at("fixed").as_object());
  if (o.contains("mtm")) ins.mtm = fleg_from(o.at("mtm").as_object());
  if (o.contains("obs")) ins.obs = obs_from(o.at("obs").as_object());
  ins.forecast = get_i(o, "forecast", 0);
  ins.convexity = get_d(o, "convexity", 0.0);
  ins.market = get_d(o, "market", 0.0);
  ins.pv_currency = get_i(o, "pv_currency", 0);
  ins.fx_num = get_i(o, "fx_num", -1);
  ins.fx_den = get_i(o, "fx_den", -1);
  ins.fx_spot = get_d(o, "fx_spot", 1.0);
  ins.fx_time = get_d(o, "fx_time", 0.0);
  ins.band_lower = get_d(o, "band_lower", 0.0);
  ins.band_upper = get_d(o, "band_upper", 0.0);
  ins.band_decay = get_d(o, "band_decay", 1.0);
  ins.turn_curve = get_i(o, "turn_curve", 0);  // TurnJump only: (curve, index) of the pinned turn
  ins.turn_index = get_i(o, "turn_index", 0);
  if (o.contains("combination"))  // Portfolio components (recursive)
    for (const auto& e : o.at("combination").as_array()) {
      const auto& c = e.as_object();
      ins.combination.push_back({get_d(c, "weight", 1.0), instrument_from_json(c.at("instrument"))});
    }
  return ins;
}

json::value instrument_to_json(const cal::Instrument& ins) {
  json::object o;
  o["quote"] = quote_to_str(ins.quote);
  o["fwd"] = fleg_to(ins.fwd);
  o["bench"] = fleg_to(ins.bench);
  o["fixed"] = xleg_to(ins.fixed);
  o["mtm"] = fleg_to(ins.mtm);
  o["obs"] = obs_to(ins.obs);
  o["forecast"] = ins.forecast;
  o["convexity"] = ins.convexity;
  o["market"] = ins.market;
  o["pv_currency"] = ins.pv_currency;
  o["fx_num"] = ins.fx_num;
  o["fx_den"] = ins.fx_den;
  o["fx_spot"] = ins.fx_spot;
  o["fx_time"] = ins.fx_time;
  o["band_lower"] = ins.band_lower;
  o["band_upper"] = ins.band_upper;
  o["band_decay"] = ins.band_decay;
  o["turn_curve"] = ins.turn_curve;  // TurnJump only: (curve, index) of the pinned turn
  o["turn_index"] = ins.turn_index;
  if (!ins.combination.empty()) {
    json::array combo;
    for (const auto& c : ins.combination) {
      json::object ce;
      ce["weight"] = c.weight;
      ce["instrument"] = instrument_to_json(c.instrument);  // recursive
      combo.push_back(std::move(ce));
    }
    o["combination"] = std::move(combo);
  }
  return o;
}

cal::BundleProblem bundle_from_json(const json::value& v) {
  const auto& o = v.as_object();
  cal::BundleProblem p;
  if (o.contains("curves"))
    for (const auto& e : o.at("curves").as_array()) p.curves.push_back(spec_from(e.as_object()));
  if (o.contains("instruments"))
    for (const auto& e : o.at("instruments").as_array()) p.instruments.push_back(instrument_from_json(e));
  return p;
}

json::value bundle_to_json(const cal::BundleProblem& p) {
  json::object o;
  json::array cs;
  for (const auto& c : p.curves) cs.push_back(spec_to(c));
  o["curves"] = cs;
  json::array is;
  for (const auto& ins : p.instruments) is.push_back(instrument_to_json(ins));
  o["instruments"] = is;
  return o;
}

namespace pf = swaps::portfolio;

// A book of positions to reprice (schema documented on the declaration). Reuses the SAME coupon parsers
// (fcpn_from/xcpn_from) as the instrument (de)serializer, so the web reuses its existing schedule builders.
pf::MultiCurveBook book_from_json(const json::value& v) {
  pf::MultiCurveBook book;
  const auto& o = v.as_object();
  if (!o.contains("positions") || !o.at("positions").is_array()) return book;
  auto floats = [](const json::object& po, const char* key, std::vector<px::FloatCoupon>& out) {
    if (po.contains(key) && po.at(key).is_array())
      for (const auto& e : po.at(key).as_array()) out.push_back(fcpn_from(e.as_object()));
  };
  for (const auto& e : o.at("positions").as_array()) {
    const auto& po = e.as_object();
    pf::MultiCurveBook::Position p;
    p.kind = (get_s(po, "kind", "swap") == "xccy") ? pf::MultiCurveBook::Kind::Xccy
                                                   : pf::MultiCurveBook::Kind::Swap;
    p.notional = get_d(po, "notional", 1.0);
    p.fixed_rate = get_d(po, "fixed_rate", 0.0);
    p.fwd_curve = get_i(po, "fwd_curve", 0);
    p.disc_curve = get_i(po, "disc_curve", 0);
    p.fixed_curve = get_i(po, "fixed_curve", p.disc_curve);  // defaults to the float discount curve
    floats(po, "float_coupons", p.float_coupons);
    if (po.contains("fixed_coupons") && po.at("fixed_coupons").is_array())
      for (const auto& c : po.at("fixed_coupons").as_array()) p.fixed_coupons.push_back(xcpn_from(c.as_object()));
    // xccy-only: the resetting foreign funding leg + its FX-forward reset roles.
    p.fx_spot = get_d(po, "fx_spot", 1.0);
    p.mtm_fwd_curve = get_i(po, "mtm_fwd_curve", 0);
    p.mtm_disc_curve = get_i(po, "mtm_disc_curve", 0);
    p.mtm_reset_num = get_i(po, "mtm_reset_num", 0);
    p.mtm_reset_den = get_i(po, "mtm_reset_den", 0);
    floats(po, "mtm_coupons", p.mtm_coupons);
    book.positions.push_back(std::move(p));
  }
  return book;
}

Eigen::VectorXd flat_x0(const cal::BundleProblem& prob, double level) {
  Eigen::VectorXd x(prob.n_knots());
  int o = 0;
  for (const auto& c : prob.curves) {
    const double v = (c.base < 0) ? level : 0.0;  // outright at the level, spread at zero
    const int ni = c.n_interp_knots();            // interp knots first, then one δ per turn
    // Interp knots seed at the level/zero; turn δ's are an overlay -> seed at 0 (no jump), NOT the level.
    for (int i = 0; i < c.n_knots(); ++i) x[o++] = (i < ni) ? v : 0.0;
  }
  return x;
}

// =================================================================================================
// BundleSession
// =================================================================================================
// A W-cache-incompatible LEAF anywhere in an instrument (including nested inside a Portfolio): FX/MtM,
// whose DF-ratio / curve-dependent notional is not a single exp(-Wx). A Portfolio of otherwise-cacheable
// components (par swaps, futures) IS W-cacheable -- its row is a weighted sum of cacheable transforms.
static bool has_noncacheable_leaf(const cal::Instrument& ins) {
  if (ins.quote == cal::QuoteKind::FxForward || ins.quote == cal::QuoteKind::XccyMtmBasis) return true;
  if (ins.quote == cal::QuoteKind::Portfolio)
    for (const auto& c : ins.combination)
      if (has_noncacheable_leaf(c.instrument)) return true;
  return false;
}

// Collect every schedule-carrying RateObservation in an instrument (Rate future obs + float-leg coupons,
// recursing into portfolio components), so the session can resolve them against the fixing context.
namespace {
void collect_sched_obs(cal::Instrument& ins, std::vector<px::RateObservation*>& out) {
  auto leg = [&](cal::FloatLeg& l) {
    for (auto& c : l.coupons)
      if (!c.obs.fixing_schedule.empty()) out.push_back(&c.obs);
  };
  if (ins.quote == cal::QuoteKind::Rate && !ins.obs.fixing_schedule.empty()) out.push_back(&ins.obs);
  leg(ins.fwd);
  leg(ins.bench);
  leg(ins.mtm);
  for (auto& w : ins.combination) collect_sched_obs(w.instrument, out);
}
}  // namespace

int BundleSession::resolve_fixings() {
  std::vector<px::RateObservation*> obs;
  for (auto& ins : prob_.instruments) collect_sched_obs(ins, obs);
  const px::PricingContext ctx{eval_date_, &fixings_};
  int missing = 0;
  for (px::RateObservation* o : obs) {
    try {
      px::resolve_into(*o, ctx);  // rewrites realized/subs in place; no recompile
    } catch (const px::MissingFixing&) {
      ++missing;  // leave the observation as-is; the instrument is un-priceable until the fixing arrives
    }
  }
  n_unresolved_ = missing;
  return missing;
}

BundleSession::BundleSession(cal::BundleProblem prob) : prob_(std::move(prob)) {
  fingerprint_ = cal::structure_fingerprint(prob_);  // the topology this session's W-cache is compiled for
  for (const auto& ins : prob_.instruments) {
    if (has_noncacheable_leaf(ins)) has_fx_ = true;  // FX/MtM (incl. inside a Portfolio) -> AAD engine
    if (ins.band_upper > ins.band_lower)
      has_band_ = true;  // soft target: compiled COLD calibrate is fine, streams frozen-Newton (soft LS)
  }
  for (const auto& c : prob_.curves) {
    if (!c.regions.empty()) has_modular_ = true;  // custom interpolation regions
    for (const auto& r : c.regions)        // only a NON-LINEAR scheme forces off the W-cache
      if (r.scheme == curve::Scheme::MonotoneCubic) has_nonlinear_ = true;
  }
  x_ = Eigen::VectorXd::Zero(prob_.n_knots());
  resolve_fixings();  // resolve any schedule-carrying observations (no-op when none carry a schedule)
}

const cal::CalibrationResult& BundleSession::calibrate(const Eigen::VectorXd& x0, const RegSpec& reg) {
  const auto t0 = std::chrono::steady_clock::now();
  if (reg.on() && reg.tension)
    // Continuous tension-energy penalty: a constant pseudo-residual block R (= sqrt(mu)*L) appended to
    // the least squares (regularize.hpp §5). Built once, off the AAD hot path, like the second-diff path.
    result_ = cal::calibrate(
        cal::linearly_regularized(prob_, cal::tension_energy_operator(prob_, reg.lambda, reg.sigma, reg.curves)),
        x0, /*use_aad=*/true);
  else if (reg.on())
    result_ = cal::calibrate(cal::smoothed(prob_, reg.lambda, reg.curves), x0, /*use_aad=*/true);
  else if (!has_nonlinear_)
    // HYBRID W-cache: cacheable rows on the fast path, any FX/MtM (or portfolio-with-FX) rows on a
    // width-reduced AAD block -- so one FX trade no longer drops the whole book to AAD.
    result_ = cal::calibrate(prob_, x0, /*use_aad=*/true);
  else
    // A non-linear region scheme (MonotoneCubic) has NO constant W at all, so the whole bundle prices on
    // the generic AAD engine (a zero-reg smoothed wrapper routes there).
    result_ = cal::calibrate(cal::smoothed(prob_, 0.0, {}), x0, /*use_aad=*/true);
  last_solve_us_ = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  result_.solve_micros = last_solve_us_;
  x_ = result_.x;
  return result_;
}

const cal::CalibrationResult& BundleSession::recalibrate(const Eigen::VectorXd& new_market,
                                                         const RegSpec& reg) {
  if (new_market.size() != prob_.n_residuals())
    throw std::runtime_error("recalibrate: market length does not match the instrument count");
  for (int i = 0; i < prob_.n_residuals(); ++i) prob_.instruments[i].market = new_market[i];
  return calibrate(x_, reg);  // warm from the current solution
}

std::vector<CurveSample> BundleSession::sample(const std::vector<double>& times) const {
  const auto C = cal::build_bundle_curves<double>(
      prob_.curves, [&](int c, int i) { return x_[prob_.offset(c) + i]; });
  std::vector<CurveSample> out(prob_.n_curves());
  for (int c = 0; c < prob_.n_curves(); ++c) {
    out[c].currency = prob_.curves[c].currency;
    out[c].t = times;
    out[c].discount.reserve(times.size());
    out[c].zero.reserve(times.size());
    out[c].forward.reserve(times.size());
    for (double t : times) {
      out[c].discount.push_back(C[c]->discount(t));
      out[c].forward.push_back(C[c]->forward(t));
      out[c].zero.push_back(t > 1e-12 ? C[c]->integral(t) / t : C[c]->forward(0.0));
    }
  }
  return out;
}

double BundleSession::model_quote(const cal::Instrument& ins) const {
  const auto C = cal::build_bundle_curves<double>(
      prob_.curves, [&](int c, int i) { return x_[prob_.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  return cal::instrument_model_quote<double>(ins, curve_of);
}

double BundleSession::residual(const cal::Instrument& ins) const {
  const auto C = cal::build_bundle_curves<double>(
      prob_.curves, [&](int c, int i) { return x_[prob_.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  return cal::instrument_residual<double>(ins, curve_of);
}

Eigen::MatrixXd BundleSession::jacobian(const RegSpec& reg) const {
  (void)reg;  // J = dq/dx is independent of any regulariser (reg only enters M through RᵀR); see header.
  return cal::aad_jacobian(prob_, x_);  // n_res x n_knots: rows = instruments, cols = knots
}

Eigen::MatrixXd BundleSession::risk_operator(const RegSpec& reg) const {
  const Eigen::MatrixXd J = jacobian(reg);                 // n_res x n_knots (shared with jacobian(), no desync)
  Eigen::MatrixXd A = J.transpose() * J;                   // n_knots x n_knots
  if (reg.on()) {
    const Eigen::MatrixXd R = reg.tension
                                  ? cal::tension_energy_operator(prob_, reg.lambda, reg.sigma, reg.curves)
                                  : cal::second_difference_operator(prob_, reg.lambda, reg.curves);
    A.noalias() += R.transpose() * R;
  }
  const Eigen::MatrixXd Ainv = A.ldlt().solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
  return Ainv * J.transpose();  // n_knots x n_res  = dx/dq
}

PortfolioReprice BundleSession::price_portfolio(const pf::MultiCurveBook& book) const {
  PortfolioReprice out;
  out.n = static_cast<int>(book.positions.size());
  if (out.n == 0) { last_price_us_ = 0.0; return out; }  // empty book: NPV/PV01 = 0, nothing to time

  // ---- pure ENGINE pricing pass (double), engine-timed -- no marshalling inside the clock ----------
  // Build the double curve handles off the CALIBRATED x, then value every position through the pricing
  // kernel. A steady_clock pair brackets exactly this pass, mirroring how calibrate() stamps its solve.
  const auto t0 = std::chrono::steady_clock::now();
  const auto Cd = cal::build_bundle_curves<double>(
      prob_.curves, [&](int c, int i) { return x_[prob_.offset(c) + i]; });
  const auto curve_d = [&Cd](int i) -> const cal::CurveHandle<double>& { return *Cd[i]; };
  out.npv = book.value<double>(curve_d);
  last_price_us_ = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  out.price_us = last_price_us_;

  // ---- PV01: ONE forward-AAD pass (not part of the timed pricing pass) -----------------------------
  // Seed x as vector-duals, reprice the book once with Scalar = ad::Dual, and read d(NPV)/d(knot forward)
  // straight off the derivative vector. PV01 = 1bp · Σⱼ ∂NPV/∂xⱼ = the book's NPV change for a +1bp
  // PARALLEL shift of every fitted knot forward -- no bump-and-reprice. (Left-multiplying this same
  // gradient by risk_operator() would instead give the full per-quote delta ladder, CLAUDE.md #4.)
  // R11: pooled (heap-free) forward AAD for a narrow bundle, heap Dual beyond MaxW.
  auto pv01_pass = [&](const auto& xd) {
    using S = typename std::decay_t<decltype(xd)>::Scalar;
    const auto Cad = cal::build_bundle_curves<S>(
        prob_.curves, [&](int c, int i) { return xd[prob_.offset(c) + i]; });
    const auto curve_ad = [&Cad](int i) -> const cal::CurveHandle<S>& { return *Cad[i]; };
    const S npv_ad = book.value<S>(curve_ad);
    out.pv01 = npv_ad.derivatives().size() ? 1e-4 * npv_ad.derivatives().sum() : 0.0;
  };
  if (prob_.n_knots() <= ad::kPooledMaxW) pv01_pass(ad::seed_pooled<ad::kPooledMaxW>(x_));
  else pv01_pass(ad::seed(x_));
  return out;
}

PortfolioReprice BundleSession::price_portfolio_json(const std::string& book_json) const {
  return price_portfolio(book_from_json(json::parse(book_json)));
}

PortfolioRisk BundleSession::price_portfolio_risk(const pf::MultiCurveBook& book, const RegSpec& reg) const {
  PortfolioRisk out;
  out.n = static_cast<int>(book.positions.size());
  const int nk = prob_.n_knots();
  const int nr = prob_.n_residuals();
  if (out.n == 0) {  // empty book: nothing to reprice or risk
    out.curve_grad = Eigen::VectorXd::Zero(nk);
    out.ladder = Eigen::VectorXd::Zero(nr);
    last_risk_us_ = 0.0;
    return out;
  }

  // The risk operator M = dx/dq (n_knots x n_res). Its FORMATION (the calibration-Jacobian solve) is book-
  // INDEPENDENT, so it sits OUTSIDE the engine clock, mirroring how price_portfolio times only the book pass.
  // `reg` regularises M: a curvature/tension penalty damps the ladder's fan-out into a local key-rate hedge
  // while preserving total DV01 (R annihilates level+linear moves). Default reg={} -> the raw operator.
  const Eigen::MatrixXd M = risk_operator(reg);

  // ---- ENGINE-STAMPED risk pass: the AAD reprice (dP/dx) + the M multiply -------------------------------
  // Seed x as vector-duals, reprice the book once with Scalar = ad::Dual, and read the FULL derivative
  // vector curve_grad = dP/dx (the exact gradient the PV01 pass sums). Then ladder = curve_grad^T · M, i.e.
  // ladder[i] = Σⱼ curve_grad[j]·M[j,i] = dP/dq_i — the portfolio's delta in calibration instrument i.
  const auto t0 = std::chrono::steady_clock::now();
  // R11: pooled (heap-free) forward AAD for a narrow bundle, heap Dual beyond MaxW.
  auto grad_pass = [&](const auto& xd) {
    using S = typename std::decay_t<decltype(xd)>::Scalar;
    const auto Cad = cal::build_bundle_curves<S>(
        prob_.curves, [&](int c, int i) { return xd[prob_.offset(c) + i]; });
    const auto curve_ad = [&Cad](int i) -> const cal::CurveHandle<S>& { return *Cad[i]; };
    const S npv_ad = book.value<S>(curve_ad);
    out.npv = npv_ad.value();
    out.curve_grad = (npv_ad.derivatives().size() == nk) ? Eigen::VectorXd(npv_ad.derivatives())
                                                         : Eigen::VectorXd(Eigen::VectorXd::Zero(nk));
  };
  if (nk <= ad::kPooledMaxW) grad_pass(ad::seed_pooled<ad::kPooledMaxW>(x_));
  else grad_pass(ad::seed(x_));
  out.ladder = M.transpose() * out.curve_grad;  // (n_res x n_knots)·(n_knots) = length n_res
  last_risk_us_ = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  out.risk_us = last_risk_us_;
  return out;
}

PortfolioRisk BundleSession::price_portfolio_risk_json(const std::string& book_json, const RegSpec& reg) const {
  return price_portfolio_risk(book_from_json(json::parse(book_json)), reg);
}

bool BundleSession::same_structure(const cal::BundleProblem& p) const {
  return cal::structure_fingerprint(p) == fingerprint_;
}

bool BundleSession::same_curve_set(const cal::BundleProblem& source) const {
  if (source.curves.size() != prob_.curves.size()) return false;
  for (std::size_t c = 0; c < prob_.curves.size(); ++c) {
    if (source.curves[c].currency != prob_.curves[c].currency) return false;
    if ((source.curves[c].base < 0) != (prob_.curves[c].base < 0)) return false;  // outright vs spread
  }
  return true;
}

Eigen::MatrixXd BundleSession::cross_jacobian(const cal::BundleProblem& source) const {
  if (!same_curve_set(source))
    throw std::invalid_argument(
        "cross_jacobian: the source bundle must share this bundle's curve set (same currencies / "
        "outright-or-spread, same order) so its instruments price on this bundle's curves.");
  const int nr = source.n_residuals();  // rows: source instruments (== source ladder order)
  const int nk = prob_.n_knots();       // cols: THIS bundle's fitted knots
  // Seed THIS bundle's state as vector-duals and build its curves once; then price each SOURCE instrument's
  // model quote on those curves and read d(quote)/dx_this straight off the derivative vector (forward-AAD).
  const Eigen::Matrix<ad::Dual, Eigen::Dynamic, 1> xd = ad::seed(x_);
  const auto Cad = cal::build_bundle_curves<ad::Dual>(
      prob_.curves, [&](int c, int i) { return xd[prob_.offset(c) + i]; });
  const auto curve_ad = [&Cad](int i) -> const cal::CurveHandle<ad::Dual>& { return *Cad[i]; };
  Eigen::MatrixXd J(nr, nk);
  for (int i = 0; i < nr; ++i) {
    const ad::Dual q = cal::instrument_model_quote<ad::Dual>(source.instruments[i], curve_ad);
    if (q.derivatives().size() == nk) J.row(i) = q.derivatives().transpose();
    else J.row(i).setZero();  // a quote that doesn't touch the curve carries an empty derivative -> zero row
  }
  return J;
}

Eigen::MatrixXd BundleSession::transform_matrix(const cal::BundleProblem& source, const RegSpec& reg) const {
  return cross_jacobian(source) * risk_operator(reg);  // (nr_src x nk)·(nk x nr_this) = nr_src x nr_this
}

Eigen::MatrixXd BundleSession::cross_jacobian_json(const std::string& source_bundle_json) const {
  return cross_jacobian(bundle_from_json(json::parse(source_bundle_json)));
}

Eigen::MatrixXd BundleSession::transform_matrix_json(const std::string& source_bundle_json,
                                                     const RegSpec& reg) const {
  return transform_matrix(bundle_from_json(json::parse(source_bundle_json)), reg);
}

void BundleSession::start_streaming(const RegSpec& reg, double step_tol) {
  if (needs_recalibrate())
    throw std::runtime_error(
        "frozen-Newton streaming needs a bundle with a constant W (a MonotoneCubic region scheme has "
        "none); use recalibrate() per tick for those. FX/MtM, bands and portfolios all stream -- FX/MtM "
        "on the hybrid engine (their AAD Jacobian refreshes only on staleness).");
  // The StreamingCalibrator builds its own hybrid engine from prob_; no compiled engine is constructed
  // here (that would throw on an FX/MtM leaf, which the hybrid handles on its AAD block).
  //
  // Anchor at the market MIDS (prob_.market()). For a hard bundle these equal the curve's reprice; for a
  // banded (soft) bundle the curve sits off-market inside the bands, and the streaming feed IS the mids,
  // so anchoring at the mids keeps the drift ~0 at the first real tick.
  const Eigen::VectorXd q0 = prob_.market();
  cal::StreamingCalibrator<cal::BundleProblem>::Options opt;
  if (reg.on())
    opt.regularizer = reg.tension
                          ? cal::tension_energy_operator(prob_, reg.lambda, reg.sigma, reg.curves)
                          : cal::second_difference_operator(prob_, reg.lambda, reg.curves);
  if (step_tol > 0.0) opt.step_tol = step_tol;  // looser tol -> fewer corrector steps (speed/accuracy knob)
  stream_ = std::make_unique<cal::StreamingCalibrator<cal::BundleProblem>>(prob_, x_, q0, opt);
  stream_sum_us_ = 0;  // reset the running average for this streaming session
  stream_ticks_ = 0;
}

const Eigen::VectorXd& BundleSession::stream_update(const Eigen::VectorXd& new_market) {
  if (!stream_) throw std::runtime_error("start_streaming() must be called first");
  const auto t0 = std::chrono::steady_clock::now();
  const cal::StreamTick tick = stream_->update(new_market);
  last_solve_us_ = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  stream_sum_us_ += last_solve_us_;
  ++stream_ticks_;
  last_newton_steps_ = tick.newton_steps;
  last_refreshes_ = tick.refreshes;
  last_drift_ = tick.drift;
  x_ = stream_->current();
  return x_;
}

// =================================================================================================
// One-shot JSON dispatcher
// =================================================================================================
std::string run_json(const std::string& request) {
  try {
    const json::value req = json::parse(request);
    const auto& o = req.as_object();

    // Stateless COMPILE verb: a composer spec (curves + generic instrument rows + interpolation regions)
    // -> the resolved bundle + streaming config, the C++ analog of server/compile.py's compile_spec. It
    // PRODUCES a bundle rather than consuming one, so it is dispatched here before the 'bundle' requirement.
    // Reachable by any host through the single swaps_run_json C ABI (Excel add-in, .NET, ctypes, web), so a
    // client can build a bundle up from typed rows instead of pasting resolved JSON. `today` supplies the
    // value date when the spec omits one.
    if (o.contains("compile")) {
      const std::string today = (o.contains("today") && o.at("today").is_string())
                                    ? std::string(o.at("today").as_string().c_str()) : "";
      return json::serialize(compile_to_json(compile_spec(o.at("compile"), today)));
    }

    // Stateless GENERATE_RISK verb: one book + N bundles -> N internally-consistent risk ladders, all off
    // bundles[0]'s DFs (later bundles re-leveled onto the anchor, under-determined ones rank-completed by
    // self-quoted pillars). Like `compile`, it produces rather than consumes a bundle, so dispatch it here.
    if (o.contains("generate_risk")) return generate_risk_json(request);

    // Stateless SWAPTION verb: price European swaptions off a calibrated curve (Bachelier / SABR).
    if (o.contains("swaption")) return swaption_json(request);

    // Stateless SABR strip calibration (no bundle): fit (alpha,rho,nu) to a market vol strip.
    if (o.contains("sabr_calibrate")) return sabr_calibrate_json(request);

    // Stateless BONDS verb (no bundle): street/yield-space bond math (price<->yield, accrued, duration,
    // convexity) for a list of fixed-rate bonds.
    if (o.contains("bonds")) return bonds_json(request);

    // Stateless ASSET_SWAP verb: par asset-swap spread(s) for bonds off a calibrated bundle (curve-space).
    if (o.contains("asset_swap")) return asset_swap_json(request);

    // Stateless EXPOSURE verb: EPE/ENE/PFE counterparty-exposure profile for a swap book off a calibrated curve.
    if (o.contains("exposure")) return exposure_json(request);

    if (!o.contains("bundle")) return err("request is missing the required 'bundle' object");

    cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
    if (prob.n_curves() == 0) return err("bundle has no curves");
    BundleSession sess(std::move(prob));
    const cal::BundleProblem& P = sess.problem();

    Eigen::VectorXd x0;
    if (o.contains("x0")) {
      const auto xv = get_da(o, "x0");
      x0 = Eigen::Map<const Eigen::VectorXd>(xv.data(), static_cast<Eigen::Index>(xv.size()));
      if (x0.size() != P.n_knots()) return err("x0 length does not match the bundle's knot count");
    } else {
      x0 = flat_x0(P);
    }

    RegSpec reg;
    if (o.contains("regularize")) {
      const auto& r = o.at("regularize").as_object();
      reg.lambda = get_d(r, "lambda", 0.0);
      reg.curves = get_ia(r, "curves");
      reg.tension = get_b(r, "tension", false);  // continuous tension energy vs discrete second-difference
      reg.sigma = get_d(r, "sigma", 0.0);        // tension parameter (tension=true); 0 => pure curvature
    }

    const cal::CalibrationResult& res = sess.calibrate(x0, reg);

    json::object out;
    {
      json::object c;
      c["iterations"] = res.iterations;
      c["rms_residual"] = res.rms_residual;
      c["stationarity"] = res.stationarity;
      c["info"] = res.info;
      out["calibration"] = c;
    }
    out["x"] = da(sess.x());

    if (o.contains("sample_times")) {
      const auto times = get_da(o, "sample_times");
      json::array carr;
      for (const auto& s : sess.sample(times)) {
        json::object co;
        co["currency"] = s.currency;
        co["t"] = da(s.t);
        co["discount"] = da(s.discount);
        co["zero"] = da(s.zero);
        co["forward"] = da(s.forward);
        carr.push_back(co);
      }
      out["curves"] = carr;
    }

    if (o.contains("price")) {
      json::array parr;
      for (const auto& e : o.at("price").as_array()) {
        const cal::Instrument ins = instrument_from_json(e);
        json::object po;
        po["model_quote"] = sess.model_quote(ins);
        po["residual"] = sess.residual(ins);
        parr.push_back(po);
      }
      out["priced"] = parr;
    }

    // Book pricing / risk / cross-bundle transform through the JSON seam (portfolio / portfolio_risk / risk /
    // transform), so every risk operation is reachable by any client (web, Excel add-in, .NET, CLI) without
    // the C++ Session object. These arms are GENERATED from api/api_surface.py by tools/gen_dispatch.py — the
    // SAME descriptor the pybind + Excel bindings come from — so the stateless seam can't drift from them.
#include "run_json_dispatch.gen.inc"

    return json::serialize(json::value(std::move(out)));
  } catch (const std::exception& ex) {
    return err(ex.what());
  }
}

}  // namespace swaps::api
