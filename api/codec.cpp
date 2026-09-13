// JSON <-> engine object graph: the definitions behind include/swaps/api/codec.hpp (read its contract first).
// Moved out of api/bundle_api.cpp on 2026-09-13 (E7 stage 2). The one change of substance: every reader now
// assigns ONLY when the field is present, so an absent field keeps the struct's own default instead of a
// literal restated here -- the struct is the single source of each default.
#include "swaps/api/codec.hpp"

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"  // RegSpec, CurveSample
#include "swaps/api/json_util.hpp"   // vecf
#include "swaps/build/date.hpp"
#include "swaps/calibration/consistent_risk.hpp"  // ConsistentRiskRequest, ConsistentRisk
#include "swaps/calibration/diagnostics.hpp"  // QuoteDiagnostic, CalibrationReport
#include "swaps/derive/bond_rv.hpp"  // BondUniverseRequest, GovvieFitRequest, SwapSpreadRequest
#include "swaps/trade/csa.hpp"    // discount_index_for
#include "swaps/trade/trade.hpp"  // Trade::vanilla_swap / to_position

namespace swaps::api {

namespace json = boost::json;
namespace px = swaps::pricing;
namespace curve = swaps::curve;
namespace pf = swaps::portfolio;
namespace der = swaps::derive;

namespace {

// ---- readers: assign ONLY when present (an explicit null is absent); the wrong type throws ---------------
bool present(const json::object& o, const char* k) { return o.contains(k) && !o.at(k).is_null(); }
std::string str(const json::value& v) {
  const auto& s = v.as_string();
  return std::string(s.begin(), s.end());
}
void into(const json::object& o, const char* k, double& out) {
  if (present(o, k)) out = o.at(k).to_number<double>();
}
void into(const json::object& o, const char* k, int& out) {
  if (present(o, k)) out = static_cast<int>(o.at(k).to_number<long long>());
}
void into(const json::object& o, const char* k, bool& out) {
  if (present(o, k)) out = o.at(k).as_bool();
}
void into(const json::object& o, const char* k, std::string& out) {
  if (present(o, k)) out = str(o.at(k));
}
void into(const json::object& o, const char* k, swaps::build::Date& out) {
  if (present(o, k)) out = swaps::build::Date::from_iso(str(o.at(k)));
}
void into(const json::object& o, const char* k, std::vector<double>& out) {
  if (!present(o, k)) return;
  out.clear();
  for (const auto& e : o.at(k).as_array()) out.push_back(e.to_number<double>());
}
void into(const json::object& o, const char* k, std::vector<int>& out) {
  if (!present(o, k)) return;
  out.clear();
  for (const auto& e : o.at(k).as_array()) out.push_back(static_cast<int>(e.to_number<long long>()));
}
void into(const json::object& o, const char* k, Eigen::VectorXd& out) {
  if (!present(o, k)) return;
  std::vector<double> v;
  into(o, k, v);
  out = Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
}
template <class T>
void into(const json::object& o, const char* k, std::optional<T>& out) {
  if (!present(o, k)) return;
  T v{};
  into(o, k, v);
  out = v;
}
// A field the library has no default for.
const json::value& need(const json::object& o, const char* k, const std::string& why) {
  if (!present(o, k)) throw std::invalid_argument(why);
  return o.at(k);
}

// ---- names <-> enums ------------------------------------------------------------------------------
const char* quote_to_str(cal::QuoteKind q) {
  switch (q) {
    case cal::QuoteKind::ParRate: return "ParRate";
    case cal::QuoteKind::ParSpread: return "ParSpread";
    case cal::QuoteKind::Rate: return "Rate";
    case cal::QuoteKind::FxForward: return "FxForward";
    case cal::QuoteKind::XccyMtmBasis: return "XccyMtmBasis";
    case cal::QuoteKind::Portfolio: return "Portfolio";
    case cal::QuoteKind::TurnJump: return "TurnJump";
    case cal::QuoteKind::ZeroCouponRate: return "ZeroCouponRate";
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
  if (s == "ZeroCouponRate") return cal::QuoteKind::ZeroCouponRate;
  if (s == "ParRate") return cal::QuoteKind::ParRate;
  throw std::invalid_argument("unknown quote kind: " + s);
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
pf::MultiCurveBook::Kind position_kind_from_str(const std::string& s) {
  if (s == "swap") return pf::MultiCurveBook::Kind::Swap;
  if (s == "xccy") return pf::MultiCurveBook::Kind::Xccy;
  throw std::invalid_argument("book: unknown position kind '" + s + "' (swap | xccy)");
}
swaps::trade::Pay pay_from_str(const std::string& s) {
  if (s == "fixed") return swaps::trade::Pay::Fixed;
  if (s == "float") return swaps::trade::Pay::Float;
  throw std::invalid_argument("book: a trade's 'pay' must be \"fixed\" or \"float\", got '" + s + "'");
}

der::GovvieModel govvie_model_from_str(const std::string& s) {
  if (s == "spline") return der::GovvieModel::Spline;
  if (s == "nelson_siegel") return der::GovvieModel::NelsonSiegel;
  if (s == "svensson") return der::GovvieModel::Svensson;
  throw std::invalid_argument("govvie_fit: unknown model '" + s + "' (spline | nelson_siegel | svensson)");
}
const char* govvie_model_to_str(der::GovvieModel m) {
  switch (m) {
    case der::GovvieModel::Spline: return "spline";
    case der::GovvieModel::NelsonSiegel: return "nelson_siegel";
    case der::GovvieModel::Svensson: return "svensson";
  }
  return "spline";
}
der::SwapSpreadType spread_type_from_str(const std::string& s) {
  if (s == "headline") return der::SwapSpreadType::HeadlineYield;
  if (s == "matched_maturity") return der::SwapSpreadType::MatchedMaturity;
  throw std::invalid_argument("swap_spread: unknown spread_type '" + s + "' (headline | matched_maturity)");
}

// ---- per-object parse -------------------------------------------------------------------------
px::RateObservation obs_from(const json::object& o) {
  px::RateObservation r;
  into(o, "sub_start", r.sub_start);
  into(o, "sub_end", r.sub_end);
  into(o, "weight", r.weight);
  into(o, "realized", r.realized);
  into(o, "tau_index", r.tau_index);
  into(o, "fixing_step", r.fixing_step);
  into(o, "fixing_step3", r.fixing_step3);
  into(o, "compounded", r.compounded);
  into(o, "realized_factor", r.realized_factor);
  // E2: an optional fixing SCHEDULE (index + per-day dates/accruals) — when present the engine RESOLVES
  // realized + forecast subs from the pricing context's fixing table instead of reading a baked `realized`.
  into(o, "fixing_index", r.fixing_index);
  if (o.contains("fixing_schedule"))
    for (const auto& e : o.at("fixing_schedule").as_array()) {
      const auto& d = e.as_object();
      px::FixingDay fd;
      into(d, "fixing_date", fd.fixing_date);
      into(d, "accrual", fd.accrual);
      into(d, "t_start", fd.t_start);
      into(d, "t_end", fd.t_end);
      into(d, "weight", fd.weight);
      r.fixing_schedule.push_back(fd);
    }
  return r;
}
px::FloatCoupon fcpn_from(const json::object& o) {
  px::FloatCoupon c;
  if (o.contains("obs")) c.obs = obs_from(o.at("obs").as_object());
  into(o, "pay", c.pay);
  into(o, "tau_pay", c.tau_pay);
  into(o, "spread", c.spread);
  into(o, "scale", c.scale);
  into(o, "reset_time", c.reset_time);
  if (o.contains("accrual_start") && o.contains("accrual_end")) {  // the accrual period (E3-S2; may start in the past)
    c.accrual_set = true;
    into(o, "accrual_start", c.accrual_start);
    into(o, "accrual_end", c.accrual_end);
  }
  into(o, "reset_fx", c.reset_fx);  // a seasoned MtM coupon's FIXED FX reset
  return c;
}
px::FixedCoupon xcpn_from(const json::object& o) {
  px::FixedCoupon c;
  into(o, "pay", c.pay);
  into(o, "tau", c.tau);
  into(o, "scale", c.scale);
  return c;
}
void float_coupons_into(const json::object& o, const char* k, std::vector<px::FloatCoupon>& out) {
  if (o.contains(k))
    for (const auto& e : o.at(k).as_array()) out.push_back(fcpn_from(e.as_object()));
}
cal::FloatLeg fleg_from(const json::object& o) {
  cal::FloatLeg L;
  float_coupons_into(o, "coupons", L.coupons);
  into(o, "forecast", L.forecast);
  into(o, "discount", L.discount);
  into(o, "reset_num", L.reset_num);
  into(o, "reset_den", L.reset_den);
  into(o, "fx_spot", L.fx_spot);
  return L;
}
cal::FixedLeg xleg_from(const json::object& o) {
  cal::FixedLeg L;
  if (o.contains("coupons"))
    for (const auto& e : o.at("coupons").as_array()) L.coupons.push_back(xcpn_from(e.as_object()));
  into(o, "discount", L.discount);
  return L;
}
cal::BundleCurveSpec spec_from(const json::object& o) {
  cal::BundleCurveSpec s;
  std::vector<double> legacy_meeting, legacy_back;  // legacy Flat+Hermite layout (pre-regions)
  into(o, "meeting", legacy_meeting);
  into(o, "back", legacy_back);
  into(o, "base", s.base);
  into(o, "currency", s.currency);
  if (o.contains("regions"))
    for (const auto& e : o.at("regions").as_array()) {
      const auto& ro = e.as_object();
      curve::CurveModule m;
      if (ro.contains("scheme")) m.scheme = scheme_from_str(str(ro.at("scheme")));
      into(ro, "knots", m.knots);
      into(ro, "sigma", m.sigma);            // tension hyperparameter (Scheme::Tension only); else ignored
      into(ro, "reg_lambda", m.reg_lambda);  // per-region smoothing weight (Phase 1); <0 = inherit
      into(ro, "reg_sigma", m.reg_sigma);    // per-region tension-energy σ (Phase 2); <0 = inherit
      s.regions.push_back(std::move(m));
    }
  // Calibration TURNS (docs/turns-calibration.md, Mode 2): an OPTIONAL array of overlay windows. Absent
  // -> empty, byte-identical to a turn-free curve. Each δ appends one free state var after the interp knots.
  if (o.contains("turns"))
    for (const auto& e : o.at("turns").as_array()) {
      const auto& to = e.as_object();
      px::Turn t;
      into(to, "start", t.start);
      into(to, "end", t.end);
      s.turns.push_back(t);
    }
  // Backward-compat: a legacy bundle that named the Flat(meeting)+Hermite(back) layout instead of
  // `regions` is normalised to the equivalent two-region layout (identical to the old modules()).
  if (s.regions.empty() && !(legacy_meeting.empty() && legacy_back.empty()))
    s.regions = curve::flat_hermite(legacy_meeting, legacy_back);
  return s;
}

// One bond row of an RV request, stamped with the request's shared yield convention.
swaps::build::BondId bond_id_from(const json::object& o, const std::string& convention, const char* verb) {
  const auto term = [&](const char* k) -> const json::value& {
    return need(o, k, std::string(verb) + ": each bond needs '" + k + "'");
  };
  swaps::build::BondId b;
  b.id = str(term("id"));
  b.yield_conv = convention;
  b.issue = swaps::build::Date::from_iso(str(term("issue")));
  b.maturity = swaps::build::Date::from_iso(str(term("maturity")));
  b.coupon = term("coupon").to_number<double>();
  into(o, "first_coupon", b.first_coupon);  // when-issued / odd first period
  return b;
}
std::vector<swaps::build::BondId> bonds_from(const json::object& o, const std::string& convention, const char* verb) {
  std::vector<swaps::build::BondId> u;
  for (const auto& e : need(o, "bonds", std::string(verb) + ": missing 'bonds' array").as_array())
    u.push_back(bond_id_from(e.as_object(), convention, verb));
  return u;
}
void settlement_into(const json::object& o, der::SettlementOverride& s) {
  into(o, "settle_calendar", s.calendar);
  into(o, "settle_lag", s.lag);
}

// ---- per-object serialize ----------------------------------------------------------------------
json::object obs_to(const px::RateObservation& r) {
  json::object o;
  o["sub_start"] = vecf(r.sub_start);
  o["sub_end"] = vecf(r.sub_end);
  o["weight"] = vecf(r.weight);
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
      json::object day;
      day["fixing_date"] = d.fixing_date;
      day["accrual"] = d.accrual;
      day["t_start"] = d.t_start;
      day["t_end"] = d.t_end;
      day["weight"] = d.weight;
      sch.push_back(std::move(day));
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
  // The accrual period rides the wire only when it differs from the observation window (a seasoned or
  // observation-shifted coupon): the common coupon stays byte-identical to the legacy document, and the
  // structural equality / fingerprint compare EFFECTIVE dates, so a resent document is the same structure.
  const bool same_as_window = !c.obs.sub_start.empty() && !c.obs.sub_end.empty() &&
                              c.accrual_start == c.obs.sub_start.front() && c.accrual_end == c.obs.sub_end.back();
  if (c.accrual_set && !same_as_window) {
    o["accrual_start"] = c.accrual_start;
    o["accrual_end"] = c.accrual_end;
  }
  if (c.reset_fx >= 0.0) o["reset_fx"] = c.reset_fx;
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
    mo["knots"] = vecf(m.knots);
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

}  // namespace

// =================================================================================================
// Public codecs
// =================================================================================================
cal::Instrument instrument_from_json(const json::value& v) {
  const auto& o = v.as_object();
  cal::Instrument ins;
  if (o.contains("quote")) ins.quote = quote_from_str(str(o.at("quote")));
  if (o.contains("fwd")) ins.fwd = fleg_from(o.at("fwd").as_object());
  if (o.contains("bench")) ins.bench = fleg_from(o.at("bench").as_object());
  if (o.contains("fixed")) ins.fixed = xleg_from(o.at("fixed").as_object());
  if (o.contains("mtm")) ins.mtm = fleg_from(o.at("mtm").as_object());
  if (o.contains("obs")) ins.obs = obs_from(o.at("obs").as_object());
  into(o, "forecast", ins.forecast);
  into(o, "convexity", ins.convexity);
  into(o, "market", ins.market);
  into(o, "pv_currency", ins.pv_currency);
  into(o, "fx_num", ins.fx_num);
  into(o, "fx_den", ins.fx_den);
  into(o, "fx_spot", ins.fx_spot);
  into(o, "fx_time", ins.fx_time);
  into(o, "band_lower", ins.band_lower);
  into(o, "band_upper", ins.band_upper);
  into(o, "band_decay", ins.band_decay);
  into(o, "turn_curve", ins.turn_curve);  // TurnJump only: (curve, index) of the pinned turn
  into(o, "turn_index", ins.turn_index);
  if (o.contains("combination"))  // Portfolio components (recursive)
    for (const auto& e : o.at("combination").as_array()) {
      const auto& c = e.as_object();
      cal::WeightedInstrument w;
      into(c, "weight", w.weight);
      w.instrument = instrument_from_json(c.at("instrument"));
      ins.combination.push_back(std::move(w));
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

pf::MultiCurveBook book_from_json(const json::value& v) {
  pf::MultiCurveBook book;
  const auto& o = v.as_object();

  // ---- TYPED TRADES (the trade:: domain's production entry) ---------------------------------------
  // Booked deals plus the binding that connects index NAMES to this bundle's curve roles. Each trade becomes a
  // trade::Trade and materialises via to_position(vd) — rolling under ITS OWN index's conventions (conventions
  // DB), never one shared SwapConv. The discount index is trade::discount_index_for: the CSA's collateral
  // currency OIS when a CSA is given, else the explicit "discount_index"; both resolved through "curve_roles".
  if (o.contains("trades")) {
    const swaps::build::Date vd = swaps::build::Date::from_iso(
        str(need(o, "value_date", "book: typed 'trades' require a 'value_date' (ISO)")));
    std::map<std::string, int> roles;
    if (o.contains("curve_roles"))
      for (const auto& kv : o.at("curve_roles").as_object())
        roles[std::string(kv.key())] = static_cast<int>(kv.value().as_int64());
    const auto role_of = [&](const std::string& id, const char* what) -> int {
      const auto it = roles.find(id);
      if (it == roles.end())
        throw std::invalid_argument(std::string("book: curve_roles has no entry for ") + what + " '" +
                                    id + "'");
      return it->second;
    };
    for (const auto& e : o.at("trades").as_array()) {
      const auto& to = e.as_object();
      const auto term = [&](const char* k) -> const json::value& {
        return need(to, k, std::string("book: every typed trade needs '") + k + "'");
      };
      const std::string index = str(term("index"));
      if (index.empty()) throw std::invalid_argument("book: every typed trade needs an 'index'");
      std::optional<swaps::trade::CSA> csa;
      if (to.contains("csa")) {
        std::string ccy;
        into(to.at("csa").as_object(), "collateral_currency", ccy);
        csa = swaps::trade::CSA::cash(ccy);
      }
      std::string explicit_discount, id, currency;
      into(to, "discount_index", explicit_discount);
      into(to, "id", id);
      into(to, "currency", currency);
      const std::string disc_id = swaps::trade::discount_index_for(csa, explicit_discount);
      swaps::trade::Trade t = swaps::trade::Trade::vanilla_swap(
          id, term("notional").to_number<double>(), pay_from_str(str(term("pay"))),
          term("fixed_rate").to_number<double>(), currency, index,
          swaps::build::Date::from_iso(str(term("effective"))), swaps::build::Date::from_iso(str(term("maturity"))),
          role_of(index, "trade index"), role_of(disc_id, "discount index"));
      book.positions.push_back(t.to_position(vd));  // per-trade conventions from the trade's own index
    }
  }

  if (!o.contains("positions")) return book;
  for (const auto& e : o.at("positions").as_array()) {
    const auto& po = e.as_object();
    pf::MultiCurveBook::Position p;
    if (po.contains("kind")) p.kind = position_kind_from_str(str(po.at("kind")));
    into(po, "notional", p.notional);
    into(po, "fixed_rate", p.fixed_rate);
    into(po, "fwd_curve", p.fwd_curve);
    into(po, "disc_curve", p.disc_curve);
    float_coupons_into(po, "float_coupons", p.float_coupons);
    if (po.contains("fixed_coupons"))
      for (const auto& c : po.at("fixed_coupons").as_array()) p.fixed_coupons.push_back(xcpn_from(c.as_object()));
    // Which curve discounts the fixed leg is not a decoder's guess. It used to default to disc_curve HERE while
    // the struct defaults it to 0, so the same position priced two ways depending on the entry point.
    if (!p.fixed_coupons.empty())
      need(po, "fixed_curve", "book: a position with fixed_coupons needs 'fixed_curve' (the fixed leg's discount curve)");
    into(po, "fixed_curve", p.fixed_curve);
    // xccy-only: the resetting foreign funding leg + its FX-forward reset roles.
    into(po, "fx_spot", p.fx_spot);
    into(po, "mtm_fwd_curve", p.mtm_fwd_curve);
    into(po, "mtm_disc_curve", p.mtm_disc_curve);
    into(po, "mtm_reset_num", p.mtm_reset_num);
    into(po, "mtm_reset_den", p.mtm_reset_den);
    float_coupons_into(po, "mtm_coupons", p.mtm_coupons);
    book.positions.push_back(std::move(p));
  }
  return book;
}

swaps::build::StreetBondRequest street_bond_request_from_json(const json::object& o) {
  swaps::build::StreetBondRequest r;
  into(o, "value_date", r.value_date);
  for (const auto& e : need(o, "bonds", "bonds: missing 'bonds' array").as_array()) {
    const auto& bo = e.as_object();
    const auto term = [&](const char* k) -> const json::value& {
      return need(bo, k, std::string("bonds: each bond needs '") + k + "'");
    };
    swaps::build::StreetBondQuote q;
    q.terms.settle = swaps::build::Date::from_iso(str(term("settle")));
    q.terms.maturity = swaps::build::Date::from_iso(str(term("maturity")));
    q.terms.coupon = term("coupon").to_number<double>();
    into(bo, "convention", q.terms.convention);
    into(bo, "issue", q.terms.issue);
    into(bo, "dated", q.terms.dated);
    into(bo, "first_coupon", q.terms.first_coupon);
    into(bo, "freq", q.terms.freq);
    into(bo, "clean", q.clean);
    into(bo, "yield", q.yield);
    r.bonds.push_back(std::move(q));
  }
  return r;
}

json::object street_analytics_to_json(const std::vector<px::StreetAnalytics>& rows) {
  std::vector<double> clean, dirty, accrued, ytm, mdur, macdur, convx;
  for (const px::StreetAnalytics& a : rows) {
    clean.push_back(a.clean);
    dirty.push_back(a.dirty);
    accrued.push_back(a.accrued);
    ytm.push_back(a.yield);
    mdur.push_back(a.modified_duration);
    macdur.push_back(a.macaulay_duration);
    convx.push_back(a.convexity);
  }
  json::object out;
  out["clean"] = vecf(clean);
  out["dirty"] = vecf(dirty);
  out["accrued"] = vecf(accrued);
  out["ytm"] = vecf(ytm);
  out["modified_duration"] = vecf(mdur);
  out["macaulay_duration"] = vecf(macdur);
  out["convexity"] = vecf(convx);
  out["n"] = static_cast<int>(rows.size());
  return out;
}

swaps::build::DeliveryBasketRequest delivery_basket_request_from_json(const json::object& o) {
  swaps::build::DeliveryBasketRequest r;
  const auto field = [&](const char* k) -> const json::value& {
    return need(o, k, std::string("bond_future: missing '") + k + "'");
  };
  r.contract = str(field("contract"));
  r.value_date = swaps::build::Date::from_iso(str(field("value_date")));
  r.first_delivery = swaps::build::Date::from_iso(str(field("first_delivery")));
  r.futures_price = field("futures_price").to_number<double>();
  r.repo = field("repo").to_number<double>();
  into(o, "delivery", r.delivery);
  into(o, "notional_coupon", r.notional_coupon);
  into(o, "round_months", r.round_months);
  for (const auto& e : field("basket").as_array()) {
    const auto& bo = e.as_object();
    const auto term = [&](const char* k) -> const json::value& {
      return need(bo, k, std::string("bond_future: each basket bond needs '") + k + "'");
    };
    swaps::build::Deliverable x;
    x.maturity = swaps::build::Date::from_iso(str(term("maturity")));
    x.coupon = term("coupon").to_number<double>();
    x.clean = term("clean").to_number<double>();
    into(bo, "id", x.id);
    into(bo, "convention", x.convention);
    into(bo, "settle", x.settle);
    into(bo, "issue", x.issue);
    into(bo, "dated", x.dated);
    into(bo, "first_coupon", x.first_coupon);
    into(bo, "freq", x.freq);
    r.basket.push_back(std::move(x));
  }
  return r;
}

json::object delivery_basket_to_json(const swaps::build::DeliveryBasketResult& r) {
  std::vector<double> cf, gross, net, irr, invoice;
  for (const swaps::build::DeliverableAnalysis& row : r.rows) {
    cf.push_back(row.result.conversion_factor);
    gross.push_back(row.result.gross_basis);
    net.push_back(row.result.net_basis);
    irr.push_back(row.result.implied_repo);
    invoice.push_back(row.result.invoice_price);
  }
  json::object out;
  out["conversion_factor"] = vecf(cf);
  out["gross_basis"] = vecf(gross);
  out["net_basis"] = vecf(net);
  out["implied_repo"] = vecf(irr);
  out["invoice_price"] = vecf(invoice);
  out["n"] = static_cast<int>(r.rows.size());
  out["ctd_index"] = 0;
  out["ctd_id"] = std::string();
  if (r.ctd) {
    out["ctd_index"] = static_cast<int>(*r.ctd);
    out["ctd_id"] = r.rows[*r.ctd].id;
  }
  return out;
}

der::BondUniverseRequest bond_universe_request_from_json(const json::object& o) {
  der::BondUniverseRequest r;
  r.value_date = swaps::build::Date::from_iso(str(need(o, "value_date", "bond_universe: missing 'value_date'")));
  r.convention = str(need(o, "convention", "bond_universe: missing 'convention' (a bonds[] row id, e.g. US-TREASURY)"));
  into(o, "settle", r.settle);
  r.bonds = bonds_from(o, r.convention, "bond_universe");
  into(o, "clean", r.clean);
  into(o, "yield", r.yield);
  return r;
}
json::object bond_universe_to_json(const der::BondUniverseResult& r) {
  json::object out;
  out["clean"] = vecf(r.clean);
  out["yield"] = vecf(r.yield);
  out["modified_duration"] = vecf(r.modified_duration);
  out["convexity"] = vecf(r.convexity);
  out["accrued"] = vecf(r.accrued);
  out["n"] = static_cast<int>(r.yield.size());
  return out;
}

der::GovvieFitRequest govvie_fit_request_from_json(const json::object& o) {
  der::GovvieFitRequest r;
  r.value_date = swaps::build::Date::from_iso(str(need(o, "value_date", "govvie_fit: missing 'value_date'")));
  r.convention = str(need(o, "convention", "govvie_fit: missing 'convention' (a bonds[] row id)"));
  settlement_into(o, r.settlement);
  r.bonds = bonds_from(o, r.convention, "govvie_fit");
  into(o, "clean", r.clean);
  need(o, "clean", "govvie_fit: missing 'clean' prices");
  into(o, "weight", r.weight);
  if (present(o, "model")) r.model = govvie_model_from_str(str(o.at("model")));
  into(o, "meeting", r.meeting);
  into(o, "back", r.back);
  into(o, "tau1", r.tau1);
  into(o, "tau2", r.tau2);
  into(o, "x0", r.x0);
  return r;
}
json::object govvie_fit_to_json(const der::GovvieFitResult& r) {
  json::object out;
  out["x"] = vecf(r.fit.x);
  out["rms_residual"] = r.fit.rms_residual;
  out["iterations"] = r.fit.iterations;
  out["residuals"] = vecf(r.residuals);
  if (r.z_spread) out["z_spread"] = vecf(*r.z_spread);
  out["model"] = govvie_model_to_str(r.model);
  out["n"] = static_cast<int>(r.residuals.size());
  return out;
}

der::SwapSpreadRequest swap_spread_request_from_json(const json::object& o) {
  der::SwapSpreadRequest r;
  const auto field = [&](const char* k) -> const json::value& {
    return need(o, k, std::string("swap_spread: missing '") + k + "'");
  };
  r.value_date = swaps::build::Date::from_iso(str(field("value_date")));
  const std::string convention = str(field("convention"));
  r.bond = bond_id_from(field("bond").as_object(), convention, "swap_spread");
  settlement_into(o, r.settlement);
  if (present(o, "spread_type")) r.type = spread_type_from_str(str(o.at("spread_type")));
  r.clean = field("clean").to_number<double>();
  r.spread = field("spread").to_number<double>();
  into(o, "index", r.index);
  into(o, "tenor", r.tenor);
  into(o, "swap_curve", r.swap_curve);
  into(o, "factor_curve", r.factor_curve);
  into(o, "anchor", r.anchor);
  return r;
}
json::object swap_spread_to_json(const der::SwapSpreadResult& r) {
  json::object out;
  out["bond_yield"] = r.derived.bond_yield;
  out["spread"] = r.derived.spread;
  out["anchor"] = r.anchor;
  json::object rows;
  rows["pin"] = instrument_to_json(r.derived.rows.pin);  // QuoteKind::Rate on the govvie factor (bond bucket)
  rows["asw"] = instrument_to_json(r.derived.rows.asw);  // Portfolio{+swap, -Rate(factor)} (the ASW basis row)
  out["rows"] = std::move(rows);
  return out;
}

AssetSwapRequest asset_swap_request_from_json(const json::object& o) {
  AssetSwapRequest r;
  const auto field = [&](const char* k) -> const json::value& {
    return need(o, k, std::string("asset_swap: missing '") + k + "'");
  };
  r.value_date = swaps::build::Date::from_iso(str(field("value_date")));
  r.bundle = bundle_from_json(field("bundle"));
  into(o, "curve", r.curve);
  for (const auto& e : field("bonds").as_array()) {
    const auto& bo = e.as_object();
    const auto term = [&](const char* k) -> const json::value& {
      return need(bo, k, std::string("asset_swap: each bond needs '") + k + "'");
    };
    swaps::build::AssetSwapBond b;
    b.convention = str(term("convention"));
    b.issue = swaps::build::Date::from_iso(str(term("issue")));
    b.settle = swaps::build::Date::from_iso(str(term("settle")));
    b.maturity = swaps::build::Date::from_iso(str(term("maturity")));
    b.coupon = term("coupon").to_number<double>();
    into(bo, "freq", b.freq);
    into(bo, "index", b.index);
    into(bo, "clean", b.clean);
    into(bo, "dirty", b.dirty);
    r.bonds.push_back(std::move(b));
  }
  return r;
}

json::object asset_swap_to_json(const std::vector<swaps::build::AssetSwapAnalytics>& rows) {
  std::vector<double> asw, clean, dirty, annuity, accrued;
  for (const swaps::build::AssetSwapAnalytics& a : rows) {
    asw.push_back(a.asw_spread);
    clean.push_back(a.clean_curve);
    dirty.push_back(a.dirty_curve);
    annuity.push_back(a.annuity);
    accrued.push_back(a.accrued);
  }
  json::object out;
  out["asw_spread"] = vecf(asw);
  out["clean_curve"] = vecf(clean);
  out["dirty_curve"] = vecf(dirty);
  out["annuity"] = vecf(annuity);
  out["accrued"] = vecf(accrued);
  out["n"] = static_cast<int>(rows.size());
  return out;
}

cal::CalibrationReportRequest calib_report_request_from_json(const json::object& o) {
  cal::CalibrationReportRequest r;
  r.bundle = bundle_from_json(need(o, "bundle", "calib_report: missing 'bundle' object"));
  into(o, "x0", r.x0);
  r.reg = reg_from_json(o);
  return r;
}

json::object calibration_report_to_json(const cal::CalibrationReport& r) {
  json::array quotes;
  quotes.reserve(r.quotes.size());
  for (const cal::QuoteReport& q : r.quotes) {
    json::object o;
    o["model"] = q.fit.model;
    o["target"] = q.fit.target;
    o["residual"] = q.fit.residual;
    o["weight"] = q.fit.weight;
    o["in_band"] = q.fit.in_band;
    o["soft"] = q.fit.soft;
    o["identifiability"] = q.identifiability;
    quotes.push_back(std::move(o));
  }
  json::object out;
  out["rms_residual"] = r.calibration.rms_residual;
  out["converged"] = r.calibration.converged;
  out["status"] = r.calibration.status;
  out["rank_deficiency"] = r.calibration.rank_deficiency;
  out["condition_number"] = r.conditioning.condition_number;
  out["singular_values"] = vecf(r.conditioning.singular_values);
  out["quotes"] = std::move(quotes);
  out["n"] = static_cast<int>(r.quotes.size());
  return out;
}

json::array quote_diagnostics_to_json(const std::vector<cal::QuoteDiagnostic>& diagnostics) {
  json::array out;
  for (const cal::QuoteDiagnostic& q : diagnostics) {
    json::object d;
    d["model"] = q.model;
    d["target"] = q.target;
    d["residual"] = q.residual;
    d["soft"] = q.soft;
    if (q.soft) {
      d["lower"] = q.lower;
      d["upper"] = q.upper;
      d["decay"] = q.decay;
    }
    d["in_band"] = q.in_band;
    d["weight"] = q.weight;
    out.push_back(std::move(d));
  }
  return out;
}

cal::ConsistentRiskRequest consistent_risk_request_from_json(const json::object& o) {
  cal::ConsistentRiskRequest r;
  r.book = book_from_json(need(o, "book", "generate_risk: missing 'book'"));
  for (const auto& b : need(o, "bundles", "generate_risk: 'bundles' must be a non-empty array").as_array())
    r.bundles.push_back(bundle_from_json(b));
  r.reg = reg_from_json(o);
  return r;
}

json::object consistent_risk_to_json(const cal::ConsistentRisk& r) {
  json::array bundles;
  for (const cal::ConsistentBundleRisk& b : r.bundles) {
    json::object bo;
    bo["ladder"] = vecf(b.ladder);
    bo["synthetic"] = vecf(b.synthetic);
    json::array knots;
    for (int k : b.synthetic_knot) knots.push_back(k);
    bo["synthetic_knot"] = std::move(knots);
    bo["npv"] = b.npv;
    bo["pv01"] = b.pv01;
    bo["ladder_dv01"] = b.ladder_dv01;
    bo["n_residuals"] = b.n_residuals;
    bo["n_synthetic"] = b.n_synthetic;
    bundles.push_back(std::move(bo));
  }
  json::object out;
  out["npv"] = r.npv;
  out["pv01"] = r.pv01;
  out["n"] = r.n;
  out["bundles"] = std::move(bundles);
  return out;
}

json::array sample_to_json(const std::vector<CurveSample>& samples) {
  json::array carr;
  for (const auto& s : samples) {
    json::object co;
    co["currency"] = s.currency;
    co["t"] = vecf(s.t);
    co["discount"] = vecf(s.discount);
    co["zero"] = vecf(s.zero);
    co["forward"] = vecf(s.forward);
    carr.push_back(std::move(co));
  }
  return carr;
}

RegSpec reg_from_json(const json::object& request) {
  RegSpec reg;
  if (!request.contains("regularize") || request.at("regularize").is_null()) return reg;
  if (!request.at("regularize").is_object())
    throw std::invalid_argument("'regularize' must be an object {lambda, curves, tension, sigma}");
  const auto& r = request.at("regularize").as_object();
  into(r, "lambda", reg.lambda);
  into(r, "curves", reg.curves);
  into(r, "tension", reg.tension);  // continuous tension energy vs discrete second-difference
  into(r, "sigma", reg.sigma);      // tension parameter (tension=true); 0 => pure curvature
  return reg;
}

}  // namespace swaps::api
