// JSON <-> engine object graph: the definitions behind include/swaps/api/codec.hpp (read its contract first).
// Moved out of api/bundle_api.cpp on 2026-09-13 (E7 stage 2). The one change of substance: every reader now
// assigns ONLY when the field is present, so an absent field keeps the struct's own default instead of a
// literal restated here -- the struct is the single source of each default.
#include "swaps/api/codec.hpp"

#include <initializer_list>
#include <limits>
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
#include "swaps/derive/scenario.hpp"  // ScenarioRequest, ScenarioResult
#include "swaps/derive/scenario_grid.hpp"  // ScenarioGridRequest, ScenarioGridResult
#include "swaps/derive/var.hpp"  // VarRequest, VarResult
#include "swaps/calibration/pnl_explain.hpp"  // PnlRequest, PnlReport
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
  // REVIEW FINDING 3 (2026-09-21): this fell back to "ParRate", so a new kind SERIALISED as a par rate --
  // a silently wrong document rather than a loud failure. The switch is exhaustive (kQuoteKindCount).
  throw std::logic_error("quote_to_str: unhandled QuoteKind (add it here and to kQuoteKindCount)");
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
  // REVIEW FINDING 3: as above -- a new Scheme used to serialise as "Hermite", silently changing a curve's
  // interpolation in any document that round-trips through the codec.
  throw std::logic_error("scheme_to_str: unhandled Scheme (add it here and to kSchemeCount)");
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
  if (o.contains("fx_fixing_time")) {  // O-X3 piece 2: when the MtM notional's FX fixes
    c.fx_fixing_set = true;
    into(o, "fx_fixing_time", c.fx_fixing_time);
  }
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
  into(o, "fx_spot_time", L.fx_spot_time);
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
      if (ro.contains("reg_sigma") && ro.at("reg_sigma").to_number<double>() >= 0.0)
        throw std::invalid_argument("region 'reg_sigma': the tension-energy regulariser (and its per-region sigma) was retired on 2026-09-22; use reg_lambda");
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
  if (c.fx_fixing_set) o["fx_fixing_time"] = c.fx_fixing_time;
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
  if (L.fx_spot_time != 0.0) o["fx_spot_time"] = L.fx_spot_time;  // emitted only when set: documents stay byte-identical
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
  into(o, "fx_spot_time", ins.fx_spot_time);
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
  if (ins.fx_spot_time != 0.0) o["fx_spot_time"] = ins.fx_spot_time;
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
  if (o.contains("currency_codes"))
    for (const auto& e : o.at("currency_codes").as_array()) p.currency_codes.push_back(str(e));
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
  if (!p.currency_codes.empty()) o["currency_codes"] = json::array(p.currency_codes.begin(), p.currency_codes.end());
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
    into(po, "fx_spot_time", p.fx_spot_time);
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
  out["swap_maturity"] = swaps::build::iso(r.swap_maturity);  // the matched swap's termination
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

// A calibration's outcome: the calibrate request's "calibration" keys, in its order, without the regularize_* request
// echo and without wall-clock time (so a golden needs no timing cut). One shape for scenario, scenario_grid and var.
json::object calibration_status_to_json(const cal::CalibrationResult& c) {
  json::object o;
  o["iterations"] = c.iterations;
  o["rms_residual"] = c.rms_residual;
  o["stationarity"] = c.stationarity;
  o["info"] = c.info;
  o["converged"] = c.converged;
  o["status"] = c.status;
  o["rank_deficiency"] = c.rank_deficiency;
  return o;
}

RegSpec reg_from_json(const json::object& request) {
  RegSpec reg;
  if (!request.contains("regularize") || request.at("regularize").is_null()) return reg;
  if (!request.at("regularize").is_object())
    throw std::invalid_argument("'regularize' must be an object {lambda, curves}");
  const auto& r = request.at("regularize").as_object();
  into(r, "lambda", reg.lambda);
  into(r, "curves", reg.curves);
  // The tension-energy operator was retired on 2026-09-22 (regularize.hpp RegSpec). A request that still asks for it is
  // refused rather than silently given the curvature penalty; the no-op spellings (tension:false, sigma:0) are accepted.
  bool tension = false;
  double sigma = 0.0;
  into(r, "tension", tension);
  into(r, "sigma", sigma);
  if (tension || sigma != 0.0)
    throw std::invalid_argument("'regularize': the tension-energy operator ('tension':true / 'sigma') was retired on 2026-09-22; the penalty is the curvature (second-difference) operator: {lambda, curves}");
  return reg;
}


// ---- the conventions registry ------------------------------------------------------------------------------------
namespace {

namespace cvd = swaps::conventions;

std::string_view view(const json::string& s) { return {s.data(), s.size()}; }

// One conventions.json object and the row it belongs to (for the messages). Readers assign ONLY when the field is
// present, so an absent field keeps the struct's own unset value; a present value of the wrong JSON type throws.
struct ConvFields {
  const json::object& o;
  std::string where;  // "index 'USD-SOFR'"

  [[noreturn]] void bad(const char* k, const std::string& what) const {
    throw std::invalid_argument("conventions: " + where + ": '" + k + "' " + what);
  }
  void required(const char* k) const {
    if (!present(o, k)) bad(k, "is required");
  }
  std::string_view text(const char* k) const {
    if (!present(o, k)) return {};
    if (!o.at(k).is_string()) bad(k, "must be a string");
    return view(o.at(k).as_string());
  }
  void integer(const char* k, int& out) const {  // never a fraction, a string or out of range
    if (!present(o, k)) return;
    const json::value& v = o.at(k);
    if (!v.is_int64() || v.as_int64() < std::numeric_limits<int>::min() || v.as_int64() > std::numeric_limits<int>::max())
      bad(k, "must be an integer");
    out = static_cast<int>(v.as_int64());
  }
  void count(const char* k, int& out) const {  // a lag, a month or step count, a number of decimals
    if (!present(o, k)) return;
    integer(k, out);
    if (out < 0) bad(k, "must be >= 0");
  }
  void number(const char* k, double& out) const {
    if (!present(o, k)) return;
    if (!o.at(k).is_number()) bad(k, "must be a number");
    out = o.at(k).to_number<double>();
  }
  void flag(const char* k, bool& out) const {
    if (!present(o, k)) return;
    if (!o.at(k).is_bool()) bad(k, "must be true or false");
    out = o.at(k).as_bool();
  }
  const json::object* object(const char* k) const {
    if (!present(o, k)) return nullptr;
    if (!o.at(k).is_object()) bad(k, "must be an object");
    return &o.at(k).as_object();
  }
  const json::array* array(const char* k) const {
    if (!present(o, k)) return nullptr;
    if (!o.at(k).is_array()) bad(k, "must be an array");
    return &o.at(k).as_array();
  }
  long serial(const char* k, const json::value& v) const {  // a real calendar date -> its Unix-day serial
    if (!v.is_string()) bad(k, "must be a YYYY-MM-DD date");
    try {
      return swaps::build::Date::from_iso(std::string(view(v.as_string()))).serial();
    } catch (const std::invalid_argument& e) {
      bad(k, e.what());
    }
  }
  std::string_view date(const char* k) const {  // kept as text, checked as a date
    if (present(o, k)) (void)serial(k, o.at(k));
    return text(k);
  }
  // One leg slot, several spellings: at most one of `names` may be given.
  const json::object* leg(std::initializer_list<const char*> names) const {
    const json::object* found = nullptr;
    const char* first = nullptr;
    for (const char* n : names) {
      const json::object* l = object(n);
      if (!l) continue;
      if (found) bad(n, std::string("names the same leg as '") + first + "'");
      found = l;
      first = n;
    }
    return found;
  }
};

std::string row_name(const char* family, std::string_view id) { return std::string(family) + " '" + std::string(id) + "'"; }

cvd::LegConv leg_from_json(const json::object* leg, const std::string& where) {
  cvd::LegConv l;
  if (!leg) return l;
  const ConvFields f{*leg, where};
  l.index = f.text("index");
  l.day_count = f.text("day_count");
  l.frequency = f.text("frequency");
  l.compounding = f.text("compounding");
  f.flag("carries_spread", l.carries_spread);
  f.flag("notional_resets", l.notional_resets);
  f.flag("flat", l.flat);
  return l;
}

void product_from_json(cvd::OverlayBatch& b, std::string_view id, const json::object& o) {
  const ConvFields f{o, row_name("product", id)};
  f.required("description");
  cvd::ProductConv p;
  p.id = id;
  p.type = f.text("type");
  p.currency = f.text("currency");
  p.calendar = f.text("calendar");
  p.bdc = f.text("bdc");
  p.frequency = f.text("frequency");
  p.discount_index = f.text("discount_index");
  p.pair = f.text("pair");
  p.base_currency = f.text("base_currency");
  f.count("spot_lag", p.spot_lag);
  f.count("payment_lag", p.payment_lag);
  f.count("exchange_lag_initial", p.exchange_lag_initial);
  f.count("exchange_lag_intermediate", p.exchange_lag_intermediate);
  f.count("exchange_lag_final", p.exchange_lag_final);
  f.count("fx_reset_fixing_lag", p.fx_reset_fixing_lag);
  p.fx_reset_calendar = f.text("fx_reset_calendar");
  f.flag("zero_coupon", p.zero_coupon);
  p.fixed = leg_from_json(f.leg({"fixed_leg"}), f.where);
  p.floating = leg_from_json(f.leg({"float_leg", "spread_leg", "usd_leg"}), f.where);
  p.other = leg_from_json(f.leg({"flat_leg", "eur_leg"}), f.where);
  b.products.push_back(p);
}

void index_from_json(cvd::OverlayBatch& b, std::string_view id, const json::object& o) {
  const ConvFields f{o, row_name("index", id)};
  cvd::IndexConv x;
  x.id = id;
  x.currency = f.text("currency");
  x.type = f.text("type");
  x.day_count = f.text("day_count");
  x.calendar = f.text("calendar");
  x.par_product = f.text("par_product");
  x.tenor = f.text("tenor");
  f.count("fixing_lag", x.fixing_lag);
  f.count("publication_lag", x.publication_lag);
  b.indices.push_back(x);
}

void bond_from_json(cvd::OverlayBatch& b, std::string_view id, const json::object& o) {
  const ConvFields f{o, row_name("bond", id)};
  cvd::BondConv x;
  x.id = id;
  x.currency = f.text("currency");
  x.calendar = f.text("calendar");
  x.day_count = f.text("day_count");
  x.frequency = f.text("frequency");
  x.stub_discount = f.text("stub_discount");
  f.count("settle_lag", x.settle_lag);
  f.flag("final_period_simple", x.final_period_simple);
  b.bonds.push_back(x);
}

void currency_from_json(cvd::OverlayBatch& b, std::string_view code, const json::object& o) {
  const ConvFields f{o, row_name("currency", code)};
  cvd::CurrencyConv x;
  x.code = code;
  x.name = f.text("name");
  x.settlement_calendar = f.text("settlement_calendar");
  x.discount_index = f.text("discount_index");
  x.default_swap_product = f.text("default_swap_product");
  x.repo_day_count = f.text("repo_day_count");
  f.count("minor_units", x.minor_units);
  b.currencies.push_back(x);
}

void calendar_from_json(cvd::OverlayBatch& b, std::string_view id, const json::object& o) {
  const ConvFields f{o, row_name("calendar", id)};
  cvd::OverlayBatch::Calendar c;
  c.row.id = id;
  c.row.name = f.text("name");
  c.row.observance = f.text("observance");
  f.flag("sandwich", c.row.sandwich);
  const json::array* weekend = f.array("weekend");
  if (!weekend) f.bad("weekend", "is required (e.g. [5, 6]: Saturday and Sunday)");
  for (const json::value& d : *weekend) {
    if (!d.is_int64() || d.as_int64() < 0 || d.as_int64() > 6) f.bad("weekend", "days are integers 0 (Mon) .. 6 (Sun)");
    c.row.weekend_mask |= 1 << d.as_int64();
  }
  const json::array* holidays = f.array("holidays");
  const json::array* join = f.array("join");
  if (!holidays && !join) f.bad("holidays", "or 'join' is required ([] for no holidays)");
  if (holidays) {
    f.required("observance");
    for (const json::value& h : *holidays) {
      if (!h.is_object()) f.bad("holidays", "entries must be objects");
      const ConvFields r{h.as_object(), f.where + " holiday"};
      cvd::HolidayRule rule;
      rule.kind = r.text("rule");
      r.integer("month", rule.month);
      r.integer("day", rule.day);
      r.integer("weekday", rule.weekday);
      r.integer("n", rule.n);
      r.integer("days", rule.days);
      r.integer("from_year", rule.from_year);
      r.integer("to_year", rule.to_year);
      rule.observance = r.text("observance");
      r.flag("except_first_friday", rule.except_first_friday);
      c.rules.push_back(rule);
    }
  }
  if (join) {
    for (const json::value& j : *join) {
      if (!j.is_string()) f.bad("join", "entries must be calendar ids");
      c.joins.push_back(view(j.as_string()));
    }
  }
  b.calendars.push_back(std::move(c));
}

void cds_product_from_json(cvd::OverlayBatch& b, std::string_view id, const json::object& o) {
  const ConvFields f{o, row_name("cds product", id)};
  cvd::CreditConv x;
  x.id = id;
  x.currency = f.text("currency");
  x.calendar = f.text("calendar");
  x.day_count = f.text("day_count");
  x.frequency = f.text("frequency");
  x.roll = f.text("roll");
  f.number("recovery_default", x.recovery_default);
  f.count("settlement_lag", x.settlement_lag);
  f.count("protection_steps", x.protection_steps);
  b.credit_products.push_back(x);
}

void bond_future_from_json(cvd::OverlayBatch& b, std::string_view id, const json::object& o) {
  const ConvFields f{o, row_name("bond future", id)};
  cvd::BondFutureConv x;
  x.id = id;
  x.currency = f.text("currency");
  x.exchange_calendar = f.text("exchange_calendar");
  x.deliverable_convention = f.text("deliverable_convention");
  x.repo_day_count = f.text("repo_day_count");
  x.delivery = f.text("delivery");
  f.number("notional_coupon", x.notional_coupon);
  f.number("basket_min_years", x.basket_min_years);
  f.number("basket_max_years", x.basket_max_years);
  f.count("maturity_rounding_months", x.maturity_rounding_months);
  f.count("conversion_factor_decimals", x.conversion_factor_decimals);
  b.bond_futures.push_back(x);
}

void fx_pair_from_json(cvd::OverlayBatch& b, std::string_view id, const json::object& o) {
  const ConvFields f{o, row_name("fx pair", id)};
  cvd::FxPairConv x;
  x.id = id;
  x.base = f.text("base");
  x.quote = f.text("quote");
  x.calendar = f.text("calendar");
  x.premium_currency = f.text("premium_currency");
  x.delta_convention = f.text("delta_convention");
  x.atm_convention = f.text("atm_convention");
  x.xccy_product = f.text("xccy_product");
  x.forward_product = f.text("forward_product");
  f.count("spot_lag", x.spot_lag);
  if (const json::array* pillars = f.array("smile_pillars")) {
    for (const json::value& v : *pillars)
      if (!v.is_number()) f.bad("smile_pillars", "must be numbers");
    if (!pillars->empty()) {
      x.smile_pillar_lo = pillars->front().to_number<double>();
      x.smile_pillar_hi = pillars->back().to_number<double>();
    }
  }
  b.fx_pairs.push_back(x);
}

void cb_schedule_from_json(cvd::OverlayBatch& b, std::string_view currency, const json::object& o) {
  const ConvFields f{o, row_name("cb schedule", currency)};
  cvd::OverlayBatch::CbSchedule c;
  c.row.currency = currency;
  c.row.bank = f.text("bank");
  c.row.source = f.text("source");
  c.row.as_of = f.date("as_of");
  const json::array* meetings = f.array("meetings");
  if (!meetings) f.bad("meetings", "is required (an array of YYYY-MM-DD dates)");
  for (const json::value& m : *meetings) c.meetings.push_back(f.serial("meetings", m));
  b.cb_schedules.push_back(std::move(c));
}

void fixing_source_from_json(cvd::OverlayBatch& b, std::string_view index_id, const json::object& o) {
  const ConvFields f{o, row_name("fixing source", index_id)};
  cvd::FixingSourceConv x;
  x.id = index_id;
  x.provider = f.text("provider");
  x.series = f.text("series");
  x.start = f.date("start");
  x.granularity = f.text("granularity");
  b.fixing_sources.push_back(x);
}

void inflation_index_from_json(cvd::OverlayBatch& b, std::string_view id, const json::object& o) {
  const ConvFields f{o, row_name("inflation index", id)};
  cvd::InflationIndexConv x;
  x.id = id;
  x.label = f.text("label");
  x.currency = f.text("currency");
  x.calendar = f.text("calendar");
  x.interpolation = f.text("interpolation");
  x.frequency = f.text("frequency");
  f.count("observation_lag_months", x.observation_lag_months);
  b.inflation_indices.push_back(x);
}

using RowDecoder = void (*)(cvd::OverlayBatch&, std::string_view, const json::object&);

void rows_from_json(cvd::OverlayBatch& b, const ConvFields& f, const char* family, RowDecoder decode) {
  const json::object* rows = f.object(family);
  if (!rows) return;
  for (const auto& kv : *rows) {
    const std::string_view id(kv.key().data(), kv.key().size());
    if (!kv.value().is_object()) throw std::invalid_argument("conventions: " + row_name(family, id) + " must be an object");
    decode(b, id, kv.value().as_object());
  }
}

json::object listing_to_json(const cvd::Registry::Listing& l) {
  json::array baked, overlay;
  for (const auto& id : l.baked) baked.emplace_back(id);
  for (const auto& id : l.overlay) overlay.emplace_back(id);
  return json::object{{"baked", std::move(baked)}, {"overlay", std::move(overlay)}};
}

}  // namespace

cvd::OverlayBatch overlay_batch_from_json(const json::object& payload) {
  const json::object* families = ConvFields{payload, "request"}.object("conventions");
  if (!families) families = &payload;
  const ConvFields f{*families, "request"};
  // Every key is a family the registry has, the overlay switch, or a conventions.json document key it never stores.
  static constexpr std::string_view kKeys[] = {"currencies",   "calendars",      "indices",   "products",
                                               "bonds",        "credit",         "bond_futures", "fx_pairs",
                                               "cb_schedules", "fixing_sources", "inflation", "clear_overlay",
                                               "$schema",      "meta",           "day_counts"};
  for (const auto& kv : *families) {
    const std::string_view key(kv.key().data(), kv.key().size());
    bool known = false;
    for (std::string_view k : kKeys) known = known || k == key;
    if (!known) throw std::invalid_argument("conventions: unknown family '" + std::string(key) + "'");
  }
  cvd::OverlayBatch b;
  f.flag("clear_overlay", b.clear_first);
  rows_from_json(b, f, "currencies", currency_from_json);
  rows_from_json(b, f, "calendars", calendar_from_json);
  rows_from_json(b, f, "indices", index_from_json);
  rows_from_json(b, f, "products", product_from_json);
  rows_from_json(b, f, "bonds", bond_from_json);
  if (const json::object* credit = f.object("credit")) rows_from_json(b, ConvFields{*credit, "credit"}, "cds_products", cds_product_from_json);
  rows_from_json(b, f, "bond_futures", bond_future_from_json);
  rows_from_json(b, f, "fx_pairs", fx_pair_from_json);
  rows_from_json(b, f, "cb_schedules", cb_schedule_from_json);
  rows_from_json(b, f, "fixing_sources", fixing_source_from_json);
  rows_from_json(b, f, "inflation", inflation_index_from_json);
  return b;
}

json::object overlay_added_to_json(const cvd::OverlayBatch& b, int overlay_size) {
  json::object added;
  const auto put = [&added](const char* family, const auto& rows, auto id) {
    if (rows.empty()) return;
    json::array ids;
    for (const auto& r : rows) ids.emplace_back(id(r));
    added[family] = std::move(ids);
  };
  const auto by_id = [](const auto& r) { return r.id; };
  put("currencies", b.currencies, [](const cvd::CurrencyConv& c) { return c.code; });
  put("calendars", b.calendars, [](const cvd::OverlayBatch::Calendar& c) { return c.row.id; });
  put("indices", b.indices, by_id);
  put("products", b.products, by_id);
  put("bonds", b.bonds, by_id);
  put("cds_products", b.credit_products, by_id);
  put("bond_futures", b.bond_futures, by_id);
  put("fx_pairs", b.fx_pairs, by_id);
  put("cb_schedules", b.cb_schedules, [](const cvd::OverlayBatch::CbSchedule& c) { return c.row.currency; });
  put("fixing_sources", b.fixing_sources, by_id);
  put("inflation", b.inflation_indices, by_id);
  return json::object{{"added", std::move(added)}, {"overlay_size", overlay_size}};
}

json::object conventions_listing_to_json(const cvd::Registry::Listings& l) {
  return json::object{{"currencies", listing_to_json(l.currencies)},
                      {"calendars", listing_to_json(l.calendars)},
                      {"indices", listing_to_json(l.indices)},
                      {"products", listing_to_json(l.products)},
                      {"bonds", listing_to_json(l.bonds)},
                      {"cds_products", listing_to_json(l.credit_products)},
                      {"bond_futures", listing_to_json(l.bond_futures)},
                      {"fx_pairs", listing_to_json(l.fx_pairs)},
                      {"cb_schedules", listing_to_json(l.cb_schedules)},
                      {"fixing_sources", listing_to_json(l.fixing_sources)},
                      {"inflation", listing_to_json(l.inflation_indices)},
                      {"overlay_size", l.overlay_size}};
}


// ---- scenario (the `scenario` verb) ---------------------------------------------------------------------------
namespace {

// A shift_curve key names an integer curve role: a bundle's curves have no string names.
int curve_role_from_key(const std::string& key, const char* verb) {
  try {
    std::size_t pos = 0;
    const int role = std::stoi(key, &pos);
    if (pos != key.size()) throw std::invalid_argument("trailing");
    return role;
  } catch (const std::exception&) {
    throw std::invalid_argument(
        std::string(verb) + ": shift_curve key '" + key +
        "' is not an integer curve role (curves in a bundle are addressed by integer index, not name)");
  }
}

derive::ScenarioMove scenario_move_from_json(const json::object& o, const char* verb) {
  derive::ScenarioMove m;
  into(o, "name", m.name);
  into(o, "parallel_bp", m.parallel_bp);
  if (present(o, "shift_curve"))
    for (const auto& kv : o.at("shift_curve").as_object()) {
      const int role = curve_role_from_key(std::string(kv.key()), verb);
      // "0" and "00" name one role: refused, rather than silently keeping one of them.
      if (m.shift_curve_bp.count(role))
        throw std::invalid_argument(std::string(verb) + ": shift_curve names curve role " + std::to_string(role) +
                                    " twice");
      m.shift_curve_bp[role] = kv.value().to_number<double>();
    }
  if (present(o, "bump_fx"))
    for (const auto& e : o.at("bump_fx").as_array()) {
      const json::object& fo = e.as_object();
      derive::FxBump b;
      into(fo, "base", b.base);
      into(fo, "quote", b.quote);
      into(fo, "rel", b.rel);
      m.fx.push_back(std::move(b));
    }
  return m;
}

}  // namespace

derive::ScenarioRequest scenario_request_from_json(const json::object& payload) {
  const json::object* body = &payload;
  if (present(payload, "scenario")) body = &payload.at("scenario").as_object();
  const json::object& o = *body;
  derive::ScenarioRequest r;
  r.bundle = bundle_from_json(need(o, "bundle", "scenario: missing 'bundle' object"));
  into(o, "x0", r.x0);
  r.reg = reg_from_json(o);
  into(o, "sample_times", r.sample_times);
  if (present(o, "book")) r.book = book_from_json(o.at("book"));
  into(o, "fx_pivot", r.fx_pivot);
  if (present(o, "scenarios"))
    for (const auto& e : o.at("scenarios").as_array()) r.scenarios.push_back(scenario_move_from_json(e.as_object(), "scenario"));
  return r;
}

json::object scenario_result_to_json(const derive::ScenarioResult& r) {
  json::object out;
  out["n_curves"] = r.n_curves;
  out["n_knots"] = r.n_knots;
  out["calibration"] = calibration_status_to_json(r.calibration);
  {
    json::object b;
    b["x"] = vecf(r.x_base);
    if (!r.base_curves.empty()) b["curves"] = sample_to_json(r.base_curves);
    if (r.has_book) {
      b["npv"] = r.base_npv;
      b["n"] = r.n_positions;
    }
    out["base"] = std::move(b);
  }
  json::array rows;
  for (const derive::ScenarioRow& row : r.rows) {
    const derive::ScenarioMove& m = r.moves[row.move];
    json::object o;
    o["name"] = m.name;
    if (!row.curves.empty()) o["curves"] = sample_to_json(row.curves);
    if (r.has_book) {
      o["npv"] = row.npv;
      o["npv_delta"] = row.npv_delta;
    }
    if (m.parallel_bp) o["parallel_bp"] = *m.parallel_bp;  // the move as given, echoed
    if (!m.shift_curve_bp.empty()) {
      json::object shifts;
      for (const auto& [role, bp] : m.shift_curve_bp) shifts[std::to_string(role)] = bp;
      o["shift_bp"] = std::move(shifts);
    }
    if (!m.fx.empty()) {
      json::array bumps;
      for (const derive::FxBump& f : m.fx) bumps.push_back(json::object{{"base", f.base}, {"quote", f.quote}, {"rel", f.rel}});
      o["fx"] = std::move(bumps);
    }
    rows.push_back(std::move(o));
  }
  out["scenarios"] = std::move(rows);
  return json::object{{"scenario", std::move(out)}};
}


// ---- scenario grid (the `scenario_grid` verb) -----------------------------------------------------------------
namespace {

derive::ShockAxisKind shock_axis_kind_from_str(const std::string& s) {
  if (s == "parallel_bp") return derive::ShockAxisKind::ParallelBp;
  if (s == "shift_curve") return derive::ShockAxisKind::ShiftCurve;
  if (s == "fx") return derive::ShockAxisKind::Fx;
  throw std::invalid_argument("scenario_grid: axis kind '" + s + "' is not one of parallel_bp / shift_curve / fx");
}
const char* shock_axis_kind_to_str(derive::ShockAxisKind k) {
  switch (k) {
    case derive::ShockAxisKind::ParallelBp: return "parallel_bp";
    case derive::ShockAxisKind::ShiftCurve: return "shift_curve";
    case derive::ShockAxisKind::Fx: return "fx";
  }
  throw std::invalid_argument("scenario_grid: unknown axis kind");
}

derive::ShockAxis shock_axis_from_json(const json::object& o) {
  derive::ShockAxis ax;
  into(o, "label", ax.label);
  if (present(o, "kind")) ax.kind = shock_axis_kind_from_str(str(o.at("kind")));
  into(o, "role", ax.role);
  into(o, "base", ax.base);
  into(o, "quote", ax.quote);
  into(o, "values", ax.values);
  return ax;
}

}  // namespace

derive::ScenarioGridRequest scenario_grid_request_from_json(const json::object& payload) {
  const json::object* body = &payload;
  if (present(payload, "scenario_grid")) body = &payload.at("scenario_grid").as_object();
  const json::object& o = *body;
  derive::ScenarioGridRequest r;
  r.bundle = bundle_from_json(need(o, "bundle", "scenario_grid: missing 'bundle' object"));
  for (const auto& e : need(o, "axes", "scenario_grid: missing 'axes' array (1 or 2 shock axes)").as_array())
    r.axes.push_back(shock_axis_from_json(e.as_object()));
  into(o, "x0", r.x0);
  r.reg = reg_from_json(o);
  into(o, "sample_times", r.sample_times);
  if (present(o, "book")) r.book = book_from_json(o.at("book"));
  into(o, "fx_pivot", r.fx_pivot);
  return r;
}

json::object scenario_grid_result_to_json(const derive::ScenarioGridResult& r) {
  json::object out;
  out["n_curves"] = r.n_curves;
  out["n_knots"] = r.n_knots;
  out["calibration"] = calibration_status_to_json(r.calibration);
  {
    json::array axes;
    for (const derive::ShockAxis& ax : r.axes) {
      json::object a;
      a["label"] = ax.label;
      a["kind"] = shock_axis_kind_to_str(ax.kind);
      if (ax.kind == derive::ShockAxisKind::ShiftCurve) a["role"] = *ax.role;
      if (ax.kind == derive::ShockAxisKind::Fx) {
        a["base"] = ax.base;
        a["quote"] = ax.quote;
      }
      a["values"] = vecf(ax.values);
      axes.push_back(std::move(a));
    }
    out["axes"] = std::move(axes);
  }
  out["shape"] = json::array{r.n0, r.n1};
  {
    json::object b;
    b["x"] = vecf(r.x_base);
    if (!r.base_curves.empty()) b["curves"] = sample_to_json(r.base_curves);
    if (r.has_book) {
      b["npv"] = r.base_npv;
      b["n"] = r.n_positions;
    }
    out["base"] = std::move(b);
  }
  if (r.has_book) {
    json::array npv, pnl;
    for (const auto& row : r.npv) npv.push_back(vecf(row));
    for (const auto& row : r.pnl) pnl.push_back(vecf(row));
    out["npv"] = std::move(npv);
    out["pnl"] = std::move(pnl);
    out["grid_us"] = r.grid_us;
    out["n_cells"] = r.n_cells;
  }
  return json::object{{"scenario_grid", std::move(out)}};
}


// ---- VaR / ES (the `var` verb) --------------------------------------------------------------------------------------
derive::VarRequest var_request_from_json(const json::object& payload) {
  const json::object* body = &payload;
  if (present(payload, "var")) body = &payload.at("var").as_object();
  const json::object& o = *body;
  derive::VarRequest r;
  into(o, "quantiles", r.quantiles);
  into(o, "pnl", r.pnl);
  if (present(o, "bundle") || present(o, "book") || present(o, "scenarios")) {
    // Presence first, in the verb's order, so a request missing two pieces names the one the verb named.
    const json::value& bundle = need(o, "bundle", "var: needs either 'pnl' or 'bundle'+'scenarios'");
    const json::value& scenarios = need(o, "scenarios", "var: reval mode needs a 'scenarios' array of market moves");
    const json::value& book = need(o, "book", "var: reval mode needs a 'book' to reprice");
    derive::VarRevalRequest v;
    v.bundle = bundle_from_json(bundle);
    into(o, "x0", v.x0);
    v.reg = reg_from_json(o);
    v.book = book_from_json(book);
    into(o, "fx_pivot", v.fx_pivot);
    for (const auto& e : scenarios.as_array()) v.scenarios.push_back(scenario_move_from_json(e.as_object(), "var"));
    r.reval = std::move(v);
  }
  return r;
}

// Key order is the verb's, byte for byte: mode, [calibration (SC3)], [base_npv, n_positions, reval_us], n, mean_pnl,
// stdev_pnl, pnl_sorted, quantiles[{q, var, es, var_pnl, es_pnl}].
json::object var_result_to_json(const derive::VarResult& r) {
  json::object out;
  if (r.reval) {
    out["mode"] = "reval";
    out["calibration"] = calibration_status_to_json(r.reval->calibration);
    out["base_npv"] = r.reval->base_npv;
    out["n_positions"] = r.reval->n_positions;
    out["reval_us"] = r.reval->reval_us;
  } else {
    out["mode"] = "supplied";
  }
  const derive::PnlDistribution& d = r.distribution;
  out["n"] = d.n;
  out["mean_pnl"] = d.mean_pnl;
  out["stdev_pnl"] = d.stdev_pnl;
  out["pnl_sorted"] = vecf(d.pnl_sorted);
  json::array qout;
  for (const derive::VarQuantile& q : d.quantiles) {
    json::object qo;
    qo["q"] = q.q;
    qo["var"] = q.var;
    qo["es"] = q.es;
    qo["var_pnl"] = q.var_pnl;
    qo["es_pnl"] = q.es_pnl;
    qout.push_back(std::move(qo));
  }
  out["quantiles"] = std::move(qout);
  return json::object{{"var", std::move(out)}};
}

// ---- P&L explain (the `pnl` verb) -------------------------------------------------------------------------------
cal::PnlRequest pnl_request_from_json(const json::object& payload) {
  const json::object* body = &payload;
  if (present(payload, "pnl")) body = &payload.at("pnl").as_object();
  const json::object& o = *body;
  cal::PnlRequest r;
  r.bundle0 = bundle_from_json(need(o, "bundle0", "pnl: missing 'bundle0' object"));
  if (present(o, "bundle1")) r.bundle1 = bundle_from_json(o.at("bundle1"));
  r.book = book_from_json(need(o, "book", "pnl: missing 'book' object"));
  into(o, "dt_years", r.dt_years);
  into(o, "x0", r.x0);
  into(o, "x1", r.x1);
  r.reg = reg_from_json(o);
  return r;
}

json::object pnl_report_to_json(const cal::PnlReport& r) {
  json::object out;
  out["total"] = r.explain.total;
  out["carry"] = r.explain.carry;
  out["roll"] = r.explain.roll;
  out["market"] = r.explain.market;
  out["residual"] = r.explain.residual;
  out["npv_t0"] = r.explain.npv_t0;
  out["npv_t1"] = r.explain.npv_t1;
  out["market_ladder"] = vecf(r.explain.market_ladder);
  out["dq"] = vecf(r.dq);
  out["dt_years"] = r.dt_years;
  out["n"] = r.n;
  return json::object{{"pnl", std::move(out)}};
}

}  // namespace swaps::api
