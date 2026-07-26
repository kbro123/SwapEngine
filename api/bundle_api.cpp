// Implementation of the bundle API facade (include/swaps/api/bundle_api.hpp).
//
// Boost.JSON is compiled ONCE here via <boost/json/src.hpp> so no other TU pays for it. Everything else
// is the thin glue between the JSON object graph and the engine's generic instrument model.

#include "swaps/api/bundle_api.hpp"

#include <stdexcept>
#include <utility>

#include <boost/json/src.hpp>  // header-only Boost.JSON, compiled in this TU only

#include "swaps/calibration/compiled_bundle.hpp"  // CompiledBundleResidual::model_rates (streaming anchor)
#include "swaps/calibration/jacobian.hpp"         // aad_jacobian (risk operator)
#include "swaps/calibration/regularize.hpp"       // smoothed(), second_difference_operator()

namespace swaps::api {

namespace json = boost::json;
namespace px = swaps::pricing;
namespace curve = swaps::curve;

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
  }
  return "ParRate";
}
cal::QuoteKind quote_from_str(const std::string& s) {
  if (s == "ParSpread") return cal::QuoteKind::ParSpread;
  if (s == "Rate") return cal::QuoteKind::Rate;
  if (s == "FxForward") return cal::QuoteKind::FxForward;
  if (s == "XccyMtmBasis") return cal::QuoteKind::XccyMtmBasis;
  if (s == "Portfolio") return cal::QuoteKind::Portfolio;
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
  }
  return "Hermite";
}

cal::BundleCurveSpec spec_from(const json::object& o) {
  cal::BundleCurveSpec s;
  s.meeting = get_da(o, "meeting");
  s.back = get_da(o, "back");
  s.base = get_i(o, "base", -1);
  s.currency = get_i(o, "currency", 0);
  if (o.contains("regions") && o.at("regions").is_array())
    for (const auto& e : o.at("regions").as_array()) {
      const auto& ro = e.as_object();
      curve::CurveModule m;
      m.scheme = scheme_from_str(get_s(ro, "scheme", "Hermite"));
      m.knots = get_da(ro, "knots");
      s.regions.push_back(std::move(m));
    }
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
  o["meeting"] = da(s.meeting);
  o["back"] = da(s.back);
  o["base"] = s.base;
  o["currency"] = s.currency;
  if (!s.regions.empty()) {
    json::array rs;
    for (const auto& m : s.regions) {
      json::object mo;
      mo["scheme"] = scheme_to_str(m.scheme);
      mo["knots"] = da(m.knots);
      rs.push_back(mo);
    }
    o["regions"] = rs;
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

Eigen::VectorXd flat_x0(const cal::BundleProblem& prob, double level) {
  Eigen::VectorXd x(prob.n_knots());
  int o = 0;
  for (const auto& c : prob.curves) {
    const double v = (c.base < 0) ? level : 0.0;  // outright at the level, spread at zero
    for (int i = 0; i < c.n_knots(); ++i) x[o++] = v;
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

BundleSession::BundleSession(cal::BundleProblem prob) : prob_(std::move(prob)) {
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
}

const cal::CalibrationResult& BundleSession::calibrate(const Eigen::VectorXd& x0, const RegSpec& reg) {
  if (reg.on())
    result_ = cal::calibrate(cal::smoothed(prob_, reg.lambda, reg.curves), x0, /*use_aad=*/true);
  else if (!has_nonlinear_)
    // HYBRID W-cache: cacheable rows on the fast path, any FX/MtM (or portfolio-with-FX) rows on a
    // width-reduced AAD block -- so one FX trade no longer drops the whole book to AAD.
    result_ = cal::calibrate(prob_, x0, /*use_aad=*/true);
  else
    // A non-linear region scheme (MonotoneCubic) has NO constant W at all, so the whole bundle prices on
    // the generic AAD engine (a zero-reg smoothed wrapper routes there).
    result_ = cal::calibrate(cal::smoothed(prob_, 0.0, {}), x0, /*use_aad=*/true);
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

Eigen::MatrixXd BundleSession::risk_operator(const RegSpec& reg) const {
  const Eigen::MatrixXd J = cal::aad_jacobian(prob_, x_);  // n_res x n_knots
  Eigen::MatrixXd A = J.transpose() * J;                   // n_knots x n_knots
  if (reg.on()) {
    const Eigen::MatrixXd R = cal::second_difference_operator(prob_, reg.lambda, reg.curves);
    A.noalias() += R.transpose() * R;
  }
  const Eigen::MatrixXd Ainv = A.ldlt().solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
  return Ainv * J.transpose();  // n_knots x n_res  = dx/dq
}

void BundleSession::start_streaming(const RegSpec& reg) {
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
  if (reg.on()) opt.regularizer = cal::second_difference_operator(prob_, reg.lambda, reg.curves);
  stream_ = std::make_unique<cal::StreamingCalibrator<cal::BundleProblem>>(prob_, x_, q0, opt);
}

const Eigen::VectorXd& BundleSession::stream_update(const Eigen::VectorXd& new_market) {
  if (!stream_) throw std::runtime_error("start_streaming() must be called first");
  stream_->update(new_market);
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

    if (o.contains("risk") && o.at("risk").as_bool()) {
      const Eigen::MatrixXd M = sess.risk_operator(reg);
      json::array rows;
      for (int i = 0; i < M.rows(); ++i) {
        json::array row;
        for (int j = 0; j < M.cols(); ++j) row.push_back(M(i, j));
        rows.push_back(row);
      }
      out["risk_operator"] = rows;
    }

    return json::serialize(json::value(std::move(out)));
  } catch (const std::exception& ex) {
    return err(ex.what());
  }
}

}  // namespace swaps::api
