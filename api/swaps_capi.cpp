// C ABI implementation: a thin, allocation-owning wrapper over swaps::api::run_json (capi.h). This is the
// single symbol a non-C++ host binds to; all the risk machinery lives behind the JSON seam it forwards to.
#include "swaps/api/capi.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/compile.hpp"

extern "C" const char* swaps_run_json(const char* req) {
  if (!req) return nullptr;
  std::string out;
  try {
    out = swaps::api::run_json(req);  // run_json already returns {"error":...} for bad requests in-band
  } catch (const std::exception&) {
    out = "{\"error\":\"capi: exception escaped run_json\"}";  // belt-and-suspenders; run_json shouldn't throw
  } catch (...) {
    out = "{\"error\":\"capi: unknown exception\"}";
  }
  char* buf = static_cast<char*>(std::malloc(out.size() + 1));
  if (!buf) return nullptr;
  std::memcpy(buf, out.c_str(), out.size() + 1);
  return buf;
}

extern "C" void swaps_string_free(const char* s) { std::free(const_cast<char*>(s)); }

// ---- stateful calibrated-session handles (warm-recalibrating hosts; see capi.h) -----------------------
namespace {
namespace json = boost::json;
using swaps::api::BundleSession;

const char* dup_str(const std::string& s) {
  char* buf = static_cast<char*>(std::malloc(s.size() + 1));
  if (buf) std::memcpy(buf, s.c_str(), s.size() + 1);
  return buf;
}
const char* err_json(const std::string& what) { return dup_str("{\"error\":\"" + what + "\"}"); }

std::vector<double> to_vec(const json::value& v) {
  std::vector<double> out;
  if (v.is_array())
    for (const auto& e : v.as_array()) out.push_back(e.to_number<double>());
  return out;
}
json::array darr(const std::vector<double>& v) {
  json::array a;
  a.reserve(v.size());
  for (double x : v) a.push_back(x);
  return a;
}
}  // namespace

// The C-ABI session: the compiled bundle's session PLUS the smoothing the spec asked for (E3-D4: until
// 2026-09-10 the session calibrated with reg = {} and streamed unregularised, so an under-determined spec
// gave Excel a rank-deficient curve 25 bp from the web's).
struct CapiSession {
  BundleSession sess;
  swaps::api::RegSpec reg;
};

extern "C" void* swaps_session_create(const char* spec_json, const char* today) {
  if (!spec_json) return nullptr;
  try {
    auto cr = swaps::api::compile_spec(json::parse(spec_json), today ? today : "");
    const swaps::api::RegSpec reg = swaps::api::compile_reg_spec(cr);
    return static_cast<void*>(new CapiSession{BundleSession(std::move(cr.bundle)), reg});
  } catch (...) {
    return nullptr;
  }
}

extern "C" const char* swaps_session_calibrate(void* session) {
  if (!session) return err_json("null session");
  try {
    auto* cs = static_cast<CapiSession*>(session);
    BundleSession* s = &cs->sess;
    const auto& r = s->calibrate(swaps::api::flat_x0(s->problem()), cs->reg);
    if (!s->needs_recalibrate()) s->start_streaming(cs->reg);  // anchor the frozen-Newton warm path when eligible
    json::object o;
    o["regularize_applied"] = cs->reg.on();
    o["regularize_lambda"] = cs->reg.lambda;
    o["regularize_tension"] = cs->reg.tension;
    o["rms_residual"] = r.rms_residual;
    o["converged"] = r.converged;
    o["status"] = r.status;
    o["rank_deficiency"] = r.rank_deficiency;
    o["iterations"] = r.iterations;
    o["quote_diagnostics"] = json::parse(s->quote_diagnostics_json());  // per-quote in-band fit
    return dup_str(json::serialize(o));
  } catch (const std::exception& e) {
    return err_json(e.what());
  }
}

extern "C" const char* swaps_session_update(void* session, const char* market_json) {
  if (!session || !market_json) return err_json("null arg");
  try {
    auto* cs = static_cast<CapiSession*>(session);
    BundleSession* s = &cs->sess;
    const std::vector<double> v = to_vec(json::parse(market_json));
    const Eigen::VectorXd m = Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
    if (s->needs_recalibrate())
      s->recalibrate(m, cs->reg);  // non-linear region: general warm re-solve
    else
      s->stream_update(m);        // frozen-Newton µs tick over the cached statics
    return dup_str("{\"ok\":true}");
  } catch (const std::exception& e) {
    return err_json(e.what());
  }
}

extern "C" const char* swaps_session_sample(void* session, const char* times_json) {
  if (!session || !times_json) return err_json("null arg");
  try {
    BundleSession* s = &static_cast<CapiSession*>(session)->sess;
    const std::vector<double> times = to_vec(json::parse(times_json));
    json::array curves;
    for (const auto& cs : s->sample(times)) {
      json::object c;
      c["currency"] = cs.currency;
      c["t"] = darr(cs.t);
      c["discount"] = darr(cs.discount);
      c["zero"] = darr(cs.zero);
      c["forward"] = darr(cs.forward);
      curves.push_back(c);
    }
    return dup_str(json::serialize(curves));
  } catch (const std::exception& e) {
    return err_json(e.what());
  }
}

extern "C" const char* swaps_session_price(void* session, const char* book_json) {
  if (!session || !book_json) return err_json("null arg");
  try {
    BundleSession* s = &static_cast<CapiSession*>(session)->sess;
    const auto pr = s->price_portfolio_json(book_json);   // book_from_json schema, repriced off x
    json::object o;
    o["npv"] = pr.npv;
    o["pv01"] = pr.pv01;
    o["price_us"] = pr.price_us;
    o["n"] = pr.n;
    return dup_str(json::serialize(o));
  } catch (const std::exception& e) {
    return err_json(e.what());
  }
}

extern "C" void swaps_session_free(void* session) { delete static_cast<CapiSession*>(session); }
