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
#include "swaps/api/json_util.hpp"

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
// REVIEW FINDING 1 (2026-09-21): this used to concatenate the message straight into a JSON string
// literal. Exception messages embed CALLER strings -- codec.cpp throws "book: unknown position kind '" + s
// + "'" and friends -- so a bad enum value carrying a double quote returned bytes that are not JSON:
//   {"error":"book: unknown position kind 'sw"ap' (swap | xccy)"}   <- a host's parser rejects this
// Serialising through Boost.JSON escapes the message instead. (swaps_run_json was never affected: it
// delegates to run_json, which builds its error object the same way.)
const char* err_json(const std::string& what) {
  json::object o;
  o["error"] = what;
  return dup_str(json::serialize(o));
}

// REVIEW FINDING 4: swaps_session_create can only answer with a pointer, so a failed compile used to
// discard the one informative thing it had -- the compiler's diagnostic. It is recorded here and read back
// with swaps_last_error(); thread_local so two hosts' threads cannot overwrite each other's message.
std::string& last_error() {
  static thread_local std::string e;
  return e;
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
  } catch (const std::exception& e) {
    last_error() = e.what();       // FINDING 4: keep the diagnostic; the caller reads swaps_last_error()
    return nullptr;
  } catch (...) {
    last_error() = "swaps_session_create: unknown exception";
    return nullptr;
  }
}

extern "C" const char* swaps_last_error(void) {
  return last_error().empty() ? "" : last_error().c_str();
}

extern "C" const char* swaps_session_calibrate(void* session) {
  if (!session) return err_json("null session");
  try {
    auto* cs = static_cast<CapiSession*>(session);
    BundleSession* s = &cs->sess;
    const auto& r = s->calibrate(swaps::api::flat_x0(s->problem()), cs->reg);
    s->start_streaming(cs->reg);  // anchor the frozen-Newton warm path -- every bundle is eligible
    json::object o;
    o["regularize_applied"] = cs->reg.on();
    o["regularize_lambda"] = cs->reg.lambda;
    o["regularize_tension"] = cs->reg.tension;
    o["rms_residual"] = r.rms_residual;
    o["converged"] = r.converged;
    o["status"] = r.status;
    o["rank_deficiency"] = r.rank_deficiency;
    o["iterations"] = r.iterations;
    o["quote_diagnostics"] = s->quote_diagnostics();  // per-quote in-band fit (the document, E6.3)
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
    const std::vector<double> v = swaps::api::to_vec(json::parse(market_json));
    const Eigen::VectorXd m = Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
    s->stream_update(m);  // frozen-Newton µs tick over the cached statics (falls back to an LM solve itself)
    return dup_str("{\"ok\":true}");
  } catch (const std::exception& e) {
    return err_json(e.what());
  }
}

extern "C" const char* swaps_session_sample(void* session, const char* times_json) {
  if (!session || !times_json) return err_json("null arg");
  try {
    BundleSession* s = &static_cast<CapiSession*>(session)->sess;
    const std::vector<double> times = swaps::api::to_vec(json::parse(times_json));
    return dup_str(json::serialize(swaps::api::sample_to_json(s->sample(times))));  // the ONE sample codec (E6.3)
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
