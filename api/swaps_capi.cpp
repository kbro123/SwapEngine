// C ABI implementation: a thin, allocation-owning wrapper over swaps::api::run_json (capi.h). This is the
// single symbol a non-C++ host binds to; all the risk machinery lives behind the JSON seam it forwards to.
#include "swaps/api/capi.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "swaps/api/bundle_api.hpp"

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
