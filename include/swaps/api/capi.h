/* swaps C ABI -- the ONE cross-language entry point.
 *
 * A non-C++ host (an Excel XLL, a .NET wrapper, Python ctypes, ...) links this and talks to the whole
 * engine through a single string-in/string-out call: the same JSON contract as swaps::api::run_json
 * (bundle -> calibrate + optional sample / price / portfolio / portfolio_risk / transform / risk_operator).
 * This header is pure C (no C++ / Boost / Eigen), so a host that only wants the JSON seam pays nothing.
 */
#ifndef SWAPS_API_CAPI_H
#define SWAPS_API_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Run a JSON request and return a newly-allocated, NUL-terminated JSON response. The CALLER OWNS the
 * result and must release it with swaps_string_free. Engine errors are returned in-band as
 * {"error": "..."} (not via the return value); the function returns NULL only when `req` is NULL or an
 * allocation fails. Thread-safe: no shared mutable state (each call constructs its own BundleSession). */
const char* swaps_run_json(const char* req);

/* Free a string previously returned by swaps_run_json. */
void swaps_string_free(const char* s);

#ifdef __cplusplus
}
#endif

#endif /* SWAPS_API_CAPI_H */
