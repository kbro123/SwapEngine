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

/* Free a string previously returned by swaps_run_json (or any session call below). */
void swaps_string_free(const char* s);

/* ---- STATEFUL calibrated-session handles (for a warm-recalibrating host: the Excel add-in, .NET, ctypes) --
 *
 * swaps_run_json above is STATELESS — each call builds its own session and throws it away. These five
 * functions instead hand the caller an OPAQUE HANDLE to a persistent calibrated session that OWNS the
 * compiled DF=exp(-Wx) statics + structure fingerprint + solved knots, so re-solving to a new market is the
 * µs WARM path (no recompile). This is the C boundary a stateful add-in (a curve1/model1 object cache) is
 * built on. Each handle is single-threaded; do not share one across threads.
 */

/* Compile a COMPOSER SPEC (the {"compile": <spec>} JSON shape) into a persistent session. `today` supplies
 * the value date when the spec omits one (pass NULL/"" to require it). Returns an opaque handle, or NULL on a
 * parse/compile error -- in which case swaps_last_error() carries the compiler's diagnostic. Release it
 * with swaps_session_free. */
void* swaps_session_create(const char* spec_json, const char* today);

/* The diagnostic of the last failed swaps_session_create ON THIS THREAD, or "" if none. The pointer is
 * owned by the library (thread_local) and stays valid until this thread's next failing create -- copy it
 * rather than holding it. Do NOT free it. Added 2026-09-21: a NULL handle used to be the whole story. */
const char* swaps_last_error(void);

/* Cold-calibrate the session (and anchor its warm path). Returns a newly-allocated JSON string
 * {"rms_residual","rank_deficiency","iterations"} (caller frees with swaps_string_free); {"error":...} on failure. */
const char* swaps_session_calibrate(void* session);

/* WARM re-solve to a new market — `market_json` is a JSON array of the quoted-instrument targets in residual
 * order — reusing the cached statics. Returns {"ok":true} or {"error":...}. Call swaps_session_calibrate first. */
const char* swaps_session_update(void* session, const char* market_json);

/* Sample every curve at `times_json` (a JSON array of times). Returns
 * [{"currency","t","discount","zero","forward"}, ...] (caller frees). */
const char* swaps_session_sample(void* session, const char* times_json);

/* Price a book off the session's calibrated curves (NO recalibration). `book_json` is the book_from_json
 * schema — typed "trades" (the engine builds the legs) or explicit-coupon "positions". Returns
 * {"npv","pv01","price_us","n"} (caller frees). */
const char* swaps_session_price(void* session, const char* book_json);

/* Release a session handle created by swaps_session_create. Safe on NULL. */
void swaps_session_free(void* session);

#ifdef __cplusplus
}
#endif

#endif /* SWAPS_API_CAPI_H */
