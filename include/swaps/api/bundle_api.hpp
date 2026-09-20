#pragma once
// Public API facade for building, calibrating, querying and streaming a whole curve BUNDLE.
//
// This is the stable seam a web/service layer talks to (CLAUDE.md is the engine; this is its product
// contract). It has TWO faces:
//   1. A JSON <-> engine-object-graph mapping (bundle_from_json / bundle_to_json and the per-object
//      (de)serializers) so a caller in any language can DEFINE and CONSTRUCT the underlying objects --
//      curves, legs, coupons, observations, instruments, the whole BundleProblem -- as data.
//   2. A stateful BundleSession that spins up an engine instance over one BundleProblem and exposes
//      calibrate / build-curves / sample / price-an-instrument / risk-operator / streaming.
//
// The API core is QuantLib-FREE: it consumes the plain-data generic instrument model (design §3). A
// later layer can add convention-driven builders (schedule -> generic coupons via ql/extract.hpp);
// that is where index/calendar knowledge belongs (CLAUDE.md §1), not here.
//
// Boost.JSON is vendored (third_party/boost); only <boost/json/fwd.hpp> leaks into this header, so a
// consumer that never touches JSON pays nothing. The Boost.JSON implementation is compiled once inside
// api/bundle_api.cpp.

#include <Eigen/Dense>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/json/fwd.hpp>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_state.hpp"  // CurveSample, sample_bundle_curves
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"  // RegSpec, smoothing_preset
#include "swaps/calibration/streaming.hpp"
#include "swaps/calibration/structure_fingerprint.hpp"  // StampedBundle / structural_stamp (E8)
#include "swaps/portfolio/portfolio.hpp"  // MultiCurveBook — the batched reprice kernel
#include "swaps/portfolio/compiled_multi.hpp"  // CompiledMultiCurveBook — the cached streaming reprice twin
#include "swaps/pricing/fixings.hpp"
#include "swaps/api/codec.hpp"  // the JSON <-> object-graph codecs (E7)

namespace swaps::api {

namespace cal = swaps::calibration;

// ---- plain result structs (no Boost types, so they cross the header boundary cheaply) -------------

// CurveSample (a curve on a time grid) lives with sample_bundle_curves in calibration/bundle_state.hpp; codec.hpp
// aliases it here as api::CurveSample.

// RegSpec (the smoothness regulariser) lives with its operators in calibration/regularize.hpp.
using RegSpec = cal::RegSpec;

// Result of a batched portfolio reprice (BundleSession::price_portfolio). `price_us` is the ENGINE-
// measured wall time of the pure pricing pass ONLY (a steady_clock pair around the double NPV valuation
// off the calibrated curves — no JSON/marshalling), the exact analogue of last_solve_us() for calibration.
struct PortfolioReprice {
  double npv = 0.0;      // total book NPV off the currently calibrated curves (discount-currency units)
  double pv01 = 0.0;     // d(NPV) for a +1bp PARALLEL shift of every fitted knot forward, one AAD pass
  double price_us = 0.0;  // pure engine pricing time of the double NPV pass, microseconds
  int n = 0;             // number of positions priced
};

// Result of expressing a portfolio's risk in the bundle's CALIBRATION INSTRUMENTS (BundleSession::
// price_portfolio_risk). `curve_grad` is the portfolio NPV gradient wrt the fitted STATE (knot forwards),
// dP/dx, from one forward-AAD pass. `ladder` is that same risk expressed per calibration instrument:
// dP/dq = curve_grad^T · M where M = risk_operator() = dx/dq, so ladder[i] = Σⱼ curve_grad[j]·M[j,i] is the
// portfolio's delta to a unit move in instrument i's model quote (a full analytic delta ladder, no bumping).
// This is the primitive a Python layer combines across two sessions to TRANSFORM a risk ladder from one
// bundle's instruments to another's: T = J_A · M_B, delta_B = delta_A · T (see jacobian() / risk_operator()).
// `risk_us` is the ENGINE-measured wall time of the AAD reprice + the M multiply only (a steady_clock pair,
// the exact analogue of price_us / last_solve_us), also cached in last_risk_us().
struct PortfolioRisk {
  double npv = 0.0;                  // book NPV off the calibrated curves (== price_portfolio().npv)
  Eigen::VectorXd curve_grad;        // dP/dx, length n_knots (AAD gradient wrt the fitted state)
  Eigen::VectorXd ladder;            // dP/dq = curve_grad^T · M, length n_residuals (native delta ladder)
  double risk_us = 0.0;              // engine time of the AAD reprice + M multiply, microseconds
  int n = 0;                         // number of positions
};

// Result of a batched swaption VOL CUBE reprice off the calibrated curve (BundleSession::price_vol_cube_json).
// Flat SoA (parallel arrays), so the whole expiry x tenor x strike surface reprices into reused buffers on the
// engine hot path — the options analogue of price_portfolio. The curve-dependent part (per-cell forward /
// annuity / expiry) is computed once from a SINGLE curve sample over the union of all schedule times; every
// strike is then a pure Bachelier/SABR evaluation in C++. `price_us` is the ENGINE-measured pricing time (a
// steady_clock pair around the reprice only, no JSON/marshalling), the exact analogue of price_us/last_solve_us.
// Per-cell arrays have length n_cells; per-point arrays have length n_points, and point i belongs to the cell
// index point_cell[i]. All arrays are double (int-valued ones like point_cell/payer carry whole numbers) so the
// one STRUCT marshaller (VEC) covers every field.
struct VolCube {
  std::vector<double> cell_forward;        // per cell: forward swap rate off the calibrated curve
  std::vector<double> cell_annuity;        // per cell: Σ τ_i·DF_i (the numeraire)
  std::vector<double> cell_expiry_years;   // per cell: option expiry in curve time (ACT/365F)
  std::vector<double> point_cell;          // per point: which cell (index into the per-cell arrays)
  std::vector<double> strike;              // per point: absolute strike (rate)
  std::vector<double> moneyness_bp;        // per point: (strike − forward) in bp
  std::vector<double> normal_vol;          // per point: Bachelier normal vol used (SABR at the strike, or flat)
  std::vector<double> price;               // per point: swaption price (annuity units)
  std::vector<double> vega;                // per point: dV/dσ
  std::vector<double> delta;               // per point: dV/dF
  std::vector<double> gamma;               // per point: d²V/dF²
  std::vector<double> vanna;               // per point: d²V/(dF dσ)
  std::vector<double> volga;               // per point: d²V/dσ²
  std::vector<double> payer;               // per point: 1 payer, 0 receiver
  double price_us = 0.0;                   // pure engine pricing time of the cube, microseconds
  int n_cells = 0;
  int n_points = 0;
};

// Native input for the vol cube (the C++ analogue of the JSON `vol_cube` document) — so callers and benchmarks
// build a surface as typed structs, no JSON in the hot path. sabr wins over normal_vol when has_sabr; strikes
// combine absolute `strikes` + `moneyness_bp` offsets from the forward + `atm`; `payer` is honoured only when
// `payer_set`, else the OTM convention (payer above the forward) is used.
struct VolCubeCell {
  std::string expiry;
  std::string tenor;
  bool has_sabr = false;
  double sabr_alpha = 0.0, sabr_rho = 0.0, sabr_nu = 0.0;
  double sabr_beta = 0.0;  // CEV backbone in [0,1]; 0 = normal SABR (E3-F1: was dropped by every verb until 2026-09-10)
  double normal_vol = 0.0;
  bool payer_set = false;
  bool payer = true;
  std::vector<double> strikes;       // absolute strike rates
  std::vector<double> moneyness_bp;  // strikes at forward + bp/1e4
  bool atm = false;                  // include the ATM strike (forward)
};
struct VolCubeSpec {
  std::string value_date;
  std::string currency;  // optional; must agree with the index's DB currency when given
  std::string index;     // REQUIRED: the swap's DB index (carries the product conventions)
  int curve = 0;
  std::vector<VolCubeCell> cells;
};

// One swaption cell's CURVE-INDEPENDENT schedule in curve time (ACT/365F): the option expiry, the underlying
// swap start, and its annual fixed pay times + accruals. Building this walks the calendar (business-day
// adjustment / holidays) for every pay date, which is the same for every reprice — so price_vol_cube_json
// memoizes it per (value_date, currency, index, curve, expiry, tenor). See BundleSession::vol_sched_cache_.
struct SwaptionSchedule {
  double t_start = 0.0;
  double t_expiry = 0.0;
  std::vector<double> pay_time;
  std::vector<double> tau;
};

// The JSON <-> engine object-graph codecs are declared in swaps/api/codec.hpp (included above).

// The market-implied flat seed (calibration/bundle_problem.hpp), re-exported for api callers.
using swaps::calibration::flat_x0;

// ---- the session facade --------------------------------------------------------------------------
class BundleSession {
 public:
  explicit BundleSession(cal::BundleProblem prob);

  // Cold-calibrate from x0. Engine is chosen automatically: the compiled W-cache analytic path for an
  // all-linear bundle (ParRate/ParSpread/Rate), the AAD analytic path when an FX-forward / MtM-xccy
  // instrument is present (the compiled engine rejects those) or when a regulariser is requested.
  const cal::CalibrationResult& calibrate(const Eigen::VectorXd& x0, const RegSpec& reg = {});

  // ---- the object model (E4.A, 2026-09-10): ONE compiled engine, scalar inputs, structure immutable ----
  // A session compiles its hybrid engine once per STRUCTURE and every consumer borrows it: calibrate, the
  // streamer, the risk operator, the bound book's twin. The quote RHS (market targets + soft bands) is
  // per-row SCALAR data on that engine -- set_market / set_band write it with no Instrument copy and no
  // solve; resolve() then re-solves to it: one frozen-Newton tick on the shared engine seeded from the
  // current x (an LM warm solve only if the tick cannot converge, e.g. a 500 bp jump, or for a bundle with
  // no constant W). Anything else -- a knot, scheme, role, schedule, instrument, the evaluation date -- is
  // STRUCTURE: compile a new session (there is no runtime structure hash gating a warm path any more).
  void set_market(const Eigen::VectorXd& market);                          // all targets; no solve
  void set_band(int row, double lower, double upper, double decay);        // one row's band; no solve
  const cal::CalibrationResult& resolve(const RegSpec& reg = {});          // re-solve to the current inputs

  // set_market + resolve: re-solve to a NEW market vector. On a streaming session this IS a tick (the
  // streamer shares the engine); on a fresh session it starts streaming first (one Jacobian) -- so a
  // requote costs a tick, not an LM. `reg` selects the regulariser the streamer runs under.
  const cal::CalibrationResult& recalibrate(const Eigen::VectorXd& new_market, const RegSpec& reg = {});

  // The FULL quote RHS from a structurally-identical problem `p` (targets AND bands), then resolve(). `p`
  // must be the SAME structure (an O(n) structural equality, no hash -- calibration::structure_equal):
  // a different residual count, knot, scheme, role, schedule or instrument throws -- compile a new
  // session for that. A band edit is a quote change and re-anchors the streamer's active set in place.
  const cal::CalibrationResult& rebind(const cal::BundleProblem& p, const RegSpec& reg = {});
  // STAMPED rebind (E8 prototype): the same requote, but the caller hands over a bundle that carries the
  // structural stamp it was BUILT with (cal::make_stamped), so the check is two integers instead of an O(n)
  // walk of every coupon (192 us of a 320 us rebind on the 8x26 chain). The stamp is resolution-insensitive,
  // so a client's unresolved document still matches a session carrying resolved fixing schedules. A stale or
  // mismatched stamp throws exactly as a structural difference does; the instrument-count and per-row quote
  // validation are unchanged, so a wrong-sized or non-finite payload is still refused before anything moves.
  const cal::CalibrationResult& rebind(const cal::StampedBundle& b, const RegSpec& reg = {});
  // This session's STRUCTURAL stamp (cal::structural_stamp) -- the value a caller's stamped bundle must
  // carry to rebind. Unlike structure_fingerprint() it does not change when fixings resolve.
  std::uint64_t structural_stamp() const { return stamp_; }

  const cal::CalibrationResult& result() const { return result_; }
  const Eigen::VectorXd& x() const { return x_; }
  const cal::BundleProblem& problem() const { return prob_; }
  // An IDENTITY STAMP of the document this session was compiled from (structure_fingerprint.hpp), for a
  // client's own caching. NOT a gate: nothing in the engine compares it (2026-09-10) -- a structural edit
  // is a new session by construction, and rebind checks structure by an O(n) equality, not a hash.
  std::uint64_t structure_fingerprint() const { return fingerprint_; }
  // True iff `p` is STRUCTURALLY EQUAL to this session's problem -- same curves, regions, knots,
  // instrument kinds, legs, cashflow times and schedules -- differing at most in market targets and
  // bands (calibration::structure_equal: an O(n) walk, no hashing, no allocation). True for
  // same_structure(problem()) on any session, including one carrying resolved fixing schedules.
  bool same_structure(const cal::BundleProblem& p) const;
  bool has_fx() const { return has_fx_; }
  bool has_modular() const { return has_modular_; }
  // A non-linear region scheme (MonotoneCubic's value-dependent filter) has no constant W, so its bundle
  // can't ride the W-cache — calibration/streaming route through the AAD engine. LINEAR custom regions
  // (Flat/Linear/NaturalCubic/Hermite) are W-cacheable and stream at microseconds like the shipped curve.
  bool has_nonlinear() const { return has_nonlinear_; }
  // A bid/offer BAND makes the calibration a soft least-squares fit (banded instruments sit off-market
  // within their band). This still streams on the fast frozen-Newton path: the streamer drives the
  // BANDED residual (residuals_vs), so it solves the soft least-squares, not an exact reprice. So a band
  // does NOT force recalibrate(). This flag is informational (e.g. to label a soft-calibrated bundle).
  bool has_band() const { return has_band_; }
  // RETIRED 2026-09-12 — always false, kept so the exposed Session property does not change under callers.
  //
  // This was the last of the three whole-bundle vetoes. It claimed a non-linear region scheme leaves "no
  // constant W for ANY curve, so there is no compiled engine to freeze" — true when it was written, false
  // since the router began partitioning per row (1956b24): a mixed bundle now has a real compiled engine
  // for every row inside its curve's linear horizon, and the AAD block carries the rest, refreshing on
  // staleness exactly as it does for FX/MtM, which has always streamed.
  //
  // Measured before removing it, not assumed. Driving the streamer directly over the ladder's own gate ticks
  // (25 bp moves, 0.1 bp ticks and band-edge crossings, 18 ticks each): mixed_scheme reprices to 6.1e-13 and
  // desk_mixed sits within 5.4e-18 of a cold LM's objective, zero failed ticks, no Hyman branch-switch
  // pathology. Even the degenerate case — a curve MonotoneCubic from its first knot, so NO row compiles and
  // n_times() == 0 — streams every tick to 1.8e-11. See ShapeLadder.EveryRungConvergesOnTheGateTicks and
  // StreamingMixed.* ; has_nonlinear() still reports the scheme honestly, which is what callers should read.
  bool needs_recalibrate() const { return false; }

  // Query the built curves at the current x on a shared time grid.
  std::vector<CurveSample> sample(const std::vector<double>& times) const;

  // Price an ARBITRARY instrument off the current curves: its model quote (par rate / par spread /
  // future rate / FX forward) and the residual against its own stored `market`.
  double model_quote(const cal::Instrument& ins) const;
  double residual(const cal::Instrument& ins) const;

  // Per-instrument calibration diagnostics, so a SOFT (banded) fit is never silent: for each residual
  // instrument, its model quote vs target, and — when it carries a bid/offer band — whether the model
  // landed INSIDE the band and the residual's slope there (band_slope: band_decay inside, 1 outside).
  // A hard pin reports in_band=false, weight=1, residual≈0. JSON array, one entry per
  // instrument in residual order:
  //   [{"model","target","residual","soft","in_band","weight"[,"lower","upper","decay"]}, ...]
  boost::json::array quote_diagnostics() const;   // the document itself (E6.3: callers stopped re-parsing it)
  std::string quote_diagnostics_json() const;     // == serialize(quote_diagnostics())

  // The calibration Jacobian J = dq/dx (n_residuals x n_knots): ROWS are calibration instruments, COLUMNS
  // are the fitted knot forwards, so J(i,j) = ∂(model quote of instrument i)/∂x_j. This is the SAME J that
  // risk_operator() builds M from (risk_operator() calls this, so the two can never desync). One forward-AAD
  // pass over the stacked residual (residual = q − market for the linear ParRate/ParSpread/Rate instruments,
  // so d(residual)/dx == dq/dx). A Python layer transforms a risk ladder from bundle A to bundle B via
  // T = J_A · M_B (n_res_A x n_res_B), delta_B = delta_A · T. `reg` is accepted for signature symmetry with
  // risk_operator() but does not affect J (a regulariser changes M through RᵀR, never the quote Jacobian).
  Eigen::MatrixXd jacobian(const RegSpec& reg = {}) const;

  // The analytic risk operator M = dx/dq (n_knots x n_residuals) by the implicit-function theorem on the
  // (regularised) least-squares condition: M = pinv([J; R]) restricted to the J rows, times
  // D = diag(−∂r/∂q). Two things the naive (JᵀJ + RᵀR)⁻¹ Jᵀ got wrong: (1) D -- a residual is NOT always
  // q_model − q: a banded row is w(q_model)·(q_model − q) (−∂r/∂q = w, i.e. `decay` inside the band) and
  // an FX forward is (ln F − ln q)/T (−∂r/∂q = 1/(q·T)); without D those columns were overstated by 1/decay
  // and by q·T. (2) The pseudo-inverse is a rank-thresholded COD, so a rank-deficient bundle gives the
  // min-norm operator on the identified directions and 0 along the null space -- a tolerance-free LDLT of a
  // singular JᵀJ returned garbage in EVERY column, pinned knots included. Left-multiply a portfolio's
  // d(NPV)/dx (one AAD pass) by M for a full analytic delta ladder, no bumping (CLAUDE.md #4).
  Eigen::MatrixXd risk_operator(const RegSpec& reg = {}) const;
  // D above: −∂r_i/∂q_i per instrument at the calibrated x (1 for a hard pin).
  Eigen::VectorXd residual_market_scale() const;

  // ---- batched portfolio reprice (the web "reprice N random swaps" feature) -----------------------
  // Reprice a full multi-curve + xccy book off the CURRENTLY CALIBRATED curves (the current x) and report
  // {npv, pv01, price_us, n}. The NPV is a pure double pass through the pricing kernel (float_leg_pv /
  // annuity / xccy_mtm_leg_pv), timed by a steady_clock around that pass ONLY — the engine stamps the
  // pricing time exactly as calibrate() stamps last_solve_us(), with no Python marshalling inside price_us.
  // PV01 is d(NPV) for a +1bp parallel shift of every fitted knot forward, from ONE forward-AAD pass (no
  // bump-and-reprice). Also cached in last_price_us(). The book references curves by their bundle index.
  PortfolioReprice price_portfolio(const swaps::portfolio::MultiCurveBook& book) const;
  // Convenience for a language binding: parse a book JSON document (schema on book_from_json) and reprice.
  PortfolioReprice price_portfolio_json(const std::string& book_json) const;

  // ---- CACHED (warm/streaming) portfolio reprice --------------------------------------------------
  // Bind a book for REPEATED repricing: build & cache a portfolio::CompiledMultiCurveBook — the multi-curve
  // W-cache twin of price_portfolio's templated kernel — ONCE, so every reprice_bound() reuses it. This is
  // the AMORTIZED path for a live book repriced every streaming tick against the recalibrating curve: the
  // one-time W build (DF = exp(-W_all x) compiled once from the bundle's curve structures) pays for itself
  // across ticks, exactly as the session already amortizes engine_/reg_R_ and VolSurface amortizes a fixed
  // cell set. The book is COPIED. The cache is dropped on any STRUCTURAL bundle change (invalidate_engine)
  // and rebuilt lazily on the next reprice_bound(). NB: the one-shot price_portfolio() deliberately stays
  // on the templated path — building the W-cache does NOT pay off on a single cold reprice (that was an
  // earlier net regression), so the compiled twin is reserved for THIS cached path.
  void bind_portfolio(const swaps::portfolio::MultiCurveBook& book);
  // Reprice the CURRENTLY BOUND book off the current calibrated x through the cached compiled kernel and
  // report {npv, pv01, price_us, n}. NPV rides the compiled W-cache (with the non-cacheable minority — Xccy
  // / compounded / moment — on the templated fallback, the hybrid split); PV01 is the +1bp parallel-knot-
  // shift directional derivative (ANALYTIC on the compiled half, one AAD pass on any fallback). price_us
  // times the NPV pass ONLY, exactly as price_portfolio does (also cached in last_price_us()). Allocation-
  // free on an all-compilable book — the streaming hot path. Throws if no book is bound.
  PortfolioReprice reprice_bound() const;
  // True once bind_portfolio() has been called (the compiled twin may be rebuilt lazily, but the bound book
  // is retained until the next bind_portfolio()).
  bool has_bound_portfolio() const { return static_cast<bool>(bound_book_); }

  // ---- batched swaption VOL CUBE reprice off the calibrated curve (the options hot path) ----------
  // Price a whole expiry x tenor x strike surface off the CURRENTLY CALIBRATED curve in one pass. The spec
  // JSON is {value_date, currency?, index?, curve?, cells:[{expiry, tenor, sabr?{alpha,rho,nu} | normal_vol?,
  // strikes?:[abs...], moneyness_bp?:[offsets...], atm?:bool, payer?:bool}]}. Curve-dependent work (each
  // cell's forward/annuity) is done once from a SINGLE sample() over the union of all schedule times; every
  // strike is then a pure Bachelier/SABR eval. Strikes default to OTM (payer above the forward, receiver
  // below) unless `payer` is set. Reuses the calibrated/streaming session, so a live vol surface reprices with
  // no recalibration — the options analogue of price_portfolio. Result is flat SoA + engine-stamped price_us.
  // `price_vol_cube` is the NATIVE entry point (typed VolCubeSpec, no JSON — what benchmarks and native clients
  // call); `price_vol_cube_json` is the thin parse-then-call wrapper for the run_json / pybind seam.
  VolCube price_vol_cube(const VolCubeSpec& spec) const;
  VolCube price_vol_cube_json(const std::string& spec_json) const;

  // Express a book's risk in THIS bundle's calibration instruments: one forward-AAD pass gives the NPV and
  // its state gradient curve_grad = dP/dx, then ladder = curve_grad^T · M (M = risk_operator(reg)) is the
  // delta per calibration quote — ladder[i] = Σⱼ curve_grad[j]·M[j,i]. The AAD reprice + the M multiply are
  // ENGINE-timed into risk_us (also last_risk_us()); the M FORMATION (the calibration Jacobian solve, book-
  // independent) is outside the clock, mirroring how price_portfolio times only the book-dependent pass.
  // `reg` regularises the RISK OPERATOR (independently of the calibration reg): with a curvature/tension
  // penalty it damps the alternating-sign "fan-out" of a delta ladder over collinear instruments into a
  // localized key-rate hedge. Because R annihilates constant+linear forward moves (regularize.hpp), the
  // TOTAL (parallel) DV01 is preserved exactly — only the ladder's SHAPE is stabilised. Default {} = raw M.
  PortfolioRisk price_portfolio_risk(const swaps::portfolio::MultiCurveBook& book, const RegSpec& reg = {}) const;
  // Convenience for a language binding: parse a book JSON document (same schema as book_from_json) and risk it.
  PortfolioRisk price_portfolio_risk_json(const std::string& book_json, const RegSpec& reg = {}) const;

  // ---- cross-bundle risk transform (the "risk reprojection" / adaptor Jacobian) --------------------
  // Express one bundle's risk in ANOTHER's instruments -- e.g. remap a 23-knot build's ladder onto a
  // 10-tenor reporting grid. The invariant is the CURVE; instruments are coordinates on it. Analytic, one
  // forward-AAD pass -- NO bumping, NO re-calibration loop (this replaces the Python finite-difference adaptor).
  //
  // cross_jacobian(source) = dq_source/dx_this  (n_res_source x n_knots_this): each source instrument's model
  // quote priced on THIS bundle's curve, differentiated wrt this bundle's knots. REQUIRES the two bundles to
  // share a curve SET (same currencies / outright-or-spread, same order) so source's curve-index references
  // are valid here -- checked by same_curve_set(); a mismatch throws.
  Eigen::MatrixXd cross_jacobian(const cal::BundleProblem& source) const;
  // T = cross_jacobian(source) · risk_operator(reg)  (n_res_source x n_res_this): a ladder in `source`'s
  // instruments maps to THIS bundle's by delta_this = delta_source · T. Matched states -> the exact
  // J_source · M_this; a finer `source` -> the least-squares projection onto this bundle's pillars (total
  // DV01 preserved, reg's null space excludes level shifts). Book-INDEPENDENT: build once, apply by matvec.
  Eigen::MatrixXd transform_matrix(const cal::BundleProblem& source, const RegSpec& reg = {}) const;
  // JSON conveniences for a language binding: parse a source bundle document (same schema as the ctor).
  Eigen::MatrixXd cross_jacobian_json(const std::string& source_bundle_json) const;
  Eigen::MatrixXd transform_matrix_json(const std::string& source_bundle_json, const RegSpec& reg = {}) const;
  // True iff `source` shares this bundle's curve set (count + per-curve currency + outright/spread), so a
  // book/ladder in one is meaningful in the other. transform_matrix()/cross_jacobian() throw when false.
  bool same_curve_set(const cal::BundleProblem& source) const;

  // ---- streaming (any bundle with a constant W: hard, banded, portfolio, or mixed FX/MtM) ---------
  // Anchor a StreamingCalibrator at the current x; each stream_update(q) re-solves to the exact curve
  // for the new market q (frozen-Newton off the cached Jacobian, refreshed only on staleness). A mixed
  // FX/MtM bundle streams on the hybrid engine (its FX rows refresh their AAD Jacobian only on staleness).
  // Throws only for a bundle with no constant W at all (a MonotoneCubic region scheme) — recalibrate those.
  // step_tol > 0 overrides the frozen-Newton convergence tolerance (||dx||_inf); 0 keeps the exact default
  // (machine-precision reprice each tick). A looser tol stops in fewer corrector steps — a speed/accuracy
  // knob for a live viewer — while still refreshing the Jacobian on staleness (so it stays robust).
  void start_streaming(const RegSpec& reg = {}, double step_tol = 0.0);
  const Eigen::VectorXd& stream_update(const Eigen::VectorXd& new_market);
  // A FOUR-NUMBER requote tick (K5', 2026-09-14): every row's target AND band. Every quote is validated first (a target outside
  // its band, an inverted band or a decay outside [0, 1] is refused and nothing changes); moved bands land on the shared engine
  // and on the streamer in place -- a band that only moves is a row re-scale on the tick, not a Jacobian refresh. A change in
  // WHICH rows are banded re-anchors the streamer (one Jacobian).
  const Eigen::VectorXd& stream_update(const Eigen::VectorXd& target, const Eigen::VectorXd& lower,
                                      const Eigen::VectorXd& upper, const Eigen::VectorXd& decay);
  // True once start_streaming() (or a resolve/recalibrate/rebind, which stream) has armed the session. A
  // fixings / evaluation-date change recompiles the engine and rebuilds the streamer lazily on the next
  // tick (anchored at the current x), so the flag stays true across it.
  bool streaming() const { return stream_armed_; }

  // ---- solve-time telemetry (measured by default; a steady_clock pair is ~100ns, <0.1% of a solve) --
  // Every calibrate/recalibrate/stream_update stamps the ENGINE-measured wall time of the solve itself
  // (no marshalling). Callers should report these instead of timing across a language boundary.
  double last_solve_us() const { return last_solve_us_; }        // most recent solve, any path (µs)
  double last_price_us() const { return last_price_us_; }        // most recent price_portfolio pricing pass (µs)
  double last_risk_us() const { return last_risk_us_; }          // most recent price_portfolio_risk pass (µs)
  // Running mean of stream_update solve times since the last start_streaming(); 0 before the first tick.
  double stream_avg_us() const { return stream_ticks_ ? stream_sum_us_ / stream_ticks_ : 0.0; }
  long stream_ticks() const { return stream_ticks_; }            // stream_update calls since start_streaming
  // Frozen-Newton introspection for the LAST stream_update (all 0 on the pure fast path): Gauss-Newton
  // steps taken, analytic-Jacobian refreshes triggered, and the market drift that provoked them.
  int last_newton_steps() const { return last_newton_steps_; }
  int last_refreshes() const { return last_refreshes_; }
  double last_drift() const { return last_drift_; }
  // Health of the LAST tick: did the frozen-Newton corrector reach step_tol? A tick that hit the refresh
  // cap is reported here and NOT committed (x() keeps the last converged curve). `last_rescales` counts
  // band-edge crossings handled by the cheap frozen-row re-scale (no Jacobian recompute) this tick.
  bool last_converged() const { return last_converged_; }
  int last_rescales() const { return last_rescales_; }
  // WHY the last tick ended: calibration::StreamStatus as an int (0 converged, 1 step cap, 2 refresh cap,
  // 3 band re-scale budget, 4 non-finite, 5 diverged) and its text. A failed tick is never committed and
  // the streamer's anchor is restored to the last committed curve, so the NEXT tick cannot report a
  // stale curve as converged (the 2026-09-09 C1 finding).
  int last_status() const { return last_status_; }
  const char* last_reason() const { return last_reason_; }

  // ---- fixings as pricing context (E2) -----------------------------------------------------------
  // Any observation carrying a fixing_schedule is RESOLVED from this session's fixing table against the
  // evaluation date: past days -> `realized` (throwing/flagging MissingFixing if absent), future days ->
  // forecast sub-periods. Resolution runs on construction and on every set_evaluation_date/set_fixings.
  // It rewrites `realized`/subs in prob_ -- STRUCTURE for the compiled tables (registered times, baked
  // constants) -- so the engine, the streamer and the bound book's compiled twin are all dropped and
  // rebuilt on their next use (the streamer re-anchors at the current x; the book re-resolves its seasoned
  // coupons). Until 2026-09-10 the streamer and the bound book kept the stale copies (E3-D2/D3). Making a
  // fixing a scalar update of a compiled row (daily boundaries registered at compile) is E4.A.3.

  // Set the evaluation date (integer serial the caller defines); fixings strictly before it are fixed.
  void set_evaluation_date(int serial) { pricing::check_date_serial(serial, "set_evaluation_date"); eval_date_ = serial; resolve_fixings(); }
  // Upsert fixings for one index and re-resolve every schedule-carrying observation in place. Returns the
  // number of observations still un-priceable (a required past fixing is missing) after the update.
  int set_fixings(const std::string& index, const std::vector<std::pair<int, double>>& rows) {
    fixings_.bulk_set(index, rows);  // notifies (unused here; the session drives resolution directly)
    return resolve_fixings();
  }
  int evaluation_date() const { return eval_date_; }
  int n_unresolved() const { return n_unresolved_; }
  const swaps::pricing::FixingTable& fixings() const { return fixings_; }

 private:
  // A BOOK's fixings-resolvable coupons (a seasoned trade's accruing period, or a "positions" row that
  // carries a fixing_schedule) are resolved against THIS session's evaluation date + fixing table before
  // pricing. Returns a resolved copy, or nullopt when nothing in the book carries a schedule (no copy).
  // Throws (with the index/date) when a required past fixing is missing -- never prices it as zero.
  std::optional<swaps::portfolio::MultiCurveBook> resolve_book(const swaps::portfolio::MultiCurveBook& book) const;
  // Collect pointers to every schedule-carrying observation in prob_ (futures obs + float-leg coupons,
  // recursing into portfolio components), and resolve them against {eval_date_, fixings_}. Returns the
  // count that could not resolve (missing a past fixing). Pointers are stable: prob_ is not resized.
  int resolve_fixings();

  // ---- the cached warm engine (fingerprint-keyed W-cache reuse) ----------------------------------
  // The hybrid residual engine is compiled ONCE per problem STRUCTURE and reused across every solve:
  // recalibrate()/rebind() overwrite only the quote RHS (engine_->set_quotes) and re-solve warm on the
  // same engine -- no W rebuild, no batch re-registration, no MtM re-guard. The tension pseudo-residual
  // block R is likewise structure-only (given fixed reg params), so it is cached keyed on (lambda, sigma,
  // curves) and recomputed only when those change. invalidate_engine() drops the engine, R, the book's
  // compiled twin AND the streamer (it borrows the engine) whenever prob_ mutates STRUCTURALLY -- today
  // that is fixings resolution (it rewrites observation times/realized in place). The streamer is rebuilt
  // lazily by the next stream_update/resolve when the session is armed. Mutable + const ensure: the engine
  // is a pure cache of prob_'s structure, so const queries (jacobian) may build it lazily.
  void invalidate_engine() { engine_.reset(); reg_R_valid_ = false; cbook_.reset(); resolved_book_.reset(); stream_.reset(); }
  // The warm re-solve behind resolve/recalibrate/rebind (one streamed tick, LM fallback).
  const cal::CalibrationResult& warm_solve(const RegSpec& reg);
  void record_tick(const cal::StreamTick& tick, double solve_us);
  void rebuild_cbook() const;
  cal::HybridBundleResidual& ensure_engine() const {
    if (!engine_) engine_ = std::make_unique<cal::HybridBundleResidual>(prob_);
    return *engine_;
  }
  const Eigen::MatrixXd& ensure_reg_R(const RegSpec& reg) const;  // cached tension block (structure-only)

  cal::BundleProblem prob_;
  std::uint64_t fingerprint_ = 0;  // structure hash at compile time (the warm-vs-recompile switch)
  Eigen::VectorXd x_;
  Eigen::VectorXd parallel_dir_;  // pricing::parallel_direction(prob_.curves): the PV01 direction, fixed per structure
  cal::CalibrationResult result_;
  bool has_fx_ = false;
  bool has_modular_ = false;
  bool has_nonlinear_ = false;
  bool has_band_ = false;
  std::unique_ptr<cal::StreamingCalibrator<cal::BundleProblem>> stream_;
  // The cached warm/streaming reprice twin (bind_portfolio/reprice_bound). bound_book_ is the copied book;
  // cbook_ is its compiled W-cache twin, built once on bind and rebuilt lazily after invalidate_engine()
  // drops it (a structural bundle change moves W). Mutable so the const reprice_bound() can rebuild lazily.
  mutable std::unique_ptr<swaps::portfolio::MultiCurveBook> bound_book_;     // the UNRESOLVED book as bound
  mutable std::unique_ptr<swaps::portfolio::MultiCurveBook> resolved_book_;  // its fixings-resolved copy (the twin's source)
  mutable std::unique_ptr<swaps::portfolio::CompiledMultiCurveBook> cbook_;
  // Streaming session state (the streamer itself may be dropped by invalidate_engine and rebuilt lazily).
  RegSpec stream_reg_;
  double stream_step_tol_ = 0.0;
  bool stream_armed_ = false;
  bool bands_changed_ = false;  // a set_band/rebind changed a band since the streamer last anchored
  Eigen::VectorXd q_scratch_;   // the live market gathered from prob_ (reused; no per-solve allocation)
  // The cached hybrid engine + tension-block cache (see ensure_engine/ensure_reg_R above).
  std::uint64_t stamp_ = 0;  // E8: the structural stamp of the compiled document (resolution-insensitive)
  // The quote-RHS half of a rebind (targets + bands + resolve), shared by both rebind overloads once the
  // caller's structural check -- O(n) equality or the stamp -- has passed.
  const cal::CalibrationResult& rebind_quotes(const cal::BundleProblem& p, const RegSpec& reg);
  mutable std::unique_ptr<cal::HybridBundleResidual> engine_;
  mutable Eigen::MatrixXd reg_R_;
  mutable bool reg_R_valid_ = false;
  mutable Eigen::MatrixXd reg_second_diff_;  // risk_operator's legacy second-difference block (uncached, rare)
  mutable double reg_R_lambda_ = 0, reg_R_sigma_ = 0;
  mutable std::vector<int> reg_R_curves_;
  swaps::pricing::FixingTable fixings_;
  int eval_date_ = 0;
  int n_unresolved_ = 0;

  // Solve-time telemetry (stamped by calibrate/recalibrate/stream_update; see the getters above).
  double last_solve_us_ = 0;
  // Pricing-time telemetry, stamped by the (const) price_portfolio; mutable so the query stays const.
  mutable double last_price_us_ = 0;
  // Risk-time telemetry, stamped by the (const) price_portfolio_risk; mutable so the query stays const.
  mutable double last_risk_us_ = 0;
  double stream_sum_us_ = 0;   // sum of stream_update solve times since start_streaming()
  long stream_ticks_ = 0;      // count of those ticks
  int last_newton_steps_ = 0;
  int last_refreshes_ = 0;
  double last_drift_ = 0;
  bool last_converged_ = true;
  int last_rescales_ = 0;
  int last_status_ = 0;
  const char* last_reason_ = "converged";

  // ---- vol-cube reprice caches (populated by the const price_vol_cube_json; mutable so it stays const) ----
  // The swaption schedules are CURVE-INDEPENDENT, so they are built once per cell and reused across reprices
  // (killing the per-call calendar walk). The per-cell (forward, annuity) depend only on the curve state x,
  // so they are cached against the x they were computed at and reused while x is unchanged (a vol-only reprice
  // — e.g. a SABR-slider tick — then needs NO curve sample, just the Bachelier pass). Cleared when x moves.
  mutable std::unordered_map<std::string, SwaptionSchedule> vol_sched_cache_;
  mutable std::unordered_map<std::string, std::pair<double, double>> vol_fa_cache_;  // cell key -> (fwd, annuity)
  mutable Eigen::VectorXd vol_fa_x_;  // the x for which vol_fa_cache_ is valid (empty => invalid)
};

// A COMPILED vol surface: resolve a FIXED set of swaption cells' schedules ONCE (the calendar walk), pre-index
// them into a single sample grid, and pre-size the flat SoA output — then `reprice()` off the current curve
// with zero per-call schedule/key/allocation work: one curve sample (only when x moved), then a pure
// Bachelier/SABR pass writing straight into reused, index-addressed buffers. The stateful streaming analog of
// BundleSession::price_vol_cube for a live surface / a fixed swaption book repriced every tick — this is what
// beats QuantLib's native loop (price_vol_cube trades some of that away for a fresh spec on every call). The
// referenced session must outlive the surface. Not thread-safe (reuses one output buffer). To move the vols
// (a SABR-slider tick) mutate `cells()` in place and call `reprice()` — no schedule rebuild, no resample.
class VolSurface {
 public:
  explicit VolSurface(const VolCubeSpec& spec);   // resolve schedules + pre-index + pre-size (curve-independent)
  const VolCube& reprice(const BundleSession& sess) const;  // sample iff x moved, then price into reused VolCube
  // Overwrite every cell's SABR (alpha per cell, rho/nu shared) — the streaming update for a live surface (a
  // level/skew slider) without touching the schedules. alpha.size() must equal n_cells.
  void set_sabr(const std::vector<double>& alpha, double rho, double nu);
  std::vector<VolCubeCell>& cells() { return defs_; }        // mutate vols/strikes in place, then reprice()
  const std::vector<VolCubeCell>& cells() const { return defs_; }
  int n_cells() const { return static_cast<int>(defs_.size()); }
  int n_points() const { return n_points_; }

 private:
  struct Cell {
    double t_expiry = 0.0;
    std::vector<double> tau;
    std::size_t start_idx = 0;         // index into union_times_ / the sampled discount vector
    std::vector<std::size_t> pay_idx;
    int point_offset = 0;              // first output-point index for this cell
  };
  int curve_ = 0;
  int n_points_ = 0;
  std::vector<Cell> cells_;                   // resolved schedules (built once)
  std::vector<VolCubeCell> defs_;             // per-cell vol model + strike specs (mutable between reprices)
  std::vector<double> union_times_;           // the one sample grid (sorted, unique)
  mutable std::vector<double> fwd_, annuity_; // per-cell curve-dependent, refreshed when x moves
  mutable Eigen::VectorXd fa_x_;              // the x fwd_/annuity_ were computed at
  mutable VolCube out_;                       // pre-sized, reused across reprices
};

// Parse a `vol_cube` JSON document into the native VolCubeSpec (used by the JSON verb and to build a VolSurface
// from a spec string in the binding). Definition in api/options.cpp.
VolCubeSpec vol_cube_spec_from_json(const std::string& spec_json);

// One-shot stateless JSON dispatcher for a web call. Request:
//   { "bundle": {...},                         (required) the BundleProblem object graph
//     "x0": [...],                             (optional) start; else a flat guess
//     "regularize": {"lambda":1.0,"curves":[3,4],   (optional) smoothness penalty on those curves;
//                     "tension":true,"sigma":0.0},   add tension:true for the continuous tension energy
//                                                     (sigma=0 curvature, sigma>0 taut) vs 2nd-difference
//     "sample_times": [...],                   (optional) grid to sample every curve on
//     "price": [ {instrument}, ... ],          (optional) instruments to price off the solved curves
//     "risk": true }                           (optional) include the dx/dq operator
// Response mirrors it: { "calibration": {...}, "x": [...], "curves": [...], "priced": [...],
//                        "risk_operator": [[...]] }. On error: { "error": "..." }.
std::string run_json(const std::string& request);
// The PARSE-ONCE entry (E6.3): a host that already holds a parsed document calls this; the string overload
// parses and forwards. Every stateless verb has the same pair (api/api_surface.py STATELESS_VERBS).
std::string run_json(const boost::json::object& request);

}  // namespace swaps::api
