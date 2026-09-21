// Implementation of the bundle API facade (include/swaps/api/bundle_api.hpp).
//
// Boost.JSON is compiled ONCE here via <boost/json/src.hpp> so no other TU pays for it. Everything else
// is the thin glue between the JSON object graph and the engine's generic instrument model.

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/json_util.hpp"  // darr / vecf
#include "swaps/calibration/diagnostics.hpp"  // quote_diagnostics

#include "swaps/trade/csa.hpp"    // CSA -> discount index (the typed-trade book entry)
#include "swaps/trade/trade.hpp"  // Trade::vanilla_swap / to_position

#include <chrono>
#include <map>
#include <stdexcept>
#include <utility>

#include <boost/json/src.hpp>  // header-only Boost.JSON, compiled in this TU only

#include "swaps/api/compile.hpp"                   // compile_spec / compile_to_json (the 'compile' verb)
#include "swaps/api/generate_risk.hpp"             // generate_risk_json (the 'generate_risk' verb)
#include "swaps/api/options.hpp"                   // swaption_json (the 'swaption' verb)
#include "swaps/api/bond.hpp"                       // bonds_json (the 'bonds' verb)
#include "swaps/api/rv.hpp"                         // bond_universe / govvie_fit / swap_spread verbs
#include "swaps/api/exposure.hpp"                   // exposure_json (the 'exposure' verb)
#include "swaps/api/scenario.hpp"                   // scenario_json (the 'scenario' verb)
#include "swaps/api/pnl.hpp"                        // pnl_json (the 'pnl' explain verb)
#include "swaps/api/vega.hpp"                       // vega_json (the 'vega' ladder verb)
#include "swaps/api/inflation.hpp"                   // inflation_json (the 'inflation' ZCIS/YoY verb)
#include "swaps/api/fx_option.hpp"                    // fx_option_json (the 'fx_option'/'fx_vol' verb)
#include "swaps/api/bond_future.hpp"                  // bond_future_json (the 'bond_future' CTD verb)
#include "swaps/api/conventions.hpp"
#include "swaps/api/ndf.hpp"                          // ndf_json (the 'ndf' non-deliverable FX verb)
#include "swaps/api/calib_report.hpp"                 // calib_report_json (calibration diagnostics verb)
#include "swaps/api/credit.hpp"                       // credit_json (the 'credit' hazard-curve/CDS verb)
#include "swaps/api/scenario_grid.hpp"                // scenario_grid_json (the 'scenario_grid' P&L surface)
#include "swaps/api/var.hpp"                          // var_json (the 'var' full-reval VaR/ES verb)
#include "swaps/calibration/compiled_bundle.hpp"  // CompiledBundleResidual::model_rates (streaming anchor)
#include <type_traits>
#include "swaps/calibration/jacobian.hpp"         // aad_jacobian (risk operator)
#include "swaps/calibration/regularize.hpp"       // second_difference_operator(), tension_energy_operator()
#include "swaps/calibration/structure_fingerprint.hpp"  // structure_fingerprint (warm-vs-recompile switch)

namespace swaps::api {

namespace json = boost::json;
namespace px = swaps::pricing;
namespace curve = swaps::curve;
namespace ad = swaps::ad;  // Dual (forward-AAD scalar) for the PV01 pass in price_portfolio

// =================================================================================================
// The JSON <-> object-graph codecs live in api/codec.cpp (include/swaps/api/codec.hpp, E7 stage 2).
// What stays here is the session's own emit helpers.
// =================================================================================================
namespace {
json::array da(const std::vector<double>& v) { return vecf(v); }
json::array da(const Eigen::VectorXd& v) { return vecf(v); }
std::string err(const std::string& msg) {
  json::object o;
  o["error"] = msg;
  return json::serialize(json::value(std::move(o)));
}
}  // namespace

namespace pf = swaps::portfolio;

// flat_x0 lives in calibration/bundle_problem.hpp (E7 3.6); bundle_api.hpp re-exports it.

// =================================================================================================
// BundleSession
// =================================================================================================
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
      px::resolve_into(*o, ctx);  // rewrites realized/subs in place; no recompile of prob_ itself
    } catch (const px::MissingFixing&) {
      ++missing;  // leave the observation as-is; the instrument is un-priceable until the fixing arrives
    }
  }
  n_unresolved_ = missing;
  // Resolution rewrites observation sub-periods/realized constants -- STRUCTURE as far as the compiled
  // engine is concerned (registered times / batch constants). Drop the cached engine; the next solve
  // rebuilds it against the resolved problem. (No-op when nothing carries a schedule: obs is empty.)
  if (!obs.empty()) invalidate_engine();
  return missing;
}

BundleSession::BundleSession(cal::BundleProblem prob) : prob_(std::move(prob)) {
  cal::validate_problem(prob_, "BundleSession");  // E1/E2/B12: a malformed bundle is an error, not a crash/NaN
  fingerprint_ = cal::structure_fingerprint(prob_);  // an identity stamp of the compiled document (not a gate)
  stamp_ = cal::structural_stamp(prob_);  // E8: computed ONCE here; a stamped rebind compares against it
  parallel_dir_ = swaps::pricing::parallel_direction(prob_.curves);
  for (const auto& ins : prob_.instruments) {
    // `has_fx_` reports the ENGINE's partition (E6.1c: the API used to keep a second, disagreeing definition
    // that called every FX forward / MtM row non-cacheable; standalone ones ride the W-cache since 2026-09-09).
    if (cal::instrument_is_noncacheable(ins, prob_.curves)) has_fx_ = true;
    if (ins.band_upper > ins.band_lower)
      has_band_ = true;  // soft target: compiled COLD calibrate is fine, streams frozen-Newton (soft LS)
  }
  for (const auto& c : prob_.curves) {
    if (!c.regions.empty()) has_modular_ = true;  // custom interpolation regions
    for (const auto& r : c.regions)        // only a NON-LINEAR scheme forces off the W-cache
      if (!curve::scheme_is_linear(r.scheme)) has_nonlinear_ = true;  // the one definition (curve_module.hpp)
  }
  x_ = Eigen::VectorXd::Zero(prob_.n_knots());
  resolve_fixings();  // resolve any schedule-carrying observations (no-op when none carry a schedule)
}

// The cached tension pseudo-residual block R (= sqrt(mu)*L, regularize.hpp §5). R depends only on the
// bundle's STRUCTURE and the reg parameters, never on quotes or x -- so it is built once per (lambda,
// sigma, curves) and reused across every warm re-solve. invalidate_engine() drops it with the engine.
const Eigen::MatrixXd& BundleSession::ensure_reg_R(const RegSpec& reg) const {
  if (!reg_R_valid_ || reg_R_lambda_ != reg.lambda || reg_R_sigma_ != reg.sigma ||
      reg_R_curves_ != reg.curves) {
    reg_R_ = cal::tension_energy_operator(prob_, reg.lambda, reg.sigma, reg.curves);
    reg_R_lambda_ = reg.lambda;
    reg_R_sigma_ = reg.sigma;
    reg_R_curves_ = reg.curves;
    reg_R_valid_ = true;
  }
  return reg_R_;
}

const cal::CalibrationResult& BundleSession::calibrate(const Eigen::VectorXd& x0, const RegSpec& reg) {
  const auto t0 = std::chrono::steady_clock::now();
  if (reg.on() && !reg.tension) {
    // Second-difference smoothing (E6.1c, 2026-09-10): the SAME engine composition as the tension path
    // below, with the discrete curvature operator as the constant R block -- the SmoothedProblem wrapper that
    // paid an AAD sweep per LM iteration for this constant block is gone. THE SHIPPED DEFAULT since 2026-09-21
    // (tension was, 2026-09-13 .. 2026-09-21; see regularize.hpp smoothing_preset).
    const cal::HybridBundleResidual& eng = ensure_engine();
    const Eigen::MatrixXd R = cal::second_difference_operator(prob_, reg.lambda, reg.curves);
    const cal::RegularizedEngine<cal::HybridBundleResidual> composed(eng, R);
    result_ = cal::calibrate_with(composed, prob_.n_knots(), prob_.n_residuals() + static_cast<int>(R.rows()), x0);
  } else {
    // EVERY other path drives the ONE cached hybrid engine (compiled W-cache rows + width-reduced AAD
    // rows; for a non-linear scheme the hybrid routes every row to the AAD block, so it subsumes the old
    // pre-E4.A AAD-only escape hatch). The engine is built once per structure and reused across warm re-solves
    // -- construction (W build, batch registration, MtM guard) is no longer paid per calibrate call.
    const cal::HybridBundleResidual& eng = ensure_engine();
    if (reg.on() && reg.tension) {
      // Tension-energy penalty as an ENGINE composition: instrument rows keep the compiled/analytic
      // residual + Jacobian, the constant R block costs a GEMV -- the regularised solve now rides the
      // W-cache instead of falling to a per-iteration AAD sweep over the wrapper problem.
      const Eigen::MatrixXd& R = ensure_reg_R(reg);
      const cal::RegularizedEngine<cal::HybridBundleResidual> composed(eng, R);
      result_ = cal::calibrate_with(composed, prob_.n_knots(),
                                    prob_.n_residuals() + static_cast<int>(R.rows()), x0);
    } else {
      result_ = cal::calibrate_with(eng, prob_.n_knots(), prob_.n_residuals(), x0);
    }
  }
  last_solve_us_ = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  result_.solve_micros = last_solve_us_;
  x_ = result_.x;
  return result_;
}

void BundleSession::set_market(const Eigen::VectorXd& market) {
  if (market.size() != prob_.n_residuals())
    throw std::runtime_error("set_market: market length does not match the instrument count");
  if (!market.allFinite()) throw std::runtime_error("set_market: market contains a non-finite quote");
  cal::validate_targets(prob_.instruments, market, "set_market");  // K5': a target outside its band, before anything changes
  for (int i = 0; i < prob_.n_residuals(); ++i) prob_.instruments[i].market = market[i];
  if (engine_) engine_->set_market(market);  // scalar row data on the ONE compiled engine; no copy, no recompile
}

void BundleSession::set_band(int row, double lower, double upper, double decay) {
  if (row < 0 || row >= prob_.n_residuals()) throw std::runtime_error("set_band: row out of range");
  if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(decay))
    throw std::runtime_error("set_band: non-finite band");
  cal::Instrument& ins = prob_.instruments[row];
  cal::validate_quote(ins.market, lower, upper, decay, "set_band", row);  // K5': the band must contain the row's target
  ins.band_lower = lower;
  ins.band_upper = upper;
  ins.band_decay = decay;
  has_band_ = false;
  for (const auto& i : prob_.instruments) has_band_ = has_band_ || (i.band_upper > i.band_lower);
  if (engine_) engine_->set_quote(row, ins.market, lower, upper, decay);
  bands_changed_ = true;  // the streamer's active-set table must be re-read before its next tick
}

const cal::CalibrationResult& BundleSession::resolve(const RegSpec& reg) { return warm_solve(reg); }

const cal::CalibrationResult& BundleSession::recalibrate(const Eigen::VectorXd& new_market,
                                                         const RegSpec& reg) {
  if (new_market.size() != prob_.n_residuals())
    throw std::runtime_error("recalibrate: market length does not match the instrument count");
  if (!new_market.allFinite())
    throw std::runtime_error("recalibrate: market contains a non-finite quote");
  set_market(new_market);
  return warm_solve(reg);
}

const cal::CalibrationResult& BundleSession::rebind(const cal::StampedBundle& b, const RegSpec& reg) {
  // The structural check is the stamp (see the header). Everything else a rebind refuses -- a different
  // instrument count, a non-finite or out-of-band quote -- is still refused by the O(n) body below.
  cal::require_stamp_match(b.stamp(), stamp_, "rebind");
  return rebind_quotes(b.problem(), reg);
}

const cal::CalibrationResult& BundleSession::rebind(const cal::BundleProblem& p, const RegSpec& reg) {
  if (p.n_residuals() != prob_.n_residuals())
    throw std::runtime_error("rebind: instrument count differs — the structure changed (compile a new session)");
  // Same COUNT is not the same STRUCTURE: a bundle with the same number of instruments but different
  // knots/tenors/regions/roles would rebind its new quotes onto the old W-cache rows (a 7y quote applied
  // to the old 5y row). An O(n) structural equality (no hash) is the contract.
  if (!same_structure(p))
    throw std::runtime_error("rebind: the structure differs (a knot, scheme, role, schedule or instrument changed) — "
                             "compile a new session; rebind carries market targets and bands only");
  return rebind_quotes(p, reg);
}

// The quote-RHS half of a rebind, shared by both overloads: the structural check is the caller's (an O(n)
// equality, or the stamp). Everything here is per-row quote data -- no structure is read.
const cal::CalibrationResult& BundleSession::rebind_quotes(const cal::BundleProblem& p, const RegSpec& reg) {
  for (int i = 0; i < prob_.n_residuals(); ++i) {
    const cal::Instrument& src = p.instruments[i];
    if (!std::isfinite(src.market) || !std::isfinite(src.band_lower) || !std::isfinite(src.band_upper) || !std::isfinite(src.band_decay))
      throw std::runtime_error("rebind: instrument " + std::to_string(i) + " carries a non-finite quote or band");
    cal::validate_quote(src.market, src.band_lower, src.band_upper, src.band_decay, "rebind", i);  // K5'
  }
  bool bands_changed = false;
  for (int i = 0; i < prob_.n_residuals(); ++i) {  // the FULL quote RHS: target AND soft-quote band
    cal::Instrument& dst = prob_.instruments[i];
    const cal::Instrument& src = p.instruments[i];
    dst.market = src.market;
    if (dst.band_lower != src.band_lower || dst.band_upper != src.band_upper || dst.band_decay != src.band_decay) {
      bands_changed = true;
      dst.band_lower = src.band_lower;
      dst.band_upper = src.band_upper;
      dst.band_decay = src.band_decay;
    }
  }
  if (bands_changed) {
    has_band_ = false;
    for (const auto& i : prob_.instruments) has_band_ = has_band_ || (i.band_upper > i.band_lower);
    bands_changed_ = true;
  }
  if (engine_) engine_->set_quotes(prob_);  // scalar row data: no Instrument copy, no recompile
  return warm_solve(reg);
}

namespace {
bool same_reg(const RegSpec& a, const RegSpec& b) {
  return a.lambda == b.lambda && a.sigma == b.sigma && a.tension == b.tension && a.curves == b.curves;
}
}  // namespace

// The warm re-solve: ONE frozen-Newton tick on the shared engine, seeded from the current x. A session that
// is not streaming yet starts (one Jacobian at x against the live market); a band edit re-anchors the
// streamer's active set; a regulariser change restarts the streamer under the new one. The tick either
// converges (committed; result_ reports "streamed") or -- an intrinsically large move -- falls back to an LM
// warm solve from x, after which the streamer is re-anchored at the LM solution. EVERY bundle takes this
// path since 2026-09-12: a value-dependent region no longer diverts to a cold solve (see needs_recalibrate).
const cal::CalibrationResult& BundleSession::warm_solve(const RegSpec& reg) {
  q_scratch_.resize(prob_.n_residuals());
  for (int i = 0; i < prob_.n_residuals(); ++i) q_scratch_[i] = prob_.instruments[i].market;
  const auto t0 = std::chrono::steady_clock::now();
  if (!stream_ || !same_reg(stream_reg_, reg)) {
    start_streaming(reg, stream_ ? stream_step_tol_ : 0.0);  // anchored at (x_, live market)
  } else if (bands_changed_) {
    stream_->resync(prob_, x_, q_scratch_);
    bands_changed_ = false;
  }
  const cal::StreamTick tick = stream_->update(q_scratch_);
  record_tick(tick, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  if (tick.converged) {
    x_ = stream_->current();
    stamp_streamed(tick, ensure_engine().residuals(x_));  // re-evaluated at x_: the engine's market IS the live market now
    return result_;
  }
  calibrate(x_, reg);  // the LM fallback, warm from x_
  stream_->resync(prob_, x_, q_scratch_);
  return result_;
}

void BundleSession::stamp_streamed(const cal::StreamTick& tick, const Eigen::VectorXd& r) {
  x_ = stream_->current();
  result_.x = x_;
  result_.converged = true;
  result_.status = "streamed (frozen-Newton tick on the shared engine)";
  result_.info = 2;
  result_.iterations = tick.newton_steps;
  result_.rms_residual = std::sqrt(r.squaredNorm() / std::max<Eigen::Index>(1, r.size()));
  result_.stationarity = 0.0;  // not evaluated on the streamed path (the tick converged to step_tol)
  result_.solve_micros = last_solve_us_;
}

void BundleSession::record_tick(const cal::StreamTick& tick, double solve_us) {
  last_solve_us_ = solve_us;
  stream_sum_us_ += solve_us;
  ++stream_ticks_;
  last_newton_steps_ = tick.newton_steps;
  last_refreshes_ = tick.refreshes;
  last_drift_ = tick.drift;
  last_converged_ = tick.converged;
  last_rescales_ = tick.rescales;
  last_status_ = static_cast<int>(tick.status);
  last_reason_ = tick.reason();
}

std::vector<CurveSample> BundleSession::sample(const std::vector<double>& times) const {
  return cal::sample_bundle_curves(prob_, x_, times);
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

json::array BundleSession::quote_diagnostics() const {
  return quote_diagnostics_to_json(cal::quote_diagnostics(prob_, x_));  // the struct is calibration/diagnostics.hpp's
}
std::string BundleSession::quote_diagnostics_json() const { return json::serialize(quote_diagnostics()); }

Eigen::MatrixXd BundleSession::jacobian(const RegSpec& reg) const {
  (void)reg;  // J = dq/dx is independent of any regulariser (reg only enters M through RᵀR); see header.
  // The cached hybrid engine's ANALYTIC Jacobian (W-cache rows analytic, FX/MtM rows width-reduced AAD)
  // -- pin-tested equal to the full AAD sweep, and reuses the session's compiled engine instead of an
  // n_knots-wide AAD pass per call. risk_operator()/transform_matrix() flow through here and inherit it.
  return ensure_engine().jacobian(x_);  // n_res x n_knots: rows = instruments, cols = knots
}

Eigen::VectorXd BundleSession::residual_market_scale() const {
  return cal::residual_market_scale(prob_.instruments);  // the one definition (problem.hpp, E6.1c)
}

Eigen::MatrixXd BundleSession::risk_operator(const RegSpec& reg) const {
  const Eigen::MatrixXd J = jacobian(reg);  // n_res x n_knots (shared with jacobian(), no desync)
  const int m = static_cast<int>(J.rows()), n = static_cast<int>(J.cols());
  // Stack the regulariser rows under J: the IFT on min ||r||² + ||Rx||² reads [J; R]ᵀ[J; R] dx = Jᵀ D dq.
  Eigen::MatrixXd S = J;
  if (reg.on()) {
    // The tension block comes from the session cache (ensure_reg_R: structure-only, built once per reg) --
    // E3-D6: rebuilding it here cost 4,449 allocations per risk_operator call.
    const Eigen::MatrixXd& R = reg.tension ? ensure_reg_R(reg)
                                           : (reg_second_diff_ = cal::second_difference_operator(prob_, reg.lambda, reg.curves));
    S.resize(m + R.rows(), n);
    S << J, R;
  }
  // Rank-thresholded pseudo-inverse (the SAME kRankThreshold the streamer and LM use): min-norm on the
  // identified directions, exactly (SᵀS)⁻¹Sᵀ when S has full column rank, and never a singular solve.
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
  cod.setThreshold(cal::kRankThreshold);
  cod.compute(S);
  const Eigen::MatrixXd P = cod.pseudoInverse();  // n_knots x (n_res + n_reg)
  return P.leftCols(m) * residual_market_scale().asDiagonal();  // n_knots x n_res = dx/dq
}

std::optional<pf::MultiCurveBook> BundleSession::resolve_book(const pf::MultiCurveBook& book) const {
  bool any = false;
  for (const auto& p : book.positions) {
    for (const auto& c : p.float_coupons) any = any || !c.obs.fixing_schedule.empty();
    for (const auto& c : p.mtm_coupons) any = any || !c.obs.fixing_schedule.empty();
  }
  if (!any) return std::nullopt;
  pf::MultiCurveBook out = book;
  const px::PricingContext ctx{eval_date_, &fixings_};
  for (auto& p : out.positions) {
    for (auto* leg : {&p.float_coupons, &p.mtm_coupons})
      for (auto& c : *leg) {
        if (c.obs.fixing_schedule.empty()) continue;
        try {
          px::resolve_into(c.obs, ctx);
        } catch (const px::MissingFixing& e) {
          throw std::runtime_error(std::string("book: a seasoned coupon needs a past fixing the session does not "
                                               "have (set_fixings / set_evaluation_date): ") + e.what());
        }
      }
  }
  return out;
}

PortfolioReprice BundleSession::price_portfolio(const pf::MultiCurveBook& book_in) const {
  PortfolioReprice out;
  out.n = static_cast<int>(book_in.positions.size());
  if (out.n == 0) { last_price_us_ = 0.0; return out; }  // empty book: NPV/PV01 = 0, nothing to time
  const auto resolved = resolve_book(book_in);          // seasoned coupons: realized part from the fixings
  const pf::MultiCurveBook& book = resolved ? *resolved : book_in;

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
  // straight off the derivative vector. PV01 = 1bp · Σⱼ uⱼ·∂NPV/∂xⱼ along u = pricing::parallel_direction: the book's
  // NPV change for +1bp on EVERY curve's forward once (outright curves' interpolation knots; a spread curve inherits it
  // and a turn delta is not a level -- the all-ones direction counted spread curves twice and moved turns, fixed
  // 2026-09-14, tests/spread_pv01_var_repro_test.cpp) -- no bump-and-reprice. (Left-multiplying this same
  // gradient by risk_operator() would instead give the full per-quote delta ladder, CLAUDE.md #4.)
  // A WIDTH-ONE directional dual (ad::seed_directional, 2026-09-10): every knot seeded with the same unit
  // derivative, so one heap-free pass returns Σⱼ ∂NPV/∂xⱼ directly -- the quantity PV01 needs -- instead of
  // a full-width gradient (E3-A4/D7: 96 % of the one-shot's 71k allocations were the 208-wide heap duals of
  // that pass, spent to compute one sum). Identical to the gradient's sum to rounding.
  {
    const auto xd = ad::seed_directional(x_, parallel_dir_);
    const auto Cad = cal::build_bundle_curves<ad::DualDir>(
        prob_.curves, [&](int c, int i) { return xd[prob_.offset(c) + i]; });
    const auto curve_ad = [&Cad](int i) -> const cal::CurveHandle<ad::DualDir>& { return *Cad[i]; };
    const ad::DualDir npv_ad = book.value<ad::DualDir>(curve_ad);
    out.pv01 = npv_ad.derivatives().size() ? 1e-4 * npv_ad.derivatives()[0] : 0.0;
  }
  return out;
}

PortfolioReprice BundleSession::price_portfolio_json(const std::string& book_json) const {
  return price_portfolio(book_from_json(json::parse(book_json)));
}

// ---- CACHED (warm/streaming) portfolio reprice ---------------------------------------------------
// bind_portfolio builds the compiled W-cache twin ONCE; reprice_bound reuses it every call. The one-time
// W build is amortized across ticks (a live book repriced against the recalibrating curve), which is the
// ONLY regime where the compiled kernel wins — a single cold reprice keeps paying the templated path
// (price_portfolio), so this is a SEPARATE entry point, not a swap-in.
void BundleSession::bind_portfolio(const pf::MultiCurveBook& book) {
  bound_book_ = std::make_unique<pf::MultiCurveBook>(book);  // UNRESOLVED: the twin re-resolves on every rebuild
  rebuild_cbook();
}

// The compiled twin is built from the book RESOLVED against the CURRENT fixings/evaluation date -- so a
// fixing that arrives after bind_portfolio reaches the seasoned coupons on the next reprice (E3-D3: the
// old code resolved once at bind and rebuilt the twin from that stale copy: 27 % NPV error).
void BundleSession::rebuild_cbook() const {
  const auto resolved = resolve_book(*bound_book_);  // seasoned coupons: realized part from the fixings
  resolved_book_ = std::make_unique<pf::MultiCurveBook>(resolved ? *resolved : *bound_book_);
  cbook_ = std::make_unique<pf::CompiledMultiCurveBook>(prob_.curves, *resolved_book_);
}

PortfolioReprice BundleSession::reprice_bound() const {
  if (!bound_book_)
    throw std::runtime_error("reprice_bound(): bind_portfolio() must be called first");
  // Lazy rebuild: invalidate_engine() drops the twin on a structural bundle change (fixings), keeps the book.
  if (!cbook_) rebuild_cbook();

  PortfolioReprice out;
  out.n = static_cast<int>(resolved_book_->positions.size());
  if (out.n == 0) { last_price_us_ = 0.0; return out; }  // empty book: NPV/PV01 = 0, nothing to time

  // Pure ENGINE pricing pass, engine-timed exactly as price_portfolio times its double NPV pass — but here
  // the DF = exp(-W_all x) matvec + gathered coupon/annuity reduce replaces the per-coupon virtual walk.
  const auto t0 = std::chrono::steady_clock::now();
  out.npv = cbook_->npv(x_);
  last_price_us_ = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  out.price_us = last_price_us_;

  // PV01: +1bp parallel-knot-shift directional derivative (analytic on the compiled half, one AAD pass on
  // any fallback) — outside the pricing clock, matching how price_portfolio times only the NPV pass.
  out.pv01 = cbook_->pv01(x_);
  return out;
}

PortfolioRisk BundleSession::price_portfolio_risk(const pf::MultiCurveBook& book_in, const RegSpec& reg) const {
  PortfolioRisk out;
  out.n = static_cast<int>(book_in.positions.size());
  const int nk = prob_.n_knots();
  const int nr = prob_.n_residuals();
  if (out.n == 0) {  // empty book: nothing to reprice or risk
    out.curve_grad = Eigen::VectorXd::Zero(nk);
    out.ladder = Eigen::VectorXd::Zero(nr);
    last_risk_us_ = 0.0;
    return out;
  }
  const auto resolved = resolve_book(book_in);  // seasoned coupons: realized part from the fixings
  const pf::MultiCurveBook& book = resolved ? *resolved : book_in;

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
  return cal::structure_equal(prob_, p);  // O(n) equality; resolution-insensitive on fixing schedules
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
  // The streamer BORROWS the session's one compiled engine (the object model): a scalar set_quotes on the
  // session is what it prices next tick. It is dropped with the engine (invalidate_engine) and rebuilt
  // here lazily by the next stream_update/resolve.
  stream_ = std::make_unique<cal::StreamingCalibrator<cal::BundleProblem>>(ensure_engine(), prob_, x_, q0, opt);
  stream_reg_ = reg;
  stream_step_tol_ = step_tol;
  stream_armed_ = true;
  bands_changed_ = false;
  stream_sum_us_ = 0;  // reset the running average for this streaming session
  stream_ticks_ = 0;
}

const Eigen::VectorXd& BundleSession::stream_update(const Eigen::VectorXd& new_market) {
  if (!stream_) {
    if (!stream_armed_) throw std::runtime_error("start_streaming() must be called first");
    start_streaming(stream_reg_, stream_step_tol_);  // rebuilt after a fixings / evaluation-date recompile
  }
  // Eigen's size asserts are compiled out under -DNDEBUG, so a wrong-length market would be a silent
  // heap over-read/over-write inside the residual kernel (recalibrate() already checks; this must too).
  if (new_market.size() != prob_.n_residuals())
    throw std::runtime_error("stream_update: market length does not match the instrument count");
  if (!new_market.allFinite())
    throw std::runtime_error("stream_update: market contains a non-finite quote");
  cal::validate_targets(prob_.instruments, new_market, "stream_update");  // K5': before a re-anchor could commit an out-of-band market
  // P3: the session's quotes ARE the market this tick solves to -- on the instruments and the shared engine, converged or not -- so
  // resolve / residual / quote_diagnostics afterwards read it. Until 2026-09-15 only a FAILED tick committed (in its fallback).
  cal::commit_targets(prob_.instruments, engine_.get(), new_market);
  if (bands_changed_) {  // a set_band since the last anchor: re-read the active-set table first
    stream_->resync(prob_, x_, new_market);
    bands_changed_ = false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const cal::StreamTick tick = stream_->update(new_market);
  record_tick(tick, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  x_ = stream_->current();  // the streamer never commits a half-solved curve; this is the last GOOD x
  if (tick.converged) {
    // result() reports THIS tick, as it does after resolve/recalibrate (the same stamp). Until 2026-09-21 only the
    // fallback below wrote result_, so on a healthy streaming session result().x was the LAST LM SOLVE -- a curve
    // that could be hundreds of ticks stale, reported converged -- while x() and the return value moved on. Found
    // by a soak probe that read it and measured a 5e-2 "error" that was just the market's drift since that solve.
    stamp_streamed(tick, stream_->last_residual());  // no residual evaluation on the gated tick (see the header)
  } else {
    // THE LM FALLBACK (2026-09-12) — the same one warm_solve has always had, so both entry points now agree
    // about what a tick that cannot converge means. Without it they disagreed in a way that mattered: a
    // failed tick left x_ at the PREVIOUS solution and returned it, so a caller that did not check
    // last_converged() priced the new market off a stale curve, silently — and it was not self-correcting,
    // since the next tick starts from the same anchor with a larger accumulated move and is MORE likely to
    // fail again.
    //
    // The two signals keep their distinct meanings, as they already do on warm_solve: last_converged() /
    // last_status() / last_reason() report the streaming TICK's health (did the fast path work), while
    // result().converged reports the SOLVE (was the market incorporated). A fallback tick is false and true
    // respectively. A caller watching streaming health still sees every failure; a caller pricing off the
    // curve gets the new market either way.
    calibrate(x_, stream_reg_);     // warm from the last good x (the quotes were committed before the tick), as warm_solve does
    stream_->resync(prob_, x_, new_market);
    x_ = result_.x;
  }
  return x_;
}

const Eigen::VectorXd& BundleSession::stream_update(const Eigen::VectorXd& target, const Eigen::VectorXd& lower,
                                                   const Eigen::VectorXd& upper, const Eigen::VectorXd& decay) {
  cal::validate_quotes(target, lower, upper, decay, prob_.n_residuals(), "stream_update");  // refused whole: nothing changes
  // In place when only band VALUES moved; a change in which rows are banded re-anchors (one Jacobian) before the tick.
  bands_changed_ = cal::requote_bands(prob_.instruments, engine_.get(), stream_.get(), lower, upper, decay, has_band_) || bands_changed_;
  return stream_update(target);
}

// =================================================================================================
// One-shot JSON dispatcher
// =================================================================================================
std::string run_json(const std::string& request) {
  try {
    return run_json(json::parse(request).as_object());  // parse ONCE (E6.3/D11): every verb takes the object
  } catch (const std::exception& e) {
    return err(e.what());
  }
}

std::string run_json(const json::object& o) {
  try {

    // Stateless COMPILE verb: a composer spec (curves + generic instrument rows + interpolation regions)
    // -> the resolved bundle + streaming config, the C++ analog of server/compile.py's compile_spec. It
    // PRODUCES a bundle rather than consuming one, so it is dispatched here before the 'bundle' requirement.
    // Reachable by any host through the single swaps_run_json C ABI (Excel add-in, .NET, ctypes, web), so a
    // client can build a bundle up from typed rows instead of pasting resolved JSON. `today` supplies the
    // value date when the spec omits one.
    if (o.contains("compile")) {
      const std::string today = (o.contains("today") && o.at("today").is_string())
                                    ? std::string(o.at("today").as_string().c_str()) : "";
      auto cr = compile_spec(o.at("compile"), today);
      // One-shot convenience (the demo / Excel / any stateless C-ABI client): when the request ALSO
      // carries `sample_times`, compile THEN calibrate THEN sample in a SINGLE call — so the caller sends
      // a small composer spec and receives curve samples, never the large resolved bundle. Rewrite to the
      // resolved `bundle` request and reuse the calibrate+sample path below (identical output shape;
      // `regularize`/`x0`/`price` all carry through). Without `sample_times` this returns the resolved
      // structure exactly as before — backward-compatible.
      if (!o.contains("sample_times")) return json::serialize(compile_to_json(cr));
      json::object req2 = o;
      req2.erase("compile");
      req2.erase("today");
      req2["bundle"] = bundle_to_json(cr.bundle);
      // The spec's smoothing is the DEFAULT regulariser of the rewrite (E3-D4): a caller that sends a
      // composer spec gets the same curve the web shows; an explicit `regularize` still wins.
      if (!req2.contains("regularize")) {
        const RegSpec reg = compile_reg_spec(cr);
        if (reg.on()) {
          json::object r;
          r["lambda"] = reg.lambda;
          r["curves"] = json::array(reg.curves.begin(), reg.curves.end());
          r["tension"] = reg.tension;
          r["sigma"] = reg.sigma;
          req2["regularize"] = std::move(r);
        }
      }
      return run_json(req2);  // the compile+sample re-entry: no re-serialise, no re-parse (E6.3)
    }

    // Stateless GENERATE_RISK verb: one book + N bundles -> N internally-consistent risk ladders, all off
    // bundles[0]'s DFs (later bundles re-leveled onto the anchor, under-determined ones rank-completed by
    // self-quoted pillars). Like `compile`, it produces rather than consumes a bundle, so dispatch it here.
    // Every STATELESS verb (it builds its own session, or needs none). The table is GENERATED from
    // api/api_surface.py STATELESS_VERBS by tools/gen_dispatch.py: ONE list names them, ONE loop
    // dispatches them, and the Excel/pybind generators read the same list (E6.3; 22 hand-written
    // if-arms before, each re-parsing the request string).
#include "run_json_stateless.gen.inc"

    // Stateless SWAPTION verb: price European swaptions off a calibrated curve (Bachelier / SABR).

    // Stateless SABR strip calibration (no bundle): fit (alpha,rho,nu) to a market vol strip.

    // Stateless BONDS verb (no bundle): street/yield-space bond math (price<->yield, accrued, duration,
    // convexity) for a list of fixed-rate bonds.

    // Stateless ASSET_SWAP verb: par asset-swap spread(s) for bonds off a calibrated bundle (curve-space).

    // Stateless BOND_FUTURE verb: CTD selection, conversion factors, gross/net basis, implied repo.

    // Stateless INFLATION verb: ZCIS/YoY breakeven-inflation curve calibration + index/breakeven output.

    // Stateless CREDIT verb: hazard-rate survival curve calibrated to a par CDS-spread strip.

    // Stateless FX_OPTION / FX_VOL verb: Garman-Kohlhagen vanilla FX options + delta-quoted smile.

    // Conventions registry (P2): add/override market conventions at runtime; list what the engine knows.

    // Stateless NDF verb: non-deliverable FX forwards / NDS (covered-interest-parity, linear, no vol).

    // Stateless CALIB_REPORT verb: calibration diagnostics (Jacobian condition number + identifiability).

    // Stateless RV verbs (api/rv.cpp): batched bond-universe analytics; the minimum-pricing-error govvie
    // fit (spline / Nelson-Siegel / Svensson) with the per-bond RV ladder; and the headline swap-spread
    // derivation returning the {pin, asw} asset-swap BASIS rows as instrument JSON.

    // Stateless EXPOSURE verb: EPE/ENE/PFE counterparty-exposure profile for a swap book off a calibrated curve.

    if (!o.contains("bundle")) return err("request is missing the required 'bundle' object");

    cal::BundleProblem prob = bundle_from_json(o.at("bundle"));
    if (prob.n_curves() == 0) return err("bundle has no curves");
    BundleSession sess(std::move(prob));
    const cal::BundleProblem& P = sess.problem();

    Eigen::VectorXd x0;
    if (o.contains("x0")) {
      const auto xv = darr(o, "x0");
      x0 = Eigen::Map<const Eigen::VectorXd>(xv.data(), static_cast<Eigen::Index>(xv.size()));
      if (x0.size() != P.n_knots()) return err("x0 length does not match the bundle's knot count");
    } else {
      x0 = flat_x0(P);
    }

    const RegSpec reg = reg_from_json(o);  // the ONE RegSpec-from-JSON (E6.3)

    const cal::CalibrationResult& res = sess.calibrate(x0, reg);

    json::object out;
    {
      json::object c;
      c["iterations"] = res.iterations;
      c["rms_residual"] = res.rms_residual;
      c["stationarity"] = res.stationarity;
      c["info"] = res.info;
      c["converged"] = res.converged;  // LM ended at a stationary point with a finite result
      c["status"] = res.status;        // the LM stopping reason, in words
      c["regularize_applied"] = reg.on();  // which smoothing this calibration ran under (E3-D4)
      c["regularize_lambda"] = reg.lambda;
      c["regularize_tension"] = reg.tension;
      c["rank_deficiency"] = res.rank_deficiency;  // >0: the instrument set under-determines the curve
      out["calibration"] = c;
    }
    out["x"] = da(sess.x());
    out["quote_diagnostics"] = sess.quote_diagnostics();  // per-quote in-band fit (soft never silent)

    if (o.contains("sample_times")) out["curves"] = sample_to_json(sess.sample(darr(o, "sample_times")));

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
