#pragma once
#include <cstdio>  // validate_quote's failure message
// The calibration problem: free variables x = knot forwards, residual vector r(x) in RATE units.
//
// QuantLib-free and templated on Scalar (CLAUDE.md §1): the schedules are extracted from QuantLib
// once (swaps/ql/extract.hpp) and stored as plain data here; residuals(x) rebuilds the curve and
// reprices every instrument with the templated kernel. With Scalar = double this drives the LM
// solve; with Scalar = AutoDiffScalar (Phase 3) the SAME code yields the analytic Jacobian.
//
// Residual convention (CLAUDE.md §2): everything is in rate units so futures (quoted as prices) do
// not swamp swaps. A futures market quote of price P contributes target rate (1 - P/100); the model
// side is the reference rate plus its convexity adjustment.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace swaps::calibration {

// =================================================================================================
// GENERIC INSTRUMENT MODEL (docs/generic-instrument-pipeline.md §3)
// =================================================================================================
// An instrument = LEGS + a quote transform + a market quote. Nothing here names an index, a currency,
// a calendar or a convention (CLAUDE.md §1, design §6): a leg is a list of generic FloatCoupon /
// FixedCoupon (pricing/cashflows.hpp) and a curve role is an integer. "SOFR OIS", "EURIBOR 3M swap",
// "FF/SOFR basis", "1M averaging future" are all THIS type with different DATA, built by a test.

// Curve ROLES belong to the LEG, not the instrument. That is the whole point: a tenor-basis or
// cross-curve instrument is just two legs with different `forecast` curves, and forecast != discount
// (multi-curve) is a leg-local fact — neither needs a new instrument type.
struct FloatLeg {
  std::vector<pricing::FloatCoupon> coupons;
  int forecast = 0;  // curve that FORECASTS this leg's index fixings
  int discount = 0;  // curve that DISCOUNTS this leg's payments
  // MtM (mark-to-market cross-currency) notional-reset roles. -1/-1 => a plain constant-notional leg
  // (the default). When set, this is an FX-resettable leg whose coupon notional is
  // fx_spot · DF[reset_num](reset)/DF[reset_den](reset) -- priced by pricing::xccy_mtm_leg_pv on the templated
  // path and, since 2026-09-09, EXACTLY on the W-cache too (BundleFloatBatch::add_mtm: a product of registered DFs).
  int reset_num = -1;   // FX-forward NUMERATOR (foreign) discount curve role
  int reset_den = -1;   // FX-forward DENOMINATOR (domestic) discount curve role
  // A COMPLETE MtM leg: both reset roles set (the compiled batch prices it as a product of registered DFs).
  bool mtm_complete() const { return forecast >= 0 && discount >= 0 && reset_num >= 0 && reset_den >= 0; }
  double fx_spot = 1.0;  // FX spot for the notional reset
  // O-X3 (2026-09-14): fx_spot is the SPOT-DATE quote and fx_spot_time the curve time of that spot date, so period i's
  // notional is fx_spot · ratio(reset_i) / ratio(fx_spot_time), ratio(t) = DF[reset_num](t)/DF[reset_den](t). 0 = today.
  double fx_spot_time = 0.0;
};

struct FixedLeg {
  std::vector<pricing::FixedCoupon> coupons;
  int discount = 0;  // curve that DISCOUNTS this leg's payments (the annuity curve)
};

// The quote transform. ALL are in RATE units (CLAUDE.md §2) so a futures row cannot outweigh a swap row.
enum class QuoteKind {
  ParRate,    // float_leg_pv(fwd) / annuity(fixed)
  ParSpread,  // (float_leg_pv(bench) - float_leg_pv(fwd)) / annuity(fixed)
  Rate,       // rate(obs) + convexity
  ZeroCouponRate, // ANNUALLY-COMPOUNDED zero-coupon par rate: r = (1 + τ·q)^(1/τ) − 1 with q the ParRate
                  // quotient of the same legs and τ the ONE fixed accrual (BRL DI×Pre: fixed pays
                  // (1+r)^τ − 1 at maturity vs CDI compounded to maturity, BUS/252). A NONLINEAR transform
                  // of ParRate (zero_coupon_transform); rides the W-cache with a chain-rule row scale, and
                  // the band residual applies to the TRANSFORMED quote.
  // Cross-currency (multi-currency). BOTH are W-cacheable in their STANDARD form (compiled_bundle.hpp):
  // a standalone FxForward's log-residual is affine in x (constant Jacobian row), and a MtM basis with a
  // funding leg is priced EXACTLY on the W-cache since 2026-09-09 (BundleFloatBatch::add_mtm; the retired mtm_funding_term_negligible
  // guard proves the dropped term is 0). Only a MtM whose funding leg is incomplete or seasoned rides the
  // width-reduced AAD block of the hybrid engine (hybrid_residual.hpp); an FX forward or MtM NESTED in a
  // Portfolio is a term of the compiled row model like any other (2026-09-22).
  FxForward,      // FX-forward point: fx_spot · DF[fx_num](fx_time)/DF[fx_den](fx_time) (pins fx_num vs fx_den)
  XccyMtmBasis,   // MtM (FX-resettable-notional) xccy basis: par basis incl. the resetting funding leg
  Portfolio,      // linear combination of component instruments: model quote = Σ weight·quote(component).
                  // `market` is the COMBINED quote (a butterfly/condor spread), so you calibrate to the
                  // combo directly without pinning each leg's outright rate. Components are full nested
                  // Instruments, so portfolios compose. ONE residual, no knots (knots are in the curve
                  // spec). W-CACHEABLE when every component is: the components register as weighted TERMS
                  // onto the one row (compiled_bundle.hpp's row model) -- FX forwards, MtM bases and
                  // zero-coupon rates included; hybrid_residual.hpp is the authority on what is not.
  TurnJump,       // a TURN's jump δ (docs/turns-calibration.md). The model quote is the raw overlay state
                  // variable δ of (turn_curve, turn_index) -- a STATE-PIN, LINEAR in x (Jacobian row is a
                  // unit vector at δ's state index). Almost always BANDED (target/lower/upper): the band's
                  // target regularises δ so it is always identifiable; bracketing futures then sharpen it.
                  // It IS W-cacheable (linear, no DF), but its Jacobian entry is direct (∂δ/∂x), not via DF.
  Npv,            // A POSITION'S NET PRESENT VALUE, in PV units (architecture review item 1/4, 2026-10-06): the
                  // model quote is  pv(fwd) − pv(bench) + fx_spot·pv_bare(mtm) − pv(fixed)  over whichever legs
                  // carry coupons, with the FIXED leg a list of dated amounts τ·scale (a swap's fixed rate, a
                  // stepped schedule, a principal exchange, an xccy notional exchange are all DATA in `scale`,
                  // not fields). A book is a BundleProblem of Npv rows (calibration/book_rows.hpp
                  // position_instrument / book_problem): the ONE compiled engine then reprices the book
                  // (model_rates), gives its parallel PV01 (directional_into) and per-position key-rate risk
                  // (jacobian) -- no book-specific partials, predicates or fallbacks. LINEAR in the DFs (no
                  // annuity division), so it rides the W-cache as a Src::Npv term; the usual AAD routing
                  // (Instrument::noncacheable) applies to a compounded or seasoned leg. A position's notional
                  // is a Portfolio weight. As a CALIBRATION row it is in PV units, not rate units: the caller
                  // weights it (CLAUDE.md §2) -- the engine does not.
};

// The tripwire for adding a quote kind: appending one moves TurnJump and breaks this assert, which names
// every place that must learn about it. A -Wswitch warning alone is not enough -- this build has no
// -Werror, so a warning scrolls past (REVIEW FINDING 3, 2026-09-21).
inline constexpr int kQuoteKindCount = 9;
static_assert(static_cast<int>(QuoteKind::Npv) + 1 == kQuoteKindCount,
              "a QuoteKind was added or reordered: update instrument_model_quote / instrument_residual "
              "(problem.hpp), the compiled batches (compiled_bundle.hpp), the codec's to/from string "
              "(api/codec.cpp) and kQuoteKindCount itself");

// A PIECEWISE-LINEAR RESIDUAL MAP (constraint rows, architecture review item 3, 2026-10-07). A row's residual is a
// continuous piecewise-linear function of its model value q that is ZERO at the target m: breakpoints b_0 < .. <
// b_{n-1} cut the line into n+1 pieces with slopes s_0 .. s_n. A plain row is {no breakpoints, slope 1}; the Huber
// bid/offer band is {{lower, upper}, {1, decay, 1}}; a one-sided bound would be {{edge}, {1, 0}}; a stiff pin is
// {{}, {w}} at target = edge. This is THE description every layer reads -- the AAD residual (residual<Scalar>), the
// compiled row map (residual_d), the streamer's edge walk (piece_of / invert / at_break), the risk scale
// (target_slope) and the diagnostics (slope_at) -- so a row's penalty shape is defined ONCE. Until 2026-10-07 the
// band's shape was written in four places (band_residual / band_residual_d / band_slope / the streamer's inline
// `side == 0 ? decay : 1`), each excluding FX forwards by name.
// Two numeric forms, each the association its tier always had (so every pinned number is bit for bit unchanged):
// residual<Scalar> keeps q the plain operand (q + k, (q − m)·s: one dual op), residual_d accumulates from the
// nearest breakpoint (r(b) + (q − b)·s). Equality ON a breakpoint resolves toward the target's piece (the band's
// convention: an edge is inside). Fixed storage: no heap on the hot path.
struct PenaltyMap {
  static constexpr int kMaxBreaks = 3;
  int n = 0;                                        // breakpoints
  double b[kMaxBreaks] = {0.0, 0.0, 0.0};           // ascending
  double s[kMaxBreaks + 1] = {1.0, 1.0, 1.0, 1.0};  // the n + 1 piece slopes

  static PenaltyMap plain() { return {}; }
  static PenaltyMap band(double lower, double upper, double decay) {
    if (!(upper > lower)) return {};
    PenaltyMap m;
    m.n = 2;
    m.b[0] = lower;
    m.b[1] = upper;
    m.s[0] = 1.0;
    m.s[1] = decay;
    m.s[2] = 1.0;
    return m;
  }
  bool is_plain() const { return n == 0 && s[0] == 1.0; }
  bool operator==(const PenaltyMap& o) const {
    if (n != o.n) return false;
    for (int j = 0; j < n; ++j) if (b[j] != o.b[j]) return false;
    for (int i = 0; i <= n; ++i) if (s[i] != o.s[i]) return false;
    return true;
  }
  // The piece the target m lies in: a breakpoint equal to m joins the piece with the SMALLER slope (a target on a
  // band edge is inside the band; one on a bound's edge is on its dead side).
  int target_piece(double m) const {
    int tp = 0;
    while (tp < n && b[tp] < m) ++tp;
    if (tp < n && b[tp] == m && s[tp + 1] < s[tp]) ++tp;
    return tp;
  }
  // The piece a value q lies in, given the target's piece: equality on a breakpoint resolves toward the target.
  template <class Q>
  int piece_of(const Q& q, int tp) const {
    int i = tp;
    while (i > 0 && q < Q(b[i - 1])) --i;
    while (i < n && q > Q(b[i])) ++i;
    return i;
  }
  // r at breakpoint j (between pieces j and j+1), walked from the target in piece tp.
  double at_break(int j, double m, int tp) const {
    if (j >= tp) {
      double r = (b[tp] - m) * s[tp];
      for (int t = tp; t < j; ++t) r += (b[t + 1] - b[t]) * s[t + 1];
      return r;
    }
    double r = (b[tp - 1] - m) * s[tp];
    for (int t = tp - 1; t > j; --t) r += (b[t - 1] - b[t]) * s[t];
    return r;
  }
  // The piece a RESIDUAL value r lies in (the map is monotone, every slope >= 0): the streamer reads a row's side off
  // its residual without re-pricing the quote.
  int piece_from_residual(double r, double m, int tp) const {
    int i = tp;
    while (i < n && r > at_break(i, m, tp)) ++i;
    while (i > 0 && r < at_break(i - 1, m, tp)) --i;
    return i;
  }
  // AAD form: q stays the plain operand of ONE op (bit for bit the pre-2026-10-07 band_residual).
  template <class Scalar>
  Scalar residual(const Scalar& q, double m) const {
    const int tp = n == 0 ? 0 : target_piece(m);
    const int i = n == 0 ? 0 : piece_of(q, tp);
    if (i == tp) {
      if (s[i] == 1.0) return q - Scalar(m);
      return (q - Scalar(m)) * s[i];
    }
    const int j = i > tp ? i - 1 : i;
    const double k = at_break(j, m, tp) - s[i] * b[j];
    if (s[i] == 1.0) return q + Scalar(k);
    return q * s[i] + Scalar(k);
  }
  // THE TWO-EDGE VIEW: the map specialised to the band shape (two breakpoints around the target, the shape every
  // streamed band has), as five flat doubles -- what the hot consumers hold (the compiled row map per banded row per
  // Newton step, the streamer's walk per step per tracked row). Its evaluators are the general ones UNROLLED (three
  // comparisons, no loops, no target-piece search): the general map carried 88 bytes and a piece walk onto a 2.3 us
  // tick (+6..20 %, measured 2026-10-07). penalty_map_test pins the view against the general walk and against the
  // pre-2026-10-07 expressions verbatim. The view is DERIVED from the map (two_edge()), never written by hand.
  struct TwoEdge {
    double lo = 0.0, hi = 0.0, s_lo = 1.0, s_in = 1.0, s_hi = 1.0;
    std::pair<double, double> residual_d(double q, double m) const {
      if (q > hi) return {(hi - m) * s_in + s_hi * (q - hi), s_hi};  // s == 1: ×1.0 is exact, the old form bit for bit
      if (q < lo) return {(lo - m) * s_in + s_lo * (q - lo), s_lo};
      return {s_in * (q - m), s_in};
    }
    double r_at_edge(int side, double m) const { return ((side > 0 ? hi : lo) - m) * s_in; }
    int side_from_residual(double r, double m) const {
      if (r > (hi - m) * s_in) return +1;
      if (r < (lo - m) * s_in) return -1;
      return 0;
    }
    int side_from_value(double v) const { return v > hi ? +1 : (v < lo ? -1 : 0); }
    double invert(int side, double r, double m) const {
      if (side > 0) return hi + (r - (hi - m) * s_in);
      if (side < 0) return lo + (r - (lo - m) * s_in);
      return m + r / s_in;
    }
    double edge(int side) const { return side > 0 ? hi : lo; }
    double slope(int side) const { return side == 0 ? s_in : (side > 0 ? s_hi : s_lo); }
    double width() const { return hi - lo; }
  };
  bool is_two_edge(double m) const { return n == 2 && b[0] <= m && m <= b[1]; }
  TwoEdge two_edge() const {
    if (n != 2) throw std::logic_error("PenaltyMap::two_edge: not a two-edge (band) map -- the general row map is stage B");
    return TwoEdge{b[0], b[1], s[0], s[1], s[2]};
  }
  // Double form {r, dr/dq} (bit for bit the pre-2026-10-07 band_residual_d): the two-edge view when the shape is a
  // band with the target inside, else the general piece walk. residual_d_general is the general path, exposed so a
  // test can pin the two agree.
  std::pair<double, double> residual_d(double q, double m) const {
    if (is_two_edge(m)) return two_edge().residual_d(q, m);
    return residual_d_general(q, m);
  }
  std::pair<double, double> residual_d_general(double q, double m) const {
    const int tp = n == 0 ? 0 : target_piece(m);
    const int i = n == 0 ? 0 : piece_of(q, tp);
    if (i == tp) return {s[i] == 1.0 ? q - m : s[i] * (q - m), s[i]};
    const int j = i > tp ? i - 1 : i;
    const double rb = at_break(j, m, tp);
    return {s[i] == 1.0 ? rb + (q - b[j]) : rb + s[i] * (q - b[j]), s[i]};
  }
  double slope_at(double q, double m) const { return s[n == 0 ? 0 : piece_of(q, target_piece(m))]; }
  double target_slope(double m) const { return s[n == 0 ? 0 : target_piece(m)]; }
  // q on piece i from its residual r (the piece's slope must be > 0): the target piece from m, any other from its
  // breakpoint on the target side (the streamer's side_of inversion, bit for bit).
  double invert(int i, double r, double m, int tp) const {
    if (i == tp) return s[i] == 1.0 ? m + r : m + r / s[i];
    const int j = i > tp ? i - 1 : i;
    const double rb = at_break(j, m, tp);
    return s[i] == 1.0 ? b[j] + (r - rb) : b[j] + (r - rb) / s[i];
  }
};

// Forward declarations for the recursive Portfolio components (each component is itself an Instrument).
struct Instrument;
struct WeightedInstrument;

// One calibration instrument.
//
// Which members a given `quote` reads (the others are ignored and may stay default-constructed):
//   ParRate   : `fwd` (the float leg, enters POSITIVELY) and `fixed` (the annuity).
//   ParSpread : `fwd` (the SPREAD/quoted leg, enters NEGATIVELY), `bench` (the benchmark leg, enters
//               POSITIVELY) and `fixed`. The two float legs are INDEPENDENT — different frequency,
//               day count, spread and forecast curve are all fine (design §6.7).
//   Rate      : `obs`, `forecast` and `convexity`.
//
// Sign convention for ParSpread: the quoted spread s satisfies pv(fwd) + s·annuity = pv(bench), hence
// s = (pv_bench − pv_fwd)/annuity.
struct Instrument {
  QuoteKind quote = QuoteKind::ParRate;
  FloatLeg fwd;
  FloatLeg bench;
  FixedLeg fixed;
  pricing::RateObservation obs;  // Rate only
  int forecast = 0;              // Rate only: the curve that forecasts `obs`
  // Rate only. An INPUT NUMBER (design §3): the convexity MODEL (Hull-White etc.) lives in tests.
  double convexity = 0.0;
  double market = 0.0;  // the market quote, in the units of `quote` (always rate units)

  // Bid/offer BAND (soft calibration target). When `band_upper > band_lower` (bounds in the quote's rate
  // units) the residual is the HUBER band residual (band_residual()): a `band_decay`-slope pull to the mid
  // inside [lower, upper], a unit-slope pull to the nearer EDGE outside it, continuous at the edges. So
  // the solver treats any model value within [lower, upper] as ~satisfied and spends its freedom on the
  // hard targets, while an overlapping instrument that cannot be hit exactly settles inside its band. The
  // default (band_upper <= band_lower, band_decay = 1) leaves the residual as the plain (q − market).
  double band_lower = 0.0, band_upper = 0.0, band_decay = 1.0;

  // Portfolio (QuoteKind::Portfolio) components: model quote = Σ weight·model_quote(component). Ignored
  // for every other quote kind. Defined out-of-line below (recursive type).
  std::vector<WeightedInstrument> combination;

  // The currency the quote/residual is expressed in (multi-currency). Consulted ONLY by a cross-
  // currency quote that mixes legs of different currencies (to name the PV numeraire); every single-
  // currency quote ignores it. Default 0 keeps existing instruments byte-identical.
  int pv_currency = 0;

  // FxForward only: F = fx_spot · DF[fx_num](fx_time) / DF[fx_den](fx_time). fx_num is the FOREIGN
  // (collateral) curve this pins (e.g. EUR-in-USD), fx_den the DOMESTIC (e.g. SOFR). `market` is the
  // outright forward. The residual is in rate/implied-basis units: (ln F_model − ln F_market)/fx_time.
  int fx_num = -1, fx_den = -1;
  double fx_spot = 1.0;
  double fx_time = 0.0;
  double fx_spot_time = 0.0;  // O-X3: curve time of the spot date fx_spot is quoted for (0 = today): F(fx_spot_time) = fx_spot
  // XccyMtmBasis only: the resetting-notional funding leg (its reset_num/reset_den/fx_spot on the leg
  // define the FX-forward notional). fwd = the pinned curve's self-forecast leg, bench = the other-
  // currency forecast leg, fixed = the annuity — all discounted on the pinned (collateral) curve.
  FloatLeg mtm;

  // TurnJump only: which turn this instrument pins. `turn_curve` is the curve carrying the turn and
  // `turn_index` its position in that curve's `turns` list. The model quote is that turn's jump δ; the
  // residual is (banded) δ − market, with `market` the target jump (rate units). See QuoteKind::TurnJump.
  int turn_curve = 0, turn_index = 0;

  // Set the FULL quote RHS -- target and soft-quote band -- from any target-shaped object exposing
  // {target, band_lower, band_upper, band_decay}. This is THE one hand-off from a market quote into a
  // calibration instrument: market::Quote::to_target() (market/quote.hpp CalibrationTarget) and the
  // API compiler's wire quote (api/compile.cpp) both feed THIS setter, so the band semantics above are
  // defined exactly once -- here. Duck-typed on purpose: the calibration layer sits BELOW the market
  // layer and must not include it; any POD with those four fields is a valid source.
  template <class Target>
  void set_target(const Target& t) {
    market = t.target;
    band_lower = t.band_lower;
    band_upper = t.band_upper;
    band_decay = t.band_decay;
  }

  // ---- what the instrument SAYS of itself (2026-09-22): every consumer asks these, none re-derives them ----
  // The curve references this quote kind READS, in the order it reads them -- the one encoding of "which members
  // a given `quote` reads" (the comment above). fn(curve_index, role) per role; a Portfolio walks its components.
  // Behind validate_problem's range check, bundle_adjacency's dependency edges, the AAD block's touched set and
  // primary_curve. (The four copies it replaced disagreed: the staged solver read a TurnJump's and a Portfolio's
  // DEFAULT fwd/fixed legs as curve 0, the AAD block seeded a ParRate's unused bench/mtm legs.)
  template <class Fn>
  void for_each_curve_ref(Fn&& fn) const;
  // The curve this instrument primarily PINS: its FIRST curve reference (an FX forward its numerator, a turn its
  // curve, a leg-based quote its fwd leg's forecast, a Portfolio its first component's). Defined out-of-line.
  int primary_curve() const;
  // A GENERAL penalty map as DATA (constraint rows stage B, 2026-10-09): when set it IS the row's residual map and the
  // four band numbers are ignored -- a one-sided band, a bound with a dead zone, any monotone piecewise-linear
  // penalty (validate_instrument: breakpoints ascending, every slope >= 0). JSON "penalty": {breaks, slopes}.
  std::optional<PenaltyMap> penalty_map;
  // This row's penalty shape: penalty_map when set, else the Huber band from the four quote numbers, else plain. A
  // standalone FX forward NEVER carries one (its residual is the log-basis map; the band is ignored there) -- that
  // exclusion lives HERE and nowhere else now.
  PenaltyMap penalty() const;
  // Any COMPOUNDED (RFR lookback/lockout product) observation anywhere in this instrument (components included).
  bool has_compounded_obs() const;
  // True iff this instrument is a SHAPE the compiled batch cannot express, so it must ride the AAD tier: a
  // compounded observation anywhere, or an incomplete / SEASONED MtM funding leg. Every other kind, nested in a
  // Portfolio or not, compiles (hybrid_residual.hpp is the router; this is the instrument's own answer).
  bool noncacheable() const;
};

// A weighted component of a Portfolio instrument. Holds a full Instrument by value, so portfolios nest.
struct WeightedInstrument {
  double weight = 1.0;
  Instrument instrument;
};

template <class Fn>
void Instrument::for_each_curve_ref(Fn&& fn) const {
  const Instrument& ins = *this;
  const auto leg = [&](const FloatLeg& l, const char* fc, const char* dc, const char* rn, const char* rd) {
    fn(l.forecast, fc);
    fn(l.discount, dc);
    if (l.reset_num >= 0) fn(l.reset_num, rn);
    if (l.reset_den >= 0) fn(l.reset_den, rd);
  };
  switch (ins.quote) {
    case QuoteKind::Rate: fn(ins.forecast, "forecast"); break;
    case QuoteKind::FxForward: fn(ins.fx_num, "fx_num"); fn(ins.fx_den, "fx_den"); break;
    case QuoteKind::TurnJump: fn(ins.turn_curve, "turn"); break;  // the δ state lives on the turn's curve
    case QuoteKind::Portfolio:
      for (const auto& c : ins.combination) c.instrument.for_each_curve_ref(fn);
      break;
    case QuoteKind::ParRate:
    case QuoteKind::ZeroCouponRate:
    case QuoteKind::ParSpread:
    case QuoteKind::XccyMtmBasis:
      leg(ins.fwd, "forecast", "discount", "reset_num", "reset_den");
      if (ins.quote == QuoteKind::ParSpread || ins.quote == QuoteKind::XccyMtmBasis)
        leg(ins.bench, "benchmark forecast", "benchmark discount", "benchmark reset_num", "benchmark reset_den");
      if (ins.quote == QuoteKind::XccyMtmBasis)
        leg(ins.mtm, "mtm forecast", "mtm discount", "mtm reset_num", "mtm reset_den");
      fn(ins.fixed.discount, "fixed discount");
      break;
    case QuoteKind::Npv:  // reads exactly the legs that carry coupons (a float-only position has no fixed curve)
      if (!ins.fwd.coupons.empty()) leg(ins.fwd, "forecast", "discount", "reset_num", "reset_den");
      if (!ins.bench.coupons.empty()) leg(ins.bench, "benchmark forecast", "benchmark discount", "benchmark reset_num", "benchmark reset_den");
      if (!ins.mtm.coupons.empty()) leg(ins.mtm, "mtm forecast", "mtm discount", "mtm reset_num", "mtm reset_den");
      if (!ins.fixed.coupons.empty()) fn(ins.fixed.discount, "fixed discount");
      break;
  }
}

inline int Instrument::primary_curve() const {
  int first = 0;
  bool seen = false;
  for_each_curve_ref([&](int c, const char*) { if (!seen) { first = c; seen = true; } });
  return first;  // an empty Portfolio pins nothing: 0, as before
}

inline bool Instrument::has_compounded_obs() const {
  const auto leg = [](const FloatLeg& l) {
    for (const auto& c : l.coupons)
      if (c.obs.compounded) return true;
    return false;
  };
  if (quote == QuoteKind::Rate && obs.compounded) return true;
  if (leg(fwd) || leg(bench) || leg(mtm)) return true;
  for (const auto& c : combination)
    if (c.instrument.has_compounded_obs()) return true;
  return false;
}

inline bool Instrument::noncacheable() const {
  if (has_compounded_obs()) return true;
  if (quote == QuoteKind::XccyMtmBasis) {
    if (!mtm.mtm_complete()) return true;
    for (const auto& c : mtm.coupons)
      if (c.seasoned_mtm()) return true;  // a seasoned coupon prices on the templated kernel
    return false;
  }
  if (quote == QuoteKind::Npv && !mtm.coupons.empty()) {
    // The batch's MtM leg is the bare reset ratio at the reset time: a spot-date roll-back (fx_spot_time != 0, O-X3)
    // is a curve-dependent factor it does not carry, so such a position prices on the templated kernel.
    if (!mtm.mtm_complete() || mtm.fx_spot_time != 0.0) return true;
    for (const auto& c : mtm.coupons)
      if (c.seasoned_mtm()) return true;
  }
  if (quote == QuoteKind::Portfolio)
    for (const auto& c : combination)
      if (c.instrument.noncacheable()) return true;
  return false;
}

// Compile-time detection: does the curve object a CurveOf accessor returns expose turn_jump(int)? True
// for the bundle's CurveHandle, false for a bare ModularCurve (single-curve CalibrationProblem, which
// never carries a TurnJump instrument). Lets instrument_model_quote's TurnJump branch stay well-formed
// for BOTH curve types via `if constexpr`.
template <class C, class = void>
struct has_turn_jump : std::false_type {};
template <class C>
struct has_turn_jump<C, std::void_t<decltype(std::declval<const C&>().turn_jump(0))>> : std::true_type {};

// Bid/offer band residual for a model quote q against market mid m (see the Instrument band fields).
// The band exists so that OVERLAPPING instruments that cannot all be reconciled exactly (1M vs 3M futures,
// a future vs a swap at the same pillar) can each sit off their mid within a bid/offer tolerance. The
// residual is HUBER-shaped: a reduced-rate pull to the mid INSIDE the band, full-slope pull to the nearer
// EDGE outside it, continuous at the edges:
//     inside  [lower, upper] : r = decay·(q − m)
//     above   upper          : r = decay·(upper − m) + (q − upper)
//     below   lower          : r = decay·(lower − m) + (q − lower)
// so dr/dq is exactly `decay` inside and exactly 1 outside, with no ramp in between. That is deliberate:
// the previous smooth Gaussian ramp (w = decay + (1−decay)(1 − e^{−z²})) made r an S-curve whose slope
// overshoots above 1 just outside the edge, which (a) gives the summed-squares objective MULTIPLE minima
// along the direction the quotes barely see (two stationary fits 110 bp apart in a knot at the same
// market, chosen by the seed), and (b) turns the Jacobian into a moving target at every edge, so the
// frozen-Newton streamer converged, silently, to points up to 147 bp from the least-squares optimum. The
// piecewise-linear residual is monotone with a single zero, so r² is convex in q; and its Jacobian is
// piecewise CONSTANT (row = slope·∂q/∂x with slope ∈ {decay, 1}), which is what lets the streamer keep
// the quote Jacobian frozen and treat a band crossing as a cheap row re-scale (streaming.hpp).
// Far outside the band the pull is to the EDGE (plus the decay pull to mid), not to the mid at full
// weight: continuity at the edge forces that, and it is the right reading of a bid/offer tolerance.
// No band (upper <= lower): the plain residual q − m. AAD-safe: branch selection on q (one-sided
// derivative exactly at an edge), constants folded so `q` is always the plain-scalar operand.
// The three band forms are the PenaltyMap's band instance (2026-10-07): the shape is written once, above.
template <class Scalar>
Scalar band_residual(const Scalar& q, double market, double lower, double upper, double decay) {
  return PenaltyMap::band(lower, upper, decay).residual<Scalar>(q, market);
}
inline std::pair<double, double> band_residual_d(double q, double market, double lower, double upper, double decay) {
  return PenaltyMap::band(lower, upper, decay).residual_d(q, market);
}
inline double band_slope(double q, double lower, double upper, double decay) {
  return PenaltyMap::band(lower, upper, decay).slope_at(q, 0.5 * (lower + upper));  // a band's target piece is the band
}

inline PenaltyMap Instrument::penalty() const {
  if (quote == QuoteKind::FxForward) return PenaltyMap::plain();
  if (penalty_map) return *penalty_map;
  return PenaltyMap::band(band_lower, band_upper, band_decay);
}

// ZeroCouponRate: the annually-compounded rate r with (1+r)^τ − 1 == τ·q, i.e. r = (1+τq)^(1/τ) − 1, and
// dr/dq = (1+τq)^(1/τ − 1). Written with exp/log so the AAD scalar types carry the derivative.
template <class Scalar>
Scalar zero_coupon_transform(const Scalar& q, double tau) {
  using std::exp; using std::log;
  return exp(log(Scalar(1.0) + Scalar(tau) * q) / tau) - Scalar(1.0);
}
inline std::pair<double, double> zero_coupon_transform_d(double q, double tau) {  // {r, dr/dq}
  const double base = 1.0 + tau * q;
  const double r = std::exp(std::log(base) / tau) - 1.0;
  return {r, std::exp((1.0 / tau - 1.0) * std::log(base))};
}
// The single fixed accrual τ of a ZeroCouponRate instrument (its fixed leg IS one coupon; anything else is a
// build error, never a silent Σ).
inline double zero_coupon_tau(const Instrument& ins) {
  if (ins.fixed.coupons.size() != 1)
    throw std::invalid_argument("ZeroCouponRate instrument must have exactly ONE fixed coupon (the zero-coupon accrual)");
  return ins.fixed.coupons.front().tau;
}

// Model quote of an instrument, per the design §3 table. `C(role)` maps a curve role index to the
// curve object (anything with `Scalar discount(double)`); a single-curve problem passes a lambda that
// returns its one curve for every role.
// INPUT VALIDATION of one instrument's shape (E3-B12, 2026-09-10): an instrument with an empty leg priced to
// NaN on both the templated and the compiled route with no exception (annuity() returns 0 for an empty leg
// and the par rate divides by it). Called once at engine construction (HybridBundleResidual) and at the
// session seam; `where` names the caller in the message. Only SHAPE is checked here -- curve-index ranges
// need the bundle and are checked by validate_problem (bundle_problem.hpp).
// A QUOTE is four numbers {target, lower, upper, decay} (owner, 2026-09-14). A band (upper > lower) exists only to give the solve
// freedom around its target, so the target must lie INSIDE it: a target outside [lower, upper] leaves the band residual negative
// (or positive) at the edge it lies beyond, where the squared residual turns concave -- the streamed and the cold solve can then
// settle on different fits. Such a quote is REFUSED before any solve, as are an inverted band (upper < lower), a decay outside
// [0, 1] on a real band, and a non-finite number. upper == lower is no band (the plain residual q - target). `where` / `row`
// name the rejected quote; the message is built only on failure, because this runs on every streamed tick.
inline void validate_quote(double target, double lower, double upper, double decay, const char* where, int row) {
  // (An inverted band, upper < lower, fails the target test: no target lies in it. The branch below names it.)
  if (std::isfinite(target) && std::isfinite(lower) && std::isfinite(upper) && std::isfinite(decay) &&
      (upper == lower || (target >= lower && target <= upper && decay >= 0.0 && decay <= 1.0)))
    return;
  const auto num = [](double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.12g", v);
    return std::string(buf);
  };
  std::string what;
  if (!std::isfinite(target) || !std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(decay))
    what = "a non-finite quote (target, lower, upper, decay)";
  else if (upper < lower)
    what = "an inverted band: upper " + num(upper) + " < lower " + num(lower);
  else if (target < lower || target > upper)
    what = "target " + num(target) + " outside its band [" + num(lower) + ", " + num(upper) +
           "] -- a band gives the solve freedom around its target, so the target must lie inside it";
  else
    what = "band decay " + num(decay) + " outside [0, 1]";
  throw std::invalid_argument(std::string(where) + (row >= 0 ? " row " + std::to_string(row) : std::string()) + ": " + what);
}

inline void validate_instrument(const Instrument& ins, const std::string& where) {
  validate_quote(ins.market, ins.band_lower, ins.band_upper, ins.band_decay, where.c_str(), -1);
  const auto fail = [&](const std::string& what) { throw std::invalid_argument(where + ": " + what); };
  if (ins.penalty_map) {
    const PenaltyMap& m = *ins.penalty_map;
    if (ins.quote == QuoteKind::FxForward) fail("an FX forward's residual is the log basis; it cannot carry a penalty map");
    if (m.n < 0 || m.n > PenaltyMap::kMaxBreaks) fail("a penalty map has 0.." + std::to_string(PenaltyMap::kMaxBreaks) + " breakpoints");
    for (int j = 0; j < m.n; ++j) {
      if (!std::isfinite(m.b[j])) fail("a penalty map breakpoint is not finite");
      if (j > 0 && !(m.b[j] > m.b[j - 1])) fail("penalty map breakpoints must be strictly ascending");
    }
    for (int i = 0; i <= m.n; ++i)
      if (!(m.s[i] >= 0.0) || !std::isfinite(m.s[i])) fail("a penalty map slope must be finite and >= 0 (a monotone penalty)");
    if (!std::isfinite(ins.market)) fail("a non-finite target");
  }
  const auto leg = [&](const FloatLeg& l, const char* name) {
    if (l.coupons.empty()) fail(std::string("the ") + name + " leg has no coupons");
    for (const auto& c : l.coupons)
      if (!(c.obs.tau_index > 0.0)) fail(std::string("a ") + name + " coupon has tau_index <= 0");
  };
  switch (ins.quote) {
    case QuoteKind::Rate:
      if (!(ins.obs.tau_index > 0.0)) fail("a Rate observation has tau_index <= 0");
      break;
    case QuoteKind::ParRate:
    case QuoteKind::ZeroCouponRate:
      leg(ins.fwd, "float");
      if (ins.fixed.coupons.empty()) fail("the fixed leg has no coupons");
      break;
    case QuoteKind::ParSpread:
      leg(ins.fwd, "float");
      leg(ins.bench, "benchmark");
      if (ins.fixed.coupons.empty()) fail("the fixed (annuity) leg has no coupons");
      break;
    case QuoteKind::XccyMtmBasis:
      leg(ins.fwd, "float");
      leg(ins.bench, "benchmark");
      if (ins.fixed.coupons.empty()) fail("the fixed (annuity) leg has no coupons");
      if (!(ins.mtm.fx_spot_time >= 0.0) || !std::isfinite(ins.mtm.fx_spot_time))
        fail("the MtM leg needs a finite fx_spot_time >= 0");
      break;
    case QuoteKind::FxForward:
      if (!(ins.fx_time > 0.0)) fail("an FX forward needs fx_time > 0");
      if (!(ins.fx_spot > 0.0)) fail("an FX forward needs fx_spot > 0");
      // fx_time may be BEFORE fx_spot_time: a tom-next forward delivers before spot.
      if (!(ins.fx_spot_time >= 0.0) || !std::isfinite(ins.fx_spot_time)) fail("an FX forward needs a finite fx_spot_time >= 0");
      break;
    case QuoteKind::TurnJump:
      if (ins.turn_curve < 0 || ins.turn_index < 0) fail("a TurnJump needs turn_curve and turn_index >= 0");
      break;
    case QuoteKind::Portfolio:
      if (ins.combination.empty()) fail("a Portfolio quote has no components");
      for (const auto& c : ins.combination) validate_instrument(c.instrument, where + " (portfolio component)");
      break;
    case QuoteKind::Npv: {
      const auto flt = [&](const FloatLeg& l, const char* name) {
        for (const auto& c : l.coupons)
          if (!(c.obs.tau_index > 0.0)) fail(std::string("a ") + name + " coupon has tau_index <= 0");
      };
      flt(ins.fwd, "float");
      flt(ins.bench, "benchmark");
      flt(ins.mtm, "mtm");
      if (ins.fwd.coupons.empty() && ins.bench.coupons.empty() && ins.mtm.coupons.empty() && ins.fixed.coupons.empty())
        fail("an Npv row has no cashflows on any leg");
      if (!ins.mtm.coupons.empty()) {
        if (!ins.mtm.mtm_complete()) fail("an Npv row's MtM leg needs forecast/discount/reset_num/reset_den");
        if (!(ins.mtm.fx_spot_time >= 0.0) || !std::isfinite(ins.mtm.fx_spot_time)) fail("the MtM leg needs a finite fx_spot_time >= 0");
      }
      break;
    }
  }
}

template <class Scalar, class CurveOf>
Scalar instrument_model_quote(const Instrument& ins, const CurveOf& C) {
  switch (ins.quote) {
    case QuoteKind::Portfolio: {
      // Σ weight·model_quote(component). Recursive, so a component may itself be a Portfolio.
      Scalar acc(0.0);
      for (const auto& c : ins.combination)
        acc += Scalar(c.weight) * instrument_model_quote<Scalar>(c.instrument, C);
      return acc;
    }
    case QuoteKind::Rate:
      return pricing::future_rate<Scalar>(ins.obs, ins.convexity, C(ins.forecast));
    case QuoteKind::TurnJump: {
      // The model quote is the raw turn jump δ, read straight off the (turned) curve handle. Linear in
      // x: δ IS a state variable. Only the bundle's CurveHandle exposes turn_jump; a bare ModularCurve
      // never carries a TurnJump instrument, so guard the call so BOTH curve types compile.
      using CurveT = std::decay_t<decltype(C(ins.turn_curve))>;
      if constexpr (has_turn_jump<CurveT>::value)
        return C(ins.turn_curve).turn_jump(ins.turn_index);
      else
        throw std::logic_error("TurnJump instrument requires a turned bundle curve handle");
    }
    case QuoteKind::ParSpread:  // E6.1c: the ONE ParSpread formula lives in cashflows.hpp
      return pricing::par_spread<Scalar>(ins.fwd.coupons, ins.bench.coupons, ins.fixed.coupons,
                                         C(ins.fwd.forecast), C(ins.fwd.discount), C(ins.bench.forecast),
                                         C(ins.bench.discount), C(ins.fixed.discount));
    case QuoteKind::FxForward:
      // FX-forward outright = fx_spot · DF_foreign(fx_time) / DF_domestic(fx_time). This is exactly the
      // machinery that converts a forward foreign cashflow back to the domestic currency at any date.
      if (ins.fx_spot_time == 0.0)  // the t = 0 reading, byte-identical
        return ins.fx_spot * (C(ins.fx_num).discount(ins.fx_time) / C(ins.fx_den).discount(ins.fx_time));
      // O-X3: fx_spot is the quote for the SPOT DATE, so F(fx_spot_time) = fx_spot, i.e. FX_0 = fx_spot·DF_den(t_s)/DF_num(t_s).
      return ins.fx_spot * (C(ins.fx_num).discount(ins.fx_time) / C(ins.fx_den).discount(ins.fx_time)) *
             (C(ins.fx_den).discount(ins.fx_spot_time) / C(ins.fx_num).discount(ins.fx_spot_time));
    case QuoteKind::XccyMtmBasis: {
      // Par basis of a MtM (FX-resettable-notional) xccy swap. fwd = the collateral curve's self-forecast
      // leg paying on its accrual ends (pv telescopes to the exchange pair DF(s0) − DF(eN); build::xccy_mtm_basis,
      // XB1), bench = the foreign-index forecast leg, fixed = the
      // annuity, all discounted on the collateral (pinned) curve; mtm = the resetting funding leg.
      //   b = (pv_self − pv_foreign)/annuity + mtm_leg_pv / (fx_spot · annuity)
      // The funding leg is par (SOFR-flat) so the mtm term ~0, but computing it exercises the resettable-
      // notional coupon and couples the residual to the funding (SOFR) curve.
      const Scalar ann = pricing::annuity<Scalar>(ins.fixed.coupons, C(ins.fixed.discount));
      const Scalar pv_self =
          pricing::float_leg_pv<Scalar>(ins.fwd.coupons, C(ins.fwd.forecast), C(ins.fwd.discount));
      const Scalar pv_fx =
          pricing::float_leg_pv<Scalar>(ins.bench.coupons, C(ins.bench.forecast), C(ins.bench.discount));
      const Scalar mtm = pricing::xccy_mtm_leg_pv<Scalar>(
          ins.mtm.coupons, ins.mtm.fx_spot, C(ins.mtm.forecast), C(ins.mtm.discount),
          C(ins.mtm.reset_num), C(ins.mtm.reset_den), ins.mtm.fx_spot_time);
      if (ins.mtm.fx_spot_time == 0.0) return (pv_self - pv_fx) / ann + mtm / (ins.mtm.fx_spot * ann);
      // O-X3: divide by FX_0 (the today-rate implied by the spot-date quote): the basis is then invariant to fx_spot AND
      // fx_spot_time -- exactly the compiled batch's bare reset ratio, which therefore needs no change.
      const Scalar fx0 = ins.mtm.fx_spot * (C(ins.mtm.reset_den).discount(ins.mtm.fx_spot_time) /
                                            C(ins.mtm.reset_num).discount(ins.mtm.fx_spot_time));
      return (pv_self - pv_fx) / ann + mtm / (fx0 * ann);
    }
    case QuoteKind::ZeroCouponRate:
      return zero_coupon_transform<Scalar>(
          pricing::par_rate<Scalar>(ins.fwd.coupons, ins.fixed.coupons, C(ins.fwd.forecast),
                                    C(ins.fwd.discount), C(ins.fixed.discount)),
          zero_coupon_tau(ins));
    case QuoteKind::ParRate:  // E6.1c: the ONE ParRate formula lives in cashflows.hpp
      return pricing::par_rate<Scalar>(ins.fwd.coupons, ins.fixed.coupons, C(ins.fwd.forecast),
                                       C(ins.fwd.discount), C(ins.fixed.discount));
    case QuoteKind::Npv: {
      // A position's NPV: + the fwd leg, − the bench leg, + the resetting MtM leg (fx_spot and the spot-date
      // roll-back inside xccy_mtm_leg_pv), − the fixed leg of dated amounts Σ DF·τ·scale -- each only when it
      // carries coupons. Accumulated from Scalar(0) like a Portfolio, so an AAD scalar's gradient is seeded by
      // the first curve-dependent leg (Eigen's make_coherent sizes the empty side).
      Scalar v(0.0);
      if (!ins.fwd.coupons.empty())
        v += pricing::float_leg_pv<Scalar>(ins.fwd.coupons, C(ins.fwd.forecast), C(ins.fwd.discount));
      if (!ins.bench.coupons.empty())
        v -= pricing::float_leg_pv<Scalar>(ins.bench.coupons, C(ins.bench.forecast), C(ins.bench.discount));
      if (!ins.mtm.coupons.empty())
        v += pricing::xccy_mtm_leg_pv<Scalar>(ins.mtm.coupons, ins.mtm.fx_spot, C(ins.mtm.forecast), C(ins.mtm.discount),
                                              C(ins.mtm.reset_num), C(ins.mtm.reset_den), ins.mtm.fx_spot_time);
      if (!ins.fixed.coupons.empty()) v -= pricing::annuity<Scalar>(ins.fixed.coupons, C(ins.fixed.discount));
      return v;
    }
  }
  // REVIEW FINDING 3 (2026-09-21): this used to be `case ParRate: default:`, so a NEW QuoteKind priced
  // silently as a par rate. The switch is exhaustive now -- a new kind trips -Wswitch here, the
  // kQuoteKindCount assert below, and this throw if both are somehow ignored.
  throw std::logic_error("instrument_model_quote: unhandled QuoteKind (add it to the switch)");
}

// residual = model_quote - market, in RATE units.
//
// `Rate` is spelled `rate + (convexity - market)` rather than `(rate + convexity) - market`: the two
// are algebraically identical but not bit-identical, and this association is the one the legacy
// futures residual uses, so a legacy future re-expressed as a generic instrument reprices to the last
// bit (design §2's backward-compatibility invariant).
//
// AAD note: a fully-fixed observation (empty sub-periods) makes `Rate` a genuine CONSTANT residual
// row with an empty derivative vector — aad_jacobian() zeroes such rows defensively. The quotient
// transforms never hit this: DF(pay) always carries the derivatives (see float_coupon_pv).
// A banded instrument (band_upper > band_lower) uses the Huber band residual band_residual() (decay-slope
// to mid inside, unit slope outside). The band is NOT applied to FxForward (its residual is already a
// log-basis transform).
//
// `market` is the target quote the residual is measured against. It defaults (overload below) to the
// instrument's stored mid `ins.market` — the calibration case — but the frozen-Newton STREAMER passes the
// LIVE feed instead, so the SAME residual definition drives both cold calibrate and each tick. FX needs
// the market INSIDE the log (ln F_model − ln market), so a live feed cannot be applied by subtracting
// afterward; threading it through here keeps FX (and the band `q − market` term) exact per tick. The band
// bounds stay absolute bid/offer levels, independent of the live market.
// The residual MAP applied to an already-priced model quote q: the FX log basis, or the row's penalty map (plain:
// q − market). What instrument_residual computes after pricing; exposed so a caller that needs BOTH the quote and
// the residual (the AAD block under the streamer's walk) prices once.
template <class Scalar>
Scalar instrument_residual_of_quote(const Instrument& ins, const Scalar& q, double market) {
  if (ins.quote == QuoteKind::FxForward) {
    using std::log;
    return (log(q) - std::log(market)) / ins.fx_time;
  }
  return ins.penalty().residual<Scalar>(q, market);
}

template <class Scalar, class CurveOf>
Scalar instrument_residual(const Instrument& ins, const CurveOf& C, double market) {
  if (ins.quote == QuoteKind::FxForward) {
    // Residual in RATE units (CLAUDE.md §2): the implied-basis discrepancy (ln F_model − ln F_market)/T.
    // A 1bp basis error maps to ~1bp REGARDLESS of tenor, so short-dated forwards are not swamped by 1y.
    using std::log;
    return (log(instrument_model_quote<Scalar>(ins, C)) - std::log(market)) / ins.fx_time;
  }
  const PenaltyMap pm = ins.penalty();
  // Rate keeps its bit-exact `rate + (convexity - market)` association when there is NO band (design §2's
  // backward-compatibility invariant); a banded Rate uses the general q·weight form.
  if (ins.quote == QuoteKind::Rate && pm.is_plain())
    return pricing::rate<Scalar>(ins.obs, C(ins.forecast)) + (ins.convexity - market);
  const Scalar q = instrument_model_quote<Scalar>(ins, C);
  return pm.residual<Scalar>(q, market);  // plain: q − market, bit for bit
}
// Calibration default: residual against the instrument's stored mid. Bit-identical to the pre-override
// code (same value flows into the same expressions), so every existing caller is unchanged.
template <class Scalar, class CurveOf>
Scalar instrument_residual(const Instrument& ins, const CurveOf& C) {
  return instrument_residual<Scalar>(ins, C, ins.market);
}

struct CalibrationProblem {
  // Curve topology (year fractions on the curve day count).
  std::vector<double> meeting_times;
  std::vector<double> back_times;

  // Every calibration instrument is a generic Instrument (design §3). This problem has exactly ONE
  // curve, so every leg role resolves to it and the legs' curve indices are ignored (single_curve_bundle()
  // likewise forces them all to curve 0, so the templated and compiled paths cannot diverge).
  std::vector<Instrument> instruments;

  int n_knots() const { return static_cast<int>(meeting_times.size() + back_times.size()); }
  int n_residuals() const { return static_cast<int>(instruments.size()); }

  // r in rate units for an ARBITRARY curve (anything with `Scalar discount(double)`). RESIDUAL ORDER
  // is the instruments' INSERTION order (the Jacobian rows, the W-cache batches and the warm/streaming
  // feeds all index off it; CompiledResidual fills the same rows).
  template <class Scalar, class Curve>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> price_residuals(const Curve& c) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(n_residuals());
    const auto one_curve = [&c](int) -> const Curve& { return c; };  // every role IS this curve
    int i = 0;
    for (const auto& ins : instruments) r[i++] = instrument_residual<Scalar>(ins, one_curve);
    return r;
  }

  // THE single-curve layout (flat front over the meeting dates + Hermite back), built once here so every
  // consumer (residuals, risk.hpp's operators, tests) derives it from the same place (E6.1c: it used to be
  // re-derived in five headers).
  template <class Scalar>
  curve::ModularCurve<Scalar> make_curve() const {
    return curve::make_modular_curve<Scalar>(curve::flat_hermite(meeting_times, back_times));
  }

  // r(x) with x = the knot forwards of a standalone two-region curve.
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto c = make_curve<Scalar>();
    c.set_forwards(x);
    return price_residuals<Scalar>(c);
  }
};

// D = diag(−∂r_i/∂q_i): how each residual row scales with ITS market quote (risk.hpp / BundleSession share
// this one definition, E6.1c). Identity for a hard pin (r = model − q); an FX forward's residual is
// (ln F − ln q)/T so −∂r/∂q = 1/(q·T); a Huber band's market mid enters only through the decay·(q − m)
// pull on every side of the band (band_residual), so −∂r/∂q = decay. Needs no curve.
inline Eigen::VectorXd residual_market_scale(const std::vector<Instrument>& instruments) {
  Eigen::VectorXd d = Eigen::VectorXd::Ones(static_cast<int>(instruments.size()));
  for (int i = 0; i < d.size(); ++i) {
    const Instrument& ins = instruments[static_cast<std::size_t>(i)];
    if (ins.quote == QuoteKind::FxForward) d[i] = 1.0 / (ins.market * ins.fx_time);
    else d[i] = ins.penalty().target_slope(ins.market);  // the slope at the target: decay inside a band, 1 plain
  }
  return d;
}

// NOTE: fixed-base spread calibration (the former SpreadCalibrationProblem, with its SpreadCurve) was
// TEST-ONLY -- production spreads calibrate jointly through the bundle path (SpreadHandle). Both now live
// in tests/spread_reference.hpp (namespace swaps::testing) so they don't sit in the shipped object model.

}  // namespace swaps::calibration
