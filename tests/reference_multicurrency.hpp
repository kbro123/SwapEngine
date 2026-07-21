#pragma once
// The multi-currency reference bundle builder (plan: composed-tickling-snowglobe). It grows across the
// multi-currency phases; Phase 1 builds the EUR block:
//   curve 0 = ESTR  (outright)          -- pinned by ESTR OIS par swaps (compounded overnight)
//   curve 1 = EUR3M (spread over ESTR)  -- pinned by 3M EURIBOR IRS   (the IBOR/par-coupon path)
//   curve 2 = EUR6M (spread over EUR3M) -- pinned by 6M EURIBOR IRS + 3s6s basis (tenor basis)
// EONIA is demonstrated in the test as a FIXED default spread over ESTR (ESTR + 8.5bp), overridable.
//
// This is the ONLY place currency / index / calendar / convention knowledge lives (CLAUDE.md §1): the
// engine takes extracted dates + accruals + integer curve roles and nothing else. Everything is a
// generic `Instrument` built through QuantLib's own MakeOIS / MakeVanillaSwap (so the EURIBOR coupons
// carry the IborCouponPricer that extract_ibor_obs requires) and the coupon-type-dispatch extractors.
//
// Design mirrors tests/reference_bundle.hpp: build instruments off PLACEHOLDER flat curves (the
// extracted schedules are curve-VALUE-independent -- they need only calendars), extract the generic
// instruments, set x_true, make the market self-consistent from x_true (so the joint/staged solve
// recovers x_true exactly), then wire the real spread-aware CurveHandles in as QuantLib term structures
// so QuantLib can price the SAME instruments off our discount factors for the 1e-10 oracle.

#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "reference_curve.hpp"  // build_market / build_problem (SOFR base), Market, Future, sofr_start/end
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"

namespace swaps::refbuild {

namespace cal = swaps::calibration;

// Builder-only currency vocabulary (the engine sees plain ints via BundleCurveSpec::currency).
enum Ccy { CCY_USD = 0, CCY_EUR = 1 };

struct MultiCcyBundle {
  cal::BundleProblem prob;
  Eigen::VectorXd x_true, x0;
  std::vector<int> off;  // stacked offset of each curve's block in x
  QuantLib::Date today;
  QuantLib::DayCounter dc = QuantLib::Actual365Fixed();

  // Curve role indices (into prob.curves / the stacked x). -1 = not present in this bundle.
  int ESTR = -1, EUR3M = -1, EUR6M = -1;
  int SOFR = -1, FF = -1;
  int EURUSD = -1;      // EUR collateralized in USD = ESTR + EURUSD xccy basis (a spread curve)
  double fx_spot = 0.0;  // EURUSD spot (USD per EUR), for the FX-forward no-arbitrage oracle

  // Per-index DEFAULT discount curve, resolved at build time; a per-trade override just passes a
  // different discount role into the leg builder (see make_ois_inst / make_ibor_inst).
  std::map<int, int> default_discount;

  // QuantLib state kept alive for the lifetime of the bundle (the term structures are linked into the
  // index handles the swaps price off).
  std::vector<QuantLib::RelinkableHandle<QuantLib::YieldTermStructure>> h;
  std::vector<QuantLib::ext::shared_ptr<QuantLib::YieldTermStructure>> ts;
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> curve_handles;
  QuantLib::ext::shared_ptr<QuantLib::Estr> estr;
  QuantLib::ext::shared_ptr<QuantLib::Euribor3M> eur3m;
  QuantLib::ext::shared_ptr<QuantLib::Euribor6M> eur6m;
  QuantLib::ext::shared_ptr<QuantLib::Sofr> sofr;
  QuantLib::ext::shared_ptr<QuantLib::FedFunds> fedfunds;

  int n_curves() const { return static_cast<int>(prob.curves.size()); }
  int offset(int c) const { return off[c]; }
};

// Build the EUR block. `eval` is the evaluation date.
inline MultiCcyBundle build_eur_bundle(QuantLib::Date eval = QuantLib::Date(15, QuantLib::July, 2026)) {
  using namespace QuantLib;
  MultiCcyBundle b;
  b.today = eval;
  Settings::instance().evaluationDate() = eval;
  const DayCounter dc = b.dc;
  auto t = [&](const Date& d) { return dc.yearFraction(eval, d); };

  // ---- Curve roles + specs (currency tag = EUR on every curve here). ----
  b.ESTR = 0;
  b.EUR3M = 1;
  b.EUR6M = 2;
  b.prob.curves.resize(3);
  // ESTR: a single flat front segment [0, 0.5] then a smooth Hermite back at the OIS pillars.
  const std::vector<double> estr_meet{0.5};
  const std::vector<double> estr_back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  // EUR3M / EUR6M: forward-SPREAD curves over their base, same front + a shorter back.
  const std::vector<double> eur_meet{0.5};
  const std::vector<double> eur_back{1, 2, 3, 5, 7, 10};
  b.prob.curves[b.ESTR] = {estr_meet, estr_back, -1, CCY_EUR};      // outright
  b.prob.curves[b.EUR3M] = {eur_meet, eur_back, b.ESTR, CCY_EUR};   // spread over ESTR
  b.prob.curves[b.EUR6M] = {eur_meet, eur_back, b.EUR3M, CCY_EUR};  // spread over EUR3M

  b.off.assign(3, 0);
  for (int c = 1; c < 3; ++c) b.off[c] = b.off[c - 1] + b.prob.curves[c - 1].n_knots();
  const int N = b.off[2] + b.prob.curves[2].n_knots();

  // Per-index default discount: every EUR index discounts on ESTR (CSA convention, overridable).
  b.default_discount = {{b.ESTR, b.ESTR}, {b.EUR3M, b.ESTR}, {b.EUR6M, b.ESTR}};

  // ---- QuantLib indices on placeholder handles (relinked to the real curves at the end). ----
  b.h.resize(3);
  for (auto& hh : b.h) hh.linkTo(ext::make_shared<FlatForward>(eval, 0.03, dc, Continuous));
  b.estr = ext::make_shared<Estr>(b.h[b.ESTR]);
  b.eur3m = ext::make_shared<Euribor3M>(b.h[b.EUR3M]);
  b.eur6m = ext::make_shared<Euribor6M>(b.h[b.EUR6M]);

  // ---- x_true: EUR ~3% ESTR forwards; small forward spreads for the tenor-basis curves. ----
  b.x_true.resize(N);
  auto fill = [&](int c, double level, double slope) {
    const int nk = b.prob.curves[c].n_knots();
    for (int i = 0; i < nk; ++i) b.x_true[b.off[c] + i] = level + slope * i;
  };
  fill(b.ESTR, 0.0300, 0.0004);    // ESTR forward level ~3%, gently upward
  fill(b.EUR3M, 0.0012, 0.00003);  // EURIBOR-3M / ESTR basis ~12bp
  fill(b.EUR6M, 0.0006, 0.00002);  // 3s6s basis ~6bp on top

  // ---- Instrument builders (roles + default-discount resolution live HERE, not in the engine). ----
  const int disc_estr = b.default_discount.at(b.ESTR);

  // ESTR OIS par swap (compounded overnight): forecast = discount = ESTR by default. `pay_lag` exercises
  // the payment-delay path (pay date != accrual end) end-to-end.
  auto make_ois_inst = [&](const Period& tenor, int fc, int disc, int pay_lag) {
    auto o = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.estr, 0.03).withDiscountingTermStructure(b.h[disc]).withPaymentLag(pay_lag));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParRate;
    ins.fwd = {swaps::qlx::extract_float_leg(o->overnightLeg(), eval, dc), fc, disc};
    ins.fixed = {swaps::qlx::extract_fixed_leg(o->fixedLeg(), eval, dc), disc};
    return std::make_pair(ins, o);
  };

  // EURIBOR IRS (fixed 30/360 annual vs float ACT/360 quarterly[3M]/semi[6M]): the float leg forecasts
  // its EURIBOR curve, discounts ESTR. MakeVanillaSwap attaches the IborCouponPricer extract_ibor_obs
  // needs. Returns a ParRate instrument (fixed rate is irrelevant to the par quote and is not extracted).
  auto make_ibor_inst = [&](const Period& tenor, const ext::shared_ptr<IborIndex>& idx, int fc, int disc) {
    auto s = ext::shared_ptr<VanillaSwap>(
        MakeVanillaSwap(tenor, idx, 0.03)
            .withDiscountingTermStructure(b.h[disc])
            .withFixedLegDayCount(Thirty360(Thirty360::BondBasis))
            .withFixedLegTenor(1 * Years)
            .withFixedLegConvention(ModifiedFollowing)
            .withFixedLegCalendar(TARGET())
            .withFloatingLegCalendar(TARGET()));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParRate;
    ins.fwd = {swaps::qlx::extract_float_leg(s->floatingLeg(), eval, dc), fc, disc};
    ins.fixed = {swaps::qlx::extract_fixed_leg(s->fixedLeg(), eval, dc), disc};
    return std::make_pair(ins, s);
  };

  // 3s6s tenor basis as a ParSpread: fwd = 3M leg (EUR3M), bench = 6M leg (EUR6M), ESTR-discounted, so
  // the quoted spread is on the 3M leg. Both legs are independent float legs (design §6.7).
  auto make_3s6s_inst = [&](const Period& tenor, int disc) {
    auto s3 = ext::shared_ptr<VanillaSwap>(
        MakeVanillaSwap(tenor, b.eur3m, 0.03).withDiscountingTermStructure(b.h[disc]));
    auto s6 = ext::shared_ptr<VanillaSwap>(
        MakeVanillaSwap(tenor, b.eur6m, 0.03).withDiscountingTermStructure(b.h[disc]));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParSpread;
    ins.fwd = {swaps::qlx::extract_float_leg(s3->floatingLeg(), eval, dc), b.EUR3M, disc};
    ins.bench = {swaps::qlx::extract_float_leg(s6->floatingLeg(), eval, dc), b.EUR6M, disc};
    ins.fixed = {swaps::qlx::extract_fixed_leg(s3->fixedLeg(), eval, dc), disc};
    return std::make_tuple(ins, s3, s6);
  };

  // Keep the QuantLib swaps alive (they hold the schedules the extracted legs mirror) for the oracle.
  std::vector<ext::shared_ptr<OvernightIndexedSwap>> ois_keep;
  std::vector<ext::shared_ptr<VanillaSwap>> vs_keep;

  // ESTR pillars: 6M (front) + the back tenors. One instrument per knot (square, identifiable). The 6M
  // OIS carries a 2-business-day payment delay to exercise that path; the rest use no lag.
  {
    auto [ins, o] = make_ois_inst(6 * Months, b.ESTR, disc_estr, 2);
    b.prob.instruments.push_back(ins);
    ois_keep.push_back(o);
  }
  for (double T : estr_back) {
    auto [ins, o] = make_ois_inst(Period(static_cast<int>(T), Years), b.ESTR, disc_estr, 0);
    b.prob.instruments.push_back(ins);
    ois_keep.push_back(o);
  }

  // EUR3M pillars: one 3M-EURIBOR IRS per EUR3M knot.
  {
    auto [ins, s] = make_ibor_inst(6 * Months, b.eur3m, b.EUR3M, disc_estr);
    b.prob.instruments.push_back(ins);
    vs_keep.push_back(s);
  }
  for (double T : eur_back) {
    auto [ins, s] = make_ibor_inst(Period(static_cast<int>(T), Years), b.eur3m, b.EUR3M, disc_estr);
    b.prob.instruments.push_back(ins);
    vs_keep.push_back(s);
  }

  // EUR6M pillars: one 6M-EURIBOR IRS per EUR6M knot, plus a few 3s6s basis (over-determined -- fine
  // with a self-consistent market, and more realistic).
  {
    auto [ins, s] = make_ibor_inst(6 * Months, b.eur6m, b.EUR6M, disc_estr);
    b.prob.instruments.push_back(ins);
    vs_keep.push_back(s);
  }
  for (double T : eur_back) {
    auto [ins, s] = make_ibor_inst(Period(static_cast<int>(T), Years), b.eur6m, b.EUR6M, disc_estr);
    b.prob.instruments.push_back(ins);
    vs_keep.push_back(s);
  }
  for (double T : {2.0, 5.0, 10.0}) {
    auto [ins, s3, s6] = make_3s6s_inst(Period(static_cast<int>(T), Years), disc_estr);
    b.prob.instruments.push_back(ins);
    vs_keep.push_back(s3);
    vs_keep.push_back(s6);
  }

  // ---- Wire the real spread-aware CurveHandles in as QuantLib term structures (for the oracle), and
  // make the market self-consistent from x_true. ----
  b.curve_handles = cal::build_bundle_curves<double>(
      b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i]; });
  b.ts.resize(3);
  for (int c = 0; c < 3; ++c) {
    auto tsc = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
        eval, dc, b.curve_handles[c].get());
    tsc->enableExtrapolation();
    b.ts[c] = tsc;
    b.h[c].linkTo(tsc);
  }
  for (auto& o : ois_keep) o->deepUpdate();
  for (auto& s : vs_keep) s->deepUpdate();

  // Self-consistent market so joint & staged solves recover x_true exactly.
  const Eigen::VectorXd r0 = b.prob.residuals<double>(b.x_true);
  for (int i = 0; i < static_cast<int>(b.prob.instruments.size()); ++i) b.prob.instruments[i].market += r0[i];

  // Flat per-curve start near each block's level.
  b.x0.resize(N);
  b.x0.segment(b.off[b.ESTR], b.prob.curves[b.ESTR].n_knots()).setConstant(0.030);
  b.x0.segment(b.off[b.EUR3M], b.prob.curves[b.EUR3M].n_knots()).setConstant(0.0012);
  b.x0.segment(b.off[b.EUR6M], b.prob.curves[b.EUR6M].n_knots()).setConstant(0.0006);
  return b;
}

// ===============================================================================================
// USD block (Phase 2): SOFR (outright) + Fed Funds (spread over SOFR) + PRIME (default fixed spread).
// ===============================================================================================
// FED FUNDS conventions (user-confirmed): there is ONE EFFR fixing per day. FF-OIS swaps COMPOUND it
// (identical mechanism to SOFR OIS); FF 1M futures ARITHMETIC-AVERAGE the same daily fixing (identical
// shape to our 1M SOFR averaging future). So the FF curve is a spread over SOFR pinned by FF/SOFR
// compounded basis swaps; the FF averaging future is validated separately vs QuantLib's
// OvernightIndexFuture (RateAveraging::Simple). All on QuantLib's real FedFunds index
// (UnitedStates(FederalReserve) calendar, ACT/360).

// A fully-FORECAST arithmetic-average overnight future on ANY overnight index (generalizes
// reference_curve.hpp avg_future_obs, which is hard-wired to SOFR): one sub-period per business day
// [fixingDate, d2] with weight accr/yf(fixingDate,d2), mirroring QuantLib 1.35
// OvernightIndexFuture::averagedRate() EXACTLY. Requires start > today (no realized prefix) so no
// fixing history is needed.
inline px::RateObservation avg_future_obs_idx(const QuantLib::ext::shared_ptr<QuantLib::OvernightIndex>& idx,
                                              const QuantLib::Date& today, const QuantLib::DayCounter& curveDc,
                                              const QuantLib::Date& start, const QuantLib::Date& end) {
  using namespace QuantLib;
  const Calendar fcal = idx->fixingCalendar();
  const DayCounter idc = idx->dayCounter();
  std::vector<std::pair<Date, Date>> subs;
  std::vector<double> weights;
  Date fixingDate = fcal.adjust(start, Preceding);
  for (Date d1 = start; d1 < end;) {
    const Date d2 = fcal.advance(d1, 1, Days);
    const Date d2cap = std::min(d2, end);
    const double accr = idc.yearFraction(d1, d2cap);
    QL_REQUIRE(fixingDate >= today, "avg_future_obs_idx expects a fully-forecast future (start > today)");
    subs.emplace_back(fixingDate, d2);
    weights.push_back(accr / idc.yearFraction(fixingDate, d2));
    fixingDate = d1 = d2;
  }
  bool all_one = true;
  for (double w : weights) if (w != 1.0) { all_one = false; break; }
  if (all_one) weights.clear();
  return swaps::qlx::make_observation(subs, 0.0, idc.yearFraction(start, end), today, curveDc, weights);
}

// Build the USD block: SOFR reference market (reused, curve 0) + Fed Funds spread (curve 1). PRIME is
// demonstrated as a fixed default spread over FF in the test (like EONIA), so it is not a calibrated
// curve here. `n_basis`-style extension is not needed; this is the canonical two-curve USD bundle.
inline MultiCcyBundle build_usd_bundle() {
  using namespace QuantLib;
  MultiCcyBundle b;

  // SOFR base: the full reference market (6 FOMC meetings + 12x1M/8x3M futures + par swaps). build_market
  // sets the evaluation date (rm::evaluation_date) and seeds SOFR fixings.
  b.h.resize(2);
  Market mk = build_market(b.h[0]);  // h[0] = SOFR handle (relinked below)
  b.today = mk.today;
  b.dc = mk.dc;
  b.sofr = mk.sofr;
  const DayCounter dc = mk.dc;
  const cal::CalibrationProblem sofr_prob = build_problem(mk);

  b.SOFR = 0;
  b.FF = 1;
  b.prob.curves.resize(2);
  b.prob.curves[b.SOFR] = {mk.meeting_times, mk.back_times, -1, CCY_USD};  // outright

  // FF: spread over SOFR, a flat front + Hermite back at the basis-swap pillars.
  const std::vector<double> ff_meet{0.5};
  const std::vector<double> ff_back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  b.prob.curves[b.FF] = {ff_meet, ff_back, b.SOFR, CCY_USD};

  b.off = {0, b.prob.curves[b.SOFR].n_knots()};
  const int N = b.off[b.FF] + b.prob.curves[b.FF].n_knots();

  // Fed Funds discounts on SOFR (CSA default), overridable per trade.
  b.default_discount = {{b.SOFR, b.SOFR}, {b.FF, b.SOFR}};
  const int disc_sofr = b.default_discount.at(b.SOFR);

  // x_true: SOFR forwards from the reference curve; a small FF-SOFR forward spread (~3bp).
  b.x_true.resize(N);
  {
    std::vector<double> sx(rm::reference_front_forwards.begin(), rm::reference_front_forwards.end());
    sx.insert(sx.end(), rm::reference_back_forwards.begin(), rm::reference_back_forwards.end());
    QL_REQUIRE(static_cast<int>(sx.size()) == b.prob.curves[b.SOFR].n_knots(), "SOFR knot/forward mismatch");
    for (int i = 0; i < static_cast<int>(sx.size()); ++i) b.x_true[i] = sx[i];
  }
  for (int i = 0; i < b.prob.curves[b.FF].n_knots(); ++i) b.x_true[b.off[b.FF] + i] = 0.0003 + 0.00002 * i;

  // Fed Funds index on its own handle.
  b.fedfunds = ext::make_shared<FedFunds>(b.h[b.FF]);

  // SOFR instruments (curve 0): the reference market's generic Instruments (roles already curve 0).
  for (const auto& ins : sofr_prob.instruments) b.prob.instruments.push_back(ins);

  // FF/SOFR compounded basis (ParSpread): fwd = FF-OIS overnight leg (forecast FF), bench = SOFR-OIS
  // overnight leg (forecast SOFR), SOFR-discounted. The quoted spread is on the FF leg.
  std::vector<ext::shared_ptr<OvernightIndexedSwap>> keep;
  auto basis_inst = [&](const Period& tenor) {
    auto ffo = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.fedfunds, 0.03).withDiscountingTermStructure(b.h[disc_sofr]));
    auto so = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.sofr, 0.03).withDiscountingTermStructure(b.h[disc_sofr]));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParSpread;
    ins.fwd = {swaps::qlx::extract_float_leg(ffo->overnightLeg(), b.today, dc), b.FF, disc_sofr};
    ins.bench = {swaps::qlx::extract_float_leg(so->overnightLeg(), b.today, dc), b.SOFR, disc_sofr};
    ins.fixed = {swaps::qlx::extract_fixed_leg(so->fixedLeg(), b.today, dc), disc_sofr};
    keep.push_back(ffo);
    keep.push_back(so);
    return ins;
  };
  b.prob.instruments.push_back(basis_inst(6 * Months));
  for (double T : ff_back) b.prob.instruments.push_back(basis_inst(Period(static_cast<int>(T), Years)));

  // Wire the real curves in and self-consistent market.
  b.curve_handles = cal::build_bundle_curves<double>(
      b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i]; });
  b.ts.resize(2);
  for (int c = 0; c < 2; ++c) {
    auto tsc = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
        b.today, dc, b.curve_handles[c].get());
    tsc->enableExtrapolation();
    b.ts[c] = tsc;
    b.h[c].linkTo(tsc);
  }
  for (auto& s : mk.swaps) s->deepUpdate();
  for (auto& s : keep) s->deepUpdate();

  const Eigen::VectorXd r0 = b.prob.residuals<double>(b.x_true);
  for (int i = 0; i < static_cast<int>(b.prob.instruments.size()); ++i) b.prob.instruments[i].market += r0[i];

  b.x0.resize(N);
  b.x0.head(b.prob.curves[b.SOFR].n_knots()).setConstant(0.043);
  b.x0.segment(b.off[b.FF], b.prob.curves[b.FF].n_knots()).setConstant(0.0003);
  return b;
}

// ===============================================================================================
// Cross-currency block (Phase 3): the EUR-collateralized-in-USD discount curve, from CONSTANT-notional
// EURUSD OIS xccy basis swaps. This is where SOFR and EUR curves share ONE BundleProblem.
// ===============================================================================================
// Market: post-LIBOR EURUSD xccy is SOFR (USD) vs ESTR (EUR) + basis. The EUR-collateralized-in-USD
// discount curve is DF_c(t) = DF_ESTR(t)·exp(-∫ basis), i.e. exactly a SpreadHandle(base = ESTR) whose
// spread knots are the xccy basis. Its no-arbitrage FX forward is F(t) = S · DF_c(t) / DF_SOFR(t).
//
// KEY (derived first-principles): for a constant-notional OIS xccy basis with matched principals at
// spot, the USD SOFR-flat leg discounted on SOFR is PAR (contributes 0) and the FX spot CANCELS, so the
// par basis spread reduces to a EUR-only multi-curve identity:
//   b = [ (1 − DF_c(T)) − Σ ESTR_fwd·τ·DF_c(t) ] / Σ τ·DF_c(t)
//     = par_spread(fwd = ESTR-forecast leg on DF_c, bench = DF_c-self-forecast leg on DF_c, annuity DF_c)
// So the xccy basis rides the EXISTING ParSpread machinery with NO engine change and NO FX scale (SOFR
// couples into the calibration only under MtM resets -- Phase 4). SOFR is present for the FX oracle.
inline MultiCcyBundle build_xccy_bundle(QuantLib::Date eval = QuantLib::Date(15, QuantLib::July, 2026)) {
  using namespace QuantLib;
  MultiCcyBundle b;
  b.today = eval;
  Settings::instance().evaluationDate() = eval;
  const DayCounter dc = b.dc;
  b.fx_spot = 1.10;  // EURUSD spot, USD per EUR

  b.SOFR = 0;
  b.ESTR = 1;
  b.EURUSD = 2;  // EUR-in-USD = ESTR + xccy basis
  b.prob.curves.resize(3);
  const std::vector<double> meet{0.5};
  const std::vector<double> ois_back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  const std::vector<double> xccy_back{1, 2, 3, 5, 7, 10};
  b.prob.curves[b.SOFR] = {meet, ois_back, -1, CCY_USD};        // outright
  b.prob.curves[b.ESTR] = {meet, ois_back, -1, CCY_EUR};        // outright
  b.prob.curves[b.EURUSD] = {meet, xccy_back, b.ESTR, CCY_EUR}; // spread over ESTR (the xccy basis)

  b.off = {0, b.prob.curves[b.SOFR].n_knots(),
           b.prob.curves[b.SOFR].n_knots() + b.prob.curves[b.ESTR].n_knots()};
  const int N = b.off[b.EURUSD] + b.prob.curves[b.EURUSD].n_knots();

  // Discount defaults: USD-collateralized USD on SOFR, EUR on ESTR, EUR-collateralized-in-USD on itself.
  b.default_discount = {{b.SOFR, b.SOFR}, {b.ESTR, b.ESTR}, {b.EURUSD, b.EURUSD}};

  b.x_true.resize(N);
  auto fill = [&](int c, double level, double slope) {
    const int nk = b.prob.curves[c].n_knots();
    for (int i = 0; i < nk; ++i) b.x_true[b.off[c] + i] = level + slope * i;
  };
  fill(b.SOFR, 0.0430, 0.0004);
  fill(b.ESTR, 0.0300, 0.0004);
  fill(b.EURUSD, -0.0015, 0.00002);  // xccy basis ~ -15bp (EUR-in-USD forwards below ESTR)

  b.h.resize(3);
  for (auto& hh : b.h) hh.linkTo(ext::make_shared<FlatForward>(eval, 0.03, dc, Continuous));
  b.sofr = ext::make_shared<Sofr>(b.h[b.SOFR]);
  b.estr = ext::make_shared<Estr>(b.h[b.ESTR]);

  std::vector<ext::shared_ptr<OvernightIndexedSwap>> keep;
  // A plain OIS par-rate instrument pinning `fc` (== discount), for the SOFR and ESTR base curves.
  auto ois_par = [&](const ext::shared_ptr<OvernightIndex>& idx, const Period& tenor, int role) {
    auto o = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, idx, 0.03).withDiscountingTermStructure(b.h[role]));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParRate;
    ins.fwd = {swaps::qlx::extract_float_leg(o->overnightLeg(), eval, dc), role, role};
    ins.fixed = {swaps::qlx::extract_fixed_leg(o->fixedLeg(), eval, dc), role};
    keep.push_back(o);
    return ins;
  };
  // Constant-notional EURUSD OIS xccy basis as a ParSpread: fwd = ESTR-forecast leg on EUR-in-USD,
  // bench = EUR-in-USD-self-forecast leg on EUR-in-USD, annuity on EUR-in-USD. Model quote == the
  // closed-form par basis spread b above.
  auto xccy_basis = [&](const Period& tenor) {
    auto o = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.estr, 0.03).withDiscountingTermStructure(b.h[b.EURUSD]));
    const auto leg = swaps::qlx::extract_float_leg(o->overnightLeg(), eval, dc);
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParSpread;
    // fwd forecasts the PINNED curve (EUR-in-USD, self) so primary_curve() assigns this instrument to the
    // EUR-in-USD block (not ESTR); bench forecasts ESTR. par_spread = (pv_bench - pv_fwd)/annuity = -b.
    ins.fwd = {leg, b.EURUSD, b.EURUSD};  // forecast EUR-in-USD (self) -- the pinned curve
    ins.bench = {leg, b.ESTR, b.EURUSD};  // forecast ESTR, discount EUR-in-USD
    ins.fixed = {swaps::qlx::extract_fixed_leg(o->fixedLeg(), eval, dc), b.EURUSD};
    ins.pv_currency = CCY_USD;  // a EURUSD xccy quote (the numeraire is USD); ignored by the ParSpread math
    keep.push_back(o);
    return ins;
  };

  // SOFR + ESTR base pillars (each pinned by its own OIS), then the EUR-in-USD spread by xccy basis.
  for (double T : {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0}) {
    const Period p = (T < 1.0) ? Period(6, Months) : Period(static_cast<int>(T), Years);
    b.prob.instruments.push_back(ois_par(b.sofr, p, b.SOFR));
    b.prob.instruments.push_back(ois_par(b.estr, p, b.ESTR));
  }
  for (double T : {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) {
    const Period p = (T < 1.0) ? Period(6, Months) : Period(static_cast<int>(T), Years);
    b.prob.instruments.push_back(xccy_basis(p));
  }

  b.curve_handles = cal::build_bundle_curves<double>(
      b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i]; });
  b.ts.resize(3);
  for (int c = 0; c < 3; ++c) {
    auto tsc = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
        eval, dc, b.curve_handles[c].get());
    tsc->enableExtrapolation();
    b.ts[c] = tsc;
    b.h[c].linkTo(tsc);
  }
  for (auto& s : keep) s->deepUpdate();

  const Eigen::VectorXd r0 = b.prob.residuals<double>(b.x_true);
  for (int i = 0; i < static_cast<int>(b.prob.instruments.size()); ++i) b.prob.instruments[i].market += r0[i];

  b.x0.resize(N);
  b.x0.segment(b.off[b.SOFR], b.prob.curves[b.SOFR].n_knots()).setConstant(0.043);
  b.x0.segment(b.off[b.ESTR], b.prob.curves[b.ESTR].n_knots()).setConstant(0.030);
  b.x0.segment(b.off[b.EURUSD], b.prob.curves[b.EURUSD].n_knots()).setConstant(-0.0015);
  return b;
}

// ===============================================================================================
// The DESK cross-currency build (Phase 5): the EUR-collateralized-in-USD discount curve from FX FORWARD
// POINTS (T/N, S/N, 1w..1y -- the short end) then MtM xccy basis swaps (the long end). Pure rates, no
// FX-vol model. The curve + SOFR give F(t) = S·DF_c(t)/DF_SOFR(t) for ANY date -- i.e. convert a forward
// EUR cashflow back to USD at any date. Here EUR-in-USD genuinely depends on BOTH ESTR (its spread base)
// and SOFR (the FX-forward denominator / the funding leg), so the dependency graph solves it last.
inline MultiCcyBundle build_xccy_fx_bundle(QuantLib::Date eval = QuantLib::Date(15, QuantLib::July, 2026)) {
  using namespace QuantLib;
  MultiCcyBundle b;
  b.today = eval;
  Settings::instance().evaluationDate() = eval;
  const DayCounter dc = b.dc;
  const double S = 1.10;  // EURUSD spot, USD per EUR
  b.fx_spot = S;
  auto t = [&](const Date& d) { return dc.yearFraction(eval, d); };

  b.SOFR = 0;
  b.ESTR = 1;
  b.EURUSD = 2;
  b.prob.curves.resize(3);
  const std::vector<double> meet{0.5}, ois_back{1, 2, 3, 5, 7, 10};
  const std::vector<double> xccy_meet{0.25, 0.5}, xccy_back{1, 2, 3, 5, 7, 10};
  b.prob.curves[b.SOFR] = {meet, ois_back, -1, CCY_USD};
  b.prob.curves[b.ESTR] = {meet, ois_back, -1, CCY_EUR};
  b.prob.curves[b.EURUSD] = {xccy_meet, xccy_back, b.ESTR, CCY_EUR};  // ESTR + xccy basis
  b.off = {0, b.prob.curves[b.SOFR].n_knots(),
           b.prob.curves[b.SOFR].n_knots() + b.prob.curves[b.ESTR].n_knots()};
  const int N = b.off[b.EURUSD] + b.prob.curves[b.EURUSD].n_knots();
  b.default_discount = {{b.SOFR, b.SOFR}, {b.ESTR, b.ESTR}, {b.EURUSD, b.EURUSD}};

  b.x_true.resize(N);
  auto fill = [&](int c, double level, double slope) {
    for (int i = 0; i < b.prob.curves[c].n_knots(); ++i) b.x_true[b.off[c] + i] = level + slope * i;
  };
  fill(b.SOFR, 0.0430, 0.0004);
  fill(b.ESTR, 0.0300, 0.0004);
  fill(b.EURUSD, -0.0015, 0.00002);

  b.h.resize(3);
  for (auto& hh : b.h) hh.linkTo(ext::make_shared<FlatForward>(eval, 0.03, dc, Continuous));
  b.sofr = ext::make_shared<Sofr>(b.h[b.SOFR]);
  b.estr = ext::make_shared<Estr>(b.h[b.ESTR]);

  std::vector<ext::shared_ptr<OvernightIndexedSwap>> keep;
  auto ois_par = [&](const ext::shared_ptr<OvernightIndex>& idx, const Period& tenor, int role) {
    auto o = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, idx, 0.03).withDiscountingTermStructure(b.h[role]));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParRate;
    ins.fwd = {swaps::qlx::extract_float_leg(o->overnightLeg(), eval, dc), role, role};
    ins.fixed = {swaps::qlx::extract_fixed_leg(o->fixedLeg(), eval, dc), role};
    keep.push_back(o);
    return ins;
  };
  // FX FORWARD POINT: pins EUR-in-USD (fx_num) against SOFR (fx_den) at its delivery time.
  auto fx_fwd = [&](double fx_time) {
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::FxForward;
    ins.fx_num = b.EURUSD;
    ins.fx_den = b.SOFR;
    ins.fx_spot = S;
    ins.fx_time = fx_time;
    ins.pv_currency = CCY_USD;
    return ins;
  };
  // MtM xccy basis swap: EUR self leg (fwd, primary), EUR ESTR leg (bench), annuity, + USD SOFR funding
  // leg with an FX-resetting notional (mtm). All EUR legs discount on EUR-in-USD; the funding leg on SOFR.
  auto mtm_basis = [&](const Period& tenor) {
    auto oe = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.estr, 0.03).withDiscountingTermStructure(b.h[b.EURUSD]));
    auto os = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.sofr, 0.03).withDiscountingTermStructure(b.h[b.SOFR]));
    const auto eleg = swaps::qlx::extract_float_leg(oe->overnightLeg(), eval, dc);
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::XccyMtmBasis;
    ins.fwd = {eleg, b.EURUSD, b.EURUSD};  // self-forecast (primary = EUR-in-USD)
    ins.bench = {eleg, b.ESTR, b.EURUSD};  // ESTR forecast
    ins.fixed = {swaps::qlx::extract_fixed_leg(oe->fixedLeg(), eval, dc), b.EURUSD};
    ins.mtm = {swaps::qlx::extract_float_leg(os->overnightLeg(), eval, dc), b.SOFR, b.SOFR};
    ins.mtm.reset_num = b.EURUSD;  // FX-forward notional numerator = EUR-in-USD
    ins.mtm.reset_den = b.SOFR;    // denominator = SOFR
    ins.mtm.fx_spot = S;
    ins.pv_currency = CCY_USD;
    keep.push_back(oe);
    keep.push_back(os);
    return ins;
  };

  // SOFR + ESTR base pillars.
  for (double T : {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) {
    const Period p = (T < 1.0) ? Period(6, Months) : Period(static_cast<int>(T), Years);
    b.prob.instruments.push_back(ois_par(b.sofr, p, b.SOFR));
    b.prob.instruments.push_back(ois_par(b.estr, p, b.ESTR));
  }
  // FX forward points (short end): T/N, S/N, 1w, 2w, 3w, 1m, 2m, 3m, 6m, 1y -- the standard strip.
  const Calendar fxcal = JointCalendar(TARGET(), UnitedStates(UnitedStates::Settlement));
  const Date spot = fxcal.advance(eval, 2, Days);
  std::vector<double> fx_times{t(fxcal.advance(eval, 1, Days)),    // T/N (tom-next)
                               t(fxcal.advance(spot, 1, Days))};   // S/N (spot-next)
  for (const Period& p : {Period(1, Weeks), Period(2, Weeks), Period(3, Weeks), Period(1, Months),
                          Period(2, Months), Period(3, Months), Period(6, Months), Period(1, Years)})
    fx_times.push_back(t(fxcal.advance(spot, p)));
  for (double ft : fx_times) b.prob.instruments.push_back(fx_fwd(ft));
  // MtM xccy basis swaps (long end).
  for (double T : {2.0, 3.0, 5.0, 7.0, 10.0})
    b.prob.instruments.push_back(mtm_basis(Period(static_cast<int>(T), Years)));

  // Real curves in, and self-consistent market = the model quote at x_true (works for the FX-forward log
  // residual too, where `market += r0` would take log(0)).
  b.curve_handles = cal::build_bundle_curves<double>(
      b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i]; });
  b.ts.resize(3);
  for (int c = 0; c < 3; ++c) {
    auto tsc = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
        eval, dc, b.curve_handles[c].get());
    tsc->enableExtrapolation();
    b.ts[c] = tsc;
    b.h[c].linkTo(tsc);
  }
  for (auto& s : keep) s->deepUpdate();
  const auto curve_of = [&](int i) -> const cal::CurveHandle<double>& { return *b.curve_handles[i]; };
  for (auto& ins : b.prob.instruments)
    ins.market = cal::instrument_model_quote<double>(ins, curve_of);

  b.x0.resize(N);
  b.x0.segment(b.off[b.SOFR], b.prob.curves[b.SOFR].n_knots()).setConstant(0.043);
  b.x0.segment(b.off[b.ESTR], b.prob.curves[b.ESTR].n_knots()).setConstant(0.030);
  b.x0.segment(b.off[b.EURUSD], b.prob.curves[b.EURUSD].n_knots()).setConstant(-0.0015);
  return b;
}

// ===============================================================================================
// The PROPER EUR rate-curve build: three COUPLED curves (ESTR / EURIBOR-3M / EURIBOR-6M) that form a
// dependency CYCLE and calibrate JOINTLY as one SCC (plan: composed-tickling-snowglobe). Desk build:
//   ESTR : 1M+3M €STR futures front, ESTR/3M-EURIBOR basis beyond 3y (no direct long ESTR OIS)
//   EUR3M: 3M EURIBOR futures front, 3s6s (3M-vs-6M) basis beyond 3y  (spread over ESTR)
//   EUR6M: single-period 3s6s front, outright 6M EURIBOR swaps beyond 3y  (spread over EUR3M)
// Each basis pins its fwd.forecast curve, so the edges close a full cycle ESTR<->EUR3M<->EUR6M.
// Futures carry ZERO convexity here (the convexity MODEL is orthogonal and already validated for SOFR;
// hull_white_convexity can be wired in exactly as reference_curve.hpp does).
// ===============================================================================================
inline MultiCcyBundle build_eur_curves(QuantLib::Date eval = QuantLib::Date(8, QuantLib::July, 2026)) {
  using namespace QuantLib;
  MultiCcyBundle b;
  b.today = eval;
  Settings::instance().evaluationDate() = eval;
  const DayCounter dc = b.dc;
  const Calendar cal = TARGET();
  auto t = [&](const Date& d) { return dc.yearFraction(eval, d); };
  auto nextm = [](int m, int y) { return (m == 12) ? std::make_pair(1, y + 1) : std::make_pair(m + 1, y); };

  b.ESTR = 0;
  b.EUR3M = 1;
  b.EUR6M = 2;
  b.prob.curves.resize(3);

  // ECB Governing Council policy (effective) dates -> flat-forward front knots (the €STR analog of the
  // SOFR/FOMC front). Absolute dates (data), so they are valid for any eval before the first.
  const std::vector<Date> ecb{Date(30, July, 2026),   Date(17, September, 2026), Date(29, October, 2026),
                              Date(17, December, 2026), Date(28, January, 2027),  Date(18, March, 2027)};
  std::vector<double> estr_meet;
  for (const Date& d : ecb) estr_meet.push_back(t(d));
  const double last_mtg_t = estr_meet.back();

  b.h.resize(3);
  for (auto& hh : b.h) hh.linkTo(ext::make_shared<FlatForward>(eval, 0.02, dc, Continuous));
  b.estr = ext::make_shared<Estr>(b.h[b.ESTR]);
  b.eur3m = ext::make_shared<Euribor3M>(b.h[b.EUR3M]);
  b.eur6m = ext::make_shared<Euribor6M>(b.h[b.EUR6M]);
  const DayCounter estr_dc = b.estr->dayCounter();

  const std::vector<int> swap_tenors{4, 5, 7, 10, 15, 20, 30};  // basis / outright back pillars
  auto tenor_t = [&](int y) { return t(cal.advance(eval, Period(y, Years))); };

  std::vector<ext::shared_ptr<OvernightIndexedSwap>> keep_ois;
  std::vector<ext::shared_ptr<VanillaSwap>> keep_vs;

  // ---------------- ESTR curve (0): €STR futures front + ESTR/3M basis back ----------------
  std::vector<double> estr_back;
  // 12 monthly 1M €STR averaging futures (Aug-2026 .. Jul-2027), fully forecast.
  {
    int m = 8, y = 2026;
    for (int i = 0; i < 12; ++i) {
      const Date s = cal.adjust(Date(1, Month(m), y));
      const auto [nm, ny] = nextm(m, y);
      const Date e = cal.adjust(Date(1, Month(nm), ny));
      cal::Instrument ins;
      ins.quote = cal::QuoteKind::Rate;
      ins.forecast = b.ESTR;
      ins.obs = avg_future_obs_idx(b.estr, eval, dc, s, e);
      b.prob.instruments.push_back(ins);
      m = nm;
      y = ny;
    }
  }
  // 8 IMM 3M compounded €STR futures (Sep-2027 .. Jun-2029); their end dates > last meeting are back knots.
  for (const auto& [im, iy] : std::vector<std::pair<int, int>>{
           {9, 2027}, {12, 2027}, {3, 2028}, {6, 2028}, {9, 2028}, {12, 2028}, {3, 2029}, {6, 2029}}) {
    const Date s = Date::nthWeekday(3, Wednesday, Month(im), iy);
    const Date e0 = s + Period(3, Months);
    const Date e = Date::nthWeekday(3, Wednesday, e0.month(), e0.year());
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::Rate;
    ins.forecast = b.ESTR;
    ins.obs = swaps::qlx::make_observation({{s, e}}, 0.0, estr_dc.yearFraction(s, e), eval, dc);
    b.prob.instruments.push_back(ins);
    if (t(e) > last_mtg_t) estr_back.push_back(t(e));
  }
  for (int y : swap_tenors) estr_back.push_back(tenor_t(y));

  // ESTR/3M-EURIBOR basis (>3y), pinning ESTR: fwd = ESTR OIS overnight leg (forecast ESTR = PRIMARY),
  // bench = 3M EURIBOR float leg (forecast EUR3M), both ESTR-discounted.
  auto estr3m_basis = [&](int y) {
    const Period tenor(y, Years);
    auto oe = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.estr, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
    auto s3 = ext::shared_ptr<VanillaSwap>(MakeVanillaSwap(tenor, b.eur3m, 0.03)
                                               .withDiscountingTermStructure(b.h[b.ESTR])
                                               .withFixedLegDayCount(Thirty360(Thirty360::BondBasis))
                                               .withFixedLegTenor(1 * Years)
                                               .withFixedLegCalendar(TARGET())
                                               .withFloatingLegCalendar(TARGET()));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParSpread;
    ins.fwd = {swaps::qlx::extract_float_leg(oe->overnightLeg(), eval, dc), b.ESTR, b.ESTR};  // PRIMARY = ESTR
    ins.bench = {swaps::qlx::extract_float_leg(s3->floatingLeg(), eval, dc), b.EUR3M, b.ESTR};
    ins.fixed = {swaps::qlx::extract_fixed_leg(oe->fixedLeg(), eval, dc), b.ESTR};
    keep_ois.push_back(oe);
    keep_vs.push_back(s3);
    return ins;
  };
  for (int y : swap_tenors) b.prob.instruments.push_back(estr3m_basis(y));

  // ---------------- EURIBOR-3M curve (1): 3M EURIBOR futures front + 3s6s basis back ----------------
  std::vector<double> eur3m_back;
  // 12 quarterly 3M EURIBOR futures (IMM Sep-2026 .. Jun-2029). A future settles on the ACTUAL fixing,
  // so its dates come from the INDEX (valueDate/maturityDate), NOT the par-coupon approximation.
  for (const auto& [im, iy] : std::vector<std::pair<int, int>>{
           {9, 2026}, {12, 2026}, {3, 2027}, {6, 2027}, {9, 2027}, {12, 2027},
           {3, 2028}, {6, 2028}, {9, 2028}, {12, 2028}, {3, 2029}, {6, 2029}}) {
    const Date fixing = Date::nthWeekday(3, Wednesday, Month(im), iy);
    const Date d1 = b.eur3m->valueDate(fixing);
    const Date d2 = b.eur3m->maturityDate(d1);
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::Rate;
    ins.forecast = b.EUR3M;
    ins.obs = swaps::qlx::make_observation({{d1, d2}}, 0.0, b.eur3m->dayCounter().yearFraction(d1, d2), eval, dc);
    b.prob.instruments.push_back(ins);
    if (t(d2) < 3.4) eur3m_back.push_back(t(d2));  // futures pillars to ~3y
  }
  for (int y : swap_tenors) eur3m_back.push_back(tenor_t(y));

  // 3s6s basis (>3y), pinning EUR3M: fwd = 3M leg (EUR3M = PRIMARY), bench = 6M leg (EUR6M).
  auto s3s6_basis = [&](int y, int pin_curve) {
    const Period tenor(y, Years);
    auto s3 = ext::shared_ptr<VanillaSwap>(
        MakeVanillaSwap(tenor, b.eur3m, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
    auto s6 = ext::shared_ptr<VanillaSwap>(
        MakeVanillaSwap(tenor, b.eur6m, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
    const auto l3 = swaps::qlx::extract_float_leg(s3->floatingLeg(), eval, dc);
    const auto l6 = swaps::qlx::extract_float_leg(s6->floatingLeg(), eval, dc);
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParSpread;
    // fwd = the PINNED curve's leg; bench = the other. EUR3M-pin: fwd=3M,bench=6M. EUR6M-pin: fwd=6M,bench=3M.
    if (pin_curve == b.EUR3M) {
      ins.fwd = {l3, b.EUR3M, b.ESTR};
      ins.bench = {l6, b.EUR6M, b.ESTR};
    } else {
      ins.fwd = {l6, b.EUR6M, b.ESTR};
      ins.bench = {l3, b.EUR3M, b.ESTR};
    }
    ins.fixed = {swaps::qlx::extract_fixed_leg(s3->fixedLeg(), eval, dc), b.ESTR};
    keep_vs.push_back(s3);
    keep_vs.push_back(s6);
    return ins;
  };
  for (int y : swap_tenors) b.prob.instruments.push_back(s3s6_basis(y, b.EUR3M));

  // ---------------- EURIBOR-6M curve (2): single-period 3s6s front + outright 6M swaps back ----------
  std::vector<double> eur6m_back{tenor_t(1), tenor_t(2), tenor_t(3)};
  for (int y : swap_tenors) eur6m_back.push_back(tenor_t(y));
  // Front: 3s6s basis at {6M,1,2,3} pinning EUR6M (fwd=6M leg).
  {
    auto p6 = [&](const Period& tenor) {
      auto s3 = ext::shared_ptr<VanillaSwap>(
          MakeVanillaSwap(tenor, b.eur3m, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
      auto s6 = ext::shared_ptr<VanillaSwap>(
          MakeVanillaSwap(tenor, b.eur6m, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
      cal::Instrument ins;
      ins.quote = cal::QuoteKind::ParSpread;
      ins.fwd = {swaps::qlx::extract_float_leg(s6->floatingLeg(), eval, dc), b.EUR6M, b.ESTR};  // PRIMARY=EUR6M
      ins.bench = {swaps::qlx::extract_float_leg(s3->floatingLeg(), eval, dc), b.EUR3M, b.ESTR};
      ins.fixed = {swaps::qlx::extract_fixed_leg(s6->fixedLeg(), eval, dc), b.ESTR};
      keep_vs.push_back(s3);
      keep_vs.push_back(s6);
      return ins;
    };
    b.prob.instruments.push_back(p6(Period(6, Months)));
    for (int y : {1, 2, 3}) b.prob.instruments.push_back(p6(Period(y, Years)));
  }
  // Back: outright 6M EURIBOR swaps (fixed 30/360 annual vs 6M float), pinning EUR6M.
  auto eur6m_outright = [&](int y) {
    auto s = ext::shared_ptr<VanillaSwap>(MakeVanillaSwap(Period(y, Years), b.eur6m, 0.03)
                                              .withDiscountingTermStructure(b.h[b.ESTR])
                                              .withFixedLegDayCount(Thirty360(Thirty360::BondBasis))
                                              .withFixedLegTenor(1 * Years)
                                              .withFixedLegCalendar(TARGET())
                                              .withFloatingLegCalendar(TARGET()));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParRate;
    ins.fwd = {swaps::qlx::extract_float_leg(s->floatingLeg(), eval, dc), b.EUR6M, b.ESTR};
    ins.fixed = {swaps::qlx::extract_fixed_leg(s->fixedLeg(), eval, dc), b.ESTR};
    keep_vs.push_back(s);
    return ins;
  };
  for (int y : swap_tenors) b.prob.instruments.push_back(eur6m_outright(y));

  // ---- Curve specs + parameterization (ESTR outright; EUR3M spread/ESTR; EUR6M spread/EUR3M) ----
  b.prob.curves[b.ESTR] = {estr_meet, estr_back, -1, CCY_EUR};
  // EUR3M's first futures pillar is ~0.44y, so its single flat-front knot must sit below that (the
  // Flat/Hermite join requires back.front() > meeting.back()). EUR6M's first pillar is 1y, so 0.5 is fine.
  b.prob.curves[b.EUR3M] = {{0.1}, eur3m_back, b.ESTR, CCY_EUR};
  b.prob.curves[b.EUR6M] = {{0.5}, eur6m_back, b.EUR3M, CCY_EUR};
  b.off = {0, b.prob.curves[0].n_knots(), b.prob.curves[0].n_knots() + b.prob.curves[1].n_knots()};
  const int N = b.off[2] + b.prob.curves[2].n_knots();
  b.default_discount = {{b.ESTR, b.ESTR}, {b.EUR3M, b.ESTR}, {b.EUR6M, b.ESTR}};

  // x_true: ESTR ~2% forwards; small forward spreads for the tenor-basis curves.
  b.x_true.resize(N);
  auto fill = [&](int c, double level, double slope) {
    for (int i = 0; i < b.prob.curves[c].n_knots(); ++i) b.x_true[b.off[c] + i] = level + slope * i;
  };
  fill(b.ESTR, 0.0200, 0.0003);
  fill(b.EUR3M, 0.0012, 0.00002);  // 3M-EURIBOR / ESTR basis ~12bp
  fill(b.EUR6M, 0.0008, 0.00002);  // 3s6s ~8bp on top

  // Real spread-aware curves in, self-consistent market = model quote at x_true.
  b.curve_handles = cal::build_bundle_curves<double>(
      b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i]; });
  b.ts.resize(3);
  for (int c = 0; c < 3; ++c) {
    auto tsc = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
        eval, dc, b.curve_handles[c].get());
    tsc->enableExtrapolation();
    b.ts[c] = tsc;
    b.h[c].linkTo(tsc);
  }
  for (auto& s : keep_ois) s->deepUpdate();
  for (auto& s : keep_vs) s->deepUpdate();
  const auto curve_of = [&](int i) -> const cal::CurveHandle<double>& { return *b.curve_handles[i]; };
  for (auto& ins : b.prob.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);

  b.x0.resize(N);
  b.x0.segment(b.off[b.ESTR], b.prob.curves[b.ESTR].n_knots()).setConstant(0.020);
  b.x0.segment(b.off[b.EUR3M], b.prob.curves[b.EUR3M].n_knots()).setConstant(0.0012);
  b.x0.segment(b.off[b.EUR6M], b.prob.curves[b.EUR6M].n_knots()).setConstant(0.0008);
  return b;
}

// Shift every curve-role index on an instrument by `d` (>=0 roles only), so a sub-bundle built with
// local indices {0,1,...} can be spliced into a combined bundle at a curve-index offset.
inline void remap_instrument_roles(calibration::Instrument& ins, int d) {
  auto sh = [d](int& r) { if (r >= 0) r += d; };
  sh(ins.fwd.forecast);
  sh(ins.fwd.discount);
  sh(ins.bench.forecast);
  sh(ins.bench.discount);
  sh(ins.fixed.discount);
  if (ins.quote == calibration::QuoteKind::Rate) sh(ins.forecast);
  sh(ins.fx_num);
  sh(ins.fx_den);
  sh(ins.mtm.forecast);
  sh(ins.mtm.discount);
  sh(ins.mtm.reset_num);
  sh(ins.mtm.reset_den);
}

// ===============================================================================================
// The EUR trio INSIDE the multi-currency bundle (Part B): SOFR + [ESTR/EUR3M/EUR6M cyclic SCC] +
// EUR-collateralized-in-USD, in ONE BundleProblem. Exercises the cross-TENOR cycle (the EUR trio) and
// the cross-CURRENCY coupling (EUR-in-USD -> ESTR-SCC + SOFR) in a single staged solve.
// (FF is omitted for brevity; it slots in exactly as in build_usd_bundle.)
// ===============================================================================================
inline MultiCcyBundle build_eur_multicurrency() {
  using namespace QuantLib;
  const Date eval(8, July, 2026);
  MultiCcyBundle b;
  b.today = eval;
  b.fx_spot = 1.10;
  const DayCounter dc = b.dc;
  const double S = b.fx_spot;

  // The EUR trio (ESTR=0/EUR3M=1/EUR6M=2, local), then spliced at global offset +1 (after SOFR).
  MultiCcyBundle eur = build_eur_curves(eval);  // sets eval + seeds nothing; leaves eval = 8 Jul 2026
  // SOFR reference market (build_market re-sets eval to rm::evaluation_date == 8 Jul 2026 and seeds fixings).
  RelinkableHandle<YieldTermStructure> hS;
  Market mk = build_market(hS);
  const cal::CalibrationProblem sofr = build_problem(mk);

  // Global roles: SOFR 0 | ESTR 1 | EUR3M 2 | EUR6M 3 | EUR-in-USD 4.
  b.SOFR = 0;
  b.ESTR = 1;
  b.EUR3M = 2;
  b.EUR6M = 3;
  b.EURUSD = 4;
  b.sofr = mk.sofr;
  b.estr = eur.estr;
  b.eur3m = eur.eur3m;
  b.eur6m = eur.eur6m;

  b.prob.curves.resize(5);
  b.prob.curves[b.SOFR] = {mk.meeting_times, mk.back_times, -1, CCY_USD};
  for (int c = 0; c < 3; ++c) {  // splice the EUR trio specs, shifting bases by +1
    auto spec = eur.prob.curves[c];
    if (spec.base >= 0) spec.base += 1;
    b.prob.curves[1 + c] = spec;
  }
  const std::vector<double> xccy_meet{0.25, 0.5}, xccy_back{1, 2, 3, 5, 7, 10};
  b.prob.curves[b.EURUSD] = {xccy_meet, xccy_back, b.ESTR, CCY_EUR};

  b.off.assign(5, 0);
  for (int c = 1; c < 5; ++c) b.off[c] = b.off[c - 1] + b.prob.curves[c - 1].n_knots();
  const int N = b.off[4] + b.prob.curves[4].n_knots();
  b.default_discount = {{b.SOFR, b.SOFR}, {b.ESTR, b.ESTR},   {b.EUR3M, b.ESTR},
                        {b.EUR6M, b.ESTR}, {b.EURUSD, b.EURUSD}};

  // Handles (5): SOFR's is build_market's hS; the EUR trio reuses eur.h; EUR-in-USD gets a fresh one.
  b.h = {hS, eur.h[0], eur.h[1], eur.h[2], RelinkableHandle<YieldTermStructure>{}};
  b.h[b.EURUSD].linkTo(ext::make_shared<FlatForward>(eval, 0.03, dc, Continuous));

  // Instruments: SOFR (roles already 0) + EUR trio (roles shifted +1) + EUR-in-USD (FX fwd + MtM).
  for (auto ins : sofr.instruments) b.prob.instruments.push_back(ins);
  for (auto ins : eur.prob.instruments) {
    remap_instrument_roles(ins, 1);
    b.prob.instruments.push_back(ins);
  }
  // EUR-in-USD: FX forward points (fx_num=EUR-in-USD, fx_den=SOFR) + MtM xccy basis (funding leg on SOFR).
  auto t = [&](const Date& d) { return dc.yearFraction(eval, d); };
  const Calendar fxcal = JointCalendar(TARGET(), UnitedStates(UnitedStates::Settlement));
  const Date spot = fxcal.advance(eval, 2, Days);
  std::vector<double> fx_times{t(fxcal.advance(eval, 1, Days)), t(fxcal.advance(spot, 1, Days))};
  for (const Period& p : {Period(1, Weeks), Period(2, Weeks), Period(3, Weeks), Period(1, Months),
                          Period(2, Months), Period(3, Months), Period(6, Months), Period(1, Years)})
    fx_times.push_back(t(fxcal.advance(spot, p)));
  for (double ft : fx_times) {
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::FxForward;
    ins.fx_num = b.EURUSD;
    ins.fx_den = b.SOFR;
    ins.fx_spot = S;
    ins.fx_time = ft;
    ins.pv_currency = CCY_USD;
    b.prob.instruments.push_back(ins);
  }
  std::vector<ext::shared_ptr<OvernightIndexedSwap>> keep;
  for (int y : {2, 3, 5, 7, 10}) {
    const Period tenor(y, Years);
    auto oe = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.estr, 0.03).withDiscountingTermStructure(b.h[b.EURUSD]));
    auto os = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(tenor, b.sofr, 0.03).withDiscountingTermStructure(b.h[b.SOFR]));
    const auto eleg = swaps::qlx::extract_float_leg(oe->overnightLeg(), eval, dc);
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::XccyMtmBasis;
    ins.fwd = {eleg, b.EURUSD, b.EURUSD};
    ins.bench = {eleg, b.ESTR, b.EURUSD};
    ins.fixed = {swaps::qlx::extract_fixed_leg(oe->fixedLeg(), eval, dc), b.EURUSD};
    ins.mtm = {swaps::qlx::extract_float_leg(os->overnightLeg(), eval, dc), b.SOFR, b.SOFR};
    ins.mtm.reset_num = b.EURUSD;
    ins.mtm.reset_den = b.SOFR;
    ins.mtm.fx_spot = S;
    ins.pv_currency = CCY_USD;
    keep.push_back(oe);
    keep.push_back(os);
    b.prob.instruments.push_back(ins);
  }

  // x_true = [SOFR reference forwards ; EUR trio ; EUR-in-USD basis].
  b.x_true.resize(N);
  {
    std::vector<double> sx(rm::reference_front_forwards.begin(), rm::reference_front_forwards.end());
    sx.insert(sx.end(), rm::reference_back_forwards.begin(), rm::reference_back_forwards.end());
    for (int i = 0; i < static_cast<int>(sx.size()); ++i) b.x_true[b.off[b.SOFR] + i] = sx[i];
  }
  b.x_true.segment(b.off[b.ESTR], eur.x_true.size()) = eur.x_true;  // ESTR/EUR3M/EUR6M blocks
  for (int i = 0; i < b.prob.curves[b.EURUSD].n_knots(); ++i) b.x_true[b.off[b.EURUSD] + i] = -0.0015 + 0.00002 * i;

  // Wire the real curves in and set a self-consistent market = the model quote at x_true.
  b.curve_handles = cal::build_bundle_curves<double>(
      b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i]; });
  b.ts.resize(5);
  for (int c = 0; c < 5; ++c) {
    auto tsc = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
        eval, dc, b.curve_handles[c].get());
    tsc->enableExtrapolation();
    b.ts[c] = tsc;
    b.h[c].linkTo(tsc);
  }
  for (auto& s : mk.swaps) s->deepUpdate();
  for (auto& s : keep) s->deepUpdate();
  const auto curve_of = [&](int i) -> const cal::CurveHandle<double>& { return *b.curve_handles[i]; };
  for (auto& ins : b.prob.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);

  // x0 near each block's level.
  b.x0.resize(N);
  b.x0.segment(b.off[b.SOFR], b.prob.curves[b.SOFR].n_knots()).setConstant(0.043);
  b.x0.segment(b.off[b.ESTR], eur.x0.size()) = eur.x0;  // ESTR/EUR3M/EUR6M starts
  b.x0.segment(b.off[b.EURUSD], b.prob.curves[b.EURUSD].n_knots()).setConstant(-0.0015);
  return b;
}

// ===============================================================================================
// The WHOLE multi-currency bundle: SOFR + FF + PRIME + ESTR + EUR3M + EUR6M + EONIA + EUR-in-USD — 8
// curves in one BundleProblem. USD block (SOFR/FF/PRIME), the coupled EUR trio (ESTR/EUR3M/EUR6M cyclic
// SCC), EONIA (ESTR + fixed spread), and the EUR-collateralized-in-USD xccy curve. Calibrated staged.
// ===============================================================================================
inline MultiCcyBundle build_full_multicurrency() {
  using namespace QuantLib;
  const Date eval(8, July, 2026);
  MultiCcyBundle b;
  b.today = eval;
  b.fx_spot = 1.10;
  const DayCounter dc = b.dc;
  const double S = b.fx_spot;
  auto t = [&](const Date& d) { return dc.yearFraction(eval, d); };

  MultiCcyBundle eur = build_eur_curves(eval);       // ESTR/EUR3M/EUR6M local 0,1,2 -> global 3,4,5
  RelinkableHandle<YieldTermStructure> hS;
  Market mk = build_market(hS);
  const cal::CalibrationProblem sofr = build_problem(mk);

  b.SOFR = 0;
  b.FF = 1;
  const int PRIME = 2;
  b.ESTR = 3;
  b.EUR3M = 4;
  b.EUR6M = 5;
  const int EONIA = 6;
  b.EURUSD = 7;
  b.sofr = mk.sofr;
  b.estr = eur.estr;
  b.eur3m = eur.eur3m;
  b.eur6m = eur.eur6m;
  b.fedfunds = ext::make_shared<FedFunds>(RelinkableHandle<YieldTermStructure>{});

  const Calendar usc = mk.sofr->fixingCalendar();
  const std::vector<double> spr_back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  const std::vector<int> spr_tenors{1, 2, 3, 5, 7, 10, 15, 20, 30};

  b.prob.curves.resize(8);
  b.prob.curves[b.SOFR] = {mk.meeting_times, mk.back_times, -1, CCY_USD};
  b.prob.curves[b.FF] = {{0.5}, spr_back, b.SOFR, CCY_USD};
  b.prob.curves[PRIME] = {{0.5}, spr_back, b.FF, CCY_USD};
  for (int c = 0; c < 3; ++c) {  // EUR trio, bases shifted +3
    auto spec = eur.prob.curves[c];
    if (spec.base >= 0) spec.base += 3;
    b.prob.curves[b.ESTR + c] = spec;
  }
  b.prob.curves[EONIA] = {{0.5}, spr_back, b.ESTR, CCY_EUR};
  b.prob.curves[b.EURUSD] = {{0.25, 0.5}, {1, 2, 3, 5, 7, 10}, b.ESTR, CCY_EUR};

  b.off.assign(8, 0);
  for (int c = 1; c < 8; ++c) b.off[c] = b.off[c - 1] + b.prob.curves[c - 1].n_knots();
  const int N = b.off[7] + b.prob.curves[7].n_knots();

  // Handles: SOFR=hS; FF/PRIME/EONIA/EUR-in-USD fresh; EUR trio reuses eur.h.
  b.h.assign(8, RelinkableHandle<YieldTermStructure>{});
  b.h[b.SOFR] = hS;
  b.h[b.ESTR] = eur.h[0];
  b.h[b.EUR3M] = eur.h[1];
  b.h[b.EUR6M] = eur.h[2];
  for (int c : {b.FF, PRIME, EONIA, b.EURUSD})
    b.h[c].linkTo(ext::make_shared<FlatForward>(eval, 0.03, dc, Continuous));
  b.fedfunds = ext::make_shared<FedFunds>(b.h[b.FF]);
  auto prime_idx = ext::make_shared<OvernightIndex>("PRIME", 0, USDCurrency(), usc, Actual360(), b.h[PRIME]);
  auto eonia_idx =
      ext::make_shared<OvernightIndex>("EONIA", 0, EURCurrency(), TARGET(), Actual360(), b.h[EONIA]);

  // Instruments: SOFR (roles 0) + EUR trio (roles +3).
  for (auto ins : sofr.instruments) b.prob.instruments.push_back(ins);
  for (auto ins : eur.prob.instruments) {
    remap_instrument_roles(ins, 3);
    b.prob.instruments.push_back(ins);
  }
  std::vector<ext::shared_ptr<OvernightIndexedSwap>> keep;
  // A compounded-OIS basis pinning `fc` vs `bench`, discounted on `disc`.
  auto ois_basis = [&](const ext::shared_ptr<OvernightIndex>& fi, int fc, const ext::shared_ptr<OvernightIndex>& bi,
                       int bench, int disc, const Period& tenor) {
    auto of = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(tenor, fi, 0.03).withDiscountingTermStructure(b.h[disc]));
    auto ob = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(tenor, bi, 0.03).withDiscountingTermStructure(b.h[disc]));
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParSpread;
    ins.fwd = {swaps::qlx::extract_float_leg(of->overnightLeg(), eval, dc), fc, disc};       // PRIMARY = fc
    ins.bench = {swaps::qlx::extract_float_leg(ob->overnightLeg(), eval, dc), bench, disc};
    ins.fixed = {swaps::qlx::extract_fixed_leg(ob->fixedLeg(), eval, dc), disc};
    keep.push_back(of);
    keep.push_back(ob);
    return ins;
  };
  for (int y : spr_tenors) {
    b.prob.instruments.push_back(ois_basis(b.fedfunds, b.FF, mk.sofr, b.SOFR, b.SOFR, Period(y, Years)));  // FF/SOFR
    b.prob.instruments.push_back(ois_basis(prime_idx, PRIME, b.fedfunds, b.FF, b.SOFR, Period(y, Years)));  // PRIME/FF
    b.prob.instruments.push_back(ois_basis(eonia_idx, EONIA, b.estr, b.ESTR, b.ESTR, Period(y, Years)));    // EONIA/ESTR
  }
  // EUR-in-USD: FX forward points + MtM xccy basis.
  const Calendar fxcal = JointCalendar(TARGET(), UnitedStates(UnitedStates::Settlement));
  const Date spot = fxcal.advance(eval, 2, Days);
  std::vector<double> fx_times{t(fxcal.advance(eval, 1, Days)), t(fxcal.advance(spot, 1, Days))};
  for (const Period& p : {Period(1, Weeks), Period(2, Weeks), Period(3, Weeks), Period(1, Months),
                          Period(2, Months), Period(3, Months), Period(6, Months), Period(1, Years)})
    fx_times.push_back(t(fxcal.advance(spot, p)));
  for (double ft : fx_times) {
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::FxForward;
    ins.fx_num = b.EURUSD;
    ins.fx_den = b.SOFR;
    ins.fx_spot = S;
    ins.fx_time = ft;
    ins.pv_currency = CCY_USD;
    b.prob.instruments.push_back(ins);
  }
  for (int y : {2, 3, 5, 7, 10}) {
    auto oe = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(Period(y, Years), b.estr, 0.03).withDiscountingTermStructure(b.h[b.EURUSD]));
    auto os = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(Period(y, Years), b.sofr, 0.03).withDiscountingTermStructure(b.h[b.SOFR]));
    const auto eleg = swaps::qlx::extract_float_leg(oe->overnightLeg(), eval, dc);
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::XccyMtmBasis;
    ins.fwd = {eleg, b.EURUSD, b.EURUSD};
    ins.bench = {eleg, b.ESTR, b.EURUSD};
    ins.fixed = {swaps::qlx::extract_fixed_leg(oe->fixedLeg(), eval, dc), b.EURUSD};
    ins.mtm = {swaps::qlx::extract_float_leg(os->overnightLeg(), eval, dc), b.SOFR, b.SOFR};
    ins.mtm.reset_num = b.EURUSD;
    ins.mtm.reset_den = b.SOFR;
    ins.mtm.fx_spot = S;
    ins.pv_currency = CCY_USD;
    keep.push_back(oe);
    keep.push_back(os);
    b.prob.instruments.push_back(ins);
  }

  // x_true.
  b.x_true.resize(N);
  auto fill = [&](int c, double level, double slope) {
    for (int i = 0; i < b.prob.curves[c].n_knots(); ++i) b.x_true[b.off[c] + i] = level + slope * i;
  };
  {
    std::vector<double> sx(rm::reference_front_forwards.begin(), rm::reference_front_forwards.end());
    sx.insert(sx.end(), rm::reference_back_forwards.begin(), rm::reference_back_forwards.end());
    for (int i = 0; i < static_cast<int>(sx.size()); ++i) b.x_true[b.off[b.SOFR] + i] = sx[i];
  }
  fill(b.FF, 0.0003, 0.00002);      // FF/SOFR ~3bp
  fill(PRIME, 0.0300, 0.0);         // PRIME = FF + 300bp
  b.x_true.segment(b.off[b.ESTR], eur.x_true.size()) = eur.x_true;  // ESTR/EUR3M/EUR6M
  fill(EONIA, 0.00085, 0.0);        // EONIA = ESTR + 8.5bp
  fill(b.EURUSD, -0.0015, 0.00002);

  b.curve_handles = cal::build_bundle_curves<double>(
      b.prob.curves, [&](int c, int i) { return b.x_true[b.off[c] + i]; });
  b.ts.resize(8);
  for (int c = 0; c < 8; ++c) {
    auto tsc = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
        eval, dc, b.curve_handles[c].get());
    tsc->enableExtrapolation();
    b.ts[c] = tsc;
    b.h[c].linkTo(tsc);
  }
  for (auto& s : mk.swaps) s->deepUpdate();
  for (auto& s : keep) s->deepUpdate();
  const auto curve_of = [&](int i) -> const cal::CurveHandle<double>& { return *b.curve_handles[i]; };
  for (auto& ins : b.prob.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);

  b.x0.resize(N);
  for (int c = 0; c < 8; ++c)
    b.x0.segment(b.off[c], b.prob.curves[c].n_knots()).setConstant(b.x_true[b.off[c]]);  // flat start at each block level
  b.x0.segment(b.off[b.ESTR], eur.x0.size()) = eur.x0;  // the EUR trio wants its structured start
  return b;
}

}  // namespace swaps::refbuild
