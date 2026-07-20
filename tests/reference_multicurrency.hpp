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

}  // namespace swaps::refbuild
