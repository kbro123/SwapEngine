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

  // Curve role indices (into prob.curves / the stacked x).
  int ESTR = -1, EUR3M = -1, EUR6M = -1;

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

}  // namespace swaps::refbuild
