#pragma once
// The canonical REALISTIC multi-curve bundle -- the model for bundle tests and benchmarks. Every curve
// has the real desk structure: a piecewise-flat meeting-date front + a smooth back, priced by real
// instruments (SOFR: 6 FOMC meetings + 12x1M then 8x3M SOFR futures + par swaps; each basis curve: 12x1M
// futures + basis swaps). Curve 0 (SOFR) is OUTRIGHT; the other curves are SPREADs, wired either as a
// CHAIN (c spread over c-1) or a STAR (every basis curve spread straight off SOFR). The STAR is what makes
// the branch-parallel solve exercise a fat wave (K independent basis blocks); the CHAIN is the sequential
// desk build. Self-consistent market generated from x_true, so the joint/staged solve recovers it.
//
// Owns all the QuantLib state (indices, term structures, spread-aware curve handles) that must outlive the
// BundleProblem's pricing -- so a caller just holds a RealisticBundle and uses `.prob` / `.x0` / `.x_true`.

#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"

namespace swaps::refbuild {

enum class BundleTopology { Chain, Star };

struct RealisticBundle {
  calibration::BundleProblem prob;
  Eigen::VectorXd x_true, x0;
  std::vector<int> off;  // stacked offset of each curve's block in x
  // QuantLib state kept alive for the lifetime of the bundle (the term structures are linked into the
  // forecast handles the basis swaps price off).
  std::vector<QuantLib::RelinkableHandle<QuantLib::YieldTermStructure>> h;
  std::vector<QuantLib::ext::shared_ptr<QuantLib::OvernightIndex>> idx;
  std::vector<std::unique_ptr<calibration::CurveHandle<double>>> curve_handles;
  std::vector<QuantLib::ext::shared_ptr<QuantLib::YieldTermStructure>> ts;
  int n_curves() const { return static_cast<int>(prob.curves.size()); }
};

// Build a realistic bundle: 1 SOFR base + `n_basis` FF-style basis curves. `mk`/`hS` come from
// build_market (hS is SOFR's handle, already linked to the SOFR curve wrapper by the caller or here).
inline RealisticBundle build_realistic_bundle(Market& mk,
                                              QuantLib::RelinkableHandle<QuantLib::YieldTermStructure>& hS,
                                              int n_basis, BundleTopology topo) {
  using namespace QuantLib;
  namespace cal = swaps::calibration;
  const Date today = mk.today;
  const DayCounter dc = mk.dc;
  const Calendar fcal = mk.sofr->fixingCalendar();
  const int NC = 1 + n_basis;

  RealisticBundle rb;
  const cal::CalibrationProblem sofr = build_problem(mk);  // real SOFR: meetings + 1M/3M futures + swaps

  const std::vector<Period> back_periods{18 * Months, 2 * Years,  3 * Years,  4 * Years,
                                         5 * Years,   7 * Years,  10 * Years, 12 * Years,
                                         15 * Years,  20 * Years, 25 * Years, 30 * Years};
  std::vector<double> back_t;
  for (const Period& p : back_periods) back_t.push_back(dc.yearFraction(today, today + p));

  // Curve specs: SOFR outright (reference knots); each basis curve = meeting front + basis-tenor back,
  // spread over its base (Star: base 0; Chain: base c-1).
  rb.prob.curves.resize(NC);
  rb.prob.curves[0] = swaps::pricing::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(sofr.meeting_times, sofr.back_times)};
  for (int c = 1; c < NC; ++c) {
    const int base = (topo == BundleTopology::Star) ? 0 : c - 1;
    rb.prob.curves[c] = swaps::pricing::CurveStructure{.base = base, .regions = swaps::curve::flat_hermite(sofr.meeting_times, back_t)};
  }
  rb.off.assign(NC, 0);
  for (int c = 1; c < NC; ++c) rb.off[c] = rb.off[c - 1] + rb.prob.curves[c - 1].n_knots();
  const int N = rb.off[NC - 1] + rb.prob.curves[NC - 1].n_knots();

  // x_true: SOFR ~4.3% forward level; each basis curve a small forward SPREAD to its base.
  rb.x_true.resize(N);
  for (int c = 0; c < NC; ++c) {
    const int nk = rb.prob.curves[c].n_knots();
    const double level = (c == 0) ? 0.0430 : (0.0010 + 0.0004 * c);  // distinct per-basis spread levels
    const double slope = (c == 0) ? 0.0003 : 0.0001;
    for (int i = 0; i < nk; ++i) rb.x_true[rb.off[c] + i] = level + slope * i;
  }

  // SOFR instruments (curve 0): the reference market's generic Instruments (roles already curve 0).
  for (const auto& ins : sofr.instruments) rb.prob.instruments.push_back(ins);

  // Basis indices + placeholder handles.
  rb.h.resize(NC);
  rb.idx.resize(NC);
  rb.h[0] = hS;
  for (int c = 1; c < NC; ++c) {
    rb.h[c].linkTo(ext::make_shared<FlatForward>(today, 0.04, dc, Continuous));
    rb.idx[c] = ext::make_shared<OvernightIndex>("BASIS" + std::to_string(c), 0, USDCurrency(), fcal,
                                                 Actual360(), rb.h[c]);
  }

  auto basis_inst = [&](const OvernightIndexedSwap& o, int fwd_fc, int bench_fc, int disc) {
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::ParSpread;
    ins.fwd = {swaps::qlx::extract_float_leg(o.overnightLeg(), today, dc), fwd_fc, disc};
    ins.bench = {swaps::qlx::extract_float_leg(o.overnightLeg(), today, dc), bench_fc, disc};
    ins.fixed = {swaps::qlx::extract_fixed_leg(o.fixedLeg(), today, dc), disc};
    return ins;
  };

  // Each basis curve: 12 1M averaging futures (front) + basis swaps over its base (back), SOFR-discounted.
  for (int c = 1; c < NC; ++c) {
    const int bench = (topo == BundleTopology::Star) ? 0 : c - 1;
    for (int i = 0; i < 12; ++i) {
      const auto& q = rm::futures_1m[i];
      const Date s = sofr_start(Month(q.ref_month), q.ref_year, Monthly),
                 e = sofr_end(Month(q.ref_month), q.ref_year, Monthly);
      cal::Instrument ins;
      ins.quote = cal::QuoteKind::Rate;
      ins.obs = avg_future_obs(mk, Future{s, e, false, q.price});
      ins.forecast = c;
      rb.prob.instruments.push_back(ins);
    }
    for (const Period& p : back_periods) {
      auto o = ext::shared_ptr<OvernightIndexedSwap>(
          MakeOIS(p, rb.idx[c], 0.03).withDiscountingTermStructure(hS));
      rb.prob.instruments.push_back(basis_inst(*o, c, bench, 0));
    }
  }

  // Actual forward curves at x_true, exposed to QuantLib; build_bundle_curves resolves the spread graph.
  rb.curve_handles = cal::build_bundle_curves<double>(
      rb.prob.curves, [&](int c, int i) { return rb.x_true[rb.off[c] + i]; });
  rb.ts.resize(NC);
  for (int c = 0; c < NC; ++c) {
    auto t = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
        today, dc, rb.curve_handles[c].get());
    t->enableExtrapolation();
    rb.ts[c] = t;
    rb.h[c].linkTo(t);
  }
  for (auto& s : mk.swaps) s->deepUpdate();

  // Self-consistent market from x_true (so joint & staged solves recover x_true exactly).
  const Eigen::VectorXd r0 = rb.prob.residuals<double>(rb.x_true);
  for (int i = 0; i < static_cast<int>(rb.prob.instruments.size()); ++i) rb.prob.instruments[i].market += r0[i];

  // Flat per-curve start: SOFR near its level, spreads near a small constant.
  rb.x0.resize(N);
  for (int c = 0; c < NC; ++c) {
    const int nk = rb.prob.curves[c].n_knots();
    const double start = (c == 0) ? 0.043 : (0.0010 + 0.0004 * c);
    rb.x0.segment(rb.off[c], nk).setConstant(start);
  }
  return rb;
}

}  // namespace swaps::refbuild
