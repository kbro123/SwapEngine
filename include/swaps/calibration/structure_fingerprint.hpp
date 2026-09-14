#pragma once
// STRUCTURE FINGERPRINT of a BundleProblem — a 64-bit hash of everything the compiled W-cache depends on
// (curve regions, curve roles, cashflow times, observation windows, per-coupon baked constants) but NOT the
// quoted TARGETS (Instrument.market / band_lower / band_upper / band_decay).
//
// This is the switch behind the object-oriented / hot-path split. Two problems with the SAME fingerprint
// differ only in their market levels, so a calibrated Model can WARM-tick to the new market over its
// existing W-cache (microseconds). A CHANGED fingerprint means the topology actually moved — a knot, a
// scheme, a role, an added instrument, a shifted observation window — so W must be recompiled. The OO layer
// stays put; the optimised hot path is regenerated only when the structure genuinely changes.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"

namespace swaps::calibration {

namespace fp_detail {

// FNV-1a over the raw bytes of ints / bit-cast doubles — order-sensitive, so it also captures list length
// and ordering (adding, removing or reordering a knot/coupon/instrument all change the hash).
struct FnvHasher {
  std::uint64_t h = 1469598103934665603ull;
  void mix(std::uint64_t v) { h ^= v; h *= 1099511628211ull; }
  void i(long long v) { mix(static_cast<std::uint64_t>(v)); }
  void d(double v) {
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof b);
    mix(b);
  }
  void vd(const std::vector<double>& xs) {
    i(static_cast<long long>(xs.size()));
    for (double x : xs) d(x);
  }
  void s(const std::string& str) {
    i(static_cast<long long>(str.size()));
    for (unsigned char c : str) mix(c);
  }
};

// An observation window is fully determined by its ENDPOINTS + count + the (fixed, per-index) calendar, so
// a bounded summary — size, first/last of each edge vector, and the scalars — is collision-safe against any
// real structural change (a shifted maturity, a shortened window after a fixing) while staying O(1) per
// observation rather than O(daily sub-periods). Cheap enough to be provably off any perf budget.
inline void edges(FnvHasher& H, const std::vector<double>& v) {
  H.i(static_cast<long long>(v.size()));
  if (!v.empty()) {
    H.d(v.front());
    H.d(v.back());
  }
}

inline void hash_obs(FnvHasher& H, const pricing::RateObservation& o) {
  edges(H, o.sub_start);
  edges(H, o.sub_end);
  edges(H, o.weight);
  H.d(o.realized);  // baked into the per-coupon constant, so a fixing change forces a (correct) recompile
  H.d(o.tau_index);
  H.d(o.fixing_step);
  H.d(o.fixing_step3);
  H.i(o.compounded);
  H.d(o.realized_factor);
  H.s(o.fixing_index);
  H.i(static_cast<long long>(o.fixing_schedule.size()));
  if (!o.fixing_schedule.empty()) {  // endpoints of the fixing schedule (interiors are calendar-derived)
    const auto& a = o.fixing_schedule.front();
    const auto& z = o.fixing_schedule.back();
    H.i(a.fixing_date);
    H.d(a.t_start);
    H.i(z.fixing_date);
    H.d(z.t_end);
  }
}

inline void hash_float(FnvHasher& H, const FloatLeg& lg) {
  H.i(lg.forecast);
  H.i(lg.discount);
  H.i(lg.reset_num);
  H.i(lg.reset_den);
  H.d(lg.fx_spot);
  if (lg.fx_spot_time != 0.0) H.d(lg.fx_spot_time);  // O-X3: hashed only when set (existing fingerprints unchanged)
  H.i(static_cast<long long>(lg.coupons.size()));
  for (const auto& c : lg.coupons) {
    hash_obs(H, c.obs);
    H.d(c.pay);
    H.d(c.tau_pay);
    H.d(c.spread);
    H.d(c.scale);  // FX-spot constant baked into the W-cache's per-coupon k, so it belongs to the structure
    H.d(c.reset_time);
    H.d(c.accrual_set ? c.accrual_start : (c.obs.sub_start.empty() ? -1.0 : c.obs.sub_start.front()));  // effective exchange dates
    H.d(c.accrual_set ? c.accrual_end : (c.obs.sub_end.empty() ? -1.0 : c.obs.sub_end.back()));
    H.d(c.reset_fx);
    if (c.fx_fixing_set) H.d(c.fx_fixing_time);  // O-X3 piece 2 (hashed only when set)
  }
}

inline void hash_fixed(FnvHasher& H, const FixedLeg& lg) {
  H.i(lg.discount);
  H.i(static_cast<long long>(lg.coupons.size()));
  for (const auto& c : lg.coupons) {
    H.d(c.pay);
    H.d(c.tau);
    H.d(c.scale);
  }
}

// STRUCTURAL EQUALITY (the object model, 2026-09-10): an O(n) walk over the same inputs the hash covers,
// with no hashing and no allocation -- the check behind BundleSession::same_structure / rebind. Two
// problems are structurally equal iff a compiled model of one prices the other's rows: same curves
// (roles, regions, knots), same instrument kinds, legs, cashflow times and observation windows. Quote
// fields (market, bands) are excluded. A schedule-carrying observation is compared on its FIXING SCHEDULE
// (endpoints + size), not on its resolved sub-periods / realized part -- those are the RESOLUTION of the
// schedule against a fixing table, which the session performs on its own copy (a client resends the
// unresolved document, so comparing resolution outputs made same_structure(problem()) false, E3-D16).
inline bool obs_equal(const pricing::RateObservation& a, const pricing::RateObservation& b) {
  if (a.fixing_schedule.size() != b.fixing_schedule.size()) return false;
  if (a.fixing_index != b.fixing_index) return false;
  if (a.tau_index != b.tau_index || a.fixing_step != b.fixing_step || a.fixing_step3 != b.fixing_step3) return false;
  if (a.compounded != b.compounded) return false;
  if (!a.fixing_schedule.empty()) {
    const auto &fa = a.fixing_schedule.front(), &fb = b.fixing_schedule.front();
    const auto &la = a.fixing_schedule.back(), &lb = b.fixing_schedule.back();
    return fa.fixing_date == fb.fixing_date && fa.t_start == fb.t_start && la.fixing_date == lb.fixing_date && la.t_end == lb.t_end;
  }
  if (a.sub_start != b.sub_start || a.sub_end != b.sub_end || a.weight != b.weight) return false;
  return a.realized == b.realized && a.realized_factor == b.realized_factor;
}
inline bool float_equal(const FloatLeg& a, const FloatLeg& b) {
  if (a.forecast != b.forecast || a.discount != b.discount || a.reset_num != b.reset_num || a.reset_den != b.reset_den) return false;
  if (a.fx_spot != b.fx_spot || a.fx_spot_time != b.fx_spot_time || a.coupons.size() != b.coupons.size()) return false;
  for (std::size_t i = 0; i < a.coupons.size(); ++i) {
    const auto &c = a.coupons[i], &d = b.coupons[i];
    if (c.pay != d.pay || c.tau_pay != d.tau_pay || c.spread != d.spread || c.scale != d.scale || c.reset_time != d.reset_time) return false;
    // EFFECTIVE exchange dates (the accrual period when carried, else the observation window): a document
    // resent without explicit accrual fields is the same structure as the builder's coupon.
    const auto eff_s = [](const pricing::FloatCoupon& x) { return x.accrual_set ? x.accrual_start : (x.obs.sub_start.empty() ? -1.0 : x.obs.sub_start.front()); };
    const auto eff_e = [](const pricing::FloatCoupon& x) { return x.accrual_set ? x.accrual_end : (x.obs.sub_end.empty() ? -1.0 : x.obs.sub_end.back()); };
    if (eff_s(c) != eff_s(d) || eff_e(c) != eff_e(d) || c.reset_fx != d.reset_fx) return false;
    if (c.fx_fixing_set != d.fx_fixing_set || (c.fx_fixing_set && c.fx_fixing_time != d.fx_fixing_time)) return false;
    if (!obs_equal(c.obs, d.obs)) return false;
  }
  return true;
}
inline bool fixed_equal(const FixedLeg& a, const FixedLeg& b) {
  if (a.discount != b.discount || a.coupons.size() != b.coupons.size()) return false;
  for (std::size_t i = 0; i < a.coupons.size(); ++i) {
    const auto &c = a.coupons[i], &d = b.coupons[i];
    if (c.pay != d.pay || c.tau != d.tau || c.scale != d.scale) return false;
  }
  return true;
}
inline bool instrument_equal(const Instrument& a, const Instrument& b) {
  if (a.quote != b.quote || a.forecast != b.forecast || a.pv_currency != b.pv_currency) return false;
  if (a.fx_num != b.fx_num || a.fx_den != b.fx_den || a.fx_spot != b.fx_spot || a.fx_time != b.fx_time) return false;
  if (a.fx_spot_time != b.fx_spot_time) return false;
  if (a.turn_curve != b.turn_curve || a.turn_index != b.turn_index || a.convexity != b.convexity) return false;
  if (!obs_equal(a.obs, b.obs) || !float_equal(a.fwd, b.fwd) || !float_equal(a.bench, b.bench) || !float_equal(a.mtm, b.mtm)) return false;
  if (!fixed_equal(a.fixed, b.fixed)) return false;
  if (a.combination.size() != b.combination.size()) return false;
  for (std::size_t i = 0; i < a.combination.size(); ++i) {
    if (a.combination[i].weight != b.combination[i].weight) return false;
    if (!instrument_equal(a.combination[i].instrument, b.combination[i].instrument)) return false;
  }
  return true;
}
inline bool curve_equal(const BundleCurveSpec& a, const BundleCurveSpec& b) {
  if (a.base != b.base || a.currency != b.currency) return false;
  if (a.regions.size() != b.regions.size()) return false;
  for (std::size_t i = 0; i < a.regions.size(); ++i) {
    const auto &r = a.regions[i], &t = b.regions[i];
    if (r.scheme != t.scheme || r.knots != t.knots || r.sigma != t.sigma) return false;
    if (r.reg_lambda != t.reg_lambda || r.reg_sigma != t.reg_sigma) return false;
  }
  if (a.turns.size() != b.turns.size()) return false;
  for (std::size_t i = 0; i < a.turns.size(); ++i)
    if (a.turns[i].start != b.turns[i].start || a.turns[i].end != b.turns[i].end) return false;
  return true;
}
inline void hash_instrument(FnvHasher& H, const Instrument& ins) {
  H.i(static_cast<long long>(ins.quote));
  H.i(ins.forecast);
  H.i(ins.pv_currency);
  H.i(ins.fx_num);
  H.i(ins.fx_den);
  H.d(ins.fx_spot);
  H.d(ins.fx_time);
  if (ins.fx_spot_time != 0.0) H.d(ins.fx_spot_time);
  H.i(ins.turn_curve);
  H.i(ins.turn_index);
  H.d(ins.convexity);
  hash_obs(H, ins.obs);
  hash_float(H, ins.fwd);
  hash_float(H, ins.bench);
  hash_fixed(H, ins.fixed);
  // EXCLUDED (the warm-tick RHS): ins.market, ins.band_lower, ins.band_upper, ins.band_decay.
  H.i(static_cast<long long>(ins.combination.size()));
  for (const auto& w : ins.combination) {
    H.d(w.weight);
    hash_instrument(H, w.instrument);  // a portfolio's components are Instruments too
  }
}

}  // namespace fp_detail

// The topology hash. Stable under a market re-quote; changed by any structural edit.
inline std::uint64_t structure_fingerprint(const BundleProblem& p) {
  fp_detail::FnvHasher H;
  H.i(static_cast<long long>(p.curves.size()));
  for (const auto& c : p.curves) {
    H.i(c.base);
    H.i(c.currency);
    H.i(static_cast<long long>(c.regions.size()));
    for (const auto& m : c.regions) {
      H.i(static_cast<long long>(m.scheme));
      H.d(m.sigma);
      H.d(m.reg_lambda);
      H.d(m.reg_sigma);
      H.vd(m.knots);
    }
    H.i(static_cast<long long>(c.turns.size()));
    for (const auto& t : c.turns) {
      H.d(t.start);
      H.d(t.end);
    }
  }
  H.i(static_cast<long long>(p.instruments.size()));
  for (const auto& ins : p.instruments) fp_detail::hash_instrument(H, ins);
  return H.h;
}

// The structural-equality check (no hash): see fp_detail::curve_equal / instrument_equal.
inline bool structure_equal(const BundleProblem& a, const BundleProblem& b) {
  if (a.curves.size() != b.curves.size() || a.instruments.size() != b.instruments.size()) return false;
  for (std::size_t c = 0; c < a.curves.size(); ++c)
    if (!fp_detail::curve_equal(a.curves[c], b.curves[c])) return false;
  for (std::size_t i = 0; i < a.instruments.size(); ++i)
    if (!fp_detail::instrument_equal(a.instruments[i], b.instruments[i])) return false;
  return true;
}

}  // namespace swaps::calibration
