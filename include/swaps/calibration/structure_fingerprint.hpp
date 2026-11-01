#pragma once
// STRUCTURAL IDENTITY of a BundleProblem: structure_equal (the O(n) walk a rebind and the P&L explain check) and
// its hash twin structural_stamp (E8), plus the StampedBundle a client rebinds with. "Structure" is everything the
// compiled W-cache depends on -- curve regions, roles, cashflow times, observation schedules, baked constants --
// and never the quoted TARGETS (Instrument.market / band_*), which a warm tick moves over the existing W.
//
// (Until 2026-11-01 this file also carried `structure_fingerprint`, an older RESOLUTION-SENSITIVE hash kept "for a
// client's own caching"; BundleSession::structure_fingerprint() had no reader in the engine, the tests, the
// benches or the web repo. Deleted, review item 1.4. The file keeps its name for the stamp's history.)

#include <cstdint>
#include <stdexcept>
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
    // Only when it adds information: with no carried accrual period and no fixing schedule the effective dates ARE
    // sub_start.front()/sub_end.back(), which obs_equal below compares (whole vectors) anyway.
    const bool eff_needed = c.accrual_set || d.accrual_set || !c.obs.fixing_schedule.empty();
    if (eff_needed && (eff_s(c) != eff_s(d) || eff_e(c) != eff_e(d))) return false;
    if (c.reset_fx != d.reset_fx) return false;
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
  if (a.state.size() != b.state.size()) return false;
  for (std::size_t i = 0; i < a.state.size(); ++i)
    if (a.state[i].curve != b.state[i].curve || a.state[i].index != b.state[i].index || a.state[i].coeff != b.state[i].coeff) return false;
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
    if (r.reg_lambda != t.reg_lambda) return false;
  }
  if (a.turns.size() != b.turns.size()) return false;
  for (std::size_t i = 0; i < a.turns.size(); ++i)
    if (a.turns[i].start != b.turns[i].start || a.turns[i].end != b.turns[i].end) return false;
  return true;
}
}  // namespace fp_detail

// The structural-equality check (no hash): see fp_detail::curve_equal / instrument_equal.
// ---- STRUCTURAL STAMP (E8 prototype) ---------------------------------------------------------------
// A 64-bit hash that is the HASH TWIN of structure_equal below: equal structures stamp equal, and any edit
// structure_equal would reject changes the stamp. It exists so a rebind can compare two integers instead of
// walking every coupon (measured: structure_equal is 192 us of a 320 us rebind on the 8x26 chain).
//
// Two differences from structure_fingerprint above, both REQUIRED to be the twin of the equality check:
//  1. RESOLUTION-INSENSITIVE. A coupon carrying a fixing schedule stamps its SCHEDULE endpoints, never the
//     resolved realized / realized_factor / forecast sub-periods -- exactly what obs_equal compares. So a
//     client's unresolved document and the session's resolved copy stamp the SAME, which is what makes
//     "hash once when the structure is built" hold across set_fixings / set_evaluation_date. (The older
//     resolution-SENSITIVE fingerprint that sat beside it was deleted on 2026-11-01: no reader.)
//  2. It covers the mtm leg and hashes WHOLE edge vectors (the old fingerprint summarised them by endpoints and
//     omits mtm entirely), so it cannot accept a structure the equality check would reject.
namespace fp_detail {

inline void stamp_obs(FnvHasher& H, const pricing::RateObservation& o) {
  H.i(static_cast<long long>(o.fixing_schedule.size()));
  H.s(o.fixing_index);
  H.d(o.tau_index);
  H.d(o.fixing_step);
  H.d(o.fixing_step3);
  H.i(o.compounded);
  if (!o.fixing_schedule.empty()) {  // resolution-insensitive: the schedule, not what it resolved to
    const auto& a = o.fixing_schedule.front();
    const auto& z = o.fixing_schedule.back();
    H.i(a.fixing_date);
    H.d(a.t_start);
    H.i(z.fixing_date);
    H.d(z.t_end);
    return;
  }
  H.vd(o.sub_start);
  H.vd(o.sub_end);
  H.vd(o.weight);
  H.d(o.realized);
  H.d(o.realized_factor);
}

inline void stamp_float(FnvHasher& H, const FloatLeg& lg) {
  H.i(lg.forecast);
  H.i(lg.discount);
  H.i(lg.reset_num);
  H.i(lg.reset_den);
  H.d(lg.fx_spot);
  H.d(lg.fx_spot_time);
  H.i(static_cast<long long>(lg.coupons.size()));
  for (const auto& c : lg.coupons) {
    H.d(c.pay);
    H.d(c.tau_pay);
    H.d(c.spread);
    H.d(c.scale);
    H.d(c.reset_time);
    // the EFFECTIVE exchange dates float_equal compares (the accrual period when carried, else the window)
    H.d(c.accrual_set ? c.accrual_start : (c.obs.sub_start.empty() ? -1.0 : c.obs.sub_start.front()));
    H.d(c.accrual_set ? c.accrual_end : (c.obs.sub_end.empty() ? -1.0 : c.obs.sub_end.back()));
    H.d(c.reset_fx);
    H.i(c.fx_fixing_set);
    if (c.fx_fixing_set) H.d(c.fx_fixing_time);
    stamp_obs(H, c.obs);
  }
}

inline void stamp_fixed(FnvHasher& H, const FixedLeg& lg) {
  H.i(lg.discount);
  H.i(static_cast<long long>(lg.coupons.size()));
  for (const auto& c : lg.coupons) {
    H.d(c.pay);
    H.d(c.tau);
    H.d(c.scale);
  }
}

inline void stamp_instrument(FnvHasher& H, const Instrument& ins) {
  H.i(static_cast<long long>(ins.quote));
  H.i(ins.forecast);
  H.i(ins.pv_currency);
  H.i(ins.fx_num);
  H.i(ins.fx_den);
  H.d(ins.fx_spot);
  H.d(ins.fx_time);
  H.d(ins.fx_spot_time);
  H.i(ins.turn_curve);
  H.i(ins.turn_index);
  H.d(ins.convexity);
  if (!ins.state.empty()) {
    H.i(static_cast<long long>(ins.state.size()));
    for (const auto& r : ins.state) { H.i(r.curve); H.i(r.index); H.d(r.coeff); }
  }
  stamp_obs(H, ins.obs);
  stamp_float(H, ins.fwd);
  stamp_float(H, ins.bench);
  stamp_float(H, ins.mtm);  // instrument_equal compares it; the older fingerprint omits it
  stamp_fixed(H, ins.fixed);
  // EXCLUDED, exactly as structure_equal excludes them: market, band_lower, band_upper, band_decay.
  H.i(static_cast<long long>(ins.combination.size()));
  for (const auto& w : ins.combination) {
    H.d(w.weight);
    stamp_instrument(H, w.instrument);
  }
}

}  // namespace fp_detail

// Bump when any field above changes, so two engine versions can never silently agree on a stale stamp.
inline constexpr std::uint64_t kStructuralStampVersion = 1;

inline std::uint64_t structural_stamp(const BundleProblem& p) {
  fp_detail::FnvHasher H;
  H.i(static_cast<long long>(kStructuralStampVersion));
  H.i(static_cast<long long>(p.curves.size()));
  for (const auto& c : p.curves) {
    H.i(c.base);
    H.i(c.currency);
    H.i(static_cast<long long>(c.regions.size()));
    for (const auto& m : c.regions) {
      H.i(static_cast<long long>(m.scheme));
      H.vd(m.knots);
      H.d(m.sigma);
      H.d(m.reg_lambda);
    }
    H.i(static_cast<long long>(c.turns.size()));
    for (const auto& t : c.turns) {
      H.d(t.start);
      H.d(t.end);
    }
  }
  H.i(static_cast<long long>(p.instruments.size()));
  for (const auto& ins : p.instruments) fp_detail::stamp_instrument(H, ins);
  if (!p.constraints.empty()) {
    H.i(static_cast<long long>(p.constraints.size()));
    for (const auto& ins : p.constraints) fp_detail::stamp_instrument(H, ins);
  }
  return H.h;
}

// The rebind-time check itself (P14: behaviour lives in include/swaps/**, the verb layer only calls it).
// Throws unless `have` is a real stamp of the same structure as `want`.
inline void require_stamp_match(std::uint64_t have, std::uint64_t want, const char* where) {
  if (have != 0 && have == want) return;
  throw std::runtime_error(std::string(where) +
                           ": the structural stamp differs (a knot, scheme, role, schedule or instrument changed, or "
                           "the bundle was not built by cal::make_stamped) — compile a new session; rebind carries "
                           "market targets and bands only");
}

// A bundle that CARRIES the structural stamp of ITS OWN contents. IMMUTABLE by construction: make_stamped
// takes the problem BY VALUE and moves it in, so the stamp is computed from the exact bytes this object
// owns -- a caller mutating its own copy afterwards cannot invalidate it -- and the members are private, so
// the structure cannot be edited out from under the stamp once built. That is what lets a session trust two
// integers instead of walking every coupon: anything that reaches rebind was stamped from what it carries.
//
// Quotes are NOT hashed (structure_equal ignores them too), so a requote is free: mutate targets and bands
// through quotes()/instrument(), and the stamp stays valid by construction. That is the whole design --
// hash the structure once when it is built, requote as often as you like.
class StampedBundle {
 public:
  StampedBundle() = default;
  const BundleProblem& problem() const { return problem_; }
  std::uint64_t stamp() const { return stamp_; }
  // The quote RHS, mutable: targets and bands are not structure (see above).
  Instrument& quotes(std::size_t i) { return problem_.instruments[i]; }
  std::size_t size() const { return problem_.instruments.size(); }

 private:
  friend StampedBundle make_stamped(BundleProblem);
  BundleProblem problem_;
  std::uint64_t stamp_ = 0;
};

inline StampedBundle make_stamped(BundleProblem p) {
  StampedBundle b;
  b.stamp_ = structural_stamp(p);
  b.problem_ = std::move(p);
  return b;
}

// `a_tail_constraints`: that many of a's LAST constraint rows are a's own (a session's installed smoothing rows) and
// are not compared -- b must carry exactly a's other constraint rows.
inline bool structure_equal(const BundleProblem& a, const BundleProblem& b, int a_tail_constraints = 0) {
  if (a.curves.size() != b.curves.size() || a.instruments.size() != b.instruments.size()) return false;
  const std::size_t nca = a.constraints.size() - static_cast<std::size_t>(a_tail_constraints);
  if (nca != b.constraints.size()) return false;
  for (std::size_t c = 0; c < a.curves.size(); ++c)
    if (!fp_detail::curve_equal(a.curves[c], b.curves[c])) return false;
  for (std::size_t i = 0; i < a.instruments.size(); ++i)
    if (!fp_detail::instrument_equal(a.instruments[i], b.instruments[i])) return false;
  for (std::size_t i = 0; i < nca; ++i)
    if (!fp_detail::instrument_equal(a.constraints[i], b.constraints[i])) return false;
  return true;
}

}  // namespace swaps::calibration
