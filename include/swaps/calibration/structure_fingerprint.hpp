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
  H.i(static_cast<long long>(lg.coupons.size()));
  for (const auto& c : lg.coupons) {
    hash_obs(H, c.obs);
    H.d(c.pay);
    H.d(c.tau_pay);
    H.d(c.spread);
    H.d(c.scale);  // FX-spot constant baked into the W-cache's per-coupon k, so it belongs to the structure
    H.d(c.reset_time);
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

inline void hash_instrument(FnvHasher& H, const Instrument& ins) {
  H.i(static_cast<long long>(ins.quote));
  H.i(ins.forecast);
  H.i(ins.pv_currency);
  H.i(ins.fx_num);
  H.i(ins.fx_den);
  H.d(ins.fx_spot);
  H.d(ins.fx_time);
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

}  // namespace swaps::calibration
