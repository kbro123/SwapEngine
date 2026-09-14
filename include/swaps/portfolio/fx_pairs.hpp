#pragma once
// portfolio/fx_pairs.hpp — which currency pair each cross-currency position's FX spot quotes, and the ONE place a
// position picks up its pair's factor (SC2, owner decision 2026-09-14: FX moves are exact per pair).
//
// A Kind::Xccy position's FX-reset notional is N_i = fx_spot · DF_num / DF_den (pricing/cashflows.hpp xccy_mtm_leg_pv),
// so fx_spot is rate(ccy(num) -> ccy(den)) in market/fx.hpp's direction ("1 base = rate quote"). The pair is DERIVED
// from data the bundle already holds -- the currency tag of the position's mtm_reset_num curve (base) and its
// mtm_reset_den curve (quote) -- never stored a second time on the position.
//
//   * fx_pair_slots(curves, book)            -- once per (bundle, book): the distinct (base tag, quote tag) pairs and
//                                              each position's slot (-1 for a position with no FX spot).
//   * fx_spot_under(fx_spot, slot, factors)  -- a position's spot under a move. The templated book (set_xccy_fx) and the
//                                              compiled book (CompiledMultiCurveBook::set_fx_factors) both call it.
//   * set_xccy_fx(moved, base, slots, f)     -- overwrite a pre-made copy's spots in place; allocation-free.
//
// Bitwise with scaling one position: fx_spot · factor, and a factor of 1 gives the spot back exactly.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/curve_spec.hpp"

namespace swaps::portfolio {

struct FxPairSlots {
  std::vector<int> base, quote;  // per slot: the currency tag of the num / den reset curve
  std::vector<int> slot_of;      // per position: its slot, or -1 (not Kind::Xccy)
  int n_slots() const { return static_cast<int>(base.size()); }
};

inline FxPairSlots fx_pair_slots(const std::vector<pricing::CurveStructure>& curves, const MultiCurveBook& book) {
  FxPairSlots s;
  s.slot_of.assign(book.positions.size(), -1);
  const int n = static_cast<int>(curves.size());
  for (std::size_t k = 0; k < book.positions.size(); ++k) {
    const MultiCurveBook::Position& p = book.positions[k];
    if (p.kind != MultiCurveBook::Kind::Xccy) continue;
    if (p.mtm_reset_num < 0 || p.mtm_reset_num >= n || p.mtm_reset_den < 0 || p.mtm_reset_den >= n)
      throw std::invalid_argument("fx: xccy position " + std::to_string(k) +
                                  " has an FX reset curve outside the bundle, so its currency pair is unknown");
    const int b = curves[static_cast<std::size_t>(p.mtm_reset_num)].currency;
    const int q = curves[static_cast<std::size_t>(p.mtm_reset_den)].currency;
    int slot = 0;
    while (slot < s.n_slots() &&
           !(s.base[static_cast<std::size_t>(slot)] == b && s.quote[static_cast<std::size_t>(slot)] == q))
      ++slot;
    if (slot == s.n_slots()) {
      s.base.push_back(b);
      s.quote.push_back(q);
    }
    s.slot_of[k] = slot;
  }
  return s;
}

inline double fx_spot_under(double fx_spot, int slot, std::span<const double> slot_factor) {
  return slot < 0 ? fx_spot : fx_spot * slot_factor[static_cast<std::size_t>(slot)];
}

// `moved` is a copy of `base` made once; every call overwrites only the xccy positions' spots (`slots` must come from
// `base`). Allocation-free.
inline void set_xccy_fx(MultiCurveBook& moved, const MultiCurveBook& base, const FxPairSlots& slots,
                        std::span<const double> slot_factor) {
  for (std::size_t k = 0; k < base.positions.size(); ++k)
    if (slots.slot_of[k] >= 0)
      moved.positions[k].fx_spot = fx_spot_under(base.positions[k].fx_spot, slots.slot_of[k], slot_factor);
}

}  // namespace swaps::portfolio
