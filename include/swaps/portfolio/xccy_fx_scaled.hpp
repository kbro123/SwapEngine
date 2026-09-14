#pragma once
// portfolio/xccy_fx_scaled.hpp — a book under a relative FX move (E7 stage 5.1).
//
// A MultiCurveBook position carries no currency pair, so an FX bump reaches a book as ONE compounded factor on
// every xccy position's fx_spot (its FX-reset notional); a non-xccy position is untouched. That is exact for a
// single-pair xccy book, and two bumped pairs multiply (SC2, an owner decision, is whether to refuse that).
// scenario, scenario_grid and var each did this by hand.
//
//   * xccy_fx_scaled(book, factor) — the scaled copy (a factor of 1 is an exact copy).
//   * XccyFxScaledBooks            — the compiled reprice twin per distinct factor, built on first use and cached,
//                                    so a grid's fx column reuses one compiled book across every rate cell.

#include <cmath>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "swaps/portfolio/compiled_multi.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace swaps::portfolio {

inline MultiCurveBook xccy_fx_scaled(MultiCurveBook book, double factor) {
  for (auto& p : book.positions)
    if (p.kind == MultiCurveBook::Kind::Xccy) p.fx_spot *= factor;
  return book;
}

class XccyFxScaledBooks {
 public:
  XccyFxScaledBooks(std::vector<pricing::CurveStructure> curves, MultiCurveBook book)
      : curves_(std::move(curves)), book_(std::move(book)) {}

  // The compiled book at `factor`. Factors are keyed to 1e-12, so two that round together share the book built for
  // the first of them.
  const CompiledMultiCurveBook& at(double factor) {
    const long long key = std::llround(factor * 1e12);
    const auto it = by_key_.find(key);
    if (it != by_key_.end()) return *it->second;
    auto compiled = std::make_unique<CompiledMultiCurveBook>(curves_, xccy_fx_scaled(book_, factor));
    return *(by_key_[key] = std::move(compiled));
  }
  const MultiCurveBook& book() const { return book_; }

 private:
  std::vector<pricing::CurveStructure> curves_;
  MultiCurveBook book_;
  std::map<long long, std::unique_ptr<CompiledMultiCurveBook>> by_key_;
};

}  // namespace swaps::portfolio
