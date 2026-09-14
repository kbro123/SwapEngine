#pragma once
// derive/fx_move.hpp — a move's FX bumps -> one EXACT factor per currency pair (SC2, owner decisions 2026-09-14).
// scenario, scenario_grid and var all resolve FX here.
//
// THE RULE
//   1. A bump {base B, quote Q, rel r} multiplies rate(B -> Q) by (1 + r). Bumps on the same UNORDERED pair compound in
//      request order, each taken in the orientation of that pair's first bump (a later Q/B bump divides).
//   2. The merged bumps are the edges of a currency graph, which must be a FOREST: a cycle (EURUSD, GBPUSD and EURGBP all
//      bumped) over-determines a cross and is refused, naming the pair.
//   3. Currencies no bump touches HOLD against one another (factor 1). When the request names an `fx_pivot`, they also
//      hold against the pivot.
//   4. A position on pair (A, B) takes the product of the edge factors along the path A -> B (times along an edge's
//      orientation, divided against it); A == B gives 1. No path -- the move does not determine that pair, e.g. EURGBP
//      under an EURUSD-only bump with no pivot -- is REFUSED (owner decision: no defaulted pivot currency).
// A directly bumped pair's factor is 1.0 · (1 + r), bitwise the single-pair factor the verbs used before SC2; its
// inverse 1.0 / (1 + r); a cross g1 / g2.
//
// Currency ids: the bundle's currency_codes first (id == curve currency tag), then any other ISO code a request names
// (GBP may be only an intermediate node), interned at SETUP. factors() allocates nothing: its scratch is sized at setup
// to (ids + 1) nodes, the +1 being the node every untouched currency holds against.
//
// book_fx_moves is the request-level entry: it builds the resolver only when the book has an xccy position AND some move
// bumps FX -- so only then are bundle.currency_codes required -- and resolves every move before any calibration.

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "swaps/portfolio/fx_pairs.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/curve_spec.hpp"

namespace swaps::derive {

struct FxBump {
  std::string base, quote;
  double rel = 0.0;  // rate(base -> quote)' = rate · (1 + rel)
};

struct FxBumpRef {
  int base = -1, quote = -1;  // currency ids from FxMoveResolver::intern
  double rel = 0.0;
};

class FxMoveResolver {
 public:
  // Throws if a code is empty or repeated, or if an xccy slot's currency tag has no code.
  FxMoveResolver(std::vector<std::string> bundle_codes, portfolio::FxPairSlots slots,
                 const std::optional<std::string>& pivot)
      : codes_(std::move(bundle_codes)), slots_(std::move(slots)) {
    for (std::size_t i = 0; i < codes_.size(); ++i) {
      if (codes_[i].empty())
        throw std::invalid_argument("fx: bundle.currency_codes[" + std::to_string(i) + "] is empty");
      for (std::size_t j = 0; j < i; ++j)
        if (codes_[j] == codes_[i])
          throw std::invalid_argument("fx: bundle.currency_codes names " + codes_[i] + " twice");
    }
    const int n = static_cast<int>(codes_.size());
    for (int s = 0; s < slots_.n_slots(); ++s)
      for (const int tag : {slots_.base[static_cast<std::size_t>(s)], slots_.quote[static_cast<std::size_t>(s)]})
        if (tag < 0 || tag >= n)
          throw std::invalid_argument(
              "fx: an FX move on a book with cross-currency positions needs bundle.currency_codes naming every curve "
              "currency (currency tag " + std::to_string(tag) + " has no code)");
    size_scratch();
    if (pivot) pivot_ = intern(*pivot);
  }

  // SETUP: the id of an ISO code, appending it when the bundle does not name it.
  int intern(const std::string& iso) {
    if (iso.empty()) throw std::invalid_argument("fx: an FX bump or fx_pivot names an empty currency");
    for (std::size_t i = 0; i < codes_.size(); ++i)
      if (codes_[i] == iso) return static_cast<int>(i);
    codes_.push_back(iso);
    size_scratch();
    return static_cast<int>(codes_.size()) - 1;
  }
  // SETUP: the most bumps any one factors() call will see.
  void reserve(std::size_t max_bumps) {
    max_bumps_ = std::max(max_bumps_, max_bumps);
    size_scratch();
  }

  // HOT, allocation-free: one factor per slot (slot_factor.size() == slots().n_slots()).
  void factors(std::span<const FxBumpRef> bumps, std::span<double> slot_factor) const {
    if (bumps.size() > max_bumps_) throw std::logic_error("fx: more bumps than FxMoveResolver::reserve allowed");
    if (slot_factor.size() != static_cast<std::size_t>(slots_.n_slots()))
      throw std::logic_error("fx: slot_factor does not hold one factor per slot");
    const int n = static_cast<int>(codes_.size());
    const int rest = n;
    std::fill(touched_.begin(), touched_.end(), char{0});
    int n_edges = 0;
    for (const FxBumpRef& b : bumps) {  // rule 1
      if (b.base < 0 || b.base >= n || b.quote < 0 || b.quote >= n)
        throw std::logic_error("fx: a bump names a currency id that was never interned");
      if (b.base == b.quote)
        throw std::invalid_argument("fx: a bump names " + codes_[static_cast<std::size_t>(b.base)] +
                                    " against itself");
      int e = 0;
      while (e < n_edges && !((eu_[e] == b.base && ev_[e] == b.quote) || (eu_[e] == b.quote && ev_[e] == b.base))) ++e;
      if (e == n_edges) {
        eu_[e] = b.base;
        ev_[e] = b.quote;
        eg_[e] = 1.0 * (1.0 + b.rel);
        ++n_edges;
      } else if (eu_[e] == b.base) {
        eg_[e] *= (1.0 + b.rel);
      } else {
        eg_[e] /= (1.0 + b.rel);
      }
      touched_[static_cast<std::size_t>(b.base)] = touched_[static_cast<std::size_t>(b.quote)] = 1;
    }
    for (int c = 0; c <= n; ++c) parent_[static_cast<std::size_t>(c)] = c;  // rule 2
    for (int e = 0; e < n_edges; ++e) {
      const int ru = find(eu_[e]), rv = find(ev_[e]);
      if (ru == rv)
        throw std::invalid_argument("fx: the bumps over-determine " + codes_[static_cast<std::size_t>(eu_[e])] +
                                    codes_[static_cast<std::size_t>(ev_[e])] +
                                    " (its rate already follows from the other bumped pairs)");
      parent_[static_cast<std::size_t>(ru)] = rv;
    }
    for (int c = 0; c < n; ++c)  // rule 3
      if (!touched_[static_cast<std::size_t>(c)]) add_edge(n_edges, c, rest);
    if (pivot_ >= 0 && touched_[static_cast<std::size_t>(pivot_)]) add_edge(n_edges, pivot_, rest);
    for (int s = 0; s < slots_.n_slots(); ++s) {  // rule 4
      const int a = slots_.base[static_cast<std::size_t>(s)], b = slots_.quote[static_cast<std::size_t>(s)];
      if (a == b) {
        slot_factor[static_cast<std::size_t>(s)] = 1.0;
        continue;
      }
      std::fill(seen_.begin(), seen_.end(), char{0});
      int top = 0;
      stack_[static_cast<std::size_t>(top++)] = a;
      seen_[static_cast<std::size_t>(a)] = 1;
      val_[static_cast<std::size_t>(a)] = 1.0;
      while (top > 0) {
        const int u = stack_[static_cast<std::size_t>(--top)];
        for (int e = 0; e < n_edges; ++e) {
          if (eu_[e] == u && !seen_[static_cast<std::size_t>(ev_[e])]) {
            val_[static_cast<std::size_t>(ev_[e])] = val_[static_cast<std::size_t>(u)] * eg_[e];
            seen_[static_cast<std::size_t>(ev_[e])] = 1;
            stack_[static_cast<std::size_t>(top++)] = ev_[e];
          } else if (ev_[e] == u && !seen_[static_cast<std::size_t>(eu_[e])]) {
            val_[static_cast<std::size_t>(eu_[e])] = val_[static_cast<std::size_t>(u)] / eg_[e];
            seen_[static_cast<std::size_t>(eu_[e])] = 1;
            stack_[static_cast<std::size_t>(top++)] = eu_[e];
          }
        }
      }
      if (!seen_[static_cast<std::size_t>(b)])
        throw std::invalid_argument("fx: the move does not determine " + codes_[static_cast<std::size_t>(a)] +
                                    codes_[static_cast<std::size_t>(b)] +
                                    " (a cross-currency position's pair): bump it, or name an 'fx_pivot' that the "
                                    "unbumped currencies hold against");
      slot_factor[static_cast<std::size_t>(s)] = val_[static_cast<std::size_t>(b)];
    }
  }

  const portfolio::FxPairSlots& slots() const { return slots_; }

 private:
  int find(int c) const {
    while (parent_[static_cast<std::size_t>(c)] != c) {
      parent_[static_cast<std::size_t>(c)] = parent_[static_cast<std::size_t>(parent_[static_cast<std::size_t>(c)])];
      c = parent_[static_cast<std::size_t>(c)];
    }
    return c;
  }
  void add_edge(int& n_edges, int u, int v) const {
    eu_[static_cast<std::size_t>(n_edges)] = u;
    ev_[static_cast<std::size_t>(n_edges)] = v;
    eg_[static_cast<std::size_t>(n_edges)] = 1.0;
    ++n_edges;
  }
  void size_scratch() {
    const std::size_t nodes = codes_.size() + 1, edges = max_bumps_ + nodes;
    eu_.resize(edges);
    ev_.resize(edges);
    eg_.resize(edges);
    parent_.resize(nodes);
    stack_.resize(nodes);
    val_.resize(nodes);
    touched_.resize(nodes);
    seen_.resize(nodes);
  }

  std::vector<std::string> codes_;  // id -> ISO code
  portfolio::FxPairSlots slots_;
  int pivot_ = -1;
  std::size_t max_bumps_ = 0;
  mutable std::vector<int> eu_, ev_, parent_, stack_;
  mutable std::vector<double> eg_, val_;
  mutable std::vector<char> touched_, seen_;
};

// A request's FX moves, resolved once: the book's pair slots and, per move, one factor per slot (EMPTY when the move
// moves no spot). The resolver -- and so bundle.currency_codes -- is needed only when the book has an xccy position and
// some move bumps FX.
struct BookFxMoves {
  portfolio::FxPairSlots slots;
  std::vector<std::vector<double>> factor;  // per move
};

inline BookFxMoves book_fx_moves(const std::vector<std::string>& currency_codes,
                                 const std::vector<pricing::CurveStructure>& curves,
                                 const portfolio::MultiCurveBook& book, const std::vector<std::vector<FxBump>>& moves,
                                 const std::optional<std::string>& fx_pivot) {
  BookFxMoves out;
  out.slots = portfolio::fx_pair_slots(curves, book);
  out.factor.resize(moves.size());
  bool any_bump = false;
  std::size_t max_bumps = 0;
  for (const std::vector<FxBump>& m : moves) {
    any_bump = any_bump || !m.empty();
    max_bumps = std::max(max_bumps, m.size());
  }
  if (out.slots.n_slots() == 0 || !any_bump) return out;
  FxMoveResolver fx(currency_codes, out.slots, fx_pivot);
  fx.reserve(max_bumps);
  std::vector<FxBumpRef> refs;
  refs.reserve(max_bumps);
  for (std::size_t k = 0; k < moves.size(); ++k) {
    if (moves[k].empty()) continue;
    refs.clear();
    for (const FxBump& b : moves[k]) {
      const int base = fx.intern(b.base);
      const int quote = fx.intern(b.quote);
      refs.push_back({base, quote, b.rel});
    }
    out.factor[k].assign(static_cast<std::size_t>(out.slots.n_slots()), 1.0);
    fx.factors(refs, out.factor[k]);
  }
  return out;
}

}  // namespace swaps::derive
