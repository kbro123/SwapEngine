#pragma once
// The rate INDEX reference object (the one typed reference object the builders read).
//
// The engine's calendar / day-count / convention logic is a set of string-keyed FREE functions
// (calendar.hpp / day_count.hpp / conventions.hpp) over the constexpr conventions DB (conventions_data.hpp).
// Those free functions are the single, parity-tested implementation. `Index` is a thin typed VALUE OBJECT over
// them for the one thing the builders ask by object -- an index's own observation conventions -- so an
// instrument builder can hold an Index and query it by method instead of threading raw string ids.
//
// (Until 2026-11-01 this header also carried an ORE-style typed family -- Calendar, DayCount, Convention /
// OisConvention / IborSwapConvention / BasisConvention / XccyConvention and Index::*_convention() producers --
// whose one live idiom was Index(id).par_convention().resolve(), i.e. swap_conv(currency, id). That is
// Index::swap_conv() now; the rest had no reader. Deleted, review item 1.6.)

#include <optional>
#include <string>
#include <utility>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/day_count.hpp"
#include "swaps/conventions_data.hpp"

namespace swaps::build {

namespace cvd = swaps::conventions;

// A rate INDEX -- the pivot reference object the model projects off (USD-SOFR, EUR-EURIBOR-3M, ...). PURE
// reference data: its own observation conventions (fixing calendar, day count, tenor, lags) and NO curve; a
// bundle later realises it as a Curve. Unknown ids are valid as HANDLES (conv() is nullopt) but every
// convention accessor throws for them -- there are no fallbacks (PRINCIPLES.md P2).
struct Index {
  std::string id;

  Index() = default;
  explicit Index(std::string id_) : id(std::move(id_)) {}

  std::optional<cvd::IndexConv> conv() const { return id.empty() ? std::nullopt : cvd::index(id); }
  bool known() const { return conv().has_value(); }
  std::string currency() const { auto c = conv(); return c ? sv_str(c->currency) : std::string(); }
  bool is_overnight() const { auto c = conv(); return c && c->type == std::string_view("overnight"); }
  std::string par_product() const { auto c = conv(); return c ? sv_str(c->par_product) : std::string(); }
  std::string tenor() const { auto c = conv(); return c ? sv_str(c->tenor) : std::string(); }
  int fixing_lag() const { return cvd::require_lag(cvd::require_index(id).fixing_lag, "fixing_lag", id); }  // DB lag (throws if unset)

  // The par-swap conventions this index quotes under: the DB product of (currency, index), flattened to the
  // engine's SwapConv (conventions.hpp swap_conv). The object-to-object bridge Index -> SwapConv -> Instrument.
  SwapConv swap_conv() const { return build::swap_conv(currency(), id); }
};

}  // namespace swaps::build
