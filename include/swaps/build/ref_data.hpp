#pragma once
// First-class reference-data OBJECTS (Phase 3 of the object-model rework, docs/OBJECT_MODEL_PLAN.md).
//
// The engine's calendar / day-count / convention logic is a set of string-keyed FREE functions
// (calendar.hpp / day_count.hpp / conventions.hpp) over the constexpr conventions DB (conventions_data.hpp).
// Those free functions are the single, parity-tested implementation and stay exactly as they are. THIS header
// adds thin typed VALUE OBJECTS over them, so the object model can hold an `Index` / `Calendar` / `DayCount`
// / `Convention` and query it by method, instead of threading raw string ids around. Every method delegates
// to the corresponding free function, so behaviour — and every oracle / compile-parity gate — is unchanged.
//
// The `Index` is the pivot the rest of the model references (Phase 4: an Instrument projects off an Index,
// which the bundle later realises as a Curve). It is kept PURE — its own observation conventions and NO curve.

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

// A holiday calendar addressed by its DB id (EUR / USD / USD-SOFR / USD-FED / EURUSD, or "" = weekends-only).
struct Calendar {
  std::string id;

  bool is_business_day(const Date& d) const { return swaps::build::is_business_day(id, d); }
  Date adjust(const Date& d, const std::string& bdc = "ModifiedFollowing") const {
    return swaps::build::adjust(id, d, bdc);
  }
  Date advance_bd(const Date& d, int n) const { return swaps::build::advance_bd(id, d, n); }
  Date roll(Date d, int step) const { return swaps::build::roll(id, d, step); }
  bool empty() const { return id.empty(); }
};

// A day-count convention (ACT/360, ACT/365F, 30E/360, 30U/360, ACT/ACT[.ISDA] ...).
struct DayCount {
  std::string id;
  double year_frac(const Date& d1, const Date& d2) const { return swaps::build::year_frac(id, d1, d2); }
};

// A market CONVENTION for building instruments off a (currency, index): the leg conventions a swap carries
// (calendar, business-day rule, fixed/float day counts, spot/pay lags, float frequency). It is a first-class
// object: `resolve()` flattens it to the engine's SwapConv, and the typed accessors expose each convention
// as its own object (calendar as a Calendar, each day count as a DayCount) so a caller can compose
// object-to-object instead of digging into raw strings. Normally obtained from an Index (par_convention());
// build one directly only to override the DB default.
struct Convention {
  std::string currency;
  std::string index;        // the projection index id (drives the floating-leg conventions)
  double float_freq = 0.0;  // 0 => the index/product default

  SwapConv resolve() const { return swap_conv(currency, float_freq, index); }

  Calendar calendar() const { return Calendar{resolve().calendar}; }
  DayCount fixed_day_count() const { return DayCount{resolve().fixed_dc}; }
  DayCount float_day_count() const { return DayCount{resolve().float_dc}; }
  std::string bdc() const { return resolve().bdc; }
  std::string frequency() const { return resolve().float_freq_tok; }
  int spot_lag() const { return resolve().spot_lag; }
  int pay_lag() const { return resolve().pay_lag; }
};

// A rate INDEX — the pivot reference object the model projects off (USD-SOFR, EUR-EURIBOR-3M, ...). PURE
// reference data: its own observation conventions (fixing calendar, day count, tenor, lags) and NO curve; a
// bundle later realises it as a Curve. Unknown ids stay valid (empty conventions, sensible fallbacks).
struct Index {
  std::string id;

  Index() = default;
  explicit Index(std::string id_) : id(std::move(id_)) {}

  std::optional<cvd::IndexConv> conv() const { return id.empty() ? std::nullopt : cvd::index(id); }
  bool known() const { return conv().has_value(); }
  std::string currency() const { auto c = conv(); return c ? sv_str(c->currency) : std::string(); }
  bool is_overnight() const { auto c = conv(); return c && c->type == std::string_view("overnight"); }
  DayCount day_count() const { return DayCount{index_day_count(id)}; }        // DB day count, else ACT/360
  Calendar fixing_calendar() const { return Calendar{index_calendar(id)}; }   // DB calendar, else weekends-only
  std::string par_product() const { auto c = conv(); return c ? sv_str(c->par_product) : std::string(); }
  std::string tenor() const { auto c = conv(); return c ? sv_str(c->tenor) : std::string(); }
  int fixing_lag() const { auto c = conv(); return c ? c->fixing_lag : 0; }

  // The par-swap CONVENTION this index quotes under — the object-to-object bridge Index -> Convention ->
  // Instrument. `float_freq` overrides the leg frequency (0 = the index/product default).
  Convention par_convention(double float_freq = 0.0) const { return Convention{currency(), id, float_freq}; }
};

}  // namespace swaps::build
