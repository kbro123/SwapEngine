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

// A holiday calendar addressed by its DB id (EUR / USD / USD-SOFR / USD-FED / EURUSD / ... / "NONE" = weekends-only).
struct Calendar {
  std::string id;

  bool is_business_day(const Date& d) const { return swaps::build::is_business_day(id, d); }
  Date adjust(const Date& d, const std::string& bdc) const {  // bdc REQUIRED (no default)
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
  // Calendar-aware form: only BUS/252 (Brazilian business/252) consults `cal_id`; every other day count
  // ignores it and matches the 2-arg form. Pass the product/index calendar when the id may be BUS/252.
  double year_frac(const Date& d1, const Date& d2, const std::string& cal_id) const {
    return swaps::build::year_frac(id, d1, d2, cal_id);
  }
};

// The market-CONVENTION family (ORE-style): a typed view over the conventions DB, resolving to the engine's
// flat SwapConv. `Convention` is the swap-family base — calendar / bdc / day counts / lags / frequency, each
// exposed as its own object (Calendar, DayCount) so callers compose object-to-object. The product subtypes
// (OisConvention, IborSwapConvention, BasisConvention) add ONLY what that product actually has, so the shape
// is self-documenting. Cross-currency (XccyConvention) resolves to XccyConv, a genuinely different shape, so
// it is a sibling rather than a subtype. Normally obtained from an Index (ois_convention() / swap_convention());
// construct one directly only to override the DB default.
struct Convention {
  std::string currency;
  std::string index;        // the projection index id (drives the floating-leg conventions)
  double float_freq = 0.0;  // 0 => the index/product default

  SwapConv resolve() const { return swap_conv(currency, index); }  // float_freq is informational: the DB product decides

  Calendar calendar() const { return Calendar{resolve().calendar}; }
  DayCount fixed_day_count() const { return DayCount{resolve().fixed_dc}; }
  DayCount float_day_count() const { return DayCount{resolve().float_dc}; }
  std::string bdc() const { return resolve().bdc; }
  std::string frequency() const { return resolve().float_freq_tok; }
  int spot_lag() const { return resolve().spot_lag; }
  int pay_lag() const { return resolve().pay_lag; }
};

// OIS conventions: an overnight index vs an annual fixed leg; the float leg COMPOUNDS daily.
struct OisConvention : Convention {
  OisConvention() = default;
  OisConvention(std::string ccy, std::string ix) : Convention{std::move(ccy), std::move(ix), 0.0} {}
  bool compounded() const { return true; }
  std::string fixed_frequency() const { return resolve().float_freq_tok; }  // annual for OIS
};

// Term IBOR swap conventions: a float leg on a fixed tenor (3M / 6M) vs a fixed leg; the float leg does NOT
// compound (each period is one fixing).
struct IborSwapConvention : Convention {
  IborSwapConvention() = default;
  IborSwapConvention(std::string ccy, std::string ix, double float_freq)
      : Convention{std::move(ccy), std::move(ix), float_freq} {}
  bool compounded() const { return false; }
  std::string float_tenor() const { return resolve().float_freq_tok; }  // 3M / 6M
};

// Tenor / index basis conventions: a spread leg on `index` quoted AGAINST a benchmark index (`bench_index`),
// both discounted on the currency's OIS.
struct BasisConvention : Convention {
  std::string bench_index;  // the index this basis is measured against
  BasisConvention() = default;
  BasisConvention(std::string ccy, std::string ix, std::string bench, double float_freq = 0.0)
      : Convention{std::move(ccy), std::move(ix), float_freq}, bench_index(std::move(bench)) {}
};

// Cross-currency MtM basis conventions. Resolves to XccyConv (calendar / bdc / day count / frequency of the
// resetting funding leg) — a different shape from the swap family, hence a sibling, not a Convention subtype.
struct XccyConvention {
  std::string pair;  // e.g. "EURUSD"

  XccyConv resolve() const { return xccy_conv(pair); }
  Calendar calendar() const { return Calendar{resolve().calendar}; }
  DayCount day_count() const { return DayCount{resolve().dc}; }
  std::string bdc() const { return resolve().bdc; }
  std::string frequency() const { return resolve().freq_tok; }
  int spot_lag() const { return resolve().spot_lag; }
  int pay_lag() const { return resolve().pay_lag; }
};

// A rate INDEX — the pivot reference object the model projects off (USD-SOFR, EUR-EURIBOR-3M, ...). PURE
// reference data: its own observation conventions (fixing calendar, day count, tenor, lags) and NO curve; a
// bundle later realises it as a Curve. Unknown ids are valid as HANDLES (conv() is nullopt) but every
// convention accessor throws for them — there are no fallbacks (PRINCIPLES.md P2).
struct Index {
  std::string id;

  Index() = default;
  explicit Index(std::string id_) : id(std::move(id_)) {}

  std::optional<cvd::IndexConv> conv() const { return id.empty() ? std::nullopt : cvd::index(id); }
  bool known() const { return conv().has_value(); }
  std::string currency() const { auto c = conv(); return c ? sv_str(c->currency) : std::string(); }
  bool is_overnight() const { auto c = conv(); return c && c->type == std::string_view("overnight"); }
  DayCount day_count() const { return DayCount{index_day_count(id)}; }        // DB day count (throws if unknown)
  Calendar fixing_calendar() const { return Calendar{index_calendar(id)}; }   // DB calendar (throws if unknown)
  std::string par_product() const { auto c = conv(); return c ? sv_str(c->par_product) : std::string(); }
  std::string tenor() const { auto c = conv(); return c ? sv_str(c->tenor) : std::string(); }
  int fixing_lag() const { auto c = conv(); return c ? c->fixing_lag : 0; }

  // The par-swap CONVENTION this index quotes under — the object-to-object bridge Index -> Convention ->
  // Instrument. `float_freq` overrides the leg frequency (0 = the index/product default). The typed
  // producers below give the ORE-style product-specific convention for cleaner call sites.
  Convention par_convention(double float_freq = 0.0) const { return Convention{currency(), id, float_freq}; }
  OisConvention ois_convention() const { return OisConvention{currency(), id}; }
  IborSwapConvention swap_convention(double float_freq) const {
    return IborSwapConvention{currency(), id, float_freq};
  }
  BasisConvention basis_convention(const std::string& bench_index, double float_freq = 0.0) const {
    return BasisConvention{currency(), id, bench_index, float_freq};
  }
};

}  // namespace swaps::build
