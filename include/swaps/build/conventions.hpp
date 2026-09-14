// swaps::build — resolve per-product conventions from the DB (conventions_data.hpp + the runtime registry).
// QuantLib-free analog of server/compile.py's _conv_from_product / _swap_conv / _index_day_count / _index_calendar.
//
// PRINCIPLES.md P2: NOTHING here is defaulted in code. A missing DB field throws with the row id; an unknown
// id throws; a curve that names no index resolves through currencies[ccy].default_swap_product (a DB row),
// not a currency/frequency heuristic. (Until 2026-09-08 this file carried 60 of the engine's 130 literal
// conventions: `? "USD"`, `? "ACT/360"`, `? "ModifiedFollowing"`, a float-frequency → EURIBOR product guess,
// and an EURUSD-only xccy convention.)
#ifndef SWAPS_BUILD_CONVENTIONS_HPP
#define SWAPS_BUILD_CONVENTIONS_HPP

#include <cctype>
#include <string>
#include <string_view>

#include "swaps/conventions_data.hpp"

namespace swaps::build {

namespace cvd = swaps::conventions;

// Resolved par-swap conventions (compile.py's `conv` dict). Every field is set by conv_from_product from the
// DB row; -1 / "" mean "not resolved" and the builders treat them as errors, never as a default.
struct SwapConv {
  std::string calendar, bdc, fixed_dc, float_dc, float_freq_tok, fixed_freq_tok;
  // DB products[].float_leg.compounding: how the float leg turns daily fixings into its coupon rate.
  // "compounded" (the ISDA OIS-COMPOUND product, and the default when the row is silent) telescopes to one
  // DF bracket; "averaged" is the H.15-style arithmetic average (the FF/SOFR basis leg, the FF 1M future).
  // Empty => compounded. Read by build::float_leg_from (item 17 / E4.E L2, 2026-09-10).
  std::string float_compounding;
  int spot_lag = -1, pay_lag = -1;
  bool zero_coupon = false;  // DB products[].zero_coupon: ONE period spot->maturity on both legs, quoted as an
                             // annually-compounded rate (QuoteKind::ZeroCouponRate; BRL DI×Pre). No frequencies.
  std::string product_id;  // the DB row this came from (diagnostics)
};

inline std::string sv_str(std::string_view v) { return std::string(v); }
inline std::string upper(std::string s) {
  for (char& c : s) c = char(std::toupper((unsigned char)c));
  return s;
}

// A swap-type product row (ois / irs / basis) -> SwapConv. Throws on any missing field.
inline SwapConv conv_from_product(const cvd::ProductConv& p) {
  if (p.type != "ois" && p.type != "irs" && p.type != "basis")
    throw std::invalid_argument("conventions DB: '" + sv_str(p.id) + "' is a '" + sv_str(p.type) +
                                "' product, not a swap (ois/irs/basis)");
  SwapConv c;
  c.product_id = sv_str(p.id);
  c.calendar = sv_str(cvd::require_field(p.calendar, "calendar", p.id));
  c.bdc = sv_str(cvd::require_field(p.bdc, "bdc", p.id));
  c.spot_lag = cvd::require_lag(p.spot_lag, "spot_lag", p.id);
  c.pay_lag = cvd::require_lag(p.payment_lag, "payment_lag", p.id);
  c.float_dc = sv_str(cvd::require_field(p.floating.day_count, "float/spread leg day_count", p.id));
  c.float_compounding = sv_str(p.floating.compounding);  // "" => compounded (the shipped default)
  if (!c.float_compounding.empty() && c.float_compounding != "compounded" && c.float_compounding != "averaged")
    throw std::invalid_argument("conventions DB: product '" + sv_str(p.id) + "' float_leg.compounding '" +
                                c.float_compounding + "' is neither 'compounded' nor 'averaged'");
  c.zero_coupon = p.zero_coupon;
  if (p.zero_coupon) {
    // ONE period spot->maturity on both legs (zero_coupon_swap): the row carries day counts, never frequencies.
    if (p.type == "basis") throw std::invalid_argument("conventions DB: '" + sv_str(p.id) + "' is a zero_coupon basis product — unsupported");
    if (!p.floating.frequency.empty() || !p.fixed.frequency.empty())
      throw std::invalid_argument("conventions DB: zero_coupon product '" + sv_str(p.id) + "' must not carry leg frequencies");
    c.fixed_dc = sv_str(cvd::require_field(p.fixed.day_count, "fixed_leg day_count", p.id));
    return c;
  }
  c.float_freq_tok = sv_str(cvd::require_field(p.floating.frequency, "float/spread leg frequency", p.id));
  if (p.type == "basis") {
    // A basis swap has no fixed leg; the annuity used to convert the spread is built on the QUOTED leg's
    // schedule and day count (the old code hard-wired an annual 30E/360-style fixed leg here).
    c.fixed_dc = c.float_dc;
    c.fixed_freq_tok = c.float_freq_tok;
  } else {
    c.fixed_dc = sv_str(cvd::require_field(p.fixed.day_count, "fixed_leg day_count", p.id));
    c.fixed_freq_tok = sv_str(cvd::require_field(p.fixed.frequency, "fixed_leg frequency", p.id));
  }
  return c;
}

// Conventions for a par swap on `index` (its DB par_product). With NO index the currency's DB default swap
// product is used (currencies[ccy].default_swap_product). Both paths throw on an unknown id.
inline SwapConv swap_conv(const std::string& currency, const std::string& index) {
  if (!index.empty()) {
    const cvd::IndexConv ix = cvd::require_index(index);
    return conv_from_product(cvd::require_product(cvd::require_field(ix.par_product, "par_product", ix.id)));
  }
  const cvd::CurrencyConv cc = cvd::require_currency(upper(currency));
  return conv_from_product(cvd::require_product(cvd::require_field(cc.default_swap_product, "default_swap_product", cc.code)));
}

// MtM xccy basis conventions for a currency PAIR (DB product "XCCY-MTM-<PAIR>", e.g. EURUSD).
struct XccyConv {
  std::string calendar, bdc, dc, freq_tok, product_id;
  int spot_lag = -1, pay_lag = -1;
  // Notional exchange settlement lags (the engine books every exchange ON its accrual date: only 0 is modelled)
  // and the FX reset fixing lag on fx_reset_calendar (CARR 2021: the FX sets 2 business days before each period).
  int exchange_lag_initial = -1, exchange_lag_intermediate = -1, exchange_lag_final = -1, fx_reset_lag = -1;
  std::string fx_reset_calendar;
};
inline XccyConv xccy_conv(const std::string& pair) {
  const std::string pid = "XCCY-MTM-" + upper(pair);
  const cvd::ProductConv p = cvd::require_product(pid);
  XccyConv x;
  x.product_id = pid;
  x.calendar = sv_str(cvd::require_field(p.calendar, "calendar", p.id));
  x.bdc = sv_str(cvd::require_field(p.bdc, "bdc", p.id));
  x.dc = sv_str(cvd::require_field(p.floating.day_count, "usd/quoted leg day_count", p.id));
  x.freq_tok = sv_str(cvd::require_field(p.frequency, "frequency", p.id));
  x.spot_lag = cvd::require_lag(p.spot_lag, "spot_lag", p.id);
  x.pay_lag = cvd::require_lag(p.payment_lag, "payment_lag", p.id);
  x.exchange_lag_initial = cvd::require_lag(p.exchange_lag_initial, "exchange_lag_initial", p.id);
  x.exchange_lag_intermediate = cvd::require_lag(p.exchange_lag_intermediate, "exchange_lag_intermediate", p.id);
  x.exchange_lag_final = cvd::require_lag(p.exchange_lag_final, "exchange_lag_final", p.id);
  x.fx_reset_lag = cvd::require_lag(p.fx_reset_fixing_lag, "fx_reset_fixing_lag", p.id);
  x.fx_reset_calendar = sv_str(cvd::require_field(p.fx_reset_calendar, "fx_reset_calendar", p.id));
  return x;
}

inline std::string index_day_count(const std::string& index) {
  const cvd::IndexConv ix = cvd::require_index(index);
  return sv_str(cvd::require_field(ix.day_count, "day_count", ix.id));
}
inline std::string index_calendar(const std::string& index) {
  const cvd::IndexConv ix = cvd::require_index(index);
  return sv_str(cvd::require_field(ix.calendar, "calendar", ix.id));
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_CONVENTIONS_HPP
