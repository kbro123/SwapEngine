// swaps::build — resolve per-product conventions from the DB (conventions_data.hpp). QuantLib-free analog of
// server/compile.py's _conv_from_product / _swap_conv / _index_day_count / _index_calendar. The DB is the
// single source of truth; the currency+frequency heuristic reproduces the index->par_product mapping for
// specs saved before the index selector (now the par_product field, added to the generated DB).
#ifndef SWAPS_BUILD_CONVENTIONS_HPP
#define SWAPS_BUILD_CONVENTIONS_HPP

#include <cctype>
#include <cmath>
#include <string>
#include <string_view>

#include "swaps/conventions_data.hpp"

namespace swaps::build {

namespace cvd = swaps::conventions;

// Resolved par-swap conventions (compile.py's `conv` dict).
struct SwapConv {
  std::string calendar, bdc, fixed_dc, float_dc, float_freq_tok;
  int spot_lag = 2, pay_lag = 2;
};

inline std::string sv_str(std::string_view v) { return std::string(v); }
inline std::string upper(std::string s) {
  for (char& c : s) c = char(std::toupper((unsigned char)c));
  return s;
}

inline SwapConv conv_from_product(const cvd::ProductConv& p, const std::string& fallback_cal,
                                  const std::string& fallback_freq) {
  SwapConv c;
  c.calendar = p.calendar.empty() ? fallback_cal : sv_str(p.calendar);
  c.bdc = p.bdc.empty() ? "ModifiedFollowing" : sv_str(p.bdc);
  c.fixed_dc = p.fixed.day_count.empty() ? "ACT/360" : sv_str(p.fixed.day_count);
  c.float_dc = p.floating.day_count.empty() ? "ACT/360" : sv_str(p.floating.day_count);
  c.spot_lag = p.spot_lag >= 0 ? p.spot_lag : 2;      // Python p.get("spot_lag", 2)
  c.pay_lag = p.payment_lag >= 0 ? p.payment_lag : 0;  // Python p.get("payment_lag", 0)
  c.float_freq_tok = p.floating.frequency.empty() ? fallback_freq : sv_str(p.floating.frequency);
  return c;
}

// (calendar, bdc, fixed/float day counts, spot/pay lag, float frequency) for a par swap (compile._swap_conv).
inline SwapConv swap_conv(const std::string& currency, double float_freq, const std::string& index = "") {
  const std::string cur = upper(currency.empty() ? "USD" : currency);
  if (!index.empty()) {
    if (auto ix = cvd::index(index)) {
      if (!ix->par_product.empty()) {
        if (auto p = cvd::product(ix->par_product))
          return conv_from_product(*p, ix->calendar.empty() ? "USD" : sv_str(ix->calendar), "1Y");
      }
    }
  }
  std::string pid, freq;
  if (std::abs(float_freq - 0.25) < 1e-6) { pid = "EUR-EURIBOR-3M-IRS"; freq = "3M"; }
  else if (std::abs(float_freq - 0.5) < 1e-6) { pid = "EUR-EURIBOR-6M-IRS"; freq = "6M"; }
  else { pid = (cur == "EUR") ? "EUR-ESTR-OIS" : "USD-SOFR-OIS"; freq = "1Y"; }
  if (auto p = cvd::product(pid)) return conv_from_product(*p, (cur == "EUR" ? "EUR" : "USD"), freq);
  return SwapConv{(cur == "EUR" ? "EUR" : "USD"), "ModifiedFollowing", "ACT/360", "ACT/360", freq, 2, 2};
}

// EUR/USD MtM xccy basis conventions (compile._xccy_conv). The DB's top-level "frequency" isn't in the
// generated ProductConv, so the quarterly default is applied (matches the JSON + Python fallback).
struct XccyConv {
  std::string calendar, bdc, dc, freq_tok;
  int spot_lag = 2, pay_lag = 2;
};
inline XccyConv xccy_conv() {
  if (auto p = cvd::product("XCCY-MTM-EURUSD")) {
    return {p->calendar.empty() ? "EURUSD" : sv_str(p->calendar),
            p->bdc.empty() ? "ModifiedFollowing" : sv_str(p->bdc),
            p->floating.day_count.empty() ? "ACT/360" : sv_str(p->floating.day_count),  // usd_leg -> floating
            "3M", p->spot_lag >= 0 ? p->spot_lag : 2, p->payment_lag >= 0 ? p->payment_lag : 2};
  }
  return {"EURUSD", "ModifiedFollowing", "ACT/360", "3M", 2, 2};
}

inline std::string index_day_count(const std::string& index) {
  if (!index.empty())
    if (auto ix = cvd::index(index))
      if (!ix->day_count.empty()) return sv_str(ix->day_count);
  return "ACT/360";
}
inline std::string index_calendar(const std::string& index) {  // "" -> observation falls back to weekends-only
  if (!index.empty())
    if (auto ix = cvd::index(index)) return sv_str(ix->calendar);
  return "";
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_CONVENTIONS_HPP
