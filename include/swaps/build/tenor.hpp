// swaps::build — a typed Tenor value object. Today a tenor travels as a raw string ("3M","1Y","ON",
// "IMM U27") that schedule::resolve re-parses on every use. Tenor parses that token ONCE and carries it as
// a typed quantity: for the common n<unit> tokens a normalized (count, unit) pair; for special tokens
// (ON, TN, SP, IMM codes, ISO dates) the verbatim string, deferred to schedule::resolve. Introducing it
// changes NO behaviour — resolve() ALWAYS delegates to swaps::build::resolve(token(), ...), so the typed
// wrapper produces byte-identical dates to the existing raw-string path (see tests/build_tenor_test.cpp).
#ifndef SWAPS_BUILD_TENOR_HPP
#define SWAPS_BUILD_TENOR_HPP

#include <cctype>
#include <stdexcept>
#include <string>
#include <tuple>

#include "swaps/build/date.hpp"
#include "swaps/build/schedule.hpp"

namespace swaps::build {

// A parsed tenor. Normalized tokens hold an integer count + a unit; special tokens keep the raw string and
// lean on schedule::resolve for calendar/IMM/ISO logic (never reimplemented here).
struct Tenor {
  enum class Unit { Day, Week, Month, Year };

  // Parse "3M","6M","1Y","2W","10D" into (count, unit). Anything else (ON/TN/SP, IMM like "U27", ISO
  // dates, "IMM U27") is stored verbatim as a special token and resolved via schedule::resolve.
  explicit Tenor(const std::string& token) : token_(trim(token)) {
    if (token_.empty()) throw std::invalid_argument("empty tenor");
    const char last = char(std::tolower((unsigned char)token_.back()));
    if (last == 'd' || last == 'w' || last == 'm' || last == 'y') {
      // Normalized only when the prefix is a non-empty run of digits (no sign, no fraction) — this keeps
      // "U27" (whose 'w'? no) and ISO dates out of the normalized bucket and safely into passthrough.
      const std::string prefix = token_.substr(0, token_.size() - 1);
      if (!prefix.empty() && all_digits(prefix)) {
        count_ = std::stoi(prefix);
        unit_ = unit_of(last);
        special_ = false;
        return;
      }
    }
    special_ = true;  // ON, TN, SP, IMM code, ISO date, or anything schedule::resolve understands.
  }

  bool is_special() const { return special_; }
  int count() const { return count_; }         // meaningful only when !is_special()
  Unit unit() const { return unit_; }           // meaningful only when !is_special()

  // The stored (trimmed) token, exactly as schedule::resolve expects it.
  const std::string& token() const { return token_; }
  // Canonical rendering: reconstructed "<count><UNIT>" for a normalized tenor, the raw token otherwise.
  std::string to_string() const {
    if (special_) return token_;
    return std::to_string(count_) + unit_letter(unit_);
  }

  // Resolve to a calendar date. ALWAYS delegates to the parity-tested free function so a Tenor is a pure
  // wrapper over the existing path — identical behaviour for normalized and special tokens alike.
  Date resolve(const Date& value_date, const std::string& cal_id, const std::string& bdc, int spot_lag) const {
    return swaps::build::resolve(token_, value_date, cal_id, bdc, spot_lag);
  }

  // Approximate act/365-ish size for sorting / coupon sizing (mirrors conventions::period_years:
  // D=1/365, W=7/365, M=1/12, Y=1). Special tokens have no normalized size -> 0.0.
  double years() const {
    if (special_) return 0.0;
    switch (unit_) {
      case Unit::Day:   return count_ * (1.0 / 365.0);
      case Unit::Week:  return count_ * (7.0 / 365.0);
      case Unit::Month: return count_ * (1.0 / 12.0);
      case Unit::Year:  return count_ * 1.0;
    }
    return 0.0;
  }

  // Equality / ordering by (unit, count) for the normalized case. If either side is special, fall back to
  // a stable comparison of the raw token strings (and normalized sorts before special).
  bool operator==(const Tenor& o) const {
    if (special_ || o.special_) return token_ == o.token_;
    return unit_ == o.unit_ && count_ == o.count_;
  }
  bool operator!=(const Tenor& o) const { return !(*this == o); }
  bool operator<(const Tenor& o) const {
    if (special_ || o.special_) {
      if (special_ != o.special_) return !special_;  // normalized < special
      return token_ < o.token_;
    }
    return std::tie(unit_, count_) < std::tie(o.unit_, o.count_);
  }

 private:
  static std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
  }
  static bool all_digits(const std::string& s) {
    for (const char c : s)
      if (c < '0' || c > '9') return false;
    return true;
  }
  static Unit unit_of(char lower) {
    switch (lower) {
      case 'd': return Unit::Day;
      case 'w': return Unit::Week;
      case 'm': return Unit::Month;
      default:  return Unit::Year;  // 'y'
    }
  }
  static char unit_letter(Unit u) {
    switch (u) {
      case Unit::Day:   return 'D';
      case Unit::Week:  return 'W';
      case Unit::Month: return 'M';
      case Unit::Year:  return 'Y';
    }
    return 'Y';
  }

  std::string token_;              // trimmed source token, fed verbatim to schedule::resolve
  int count_ = 0;                  // normalized count (n<unit>), valid when !special_
  Unit unit_ = Unit::Year;         // normalized unit,          valid when !special_
  bool special_ = true;            // true for ON/TN/SP/IMM/ISO passthrough tokens
};

}  // namespace swaps::build

#endif  // SWAPS_BUILD_TENOR_HPP
