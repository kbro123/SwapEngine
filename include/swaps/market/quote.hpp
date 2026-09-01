#pragma once
// First-class MARKET QUOTE value object (swaps::market).
//
// Today a market observation reaches the calibration layer as a bare `double market` on an
// Instrument (calibration/problem.hpp) — no bid, no ask, no source, no timestamp. This header adds a
// small, standalone VALUE TYPE that represents a real two-sided market and RESOLVES to the engine's
// existing calibration target + soft-quote band. It is header-only, QuantLib-free, value-semantic and
// depends on NOTHING in the calibration layer: the hand-off is the plain-POD `CalibrationTarget`
// below, whose fields map ONE-TO-ONE onto Instrument.{market, band_lower, band_upper, band_decay}.
//
// The engine's band semantics (problem.hpp, ~L103-108) that we mirror EXACTLY:
//   * band_upper >  band_lower  => a SOFT target: the residual weight decays from 1 outside the band
//                                  down to a floor `band_decay` inside [band_lower, band_upper].
//   * band_upper <= band_lower (with band_decay == 1) => a plain HARD target (a hard pin): the
//                                  residual is the unweighted (q - market). This is the engine default.
//
// So a two-sided quote becomes a soft band [bid, ask] pulling to mid; a mid-only quote becomes a hard
// pin on the mid — the exact shape the solver already understands.

#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace swaps::market {

// The hand-off CONTRACT between a market Quote and the calibration layer. A plain POD by design: its
// four fields correspond ONE-TO-ONE, and in order, to the calibration Instrument's quote fields —
//   target      <-> Instrument.market
//   band_lower  <-> Instrument.band_lower
//   band_upper  <-> Instrument.band_upper
//   band_decay  <-> Instrument.band_decay
// Copy them straight across (no transform) to feed a calibration Instrument from a Quote. Kept here,
// not pulled from problem.hpp, so the market layer stays independent of the calibration layer.
struct CalibrationTarget {
  double target = 0.0;
  double band_lower = 0.0;
  double band_upper = 0.0;
  double band_decay = 1.0;

  // Convenience predicates mirroring the engine's band test (band_upper > band_lower => soft).
  bool is_soft_band() const { return band_upper > band_lower; }
  bool is_hard_pin() const { return !is_soft_band(); }
};

// A market quote: a mid, optionally a two-sided bid/ask, and optional provenance (source + timestamp).
// Build one with the named constructors Quote::mid(...) or Quote::bid_ask(...). The default band decay
// applied to a two-sided quote is kDefaultDecay (see to_target()).
struct Quote {
  // Default soft-band weight floor handed to a TWO-SIDED quote's CalibrationTarget. Inside [bid, ask]
  // the residual weight decays down to this floor; a smaller value means the solver treats a model
  // value anywhere in the bid/ask band as more nearly satisfied. Must be in (0, 1].
  static constexpr double kDefaultDecay = 0.1;

  // ----- named constructors -------------------------------------------------------------------------

  // A one-sided quote: only a mid is known. Resolves to a HARD pin on the mid.
  static Quote mid(double m, std::string source = std::string(), double timestamp = 0.0) {
    Quote q;
    q.mid_ = m;
    q.bid_ = kNoSide;
    q.ask_ = kNoSide;
    q.source_ = std::move(source);
    q.timestamp_ = timestamp;
    return q;
  }

  // A two-sided quote: a firm bid and ask. The mid is (bid + ask) / 2. Resolves to a SOFT band [bid, ask].
  static Quote bid_ask(double bid, double ask, std::string source = std::string(),
                       double timestamp = 0.0) {
    Quote q;
    q.bid_ = bid;
    q.ask_ = ask;
    q.mid_ = 0.5 * (bid + ask);
    q.source_ = std::move(source);
    q.timestamp_ = timestamp;
    return q;
  }

  // ----- accessors ----------------------------------------------------------------------------------

  bool is_two_sided() const { return !std::isnan(bid_) && !std::isnan(ask_); }

  // The mid: (bid + ask) / 2 for a two-sided quote, else the supplied mid.
  double mid() const { return is_two_sided() ? 0.5 * (bid_ + ask_) : mid_; }
  // The bid / ask. For a one-sided quote both collapse to the mid (there is no genuine two-sided market).
  double bid() const { return is_two_sided() ? bid_ : mid_; }
  double ask() const { return is_two_sided() ? ask_ : mid_; }
  // The bid/ask spread (ask - bid), or 0 for a one-sided quote.
  double spread() const { return is_two_sided() ? (ask_ - bid_) : 0.0; }

  const std::string& source() const { return source_; }
  double timestamp() const { return timestamp_; }

  // ----- THE BRIDGE ---------------------------------------------------------------------------------

  // Map this quote to the engine's calibration target + band (see CalibrationTarget). `decay` is the
  // in-band weight floor used only for a two-sided quote.
  //   two-sided : target = mid, [band_lower, band_upper] = [bid, ask], band_decay = decay  (a SOFT band)
  //   one-sided : target = mid, band_lower = band_upper = 0, band_decay = 1                (a HARD pin)
  // The result's fields feed Instrument.{market, band_lower, band_upper, band_decay} directly.
  CalibrationTarget to_target(double decay = kDefaultDecay) const {
    if (is_two_sided()) {
      return CalibrationTarget{mid(), bid_, ask_, decay};
    }
    return CalibrationTarget{mid_, 0.0, 0.0, 1.0};
  }

 private:
  static constexpr double kNoSide = std::numeric_limits<double>::quiet_NaN();

  double mid_ = 0.0;
  double bid_ = kNoSide;  // NaN sentinel => not supplied (one-sided)
  double ask_ = kNoSide;
  std::string source_;
  double timestamp_ = 0.0;  // a plain serial/epoch field — no clock calls here
};

}  // namespace swaps::market
