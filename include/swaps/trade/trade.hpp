#pragma once
// swaps::trade — the BOOKED DEAL object the review found missing.
//
// The engine has always modelled a market QUOTE (calibration::Instrument: legs + a quote transform + a
// mid, with NO contract rate and NO trade identity) and a fast VALUATION shape (portfolio::MultiCurveBook::
// Position: resolved coupon vectors + curve roles + a signed notional + a contract rate). What sat BETWEEN
// them — a trade a desk actually books, with an id, a direction, a contract rate struck at execution and a
// settlement type — had no home. `Trade` is that object.
//
// A Trade is a persistent, human-meaningful record: "trade id XYZ, we pay fixed 3.75% on 100mm USD SOFR,
// 10y, cash settled, forecasting curve role 0 and discounting role 0 of the calibrated bundle". It is
// QuantLib-free, header-only and value-semantic like the rest of build/ — it holds resolved build::Dates,
// not tenor tokens. Its one job on the hot path is to.materialize itself into the Position the reprice
// kernel already consumes: `to_position()` runs the SAME build::float_leg / build::fixed_coupons builders
// the calibration instruments use, so a booked deal can never price under a different cashflow model than
// the curve it is valued on was calibrated with.

#include <string>
#include <utility>
#include <vector>

#include "swaps/build/conventions.hpp"   // SwapConv
#include "swaps/build/date.hpp"          // Date
#include "swaps/build/instruments.hpp"   // build::float_leg, build::fixed_coupons (cal:: / px:: aliases)
#include "swaps/portfolio/portfolio.hpp"  // MultiCurveBook::Position

namespace swaps::trade {

namespace build = swaps::build;
namespace portfolio = swaps::portfolio;

// The product family. Vanilla (OIS / IRS) fixed-vs-float swap only for now; the enum leaves room for the
// next bookable deals (basis, xccy, FRA, ...) to slot in beside it without reshaping the Trade record.
enum class Kind { Swap };

// Direction, from OUR book's point of view: which leg WE pay.
//   Fixed => payer-of-fixed  (we pay the contract fixed rate, receive the floating index).
//   Float => receiver-of-fixed (we pay floating, receive the contract fixed rate).
enum class Pay { Fixed, Float };

// Settlement style. A booked attribute; it does not change the discounted-cashflow value of a vanilla swap
// (both legs are still the same coupons), so `to_position()` ignores it — it is carried for the record /
// downstream lifecycle logic, not the NPV kernel.
enum class Settlement { Cash, Physical };

// A booked deal.
struct Trade {
  Kind kind = Kind::Swap;
  std::string id;         // trade id (booking reference)
  double notional = 0.0;  // ABSOLUTE notional (always >= 0); direction is carried by `pay`, not the sign
  Pay pay = Pay::Fixed;   // payer-of-fixed vs receiver-of-fixed
  double fixed_rate = 0.0;  // the CONTRACT rate struck at execution — a booked attribute a quote has not
  std::string currency;   // e.g. "USD"
  std::string index;      // projection index id, e.g. "USD-SOFR"

  // Resolved dates (value-semantic — tokens are resolved by the factory/caller, never stored as strings).
  build::Date trade_date;  // when the deal was struck
  build::Date effective;   // first accrual start (spot-start swaps: ~= spot(value_date))
  build::Date maturity;    // final accrual end

  Settlement settle = Settlement::Cash;

  // Bundle curve ROLES for valuation (integer indices into the calibrated bundle's per-curve handles,
  // exactly as MultiCurveBook::Position uses them). A vanilla swap forecasts one and discounts another.
  int forecast_curve = 0;
  int discount_curve = 0;

  // Factory for the common case: a vanilla fixed-vs-float swap. Stores the resolved effective/maturity
  // dates; `trade_date` defaults to `effective` (callers can overwrite the field afterwards).
  static Trade vanilla_swap(std::string id, double notional, Pay pay, double fixed_rate,
                            std::string currency, std::string index, const build::Date& effective,
                            const build::Date& maturity, int forecast_role, int discount_role,
                            Settlement settle = Settlement::Cash) {
    Trade t;
    t.kind = Kind::Swap;
    t.id = std::move(id);
    t.notional = notional;
    t.pay = pay;
    t.fixed_rate = fixed_rate;
    t.currency = std::move(currency);
    t.index = std::move(index);
    t.trade_date = effective;
    t.effective = effective;
    t.maturity = maturity;
    t.settle = settle;
    t.forecast_curve = forecast_role;
    t.discount_curve = discount_role;
    return t;
  }

  // ---- THE KEY METHOD -----------------------------------------------------------------------------------
  // Materialize this booked deal into the fast valuation Position the reprice kernel consumes. The coupon
  // schedules are built with the SAME build::float_leg / build::fixed_coupons builders the calibration
  // instruments use (par_swap), rolled from spot(value_date) under `conv` to `maturity` — so a trade can
  // never price on a different cashflow model than the curve it is valued against was calibrated with.
  //
  // FIELD-BY-FIELD mapping onto portfolio::MultiCurveBook::Position (whose value() prices a swap as
  // notional * (float_leg_pv(fwd,disc) - fixed_rate * annuity(fixed)), i.e. payer-of-fixed per unit
  // notional):
  //
  //   Position.kind          <- Kind::Swap                  (vanilla multi-curve swap branch of value()).
  //   Position.notional      <- signed_notional()           MAGNITUDE = this->notional; SIGN carries the
  //                                                          direction: +notional for a PAYER of fixed
  //                                                          (NPV = float - fixed, exactly what value()
  //                                                          computes), -notional for a RECEIVER of fixed
  //                                                          (NPV = fixed - float = -(float - fixed)). This
  //                                                          is the documented payer sign convention.
  //   Position.float_coupons <- build::float_leg(...).coupons   the floating leg, conv.float_freq_tok /
  //                                                              conv.float_dc, rolled to `maturity`.
  //   Position.fwd_curve     <- forecast_curve              curve that FORECASTS the float index.
  //   Position.disc_curve    <- discount_curve              curve that DISCOUNTS the float payments.
  //   Position.fixed_coupons <- build::fixed_coupons(...).coupons   annual fixed leg, conv.fixed_dc.
  //   Position.fixed_curve   <- discount_curve              a vanilla swap discounts BOTH legs on the same
  //                                                          curve, so the fixed leg's discount role is the
  //                                                          trade's discount_curve.
  //   Position.fixed_rate    <- fixed_rate                  the booked CONTRACT rate.
  //
  // Fields NOT populated (left at their defaults) and WHY:
  //   Position.mtm_coupons / mtm_fwd_curve / mtm_disc_curve / mtm_reset_num / mtm_reset_den / fx_spot
  //     — these belong to the Kind::Xccy branch (an FX-resetting foreign funding leg). A vanilla single-
  //       currency swap has no such leg, so they stay default (empty / 0 / 1.0) and value() never reads
  //       them for a Kind::Swap position. They are the extension point for a future Kind::Xccy trade.
  portfolio::MultiCurveBook::Position to_position(const build::Date& value_date,
                                                  const build::SwapConv& conv) const {
    portfolio::MultiCurveBook::Position p;
    p.kind = portfolio::MultiCurveBook::Kind::Swap;
    p.notional = signed_notional();

    // Floating leg: forecast on our forecast role, discount on our discount role.
    build::cal::FloatLeg fl =
        build::float_leg(value_date, conv, maturity, forecast_curve, discount_curve,
                         conv.float_freq_tok, conv.float_dc);
    p.float_coupons = std::move(fl.coupons);
    p.fwd_curve = forecast_curve;
    p.disc_curve = discount_curve;

    // Fixed leg: annual coupons discounted on the same (discount) curve, carrying the contract rate.
    build::cal::FixedLeg fx = build::fixed_coupons(value_date, conv, maturity, discount_curve);
    p.fixed_coupons = std::move(fx.coupons);
    p.fixed_curve = discount_curve;
    p.fixed_rate = fixed_rate;

    return p;
  }

  // Direction-signed notional for the payer-of-fixed valuation kernel. Payer of fixed => +, receiver => -.
  double signed_notional() const { return pay == Pay::Fixed ? notional : -notional; }
};

}  // namespace swaps::trade
