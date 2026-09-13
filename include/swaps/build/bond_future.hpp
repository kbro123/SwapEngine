// swaps::build — bond-future DELIVERY BASKET analytics (E7 stage 3.3). The `bond_future` verb's whole computation,
// moved out of api/bond_future.cpp so it is unit-testable and reachable by the oracles: for each deliverable, the
// exchange conversion factor (rounded as the exchange rounds it), gross / net basis, implied repo and invoice
// price; and the cheapest-to-deliver. Contract parameters are DATA (conventions.json bond_futures[]): the CF
// notional coupon, the maturity rounding, the repo day count, the default deliverable convention and the number of
// decimals the exchange rounds the factor to. The date-free basis math is pricing/bond_future.hpp.
//
// Pinned against CME's published factors (tests/delivery_basket_test.cpp, CME IR232 worked examples). Until
// 2026-09-13 the verb used the RAW factor everywhere, so invoice / basis / implied repo were off by F·(raw − CF).
#ifndef SWAPS_BUILD_BOND_FUTURE_HPP
#define SWAPS_BUILD_BOND_FUTURE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/build/bond.hpp"
#include "swaps/pricing/bond_future.hpp"

namespace swaps::build {

// Whole years and months from the FIRST DAY of the delivery month to maturity, the months rounded DOWN to a
// multiple of `round_months` (3: CME bond / 10-year quarters; 1: 2 / 3 / 5-year whole months). Counting from the 1st
// (CME IR232) means a maturity's day of month never costs a month; until 2026-09-13 the count ran from the
// first_delivery date's own day, which lost one for a maturity earlier in its month than that day.
struct ConversionTerm {
  int years = 0, months = 0;
};
inline ConversionTerm conversion_term(const Date& first_delivery, const Date& maturity, int round_months) {
  if (round_months <= 0 || 12 % round_months != 0)
    throw std::invalid_argument("bond future: the maturity rounding must divide 12 months");
  const int total = std::max(0, (maturity.year() - first_delivery.year()) * 12 +
                                    (int(maturity.month()) - int(first_delivery.month())));
  return {total / 12, (total % 12 / round_months) * round_months};
}

// The factor as the exchange publishes and invoices it: rounded half-up to `decimals` places (CME: 4).
inline double round_conversion_factor(double cf, int decimals) {
  const double scale = std::pow(10.0, decimals);
  return std::round(cf * scale) / scale;
}

// One bond offered into delivery. `convention` empty => the contract's deliverable convention; `settle` absent =>
// the request's value date. Dating is seasoned (`issue`) or when-issued (`dated` + `first_coupon`).
struct Deliverable {
  std::string id;  // echoed; names the CTD
  std::string convention;
  std::optional<Date> settle;
  Date maturity;
  std::optional<Date> issue, dated, first_coupon;
  double coupon = 0.0;  // annual rate (0.045 = 4.5%)
  std::optional<int> freq;
  double clean = 0.0;  // observed clean price, per unit face
};

struct DeliveryBasketRequest {
  std::string contract;  // bond_futures[] id -- required
  Date value_date, first_delivery;
  std::optional<Date> delivery;  // absent => first_delivery
  double futures_price = 0.0;    // per unit face
  double repo = 0.0;             // funding rate on the contract's repo day count
  std::optional<double> notional_coupon;  // overrides the contract row
  std::optional<int> round_months;        // overrides the contract row
  std::vector<Deliverable> basket;
};

struct DeliverableAnalysis {
  std::string id;
  pricing::DeliverableResult<double> result;
};
struct DeliveryBasketResult {
  std::vector<DeliverableAnalysis> rows;  // basket order
  std::optional<std::size_t> ctd;         // max implied repo; absent for an empty basket
};

inline DeliveryBasketResult analyze_delivery_basket(const DeliveryBasketRequest& r) {
  if (r.contract.empty())
    throw std::invalid_argument("bond future: needs 'contract' (a bond_futures[] row id, e.g. CME-TY)");
  const conventions::BondFutureConv fut = conventions::require_bond_future(r.contract);
  const double notional_coupon = r.notional_coupon.value_or(fut.notional_coupon);
  const int round_months = r.round_months.value_or(fut.maturity_rounding_months);
  const double repo_basis = day_count_basis(std::string(fut.repo_day_count));
  const Date delivery = r.delivery.value_or(r.first_delivery);

  DeliveryBasketResult out;
  std::vector<pricing::DeliverableResult<double>> results;
  for (const Deliverable& d : r.basket) {
    BondTerms t;
    t.convention = d.convention.empty() ? std::string(fut.deliverable_convention) : d.convention;
    t.maturity = d.maturity;
    t.issue = d.issue;
    t.dated = d.dated;
    t.first_coupon = d.first_coupon;
    t.coupon = d.coupon;
    t.freq = d.freq;
    const Date settle = d.settle.value_or(r.value_date);
    // Accrued at cash settlement and at delivery: the same bond built at both dates (yield space, no curve).
    t.settle = settle;
    const BuiltBond now = bond_from_terms(t, r.value_date);
    t.settle = delivery;
    const BuiltBond at_delivery = bond_from_terms(t, r.value_date);

    const ConversionTerm term = conversion_term(r.first_delivery, d.maturity, round_months);
    pricing::DeliverableInput in;
    in.clean = d.clean;
    in.conversion_factor = round_conversion_factor(
        pricing::cme_conversion_factor<double>(d.coupon, term.years, term.months, notional_coupon),
        fut.conversion_factor_decimals);
    in.accrued_now = now.accrued;
    in.accrued_delivery = at_delivery.accrued;
    // Coupons paid in (settle, delivery]: reinvested at repo by the basis math.
    const int freq = bond_frequency(t);
    Date ref_start;
    for (const Date& cd : coupon_dates_backward(t.dated ? *t.dated : *t.issue, d.maturity, freq, ref_start))
      if (cd > settle && cd <= delivery) in.interim_coupons.emplace_back(d.coupon / double(freq), double(delivery - cd));

    results.push_back(
        pricing::analyze_deliverable<double>(in, r.futures_price, r.repo, double(delivery - settle), repo_basis));
    out.rows.push_back({d.id, results.back()});
  }
  if (!results.empty()) out.ctd = pricing::select_ctd(results);
  return out;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_BOND_FUTURE_HPP
