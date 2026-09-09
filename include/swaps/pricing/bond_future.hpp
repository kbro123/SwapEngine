#pragma once
// pricing/bond_future.hpp — the deliverable-basket / cheapest-to-deliver (CTD) math for bond futures.
//
// A bond future settles by physical delivery of ONE bond from a basket of eligible government bonds; the
// short chooses which to deliver. The exchange normalises the coupons across the basket with a CONVERSION
// FACTOR (CF): the clean price, per unit face, of the deliverable priced to the futures' NOTIONAL COUPON
// yield (6% for CME US Treasury / 10-year note futures, and for Eurex), evaluated on the first delivery day
// with the remaining maturity ROUNDED DOWN to a whole number of quarters. The short's cash at delivery is
//     invoice price = futures_price · CF + accrued_at_delivery.
// The CTD is the bond that is cheapest to buy-and-deliver: the MAX implied repo rate (equivalently, to a
// first order, the MIN net basis). This header is the same dual-use shape as pricing/bond.hpp — QuantLib-
// free, templated on Scalar so Scalar=double prices and Scalar=ad::Dual yields the analytic gradient from
// the same code — and it REUSES the bond kernel: the conversion factor IS a yield-space clean price at 6%,
// so it is bond_clean_from_yield in closed form.
//
// CONVENTION (documented once): cme_conversion_factor() below is the CME 6% notional closed form (US
// Treasury bond + 2/3/5/10-year note futures; the identical algebra with a 6% notional is Eurex's Bund/Bobl/
// Schatz factor). The one nuance CME/Eurex differ on is the maturity-rounding granularity — CME rounds the
// residual to whole QUARTERS for the bond/10y and to whole MONTHS for the 2/3/5y note; the closed form here
// takes (n whole years, z residual months already rounded) so the CALLER (the build/date layer or the API
// verb) owns the rounding rule and this stays pure math. UST is done first per the engine's roadmap.

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace swaps::pricing {

// =================================================================================================
// CONVERSION FACTOR — CME/Eurex 6% notional closed form.
// =================================================================================================
// CF = price per unit face of the deliverable priced to the 6% notional yield on the first delivery day,
// with the remaining life rounded down to (n years, z months), z in {0,3,6,9} for the bond/10y. In closed
// form (per CME Rulebook; verified against building the bond and calling bond_clean_from_yield at 0.06):
//     v = z            if z < 7,   else z − 6
//     A = 1 / 1.03^(v/6)
//     B = (coupon/2)·(6 − v)/6                          (the accrued the CF is quoted CLEAN of)
//     C = 1 / 1.03^(2n)   if z < 7,   else 1 / 1.03^(2n+1)
//     D = (coupon/0.06)·(1 − C)
//     CF = A·(coupon/2 + C + D) − B
// Self-checks: a 6% bond an exact number of half-years out (z=0) has CF = 1; a 10% 20y note (z=0) has
// CF = 1.4623 (Hull). Affine in `coupon`, so AAD-clean; n,z are integer convention data.
template <class Scalar>
Scalar cme_conversion_factor(Scalar coupon, int n, int z, double notional_coupon) {  // notional coupon = contract DATA (bond_futures[].notional_coupon)
  const double semi = 1.0 + notional_coupon / 2.0;  // 1.03 for a 6% notional
  const int v = (z < 7) ? z : z - 6;
  const double A = 1.0 / std::pow(semi, double(v) / 6.0);
  const Scalar B = (coupon / Scalar(2)) * Scalar(double(6 - v) / 6.0);
  const double C = (z < 7) ? 1.0 / std::pow(semi, double(2 * n))
                           : 1.0 / std::pow(semi, double(2 * n + 1));
  const Scalar D = (coupon / Scalar(notional_coupon)) * Scalar(1.0 - C);
  return Scalar(A) * (coupon / Scalar(2) + Scalar(C) + D) - B;
}

// =================================================================================================
// BASKET / CTD math — gross & net basis, implied repo, invoice price.
// =================================================================================================
// Per-deliverable inputs already resolved from the Market + schedule (per unit face; a builder / the API
// verb fills these from dates via build/bond.hpp — this stays date-free like the rest of pricing/):
//   clean             observed clean price now
//   conversion_factor CF for this futures contract
//   accrued_now       accrued at the cash settlement date
//   accrued_delivery  accrued at the futures delivery date
//   interim_coupons   coupons PAID in (settle, delivery]: {amount, days_from_payment_to_delivery}
struct DeliverableInput {
  double clean = 0.0;
  double conversion_factor = 0.0;
  double accrued_now = 0.0;
  double accrued_delivery = 0.0;
  std::vector<std::pair<double, double>> interim_coupons;  // {amount, days_to_delivery}
};

// Invoice price the short receives at delivery, per unit face: futures · CF + accrued_at_delivery.
template <class Scalar>
Scalar bond_future_invoice_price(Scalar futures, Scalar conversion_factor, Scalar accrued_delivery) {
  return futures * conversion_factor + accrued_delivery;
}

// Gross basis (price points): clean − futures · CF. The raw richness of the cash bond to the futures,
// before financing carry. Positive ⇒ the cash bond trades rich to the delivery-normalised futures.
template <class Scalar>
Scalar bond_future_gross_basis(Scalar clean, Scalar futures, Scalar conversion_factor) {
  return clean - futures * conversion_factor;
}

// Implied repo rate (ACT/360, simple): the financing rate at which buy-bond-now / finance / deliver breaks
// exactly even — the annualised cash-and-carry return. With P_buy = clean + accrued_now the purchase dirty,
// P_inv = futures·CF + accrued_delivery the delivery proceeds, and interim coupons c_k received d_k days
// before delivery:
//     IRR = 360 · (P_inv + Σ c_k − P_buy) / (P_buy·d − Σ c_k·d_k)
// (d = days settle→delivery). The bond with the HIGHEST IRR is the CTD.
template <class Scalar>
Scalar bond_future_implied_repo(const DeliverableInput& in, Scalar futures, double days_to_delivery, double repo_basis) {  // repo_basis = 360/365 per bond_futures[].repo_day_count
  const Scalar p_buy = Scalar(in.clean + in.accrued_now);
  const Scalar p_inv = futures * Scalar(in.conversion_factor) + Scalar(in.accrued_delivery);
  Scalar coupon_sum = Scalar(0), coupon_time = Scalar(0);  // Σ c_k , Σ c_k·d_k
  for (const auto& ck : in.interim_coupons) {
    coupon_sum += Scalar(ck.first);
    coupon_time += Scalar(ck.first) * Scalar(ck.second);
  }
  const Scalar denom = p_buy * Scalar(days_to_delivery) - coupon_time;
  return Scalar(repo_basis) * (p_inv + coupon_sum - p_buy) / denom;
}

// Net basis (price points): gross basis net of carry — the forward clean price at delivery (bond bought
// dirty now and financed at `repo`, interim coupons reinvested at `repo`) minus futures·CF:
//     F_dirty = P_buy·(1 + repo·d/360) − Σ c_k·(1 + repo·d_k/360)
//     net_basis = (F_dirty − accrued_delivery) − futures·CF
// Zero exactly when repo == the implied repo rate. The CTD is (to first order) the MIN net basis.
template <class Scalar>
Scalar bond_future_net_basis(const DeliverableInput& in, Scalar futures, Scalar repo,
                             double days_to_delivery, double repo_basis) {
  const Scalar p_buy = Scalar(in.clean + in.accrued_now);
  Scalar fwd_dirty = p_buy * (Scalar(1) + repo * Scalar(days_to_delivery / repo_basis));
  for (const auto& ck : in.interim_coupons)
    fwd_dirty -= Scalar(ck.first) * (Scalar(1) + repo * Scalar(ck.second / repo_basis));
  const Scalar fwd_clean = fwd_dirty - Scalar(in.accrued_delivery);
  return fwd_clean - futures * Scalar(in.conversion_factor);
}

// Everything a caller wants for one deliverable, in one struct.
template <class Scalar>
struct DeliverableResult {
  Scalar conversion_factor = Scalar(0);
  Scalar gross_basis = Scalar(0);
  Scalar net_basis = Scalar(0);
  Scalar implied_repo = Scalar(0);
  Scalar invoice_price = Scalar(0);
};

// Analyse one deliverable against a futures price and a funding repo (the funding rate is what makes net
// basis meaningful; implied repo is intrinsic and independent of it).
template <class Scalar>
DeliverableResult<Scalar> analyze_deliverable(const DeliverableInput& in, Scalar futures, Scalar repo,
                                              double days_to_delivery, double repo_basis) {
  DeliverableResult<Scalar> r;
  r.conversion_factor = Scalar(in.conversion_factor);
  r.gross_basis = bond_future_gross_basis(Scalar(in.clean), futures, Scalar(in.conversion_factor));
  r.net_basis = bond_future_net_basis(in, futures, repo, days_to_delivery, repo_basis);
  r.implied_repo = bond_future_implied_repo(in, futures, days_to_delivery, repo_basis);
  r.invoice_price = bond_future_invoice_price(futures, Scalar(in.conversion_factor),
                                              Scalar(in.accrued_delivery));
  return r;
}

// CTD selection over a basket: the index of the MAX implied repo rate (the standard cash-and-carry CTD
// rule; ties → the first). `results[i]` corresponds to deliverable i. Returns 0 for an empty basket only if
// the caller passed one — callers should guard non-empty.
template <class Scalar>
std::size_t select_ctd(const std::vector<DeliverableResult<Scalar>>& results) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < results.size(); ++i)
    if (results[i].implied_repo > results[best].implied_repo) best = i;
  return best;
}

}  // namespace swaps::pricing
