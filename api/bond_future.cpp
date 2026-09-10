// Bond-future seam: the stateless "bond_future" run_json verb — deliverable-basket / cheapest-to-deliver
// analytics (curve-free, no bundle). For each bond in a delivery basket it returns the exchange CONVERSION
// FACTOR (CME/Eurex 6% notional closed form), the GROSS and NET basis, the IMPLIED REPO rate and the
// INVOICE price, and it selects the CTD (max implied repo). Mirrors api/bond.cpp: a builder fills the plain
// bond structs (build/bond.hpp) from dates + terms, and the templated kernel (pricing/bond_future.hpp) does
// the basis math. QuantLib-free.
//
// Units are the model's decimals (like the bonds verb): coupon/repo are absolute rates (0.06 = 6%), prices
// and basis per unit face (1.0 = par). The thin Python/Excel client converts to the 32nds / bp desk basis.
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "swaps/api/bond_future.hpp"
#include "swaps/api/json_util.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/build/date.hpp"
#include "swaps/pricing/bond_future.hpp"

namespace swaps::api {

namespace json = boost::json;
namespace b = swaps::build;
namespace cvd = swaps::conventions;
namespace px = swaps::pricing;

namespace {

// Whole (n years, z months) from the first delivery day to maturity, the month remainder rounded DOWN to
// `round_months` (3 → CME bond/10y quarters {0,3,6,9}; 1 → 2/3/5y note whole months). A partial trailing
// month (maturity day-of-month before the first-delivery day-of-month) is dropped — CME rounds the term
// DOWN before the CF calc.
void maturity_rounding(const b::Date& first_delivery, const b::Date& maturity, int round_months, int& n,
                       int& z) {
  int months = (maturity.year() - first_delivery.year()) * 12 +
               (int(maturity.month()) - int(first_delivery.month()));
  if (int(maturity.day()) < int(first_delivery.day())) --months;  // drop the partial final month
  if (months < 0) months = 0;
  n = months / 12;
  const int z_raw = months % 12;
  z = (z_raw / round_months) * round_months;
}
}  // namespace

std::string bond_future_json(const json::object& request) {
  const json::object& top = request;
  const json::object& o = top.contains("bond_future") && top.at("bond_future").is_object()
                              ? top.at("bond_future").as_object()
                              : top;
  if (!o.contains("basket") || !o.at("basket").is_array())
    throw std::invalid_argument("bond_future: missing 'basket' array");
  const std::string vd_iso = js(o, "value_date");
  const std::string fd_iso = js(o, "first_delivery");
  if (vd_iso.empty() || fd_iso.empty())
    throw std::invalid_argument("bond_future: 'value_date' and 'first_delivery' are required (YYYY-MM-DD)");
  const b::Date value = b::Date::from_iso(vd_iso);
  const b::Date first_delivery = b::Date::from_iso(fd_iso);
  const std::string del_iso = js(o, "delivery");
  const b::Date delivery = del_iso.empty() ? first_delivery : b::Date::from_iso(del_iso);
  const double futures = jd(o, "futures_price", 0.0);
  const double repo = jd(o, "repo", 0.0);
  // The CONTRACT row (bond_futures[]) supplies the conversion-factor notional coupon, the maturity rounding,
  // the deliverable bond convention and the repo day-count basis; explicit request fields override.
  const std::string contract_id = js(o, "contract");
  if (contract_id.empty()) throw std::invalid_argument("bond_future: missing 'contract' (a bond_futures[] row id, e.g. CME-TY)");
  const cvd::BondFutureConv fut = cvd::require_bond_future(contract_id);
  const double notional_coupon = jd(o, "notional_coupon", fut.notional_coupon);
  const int round_months = static_cast<int>(jd(o, "round_months", fut.maturity_rounding_months));
  const double repo_basis = b::day_count_basis(std::string(fut.repo_day_count));
  if (round_months <= 0 || 12 % round_months != 0)
    throw std::invalid_argument("bond_future: 'round_months' must divide 12");

  std::vector<double> cf_v, gross_v, net_v, irr_v, inv_v;
  std::vector<std::string> ids;
  std::vector<px::DeliverableResult<double>> results;
  const json::array& arr = o.at("basket").as_array();
  for (auto* v : {&cf_v, &gross_v, &net_v, &irr_v, &inv_v}) v->reserve(arr.size());
  results.reserve(arr.size());

  for (const auto& bv : arr) {
    const json::object& bo = bv.as_object();
    const std::string maturity_iso = js(bo, "maturity");
    if (maturity_iso.empty() || !bo.contains("coupon") || !bo.contains("clean"))
      throw std::invalid_argument("bond_future: each basket bond needs 'maturity', 'coupon' and 'clean'");
    const b::Date maturity = b::Date::from_iso(maturity_iso);
    const double coupon = jd(bo, "coupon", 0.0);
    const double clean = jd(bo, "clean", 0.0);

    const std::string conv_id = js(bo, "convention").empty() ? std::string(fut.deliverable_convention) : js(bo, "convention");
    const px::YieldConvention yc = b::yield_convention(conv_id);
    const int freq = bo.contains("freq") ? static_cast<int>(jd(bo, "freq", 0.0))
                                         : static_cast<int>(yc.freq + 0.5);
    const std::string settle_iso = js(bo, "settle");
    const b::Date settle = settle_iso.empty() ? value : b::Date::from_iso(settle_iso);

    // Build the deliverable at the cash-settlement date and again at the delivery date, to read accrued at
    // each (the yield-space clean/accrued path — no curve). WHEN-ISSUED via dated + first_coupon.
    const std::string dated_iso = js(bo, "dated"), first_cpn_iso = js(bo, "first_coupon");
    if (dated_iso.empty() != first_cpn_iso.empty())
      throw std::invalid_argument("bond_future: a when-issued bond needs BOTH 'dated' and 'first_coupon'");
    const bool is_wi = !dated_iso.empty();
    const std::string issue_iso = js(bo, "issue");
    if (!is_wi && issue_iso.empty())
      throw std::invalid_argument("bond_future: a seasoned bond needs 'issue' (dated date)");

    auto build_at = [&](const b::Date& stl) -> b::BuiltBond {
      if (is_wi)
        return b::when_issued_bond(value, b::Date::from_iso(dated_iso), b::Date::from_iso(first_cpn_iso),
                                   maturity, coupon, freq, stl, yc.stub, yc.final_period_simple);
      b::FixedBondTerms t;
      t.value_date = value;
      t.settle = stl;
      t.issue = b::Date::from_iso(issue_iso);
      t.maturity = maturity;
      t.coupon = coupon;
      t.freq = freq;
      t.stub = yc.stub;
      t.final_period_simple = yc.final_period_simple;
      return b::fixed_rate_bond(t);
    };
    const b::BuiltBond bb_now = build_at(settle);
    const b::BuiltBond bb_del = build_at(delivery);

    // Conversion factor from the ROUNDED remaining maturity at the first delivery day.
    int n = 0, z = 0;
    maturity_rounding(first_delivery, maturity, round_months, n, z);
    const double cf = px::cme_conversion_factor<double>(coupon, n, z, notional_coupon);

    // Interim coupons paid in (settle, delivery]: enumerate the coupon grid, reinvest at repo.
    px::DeliverableInput in;
    in.clean = clean;
    in.conversion_factor = cf;
    in.accrued_now = bb_now.accrued;
    in.accrued_delivery = bb_del.accrued;
    const double cpn_per_period = coupon / double(freq);
    const b::Date issue_or_dated = is_wi ? b::Date::from_iso(dated_iso) : b::Date::from_iso(issue_iso);
    b::Date ref_start;
    for (const b::Date& cd : b::coupon_dates_backward(issue_or_dated, maturity, freq, ref_start))
      if (cd > settle && cd <= delivery)
        in.interim_coupons.emplace_back(cpn_per_period, double(delivery - cd));

    const double days = double(delivery - settle);
    const px::DeliverableResult<double> r = px::analyze_deliverable<double>(in, futures, repo, days, repo_basis);
    results.push_back(r);
    cf_v.push_back(r.conversion_factor);
    gross_v.push_back(r.gross_basis);
    net_v.push_back(r.net_basis);
    irr_v.push_back(r.implied_repo);
    inv_v.push_back(r.invoice_price);
    ids.push_back(js(bo, "id"));
  }

  const std::size_t ctd = results.empty() ? 0 : px::select_ctd(results);

  json::object out;
  out["conversion_factor"] = vecf(cf_v);
  out["gross_basis"] = vecf(gross_v);
  out["net_basis"] = vecf(net_v);
  out["implied_repo"] = vecf(irr_v);
  out["invoice_price"] = vecf(inv_v);
  out["n"] = static_cast<int>(cf_v.size());
  out["ctd_index"] = static_cast<int>(ctd);
  out["ctd_id"] = results.empty() ? std::string() : ids[ctd];
  return json::serialize(json::value(std::move(out)));
}


// The STRING seam (tests, the C ABI, hosts holding raw text): parse once, then the object entry
// point above -- run_json passes its already-parsed object straight through (E6.3, D11).
std::string bond_future_json(const std::string& request) { return bond_future_json(json::parse(request).as_object()); }
}  // namespace swaps::api
