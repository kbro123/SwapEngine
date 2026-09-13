// E5 taxonomy: T1 oracle (engine number vs an independent number) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// INDEPENDENT pin of the stateless `bond_future` run_json verb (api/bond_future.cpp), written BEFORE its E7
// lift into the library. Driven through the REAL JSON seam (swaps::api::run_json), so the conventions-DB
// contract rows (bond_futures[CME-TU/FV/TY/US]), the verb's maturity rounding, the bond builder's accrued and
// the pricing/bond_future.hpp kernel are all exercised together.
//
// T1 — CME's PUBLISHED conversion factors: CME Group, "Calculating U.S. Treasury Futures Conversion Factors"
//   (Interest Rate Resource Center, IR232), worked examples 1–5:
//   https://www.cmegroup.com/trading/interest-rates/files/Calculating_U.S.Treasury_Futures_Conversion_Factors.pdf
//   Covers BOTH rounding families: whole months (TU, 3Y, FV) and whole quarters (TY, US). CME defines the factor
//   ROUNDED TO 4 dp and invoices with that rounded factor, so the verb's factor must EQUAL the published one and
//   the invoice must be F·CF_published + accrued. Until 2026-09-13 the verb used the raw factor everywhere (TY:
//   0.835650542 against the exchange's 0.8357) -- this file was written against that code first and FAILS on it.
//
// T5 — hand identities in the same inputs: gross = clean − F·CF; invoice = F·CF + accrued_at_delivery
//   (ACT/ACT ICMA from literal day counts); implied repo / net basis from the cash-and-carry formulas
//   (pricing/bond_future.hpp), incl. an interim coupon inside the delivery window. Futures / clean / repo are
//   HYPOTHETICAL round numbers (the identities do not depend on them being market data); dates, coupons,
//   maturities and CUSIPs are real.

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <cmath>
#include <string>

#include "swaps/api/bundle_api.hpp"

#include "tolerances.hpp"

namespace api = swaps::api;
namespace json = boost::json;

namespace {

struct Case {
  const char* label;
  const char* contract;        // bond_futures[] row
  const char* value_date;      // = cash settlement (no per-bond settle unless `settle` set)
  const char* settle;          // per-bond settle override ("" = value_date)
  const char* first_delivery;  // CME: the FIRST CALENDAR DAY of the delivery month
  const char* issue;           // dated date (on the maturity-anchored grid)
  const char* maturity;
  const char* cusip;
  double coupon;
  double published_cf;         // CME IR232
  // hand inputs for the T5 identities
  double futures, clean, repo;
  double acc_now_days, acc_now_period;  // ACT/ACT ICMA: days since last coupon / days in period, at settle
  double acc_del_days, acc_del_period;  // ... at delivery (= first_delivery here)
  double days_to_delivery;              // delivery − settle
  double interim_amount, interim_days;  // coupon paid in (settle, delivery] and its days to delivery (0,0 = none)
};

// clang-format off
const Case kCases[] = {
  // Ex.1 2-Year, Dec-2008: 1-1/2s of Oct 31 2010 (n=1, z=10). Oct31→Apr30 period = 181 d.
  {"TU Dec-2008", "CME-TU", "2008-11-20", "", "2008-12-01", "2008-10-31", "2010-10-31", "912828JP6", 0.015,   0.9229,
   1.0850, 1.0085, 0.0050, 20, 181, 31, 181, 11, 0.0, 0.0},
  // Ex.2 3-Year, Mar-2009: 1-1/8s of Jan 15 2012 (n=2, z=10). The DB has NO CME 3-Year (Z3N) row; its CF
  // parameters (6% notional, whole-month rounding) are identical to CME-TU's, and the verb does not filter
  // eligibility, so the TU row stands in. Mar 1 2009 is a Sunday — CME still counts from it.
  {"3Y Mar-2009", "CME-TU", "2009-02-20", "", "2009-03-01", "2009-01-15", "2012-01-15", "912828KB5", 0.01125, 0.8747,
   1.1450, 1.0010, 0.0030, 36, 181, 45, 181, 9, 0.0, 0.0},
  // Ex.3 5-Year, Dec-2008: 2-3/4s of Oct 31 2013 (n=4, z=10).
  {"FV Dec-2008", "CME-FV", "2008-11-20", "", "2008-12-01", "2008-10-31", "2013-10-31", "912828JQ4", 0.0275,  0.8653,
   1.2050, 1.0350, 0.0050, 20, 181, 31, 181, 11, 0.0, 0.0},
  // Ex.4 10-Year, Dec-2008: 3-3/4s of Nov 15 2018 (n=9, z=9 — quarter rounding from 9y11m14d). May15→Nov15 grid.
  {"TY Dec-2008", "CME-TY", "2008-11-20", "", "2008-12-01", "2008-11-15", "2018-11-15", "912828JR2", 0.0375,  0.8357,
   1.2400, 1.0300, 0.0050, 5, 181, 16, 181, 11, 0.0, 0.0},
  // Ex.5 30-Year, Dec-2008: 4-1/2s of May 15 2038 (n=29, z=3). Settles 2008-11-10, BEFORE the Nov-15 coupon:
  // accrued 179/184 (May15→Nov15), the 2.25% coupon is paid 16 days before delivery, delivery accrued 16/181.
  {"US Dec-2008", "CME-US", "2008-11-20", "2008-11-10", "2008-12-01", "2008-05-15", "2038-05-15", "912810PX0", 0.045, 0.7943,
   1.2800, 1.0200, 0.0050, 179, 184, 16, 181, 21, 0.0225, 16.0},
};
// clang-format on

json::object request_for(const Case& c) {
  json::object bond{{"id", c.cusip}, {"issue", c.issue}, {"maturity", c.maturity},
                    {"coupon", c.coupon}, {"clean", c.clean}};
  if (std::string(c.settle).size()) bond["settle"] = c.settle;
  json::object bf{{"contract", c.contract},     {"value_date", c.value_date},
                  {"first_delivery", c.first_delivery}, {"futures_price", c.futures},
                  {"repo", c.repo},             {"basket", json::array{bond}}};
  return json::object{{"bond_future", bf}};
}

double at(const json::object& o, const char* key, std::size_t i = 0) {
  return o.at(key).as_array().at(i).to_number<double>();
}

}  // namespace

// T1: the verb's conversion factor IS the published exchange factor -- for month-rounded (TU / 3Y / FV) and
// quarter-rounded (TY / US) contracts. Exact: a 4-dp factor k/1e4 is the correctly rounded double either way.
TEST(BondFutureReference, PublishedConversionFactors) {
  for (const Case& c : kCases) {
    const std::string resp = api::run_json(request_for(c));
    const json::object o = json::parse(resp).as_object();
    ASSERT_FALSE(o.contains("error")) << c.label << ": " << resp;
    ASSERT_EQ(o.at("n").as_int64(), 1) << c.label;
    const double cf = at(o, "conversion_factor");
    EXPECT_EQ(cf, c.published_cf) << c.label << " " << c.cusip;
  }
}

// T5: gross basis, invoice price, implied repo and net basis are the textbook identities in the same inputs,
// with accrued computed by hand (ACT/ACT ICMA from literal day counts) — independent of the bond builder.
TEST(BondFutureReference, HandIdentities) {
  for (const Case& c : kCases) {
    const std::string resp = api::run_json(request_for(c));
    const json::object o = json::parse(resp).as_object();
    ASSERT_FALSE(o.contains("error")) << c.label << ": " << resp;
    const double cf = at(o, "conversion_factor");
    const double F = c.futures;

    const double acc_now = (c.coupon / 2.0) * c.acc_now_days / c.acc_now_period;
    const double acc_del = (c.coupon / 2.0) * c.acc_del_days / c.acc_del_period;

    // gross basis = clean − F·CF
    EXPECT_NEAR(at(o, "gross_basis"), c.clean - F * cf, swaps::tol::literal) << c.label;
    // invoice = F·CF + accrued at delivery
    const double invoice = F * cf + acc_del;
    EXPECT_NEAR(at(o, "invoice_price"), invoice, swaps::tol::literal) << c.label;
    // ... and that is the EXCHANGE invoice: futures x the published 4-dp factor + accrued at delivery.
    EXPECT_NEAR(at(o, "invoice_price"), F * c.published_cf + acc_del, swaps::tol::literal) << c.label;

    // implied repo (ACT/360): 360·(P_inv + Σc − P_buy) / (P_buy·d − Σc·d_k)
    const double p_buy = c.clean + acc_now;
    const double d = c.days_to_delivery;
    const double irr = 360.0 * (invoice + c.interim_amount - p_buy) / (p_buy * d - c.interim_amount * c.interim_days);
    EXPECT_NEAR(at(o, "implied_repo"), irr, swaps::tol::literal) << c.label;

    // net basis: forward clean at delivery (bought dirty, financed at repo, interim coupon reinvested) − F·CF
    const double fwd_dirty = p_buy * (1.0 + c.repo * (d / 360.0)) -
                             c.interim_amount * (1.0 + c.repo * (c.interim_days / 360.0));
    EXPECT_NEAR(at(o, "net_basis"), fwd_dirty - acc_del - F * cf, swaps::tol::literal) << c.label;

    EXPECT_EQ(o.at("ctd_index").as_int64(), 0) << c.label;
    EXPECT_EQ(std::string(o.at("ctd_id").as_string().c_str()), c.cusip) << c.label;
  }
}

// T5: the CTD is the max-implied-repo deliverable. The same FV note (identical CF and accrued) at two clean
// prices: the cheaper one has the higher implied repo, so it is the CTD. It is listed SECOND so the verb's
// first-index tie rule cannot pass the test by accident.
TEST(BondFutureReference, CtdIsCheaperOfTwoIdenticalNotes) {
  const Case& fv = kCases[2];
  json::object rich{{"id", "FV-rich"}, {"issue", fv.issue}, {"maturity", fv.maturity}, {"coupon", fv.coupon},
                    {"clean", 1.0350}};
  json::object cheap{{"id", "FV-cheap"}, {"issue", fv.issue}, {"maturity", fv.maturity}, {"coupon", fv.coupon},
                     {"clean", 1.0300}};
  json::object bf{{"contract", "CME-FV"}, {"value_date", fv.value_date}, {"first_delivery", fv.first_delivery},
                  {"futures_price", fv.futures}, {"repo", fv.repo}, {"basket", json::array{rich, cheap}}};
  const std::string resp = api::run_json(json::object{{"bond_future", bf}});
  const json::object o = json::parse(resp).as_object();
  ASSERT_FALSE(o.contains("error")) << resp;
  ASSERT_EQ(o.at("n").as_int64(), 2);
  EXPECT_EQ(at(o, "conversion_factor", 0), at(o, "conversion_factor", 1));  // same note → same CF
  EXPECT_GT(at(o, "implied_repo", 1), at(o, "implied_repo", 0));
  EXPECT_EQ(o.at("ctd_index").as_int64(), 1);
  EXPECT_EQ(std::string(o.at("ctd_id").as_string().c_str()), "FV-cheap");
}
