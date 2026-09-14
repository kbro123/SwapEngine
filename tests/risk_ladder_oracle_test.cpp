// @oracle-test — the DELTA LADDER the `portfolio_risk` / `risk` / `generate_risk` run_json verbs return vs a ladder
// built from QuantLib instruments by bump-and-RECALIBRATE. DO NOT DELETE OR WEAKEN.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// See tests/ORACLE_TESTS.md before changing anything here.
//
// WHAT THIS PINS THAT NOTHING ELSE DID (draft 2026-09-14). calibration/risk.hpp is the IFT ladder clients hedge
// on. Until now it was checked only against the ENGINE's own bump-and-recalibrate (risk_test.cpp, T3 parity) and
// against hand replicas (consistent_risk_test.cpp, risk_scale_repro_test.cpp), so an error SHARED by the engine's
// instruments, its Jacobian and its book pricing cancelled out of every check. Here the reference shares none of
// those:
//   reference  QuantLib instrument objects (OvernightIndexedSwap, VanillaSwap, OvernightIndexFuture) DEFINE the
//              calibration residuals (calibration_oracle_test.cpp's fixture, repeated below); every market quote
//              is bumped +/-0.25 bp and the curve is RE-SOLVED against those QuantLib residuals; a book of QuantLib
//              OvernightIndexedSwap / VanillaSwap objects is repriced off the re-solved curve. Central difference.
//   engine     the SAME market and the SAME trades go through swaps::api::run_json: the bundle path's
//              `portfolio_risk` (ladder = (pinv(J)·D)ᵀ·dP/dx, AAD) and `risk` (M = pinv(J)·D), and the
//              stateless `generate_risk` (null_completed_ladder with the residual market scale D).
// What IS shared, stated rather than implied: the state space and the interpolator (both arms build OUR
// ModularCurve from knot forwards x; QuantLib sees it through qlx::CurveTermStructure), exactly as in
// calibration_oracle_test.cpp. So this pins instrument assembly, book assembly, the Jacobian, the IFT, D, the
// sign and the units of the ladder -- not the interpolation scheme (scheme_value_oracle_test.cpp does that).
//
// UNITS (verified in code, see NOTES): every ladder entry is dP/dq_j PER UNIT QUOTE (q in decimal rate units:
// 0.0001 = 1 bp), P = book NPV in the discount currency per unit notional, positions payer-of-fixed with a signed
// notional (portfolio.hpp: notional·(float_leg_pv − K·annuity)). Per-bp delta = ladder·1e-4. A futures row is
// quoted as a RATE (1 − price/100), so its entry is per unit RATE, i.e. the negative of a per-price delta.
//
// TOLERANCE, MEASURED (2026-09-14). The error is central-FD truncation h²/6·|P'''| plus solver noise. The first
// draft bumped by 1 bp with kRel = 2e-5 and two rows (|ladder| ≈ 37) missed by 1.4x: 2.8e-5 relative. At h = 0.25 bp
// the same rows fell 16.1x -- exactly h² -- so it was truncation, not an engine/QuantLib difference, and the bump was
// shrunk rather than the tolerance widened. At h = 0.25 bp: worst ladder row 0.087x of 2e-5 (1.7e-6 relative), worst
// operator entry 0.049x; solver noise ≤ ‖ladder‖₁·kSolveFloor/(2h) ≈ 1e2·1e-14/5e-5 = 2e-8. kRel = 5e-6 on
// max(|ref|, 1) keeps ~3x over the measured truncation and is still ~3 orders below a 365/360 day-count slip
// (1.4e-2), a dropped D (factor 4 on a banded row) or a first-order (forward-difference) bump (negative control 1).
//
// SCOPE, stated rather than implied: hard pins on instruments QuantLib can express (OIS, IBOR IRS, averaged
// overnight futures), four curves incl. two spread-over-base chains. OUT OF SCOPE: soft BANDS as a quote type
// (a banded row appears only in the D control, through the in-band identity pinned by risk_scale_repro_test.cpp),
// TURN JUMPS, BUTTERFLY/Portfolio combination rows, FX forwards and MtM XCCY basis (no QuantLib counterpart;
// D = 1/(q·T) stays pinned by risk_scale_repro_test.cpp), SYNTHETIC NULL PILLARS (the fixture is square and full
// rank; generate_risk's rank completion stays on consistent_risk_test.cpp), a regularised risk operator (reg
// changes the operator by design; not a market delta), multi-bundle re-leveling (generate_risk with N > 1).
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <boost/json.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"            // run_json, bundle_to_json, instrument_to_json
#include "swaps/api/generate_risk.hpp"         // the generate_risk verb -- named for tools/oracle_coverage.py
#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/observations.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/consistent_risk.hpp"  // generate_risk's computation -- named for oracle_coverage.py
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/risk.hpp"             // THE header this oracle exists for (null_completed_ladder)
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;
namespace b = swaps::build;
namespace api = swaps::api;
namespace json = boost::json;

namespace {

// ---- the constants of the comparison (each derived in the header comment / NOTES) ------------------------------
constexpr double kH = 2.5e-5;         // the bump: 0.25 bp in quote (rate) units (truncation measured, see header)
constexpr double kRel = 5e-6;         // relative ladder tolerance on max(|ref|, kFloor): ~3x the measured truncation
constexpr double kFloor = 1.0;        // per unit quote, per unit notional: below it the bound is absolute (5e-6)
constexpr double kSolveFloor = 1e-14; // every QuantLib re-solve must reprice its market to this (rate units)
constexpr double kXGap = 1e-11;       // verb knots vs QuantLib-residual knots: 10x calibration_oracle's measured 9.2e-13
constexpr double kNpvAbs = 5e-9;      // verb NPV vs QuantLib NPV: kXGap × ‖dP/dx‖₁ (≈ Σ|N|·A ≈ 50, ≲ 2e2) + leg PVs
                                      // (≲ 1) at curve_rel 1e-10 + JSON ULPs
constexpr int kMaxNewton = 50;
constexpr double kDecay = 0.25;       // the D control's band decay
constexpr double kBandHalfWidth = 20e-4;

swaps::build::Date eng_date(const QuantLib::Date& d) {
  std::ostringstream os;
  os << QuantLib::io::iso_date(d);
  return swaps::build::Date::from_iso(os.str());
}

// ================================================================================================================
// The calibration_oracle_test.cpp fixture, repeated VERBATIM in substance (that file's helpers live in an anonymous
// namespace). Any change there must be mirrored here; the first test below re-asserts its quote agreement.
// ================================================================================================================
enum Role { SOFR = 0, FF = 1, ESTR = 2, EUR6M = 3, NROLES = 4 };

struct Fixture {
  Date today;
  DayCounter dc = Actual365Fixed();
  cal::BundleProblem prob;                                  // arm A: our instruments
  std::vector<std::function<double()>> ql_quote;            // arm B: QuantLib's quote for the same trade
  std::vector<std::string> label;                           // one per residual row, for failure messages
  std::vector<RelinkableHandle<YieldTermStructure>> h{NROLES};
  Eigen::VectorXd x_true, x0;
  ext::shared_ptr<Sofr> sofr;
  ext::shared_ptr<FedFunds> ff;
  ext::shared_ptr<Estr> estr;
  ext::shared_ptr<Euribor6M> eur6m;
  std::vector<ext::shared_ptr<Instrument>> keep;
};

const std::vector<std::string>& tenors() {
  static const std::vector<std::string> t = {"1Y", "2Y", "3Y", "5Y", "10Y", "30Y"};
  return t;
}

template <class Index>
std::vector<double> add_ois(Fixture& f, const b::SwapConv& conv, const ext::shared_ptr<Index>& idx, int fc, int disc,
                            const std::string& name) {
  std::vector<double> knots;
  for (const std::string& t : tenors()) {
    const ext::shared_ptr<OvernightIndexedSwap> qls =
        MakeOIS(PeriodParser::parse(t), idx, 0.03)
            .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
            .withSettlementDays(conv.spot_lag)
            .withPaymentLag(conv.pay_lag)
            .withPaymentAdjustment(ModifiedFollowing);
    f.keep.push_back(qls);
    f.prob.instruments.push_back(b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
    f.ql_quote.emplace_back([qls] { return qls->fairRate(); });
    f.label.push_back(name + " " + t);
  }
  return knots;
}

std::vector<double> add_ibor(Fixture& f, const b::SwapConv& conv, int fc, int disc) {
  std::vector<double> knots;
  for (const std::string& t : tenors()) {
    const ext::shared_ptr<VanillaSwap> qls = MakeVanillaSwap(PeriodParser::parse(t), f.eur6m, 0.03)
                                                 .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
                                                 .withSettlementDays(conv.spot_lag);
    f.keep.push_back(qls);
    f.prob.instruments.push_back(b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
    f.ql_quote.emplace_back([qls] { return qls->fairRate(); });
    f.label.push_back("EURIBOR6M " + t);
  }
  return knots;
}

std::vector<double> add_averaged_futures(Fixture& f, int fc, int n) {
  std::vector<double> knots;
  const Calendar cal = f.ff->fixingCalendar();
  Date s = Date(1, static_cast<Month>(f.today.month() + 1), f.today.year());
  for (int i = 0; i < n; ++i) {
    const Date e = cal.advance(s, 1, Months, Following);
    const ext::shared_ptr<OvernightIndexFuture> qlf =
        ext::make_shared<OvernightIndexFuture>(f.ff, s, e, Handle<Quote>(), RateAveraging::Simple);
    f.keep.push_back(qlf);
    const std::string dcs = b::index_day_count("USD-FEDFUNDS"), cls = b::index_calendar("USD-FEDFUNDS");
    f.prob.instruments.push_back(b::rate_instrument(
        fc, b::observation(eng_date(f.today), eng_date(s), eng_date(e), "averaged", 0.0, dcs, cls), 0.0));
    knots.push_back(b::curve_time(eng_date(f.today), eng_date(e)));
    f.ql_quote.emplace_back([qlf] { return 1.0 - qlf->NPV() / 100.0; });  // quoted as a RATE
    f.label.push_back("FF future " + std::to_string(i + 1));
    s = e;
  }
  return knots;
}

// Row layout (27 residuals == 27 knots): SOFR OIS 0-5 | FF futures 6-8 | FF OIS 9-14 | ESTR OIS 15-20 | EURIBOR 21-26.
constexpr int kEstr10yRow = 15 + 4;  // the D control's banded row: loaded by the ESTR 12Y and EURIBOR 8Y positions

Fixture build() {
  Fixture f;
  f.today = Date(8, July, 2026);
  Settings::instance().evaluationDate() = f.today;
  f.sofr = ext::make_shared<Sofr>(f.h[SOFR]);
  f.ff = ext::make_shared<FedFunds>(f.h[FF]);
  f.estr = ext::make_shared<Estr>(f.h[ESTR]);
  f.eur6m = ext::make_shared<Euribor6M>(f.h[EUR6M]);

  const auto k0 = add_ois(f, b::swap_conv("USD", "USD-SOFR"), f.sofr, SOFR, SOFR, "SOFR");
  auto k1 = add_averaged_futures(f, FF, 3);
  const auto k1b = add_ois(f, b::swap_conv("USD", "USD-FEDFUNDS"), f.ff, FF, SOFR, "FF");
  k1.insert(k1.end(), k1b.begin(), k1b.end());
  const auto k2 = add_ois(f, b::swap_conv("EUR", "EUR-ESTR"), f.estr, ESTR, ESTR, "ESTR");
  const auto k3 = add_ibor(f, b::swap_conv("EUR", "EUR-EURIBOR-6M"), EUR6M, ESTR);

  f.prob.curves.resize(NROLES);
  f.prob.curves[SOFR] = px::CurveStructure{.base = -1, .currency = 0, .regions = cv::flat_hermite({}, k0)};
  f.prob.curves[FF] = px::CurveStructure{.base = SOFR, .currency = 0, .regions = cv::flat_hermite({}, k1)};
  f.prob.curves[ESTR] = px::CurveStructure{.base = -1, .currency = 1, .regions = cv::flat_hermite({}, k2)};
  f.prob.curves[EUR6M] = px::CurveStructure{.base = ESTR, .currency = 1, .regions = cv::flat_hermite({}, k3)};

  const double level[NROLES] = {0.030, 0.0008, 0.025, 0.0015};
  const double tilt[NROLES] = {4e-4, 2e-5, 3e-4, 1e-5};
  f.x_true.setZero(f.prob.n_knots());
  f.x0.setZero(f.prob.n_knots());
  for (int c = 0; c < NROLES; ++c) {
    const int o = f.prob.offset(c);
    for (int i = 0; i < f.prob.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i) {
      f.x_true[o + i] = level[c] + tilt[c] * i;
      f.x0[o + i] = level[c];
    }
  }
  const Eigen::VectorXd q = f.prob.residuals<double>(f.x_true) + f.prob.market();
  for (int i = 0; i < f.prob.n_residuals(); ++i) f.prob.instruments[static_cast<std::size_t>(i)].market = q[i];
  return f;
}

// ================================================================================================================
// The BOOK: the same five trades as QuantLib objects (the reference) and as engine positions (the verb's input).
// Off-market fixed rates and mixed signs so the ladder has sign changes and non-zero gamma on every curve.
// ================================================================================================================
struct Book {
  std::vector<ext::shared_ptr<Instrument>> ql;  // NPV() is signed through the Payer/Receiver type
  json::array positions;                         // engine: payer-of-fixed, signed notional
};

// One engine position from the SAME builder the calibration rows use (the calibration oracle pins these legs to
// QuantLib's fair rate at 5e-13); legs travel through the public codec's instrument_to_json coupon arrays.
json::object engine_position(const cal::Instrument& ins, double notional, double fixed_rate) {
  const json::object io = api::instrument_to_json(ins).as_object();
  json::object po;
  po["notional"] = notional;
  po["fixed_rate"] = fixed_rate;
  po["fwd_curve"] = ins.fwd.forecast;
  po["disc_curve"] = ins.fwd.discount;
  po["fixed_curve"] = ins.fixed.discount;
  po["float_coupons"] = io.at("fwd").as_object().at("coupons");
  po["fixed_coupons"] = io.at("fixed").as_object().at("coupons");
  return po;
}

template <class Index>
void add_ois_position(Fixture& f, Book& book, const b::SwapConv& conv, const ext::shared_ptr<Index>& idx,
                      const char* tenor, int fc, int disc, double notional, double fixed_rate) {
  const ext::shared_ptr<OvernightIndexedSwap> qls =
      MakeOIS(PeriodParser::parse(tenor), idx, fixed_rate)
          .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
          .withSettlementDays(conv.spot_lag)
          .withPaymentLag(conv.pay_lag)
          .withPaymentAdjustment(ModifiedFollowing)
          .withNominal(std::abs(notional))
          .receiveFixed(notional < 0.0);
  book.ql.push_back(qls);
  const cal::Instrument ins = b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0);
  book.positions.push_back(engine_position(ins, notional, fixed_rate));
}

void add_ibor_position(Fixture& f, Book& book, const b::SwapConv& conv, const char* tenor, int fc, int disc,
                       double notional, double fixed_rate) {
  const ext::shared_ptr<VanillaSwap> qls = MakeVanillaSwap(PeriodParser::parse(tenor), f.eur6m, fixed_rate)
                                               .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
                                               .withSettlementDays(conv.spot_lag)
                                               .withNominal(std::abs(notional))
                                               .receiveFixed(notional < 0.0);
  book.ql.push_back(qls);
  const cal::Instrument ins = b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0);
  book.positions.push_back(engine_position(ins, notional, fixed_rate));
}

Book make_book(Fixture& f) {
  Book book;
  add_ois_position(f, book, b::swap_conv("USD", "USD-SOFR"), f.sofr, "7Y", SOFR, SOFR, +1.0, 0.0335);
  add_ois_position(f, book, b::swap_conv("USD", "USD-SOFR"), f.sofr, "25Y", SOFR, SOFR, +0.5, 0.0290);
  add_ois_position(f, book, b::swap_conv("USD", "USD-FEDFUNDS"), f.ff, "4Y", FF, SOFR, -1.5, 0.0330);
  add_ois_position(f, book, b::swap_conv("EUR", "EUR-ESTR"), f.estr, "12Y", ESTR, ESTR, +2.0, 0.0240);
  add_ibor_position(f, book, b::swap_conv("EUR", "EUR-EURIBOR-6M"), "8Y", EUR6M, ESTR, -1.0, 0.0295);
  return book;
}

// ================================================================================================================
// The QuantLib arm: QuantLib instruments define the residuals and price the book; our state vector x.
// ================================================================================================================
class QuantLibArm {
 public:
  QuantLibArm(Fixture& f, const Book& book) : f_(&f), book_(&book) {}
  void set_market(const Eigen::VectorXd& q) { q_ = q; }

  Eigen::VectorXd residuals(const Eigen::VectorXd& x) const {
    relink(x);
    Eigen::VectorXd r(static_cast<int>(f_->ql_quote.size()));
    for (int i = 0; i < r.size(); ++i) r[i] = f_->ql_quote[static_cast<std::size_t>(i)]() - q_[i];
    return r;
  }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {  // central FD: QuantLib has no AD (a feature here)
    const int n = static_cast<int>(x.size()), m = static_cast<int>(f_->ql_quote.size());
    Eigen::MatrixXd J(m, n);
    const double eps = 1e-7;
    for (int j = 0; j < n; ++j) {
      Eigen::VectorXd up = x, dn = x;
      up[j] += eps;
      dn[j] -= eps;
      J.col(j) = (residuals(up) - residuals(dn)) / (2 * eps);
    }
    return J;
  }
  double book_npv(const Eigen::VectorXd& x) const {
    relink(x);
    double npv = 0.0;
    for (const auto& ins : book_->ql) npv += ins->NPV();
    return npv;
  }

 private:
  void relink(const Eigen::VectorXd& x) const {
    std::vector<int> off(f_->prob.curves.size());
    for (int c = 0; c < static_cast<int>(f_->prob.curves.size()); ++c) off[static_cast<std::size_t>(c)] = f_->prob.offset(c);
    curves_ = cal::build_bundle_curves<double>(
        f_->prob.curves, [&](int c, int i) { return x[off[static_cast<std::size_t>(c)] + i]; });
    ts_.clear();
    for (std::size_t c = 0; c < curves_.size(); ++c) {
      ts_.push_back(ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
          f_->today, f_->dc, curves_[c].get()));
      f_->h[c].linkTo(ts_.back());
    }
  }
  Fixture* f_;
  const Book* book_;
  Eigen::VectorXd q_;
  mutable std::vector<std::unique_ptr<cal::CurveHandle<double>>> curves_;
  mutable std::vector<ext::shared_ptr<YieldTermStructure>> ts_;
};

// A CHORD Newton re-solve: the QuantLib FD Jacobian at the base solution, factored once, steers; QuantLib's exact
// residuals decide where it lands. Each step contracts the error by ~‖J0⁻¹(J(x) − J0)‖ ≈ T·|Δx| ≲ 1e-2 for a 1 bp
// move, so ~6 steps reach the floating-point floor; it stops at the first step that fails to reduce ‖r‖∞. The
// Jacobian only steers -- the answer is the root of the QuantLib residuals whatever J0 is.
Eigen::VectorXd chord_solve(const QuantLibArm& arm, const Eigen::PartialPivLU<Eigen::MatrixXd>& lu, Eigen::VectorXd x,
                            double& r_inf, int& steps) {
  Eigen::VectorXd r = arm.residuals(x);
  r_inf = r.cwiseAbs().maxCoeff();
  for (steps = 0; steps < kMaxNewton; ++steps) {
    const Eigen::VectorXd xn = x - lu.solve(r);
    const Eigen::VectorXd rn = arm.residuals(xn);
    const double nn = rn.cwiseAbs().maxCoeff();
    if (!(nn < r_inf)) break;  // the floor: no further reduction
    x = xn;
    r = rn;
    r_inf = nn;
  }
  return x;
}

// ================================================================================================================
// The REFERENCE: computed once (plain numbers only -- no QuantLib object outlives the computation, so the per-test
// IndexManager reset in ql_test_isolation.cpp cannot touch it).
// ================================================================================================================
struct Reference {
  bool ok = false;
  std::string why;
  int n = 0;
  cal::BundleProblem prob;           // the market both arms see (arm A instruments, QuantLib-consistent quotes)
  Eigen::VectorXd x0;                // the fixture's flat cold seed (sent to the bundle verb)
  json::object book;                 // {positions: [...]} -- the verb's book
  std::vector<std::string> label;
  Eigen::VectorXd quote_gap;         // |QuantLib quote − our quote| at x_true (the calibration oracle's diagnostic)
  Eigen::VectorXd x_ql;              // the QuantLib-residual solution at the base market
  double npv_ql = 0.0;               // QuantLib book NPV at x_ql
  Eigen::VectorXd ladder;            // central FD: [P(q+h e_j) − P(q−h e_j)] / 2h, per unit quote
  Eigen::VectorXd ladder_forward;    // NEGATIVE CONTROL: [P(q+h e_j) − P(q)] / h
  Eigen::MatrixXd dxdq;              // central FD of the re-solved knots, n_knots × n_res
  double worst_solve = 0.0;          // max ‖r‖∞ over the base and all 2n bumped re-solves
  int max_steps = 0;
};

Reference compute_reference() {
  Reference R;
  SavedSettings saved;
  Fixture f = build();
  const Book book = make_book(f);
  R.n = f.prob.n_residuals();
  if (f.prob.n_knots() != R.n) {
    R.why = "the fixture is no longer square";
    return R;
  }
  R.prob = f.prob;
  R.x0 = f.x0;
  R.label = f.label;
  R.book["positions"] = book.positions;

  QuantLibArm arm(f, book);
  arm.set_market(Eigen::VectorXd::Zero(R.n));
  R.quote_gap = (arm.residuals(f.x_true) - f.prob.market()).cwiseAbs();

  const Eigen::VectorXd q = f.prob.market();
  arm.set_market(q);
  const cal::CalibrationResult base = cal::calibrate_with(arm, R.n, R.n, f.x0);  // cold, as calibration_oracle does
  if (!base.converged) {
    R.why = std::string("QuantLib-residual base calibration did not converge: ") + base.status;
    return R;
  }
  const Eigen::PartialPivLU<Eigen::MatrixXd> lu(arm.jacobian(base.x));
  double r_inf = 0.0;
  int steps = 0;
  R.x_ql = chord_solve(arm, lu, base.x, r_inf, steps);  // polish to the floor
  R.worst_solve = r_inf;
  R.max_steps = steps;
  R.npv_ql = arm.book_npv(R.x_ql);

  R.ladder.resize(R.n);
  R.ladder_forward.resize(R.n);
  R.dxdq.resize(R.n, R.n);
  for (int j = 0; j < R.n; ++j) {
    Eigen::VectorXd qu = q, qd = q;
    qu[j] += kH;
    qd[j] -= kH;
    arm.set_market(qu);
    const Eigen::VectorXd xu = chord_solve(arm, lu, R.x_ql, r_inf, steps);
    R.worst_solve = std::max(R.worst_solve, r_inf);
    R.max_steps = std::max(R.max_steps, steps);
    const double pu = arm.book_npv(xu);
    arm.set_market(qd);
    const Eigen::VectorXd xd = chord_solve(arm, lu, R.x_ql, r_inf, steps);
    R.worst_solve = std::max(R.worst_solve, r_inf);
    R.max_steps = std::max(R.max_steps, steps);
    const double pd = arm.book_npv(xd);
    R.ladder[j] = (pu - pd) / (2.0 * kH);
    R.ladder_forward[j] = (pu - R.npv_ql) / kH;
    R.dxdq.col(j) = (xu - xd) / (2.0 * kH);
  }
  R.ok = true;
  return R;
}

const Reference& reference() {
  static const Reference ref = compute_reference();
  return ref;
}

// ---- the verbs ----------------------------------------------------------------------------------------------------
json::array arr(const Eigen::VectorXd& v) {
  json::array a;
  for (Eigen::Index i = 0; i < v.size(); ++i) a.push_back(v[i]);
  return a;
}
Eigen::VectorXd nums(const json::value& v) {
  const json::array& a = v.as_array();
  Eigen::VectorXd e(static_cast<Eigen::Index>(a.size()));
  for (std::size_t i = 0; i < a.size(); ++i) e[static_cast<Eigen::Index>(i)] = a[i].to_number<double>();
  return e;
}
json::object run(const json::object& req) { return json::parse(api::run_json(req)).as_object(); }

// The bundle path: calibrate (no regularize => RegSpec{} => M = pinv(J)·D, the raw market-delta operator), then the
// generated `portfolio_risk` and `risk` arms.
json::object bundle_verb(const Reference& R, const cal::BundleProblem& p) {
  json::object req;
  req["bundle"] = api::bundle_to_json(p);
  req["x0"] = arr(R.x0);
  req["portfolio_risk"] = R.book;
  req["risk"] = true;
  return run(req);
}

// generate_risk with ONE bundle. A NEGLIGIBLE regulariser (row weight 1e-8, mu = 1e-16) keeps relevel_calibration_reg
// from applying its LIGHT tension floor, which would move x off the zero-residual root (risk_scale_repro_test.cpp's
// recipe). Its bias on x is ~mu·‖K‖·|x|/sigma_min² ≲ 1e-16·2e3·3e-2/1e-4 ≈ 6e-11 -- ~1e-8 on the ladder.
json::object generate_risk_verb(const Reference& R, const cal::BundleProblem& p) {
  json::object reg;
  reg["lambda"] = 1e-8;
  json::array cs;
  for (int c = 0; c < NROLES; ++c) cs.push_back(c);
  reg["curves"] = std::move(cs);
  reg["tension"] = true;
  reg["sigma"] = 0.0;
  json::object g;
  g["book"] = R.book;
  g["bundles"] = json::array{api::bundle_to_json(p)};
  g["regularize"] = std::move(reg);
  json::object req;
  req["generate_risk"] = std::move(g);
  return run(req);
}

double tol_row(double ref) { return kRel * std::max(std::abs(ref), kFloor); }

// Every row within tol_row; returns the worst |got − ref| / tol for the log line.
double expect_ladder(const Eigen::VectorXd& got, const Reference& R, const char* what) {
  EXPECT_EQ(got.size(), R.ladder.size()) << what;
  if (got.size() != R.ladder.size()) return 0.0;
  double worst = 0.0;
  for (int j = 0; j < R.n; ++j) {
    EXPECT_NEAR(got[j], R.ladder[j], tol_row(R.ladder[j]))
        << what << " row " << j << " (" << R.label[static_cast<std::size_t>(j)] << "), per unit quote";
    worst = std::max(worst, std::abs(got[j] - R.ladder[j]) / tol_row(R.ladder[j]));
  }
  std::cout << "  [" << what << "] worst |engine − QuantLib| / tol over " << R.n << " quotes = " << worst << "\n";
  return worst;
}

cal::BundleProblem banded(const cal::BundleProblem& p) {
  cal::BundleProblem out = p;
  cal::Instrument& ins = out.instruments[static_cast<std::size_t>(kEstr10yRow)];
  ins.band_lower = ins.market - kBandHalfWidth;  // ±20 bp around the mid: a ±1 bp bump stays inside
  ins.band_upper = ins.market + kBandHalfWidth;
  ins.band_decay = kDecay;
  return out;
}

}  // namespace

// Preconditions, so a failure below names its cause: the fixture still agrees with QuantLib quote by quote, every
// QuantLib re-solve reached the floor the tolerance assumes, and the verb's NPV is QuantLib's NPV.
TEST(RiskLadderOracle, ReferenceIsWellPosedAndTheVerbNpvIsQuantLibs) {
  const Reference& R = reference();
  ASSERT_TRUE(R.ok) << R.why;
  EXPECT_LT(R.quote_gap.maxCoeff(), 5e-13) << "the calibration_oracle fixture drifted (its own test says which row)";
  EXPECT_LT(R.worst_solve, kSolveFloor) << "a QuantLib re-solve stopped above the floor the FD noise bound assumes";
  EXPECT_LT(R.max_steps, kMaxNewton) << "the chord re-solve hit its step cap";
  std::cout << "  [reference] 2x" << R.n << " QuantLib re-solves, worst |r| = " << R.worst_solve
            << ", max chord steps = " << R.max_steps << ", QuantLib book NPV = " << R.npv_ql << "\n";

  const json::object out = bundle_verb(R, R.prob);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_TRUE(out.at("calibration").as_object().at("converged").as_bool());
  ASSERT_EQ(out.at("calibration").as_object().at("rank_deficiency").to_number<int>(), 0);
  // The verb's curve is the QuantLib-residual curve (calibration_oracle measured 9.2e-9 bp) ...
  EXPECT_LT((nums(out.at("x")) - R.x_ql).cwiseAbs().maxCoeff(), kXGap);
  // ... so the verb's book NPV must be QuantLib's book NPV on it.
  EXPECT_NEAR(out.at("portfolio_risk").as_object().at("npv").to_number<double>(), R.npv_ql, kNpvAbs);
}

// THE TEST, stateful path: `portfolio_risk` ladder = dP/dq per quote vs QuantLib bump-and-recalibrate.
TEST(RiskLadderOracle, PortfolioRiskLadderMatchesQuantLibBumpAndRecalibrate) {
  const Reference& R = reference();
  ASSERT_TRUE(R.ok) << R.why;
  const json::object out = bundle_verb(R, R.prob);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  expect_ladder(nums(out.at("portfolio_risk").as_object().at("ladder")), R, "portfolio_risk");
}

// The operator itself, book-free: `risk` returns M = dx/dq (n_knots × n_res); QuantLib's is the central FD of the
// re-solved knot forwards. Same truncation argument on x(q) (weaker nonlinearity than P(q): no book leverage).
TEST(RiskLadderOracle, RiskOperatorMatchesQuantLibKnotResponse) {
  const Reference& R = reference();
  ASSERT_TRUE(R.ok) << R.why;
  const json::object out = bundle_verb(R, R.prob);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  const json::array& rows = out.at("risk_operator").as_array();
  ASSERT_EQ(rows.size(), static_cast<std::size_t>(R.n));
  double worst = 0.0;
  for (int i = 0; i < R.n; ++i) {
    const Eigen::VectorXd row = nums(rows[static_cast<std::size_t>(i)]);
    ASSERT_EQ(row.size(), R.n) << "knot " << i;
    for (int j = 0; j < R.n; ++j) {
      EXPECT_NEAR(row[j], R.dxdq(i, j), tol_row(R.dxdq(i, j)))
          << "dx[" << i << "]/dq[" << j << "] (" << R.label[static_cast<std::size_t>(j)] << ")";
      worst = std::max(worst, std::abs(row[j] - R.dxdq(i, j)) / tol_row(R.dxdq(i, j)));
    }
  }
  std::cout << "  [risk] worst |M − QuantLib dx/dq| / tol over " << R.n * R.n << " entries = " << worst << "\n";
}

// THE TEST, stateless path: generate_risk's null-completed ladder (D applied, no synthetic pillar on a full-rank
// square bundle) and its ladder_dv01 = 1bp · Σ ladder.
TEST(RiskLadderOracle, GenerateRiskLadderMatchesQuantLibBumpAndRecalibrate) {
  const Reference& R = reference();
  ASSERT_TRUE(R.ok) << R.why;
  const json::object out = generate_risk_verb(R, R.prob);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  const json::object& b0 = out.at("bundles").as_array().at(0).as_object();
  ASSERT_EQ(b0.at("n_synthetic").to_number<int>(), 0) << "a square full-rank bundle must need no synthetic pillar";
  expect_ladder(nums(b0.at("ladder")), R, "generate_risk");
  double tol_sum = 0.0;
  for (int j = 0; j < R.n; ++j) tol_sum += tol_row(R.ladder[j]);
  EXPECT_NEAR(b0.at("ladder_dv01").to_number<double>(), 1e-4 * R.ladder.sum(), 1e-4 * tol_sum)
      << "ladder_dv01 is the book's per-bp delta summed over the quotes";
  EXPECT_NEAR(out.at("npv").to_number<double>(), R.npv_ql, kNpvAbs);
}

// NEGATIVE CONTROL 1 -- the tolerance resolves second order. A FORWARD difference at the same bump (the classic
// mis-scaled bump) carries h/2·|P''|: measured 423x the first draft's tolerance at 1 bp, so ~4x less bump and ~4x
// less tolerance leave it far above 5x. If it ever passes the ladder check, kRel has become too loose to see gamma
// and the oracle has lost its teeth.
TEST(RiskLadderOracle, NegativeControlAFirstOrderBumpFailsTheTolerance) {
  const Reference& R = reference();
  ASSERT_TRUE(R.ok) << R.why;
  double worst = 0.0;
  int at = -1;
  for (int j = 0; j < R.n; ++j) {
    const double ratio = std::abs(R.ladder_forward[j] - R.ladder[j]) / tol_row(R.ladder[j]);
    if (ratio > worst) {
      worst = ratio;
      at = j;
    }
  }
  std::cout << "  [control] forward-difference miss = " << worst << " x tol at row " << at << "\n";
  EXPECT_GT(worst, 5.0) << "a first-order bump now passes: the tolerance cannot see gamma";
}

// NEGATIVE CONTROL 2 -- the residual market scale D. The ESTR 10Y row is re-quoted as a soft band (±20 bp, decay
// 0.25) around the SAME mid. At the zero-residual fit the band is inside, so its market delta IS the hard pin's
// (the in-band identity, risk_scale_repro_test.cpp) -- the QuantLib reference applies unchanged. Both verbs must
// still match it on EVERY row (D = decay on the banded row), and the ladder with D dropped on that row (the
// pre-2026-09-13 generate_risk bug: entry / decay) must MISS it by far more than the tolerance.
TEST(RiskLadderOracle, NegativeControlDroppingTheResidualMarketScaleFails) {
  const Reference& R = reference();
  ASSERT_TRUE(R.ok) << R.why;
  ASSERT_GT(std::abs(R.ladder[kEstr10yRow]), 1.0) << "the control needs a row the book actually loads";
  const cal::BundleProblem soft = banded(R.prob);

  const json::object out = bundle_verb(R, soft);
  ASSERT_FALSE(out.contains("error")) << json::serialize(out);
  ASSERT_TRUE(out.at("calibration").as_object().at("converged").as_bool());
  ASSERT_TRUE(out.at("quote_diagnostics").as_array().at(kEstr10yRow).as_object().at("in_band").as_bool());
  const Eigen::VectorXd lad = nums(out.at("portfolio_risk").as_object().at("ladder"));
  expect_ladder(lad, R, "portfolio_risk, banded row");

  const json::object g = generate_risk_verb(R, soft);
  ASSERT_FALSE(g.contains("error")) << json::serialize(g);
  const Eigen::VectorXd glad = nums(g.at("bundles").as_array().at(0).as_object().at("ladder"));
  expect_ladder(glad, R, "generate_risk, banded row");

  // D dropped: the banded entry divided by its decay (factor 4), everything else unchanged.
  ASSERT_EQ(lad.size(), R.n);
  const double dropped = lad[kEstr10yRow] / kDecay;
  const double miss = std::abs(dropped - R.ladder[kEstr10yRow]) / tol_row(R.ladder[kEstr10yRow]);
  std::cout << "  [control] D dropped on the banded row misses by " << miss << " x tol\n";
  EXPECT_GT(miss, 1e3) << "dropping D on a banded row must be visible to this oracle (expected ~1.5e5 x tol)";
}
