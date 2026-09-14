// @oracle-test — the pnl run_json VERB vs QuantLib: npv_t0 / npv_t1 / total / carry / roll / dq / the market ladder / the
// residual, each recomputed from QuantLib OIS objects priced off our curve (api/pnl.cpp). DO NOT DELETE OR WEAKEN.
// E5 taxonomy: T1 oracle (engine number vs an independent number) | T5 properties + value pins (hand / closed-form literals, identities, FD)
//
// See tests/ORACLE_TESTS.md before changing anything here.
//
// WHAT THIS PINS THAT NOTHING ELSE DID (draft 2026-09-14). tests/pnl_explain_test.cpp calls the LIBRARY with
// hand-built curve-time coupons and checks identities and signs. No test sends a pnl request through JSON (api_test
// only lists the key in its dispatch smoke), and nothing compares carry or roll to a number the engine did not make.
//
// QuantLib expresses all three reprices of the contract (pnl_explain.hpp:12-70) natively. The evaluation date
// stays t0 throughout, so every overnight fixing is a forecast and no historic fixing is ever needed:
//   npv_t0  = NPV off our curve at x0, reference date t0 (MakeOIS's default DiscountingSwapEngine).
//   carry   = the SAME swaps, curve at x0 still anchored at t0, priced by DiscountingSwapEngine(h, false, t1, t1):
//             QuantLib's npv() sums the flows with date > t1 and divides by DF(t1)
//             (src/ql/cashflows/cashflows.cpp:425-448, event.cpp:28-39), i.e. DF'(t) = DF(t)/DF(dt) on the original
//             dates. That is exactly RebasedHandle on the drop-only book.
//   roll    = the curve at x0 re-anchored at t1 (CurveTermStructure(t1, Act365F, curve)): every flow is discounted and
//             forecast at (d - t1)/365 = t - dt, which is roll_book(shift=true), and the default engine settles at the
//             curve's reference date t1, so the flow ON t1 is excluded as the engine drops pay <= dt.
//   npv_t1  = the curve at x1 anchored at t1.
//   ladder  = dP/dq = J^{-T} g with g = dNPV/dx and J = dq/dx both from QuantLib central bumps. The problem is square
//             and hard, so the IFT ladder is exactly this and needs no LM. dq from QuantLib's own quotes.
//
// SCOPE, stated rather than implied. The dates are ALIGNED: book swaps start at t0 with no spot or payment lag and
// annual coupons, and t1 is exactly their first coupon date (2026-07-08 -> 2027-07-08, 365 days, so dt = 1.0
// exactly). The contract's one documented approximation is a coupon STRADDLING t1, which keeps its tau and drops its
// elapsed accrual. That case has no QuantLib counterpart: QuantLib would need the realized fixings the engine
// deliberately does not use. It stays on pnl_explain_test.cpp and is not claimed here.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <boost/json.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/codec.hpp"
#include "swaps/api/pnl.hpp"  // pnl_json -- named so the include-closure coverage tool sees the verb
#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/pnl_explain.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "tolerances.hpp"

namespace {

namespace ql = QuantLib;
namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;
namespace b = swaps::build;
namespace json = boost::json;
namespace tol = swaps::tol;

constexpr double kNotional = 1e6;
constexpr double kNpvAbs = tol::curve_rel * kNotional;  // 1e-4 on a 1e6 book: DF-level agreement, as pricing_test
constexpr double kQuoteTol = 5e-13;                     // calibration_oracle_test.cpp's bound on these OIS rows
// FD step: truncation h^2/6 * O(T^3) ~ 1e-9 on the 30Y row at 1e-6; roundoff ~1e-16/2h. Balanced near 5e-7.
constexpr double kFdStep = 1e-6;
// The ladder is J^{-T} g from BUMPED QuantLib numbers, so its relative error is ~ kappa(J) x (FD rel error of J and g).
// 1e-5 is looser than tol::jacobian_rel because of that kappa amplification. The product is ASSERTED at < 1/4 of it.
constexpr double kLadderRel = 1e-5;
// T5 flat-curve literals: ~20 exp() flows of 1e6 notional at a few ulp each is ~1e-9. 1e-6 is the stated margin.
constexpr double kFlatAbs = 1e-6;

b::Date eng_date(const ql::Date& d) {
  std::ostringstream os;
  os << ql::io::iso_date(d);
  return b::Date::from_iso(os.str());
}

json::array to_json(const Eigen::VectorXd& v) {
  json::array a;
  for (Eigen::Index i = 0; i < v.size(); ++i) a.push_back(v[i]);
  return a;
}
double num(const json::value& v) { return v.to_number<double>(); }

struct Trade {
  const char* tenor;
  double fixed_rate;
  double notional;  // > 0 payer of fixed, < 0 receiver (MultiCurveBook's sign)
};
const std::vector<Trade>& trades() {
  static const std::vector<Trade> t = {{"10Y", 0.032, kNotional}, {"5Y", 0.030, -2.0 * kNotional}};
  return t;
}

struct Fixture {
  ql::Date t0, t1;
  double dt = 0.0;
  ql::DayCounter dc = ql::Actual365Fixed();
  ql::RelinkableHandle<ql::YieldTermStructure> h;
  ql::ext::shared_ptr<ql::Sofr> sofr;
  cal::BundleProblem prob;                                          // the calibration bundle (one SOFR curve)
  std::vector<ql::ext::shared_ptr<ql::OvernightIndexedSwap>> quote;  // its QuantLib twins (fair rates)
  std::vector<ql::ext::shared_ptr<ql::OvernightIndexedSwap>> spot;   // the book, default engine on h
  std::vector<ql::ext::shared_ptr<ql::OvernightIndexedSwap>> carry;  // the book, npvDate = settlement = t1
  json::array positions;                                            // the book as the verb receives it
  Eigen::VectorXd x0, x1;
};

void build(Fixture& f) {
  f.t0 = ql::Date(8, ql::July, 2026);
  f.t1 = ql::Date(8, ql::July, 2027);
  f.dt = static_cast<double>(f.t1 - f.t0) / 365.0;  // == build::curve_time(t0, t1)
  ql::Settings::instance().evaluationDate() = f.t0;
  f.sofr = ql::ext::make_shared<ql::Sofr>(f.h);

  // Calibration bundle: six DB-convention SOFR OIS, one knot per instrument (square, hard).
  const b::SwapConv conv = b::swap_conv("USD", "USD-SOFR");
  std::vector<double> knots;
  for (const char* t : {"1Y", "2Y", "3Y", "5Y", "10Y", "30Y"}) {
    const ql::ext::shared_ptr<ql::OvernightIndexedSwap> qls =
        ql::MakeOIS(ql::PeriodParser::parse(t), f.sofr, 0.03)
            .withDiscountingTermStructure(f.h)
            .withSettlementDays(conv.spot_lag)
            .withPaymentLag(conv.pay_lag)
            .withPaymentAdjustment(ql::ModifiedFollowing);
    f.quote.push_back(qls);
    f.prob.instruments.push_back(b::par_swap(eng_date(f.t0), conv, eng_date(qls->maturityDate()), 0, 0, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
  }
  f.prob.curves = {px::CurveStructure{.base = -1, .currency = 0, .regions = cv::flat_hermite({}, knots)}};

  // The book: spot-starting, no payment lag, so t1 is a coupon date of every leg (the aligned scope).
  b::SwapConv bconv = conv;
  bconv.spot_lag = 0;
  bconv.pay_lag = 0;
  const auto carry_engine = ql::ext::make_shared<ql::DiscountingSwapEngine>(f.h, false, f.t1, f.t1);
  for (const Trade& tr : trades()) {
    const ql::Swap::Type type = tr.notional > 0 ? ql::Swap::Payer : ql::Swap::Receiver;
    const auto make = [&] {
      return ql::MakeOIS(ql::PeriodParser::parse(tr.tenor), f.sofr, tr.fixed_rate)
          .withType(type)
          .withNominal(std::abs(tr.notional))
          .withSettlementDays(0)
          .withPaymentLag(0)
          .withPaymentAdjustment(ql::ModifiedFollowing);
    };
    const ql::ext::shared_ptr<ql::OvernightIndexedSwap> s = make().withDiscountingTermStructure(f.h);
    const ql::ext::shared_ptr<ql::OvernightIndexedSwap> c = make().withPricingEngine(carry_engine);
    f.spot.push_back(s);
    f.carry.push_back(c);

    // Our coupons from the shipped builder, carried to the verb through the PUBLIC codec (instrument_to_json writes
    // the same coupon keys book_from_json reads: codec.cpp fcpn_to/xcpn_to vs fcpn_from/xcpn_from).
    const cal::Instrument ins = b::par_swap(eng_date(f.t0), bconv, eng_date(s->maturityDate()), 0, 0, 0.0);
    const json::object ij = api::instrument_to_json(ins).as_object();
    json::object pos;
    pos["notional"] = tr.notional;
    pos["fixed_rate"] = tr.fixed_rate;
    pos["fwd_curve"] = 0;
    pos["disc_curve"] = 0;
    pos["fixed_curve"] = 0;
    pos["float_coupons"] = ij.at("fwd").as_object().at("coupons");
    pos["fixed_coupons"] = ij.at("fixed").as_object().at("coupons");
    f.positions.push_back(std::move(pos));
  }

  f.x0.resize(f.prob.n_knots());
  f.x1.resize(f.prob.n_knots());
  for (int i = 0; i < f.prob.n_knots(); ++i) {
    f.x0[i] = 0.030 + 4e-4 * i;
    f.x1[i] = f.x0[i] + 0.0010 + 5e-5 * i;  // a non-parallel +10..12.5 bp move
  }
}

// A bundle with the fixture's structure whose market is our model quote at x (self-consistent).
cal::BundleProblem bundle_at(const Fixture& f, const Eigen::VectorXd& x) {
  cal::BundleProblem p = f.prob;
  const Eigen::VectorXd q = p.residuals<double>(x) + p.market();
  for (int i = 0; i < p.n_residuals(); ++i) p.instruments[static_cast<std::size_t>(i)].market = q[i];
  return p;
}

// QuantLib's side: OUR curve at x, anchored at a chosen reference date, behind the one SOFR handle.
class QlSide {
 public:
  explicit QlSide(Fixture& f) : f_(&f) {}
  void link(const Eigen::VectorXd& x, const ql::Date& anchor) {
    const cal::BundleProblem& p = f_->prob;
    curves_ = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
    ts_ = ql::ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(anchor, f_->dc,
                                                                                         curves_[0].get());
    f_->h.linkTo(ts_);
  }
  double book(const std::vector<ql::ext::shared_ptr<ql::OvernightIndexedSwap>>& legs) const {
    double v = 0.0;
    for (const auto& s : legs) v += s->NPV();
    return v;
  }
  Eigen::VectorXd quotes() const {
    Eigen::VectorXd q(static_cast<int>(f_->quote.size()));
    for (int i = 0; i < q.size(); ++i) q[i] = f_->quote[static_cast<std::size_t>(i)]->fairRate();
    return q;
  }
  // Central differences at t0: g = dNPV/dx and J = dq/dx.
  void bumps(const Eigen::VectorXd& x, double eps, Eigen::VectorXd& g, Eigen::MatrixXd& J) {
    const int n = static_cast<int>(x.size());
    g.resize(n);
    J.resize(static_cast<int>(f_->quote.size()), n);
    for (int j = 0; j < n; ++j) {
      Eigen::VectorXd up = x, dn = x;
      up[j] += eps;
      dn[j] -= eps;
      link(up, f_->t0);
      const double pu = book(f_->spot);
      const Eigen::VectorXd qu = quotes();
      link(dn, f_->t0);
      g[j] = (pu - book(f_->spot)) / (2.0 * eps);
      J.col(j) = (qu - quotes()) / (2.0 * eps);
    }
  }

 private:
  Fixture* f_;
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> curves_;
  ql::ext::shared_ptr<ql::YieldTermStructure> ts_;
};

json::object run_pnl(const Fixture& f, const cal::BundleProblem& p0, const cal::BundleProblem* p1,
                     const Eigen::VectorXd* x1) {
  json::object payload;
  payload["bundle0"] = api::bundle_to_json(p0);
  if (p1) payload["bundle1"] = api::bundle_to_json(*p1);
  json::object book;
  book["positions"] = f.positions;
  payload["book"] = std::move(book);
  payload["dt_years"] = f.dt;
  payload["x0"] = to_json(f.x0);
  if (x1) payload["x1"] = to_json(*x1);
  json::object req;
  req["pnl"] = std::move(payload);
  return json::parse(api::pnl_json(req)).as_object().at("pnl").as_object();  // the object seam (E6.3)
}

}  // namespace

TEST(PnlOracle, EveryComponentMatchesQuantLibOnAnAlignedHorizon) {
  Fixture f;
  build(f);
  ASSERT_EQ(f.dt, 1.0) << "2026-07-08 -> 2027-07-08 is 365 days";
  for (const json::value& pv : f.positions) {  // the aligned scope: the first flow of every leg pays exactly at t1
    const json::object& po = pv.as_object();
    EXPECT_EQ(num(po.at("float_coupons").as_array().front().as_object().at("pay")), f.dt);
    EXPECT_EQ(num(po.at("fixed_coupons").as_array().front().as_object().at("pay")), f.dt);
  }

  const cal::BundleProblem p0 = bundle_at(f, f.x0), p1 = bundle_at(f, f.x1);
  const json::object e = run_pnl(f, p0, &p1, &f.x1);

  // ---- QuantLib's reprices ---------------------------------------------------------------------------------------
  QlSide qls(f);
  qls.link(f.x0, f.t0);
  const double npv_t0 = qls.book(f.spot);
  const double carry_t1 = qls.book(f.carry);
  const Eigen::VectorXd q0 = qls.quotes();
  qls.link(f.x1, f.t0);
  const Eigen::VectorXd q1 = qls.quotes();
  qls.link(f.x0, f.t1);
  const double cr_t1 = qls.book(f.spot);
  qls.link(f.x1, f.t1);
  const double npv_t1 = qls.book(f.spot);

  EXPECT_NEAR(num(e.at("npv_t0")), npv_t0, kNpvAbs);
  EXPECT_NEAR(num(e.at("npv_t1")), npv_t1, kNpvAbs);
  EXPECT_NEAR(num(e.at("total")), npv_t1 - npv_t0, kNpvAbs);
  EXPECT_NEAR(num(e.at("carry")), carry_t1 - npv_t0, kNpvAbs);
  EXPECT_NEAR(num(e.at("roll")), cr_t1 - carry_t1, kNpvAbs);
  // Non-trivial on both time legs, or the comparison proves nothing.
  EXPECT_GT(std::abs(carry_t1 - npv_t0), 1e3 * kNpvAbs);
  EXPECT_GT(std::abs(cr_t1 - carry_t1), 1e3 * kNpvAbs);

  const Eigen::VectorXd dq_ql = q1 - q0;
  const json::array& dq = e.at("dq").as_array();
  ASSERT_EQ(static_cast<int>(dq.size()), dq_ql.size());
  for (int i = 0; i < dq_ql.size(); ++i) EXPECT_NEAR(num(dq[static_cast<std::size_t>(i)]), dq_ql[i], 2 * kQuoteTol) << i;

  // ---- the market term: QuantLib bumps, the FD error and its amplification ASSERTED inside the budget -------------
  Eigen::VectorXd g, g2;
  Eigen::MatrixXd J, J2;
  qls.bumps(f.x0, kFdStep, g, J);
  qls.bumps(f.x0, 2.0 * kFdStep, g2, J2);
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
  const double kappa = svd.singularValues()[0] / svd.singularValues()[svd.singularValues().size() - 1];
  const double fd_rel = (J - J2).cwiseAbs().maxCoeff() / J.cwiseAbs().maxCoeff() +
                        (g - g2).cwiseAbs().maxCoeff() / g.cwiseAbs().maxCoeff();
  std::cout << "  [pnl] QuantLib J kappa=" << kappa << ", FD rel err est=" << fd_rel << "\n";
  ASSERT_LT(kappa * fd_rel, 0.25 * kLadderRel) << "the ladder budget would not hold";

  const Eigen::VectorXd ladder = J.transpose().fullPivLu().solve(g);  // dP/dq, square hard problem
  const json::array& ml = e.at("market_ladder").as_array();
  ASSERT_EQ(static_cast<int>(ml.size()), ladder.size());
  double market_ql = 0.0, scale = 0.0;
  for (int i = 0; i < ladder.size(); ++i) {
    const double want = ladder[i] * dq_ql[i];
    market_ql += want;
    scale += std::abs(want);
    EXPECT_NEAR(num(ml[static_cast<std::size_t>(i)]), want, kLadderRel * std::abs(want) + kNpvAbs) << i;
  }
  EXPECT_NEAR(num(e.at("market")), market_ql, kLadderRel * scale + kNpvAbs);

  const double residual_ql = (npv_t1 - cr_t1) - market_ql;
  const double res_tol = kLadderRel * scale + 2 * kNpvAbs;
  EXPECT_NEAR(num(e.at("residual")), residual_ql, res_tol);
  EXPECT_GT(std::abs(residual_ql), 10 * res_tol) << "the second-order piece must be resolvable at this tolerance";

  std::cout << "  [pnl] QL: npv_t0=" << npv_t0 << " carry=" << carry_t1 - npv_t0 << " roll=" << cr_t1 - carry_t1
            << " market=" << market_ql << " residual=" << residual_ql << "\n";
}

// T5: on a FLAT continuously compounded curve the slide is invisible (DF(t - dt) == DF(t)/DF(dt)), so roll is 0 and
// carry is the closed-form financing of the surviving flows: sum_{pay > dt} PV_k (e^{f dt} - 1) - sum_{pay <= dt} PV_k.
// PV_k are hand formulas on the builder's coupon schedule (inputs, not engine outputs): a telescoped OIS coupon paid
// at its accrual end with tau_pay == tau_index is DF(s) - DF(e); a fixed coupon is K tau DF(pay).
TEST(PnlOracle, FlatCurveRollIsZeroAndCarryIsTheClosedFormFinancing) {
  Fixture f;
  build(f);
  const double fwd = 0.03;
  f.x0 = Eigen::VectorXd::Constant(f.prob.n_knots(), fwd);
  const json::object e = run_pnl(f, bundle_at(f, f.x0), nullptr, nullptr);  // no bundle1: dq = 0, x1 = x0

  const auto df = [&](double t) { return std::exp(-fwd * t); };
  double paid = 0.0, surv = 0.0, npv0 = 0.0;
  for (const json::value& pv : f.positions) {
    const json::object& po = pv.as_object();
    const double N = num(po.at("notional")), K = num(po.at("fixed_rate"));
    const auto add = [&](double pay, double v) {
      npv0 += v;
      (pay <= f.dt ? paid : surv) += v;
    };
    for (const json::value& cv_ : po.at("float_coupons").as_array()) {
      const json::object& c = cv_.as_object();
      const json::object& o = c.at("obs").as_object();
      const double s = num(o.at("sub_start").as_array().front()), en = num(o.at("sub_end").as_array().front());
      ASSERT_DOUBLE_EQ(num(c.at("tau_pay")), num(o.at("tau_index"))) << "the telescoping closed form needs tau_pay == tau_index";
      ASSERT_DOUBLE_EQ(num(c.at("pay")), en) << "no payment lag";
      add(num(c.at("pay")), N * (df(s) - df(en)));
    }
    for (const json::value& xv : po.at("fixed_coupons").as_array()) {
      const json::object& c = xv.as_object();
      add(num(c.at("pay")), -N * K * num(c.at("tau")) * df(num(c.at("pay"))));
    }
  }
  const double carry_cf = surv * (std::exp(fwd * f.dt) - 1.0) - paid;  // == surv e^{f dt} - npv0
  EXPECT_NEAR(num(e.at("npv_t0")), npv0, kFlatAbs);
  EXPECT_NEAR(num(e.at("carry")), carry_cf, kFlatAbs);
  EXPECT_NEAR(num(e.at("roll")), 0.0, kFlatAbs);
  EXPECT_EQ(num(e.at("market")), 0.0) << "no bundle1: dq is identically zero";
  EXPECT_NEAR(num(e.at("total")), carry_cf, kFlatAbs);
  EXPECT_NEAR(num(e.at("residual")), 0.0, 2 * kFlatAbs);
  EXPECT_GT(std::abs(carry_cf), 1e6 * kFlatAbs);
}
