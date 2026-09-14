// @oracle-test — the calib_report run_json VERB vs QuantLib: per-quote model values, and the spectrum / condition / rank /
// identifiability recomputed from a QuantLib bump-and-reprice Jacobian (api/calib_report.cpp). DO NOT DELETE OR WEAKEN.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// See tests/ORACLE_TESTS.md before changing anything here.
//
// WHAT THIS PINS THAT NOTHING ELSE DID (draft 2026-09-14). tests/calib_report_test.cpp is T3 (the verb vs a native
// session) and tests/calibration_diagnostics_test.cpp is T5 on hand-built 2x2 matrices. Neither compares the verb's
// Jacobian-derived numbers to a Jacobian the engine did not compute. Here the reference Jacobian is QuantLib's
// instrument objects repriced off our curve under central bumps of the knot forwards, and the reference spectrum and
// projector come from Eigen's BDCSVD (the verb uses JacobiSVD for sigma and a COD pseudo-inverse for M). Neither
// shares engine code with the verb.
//
// WHY THE COMPARISON IS EXACT UP TO FD NOISE
//   * SAME STATE SPACE. Both arms read OUR ModularCurve at the SAME x: the verb's calibrated state, which the test
//     recovers by calling the verb's library function (cal::calibration_report<api::BundleSession>) on the SAME
//     decoded bundle (the object seam, so no text round trip). The only difference is who turns the curve into a
//     quote: the verb's kernel (arm A) or QuantLib's OvernightIndexedSwap / VanillaSwap / OvernightIndexFuture
//     through qlx::CurveTermStructure (arm B). This mirrors calibration_oracle_test.cpp, whose quote agreement on
//     these exact instruments is measured at 1.1e-16.
//   * HARD PINS ONLY. With no band and no FX forward, D = residual_market_scale = 1 (problem.hpp:455), so the
//     residual Jacobian IS dq/dx and identifiability is the plain projector diag(J J+). A band would scale its row
//     by the decay, which is the ENGINE's band model and has no QuantLib counterpart. It stays on
//     calibration_diagnostics_test / risk_scale_repro_test.
//   * OVER-DETERMINED ON PURPOSE. A square full-rank J has projector = identity, so every identifiability is exactly
//     1 and the comparison would be vacuous. Extra no-knot rows and one exact duplicate make the projector
//     non-trivial (the duplicated pair shares its leverage) while the market stays self-consistent. The residual is
//     still zero, so the calibrated x is the generating x.
//
// TOLERANCES (derived below from QuantLib's own numbers, never from the engine's):
//   * quotes: 5e-13, the same bound calibration_oracle_test uses for these rows.
//   * FD Jacobian error ||E||_2: MEASURED, not assumed, as 4 x ||J(h) - J(2h)||_2 (Richardson: the truncation at h
//     is 1/3 of the difference, and the roundoff at h is of its order). It is capped by an a-priori
//     kFdEntry*sqrt(m*n) so it cannot silently grow. The averaged FF futures set the roundoff floor (a daily
//     DF-ratio minus 1 of ~8e-5 per day).
//   * singular values: Weyl, |sigma_i - sigma'_i| <= ||E||_2.
//   * condition number: relative <= ||E||/sigma_min + ||E||/sigma_max.
//   * identifiability: |P - P'|_2 <= 2||E||_2/sigma_min (projector perturbation, full column rank).
//   Each derived bound is ASSERTED below a ceiling so it can never silently become vacuous.
//
// SCOPE, stated rather than implied: no regulariser (R enters M through R^T R and QuantLib has no counterpart), no
// band, no FX, no turn. rank_deficiency is compared on a full-rank problem only.
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

#include "swaps/api/bundle_api.hpp"
#include "swaps/api/calib_report.hpp"  // calib_report_json -- named so the include-closure coverage tool sees the verb
#include "swaps/api/codec.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/observations.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/diagnostics.hpp"
#include "swaps/calibration/residual_engine.hpp"  // kRankThreshold
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

constexpr double kQuoteTol = 5e-13;  // calibration_oracle_test.cpp's measured-1.1e-16 bound on these same rows
// A-PRIORI per-entry FD error CEILING. The tolerances actually used are the MEASURED Richardson estimate below; this
// only stops them from silently growing vacuous.
constexpr double kFdEntry = 1e-7;
// Step: a 30Y par rate's third derivative in a knot forward is O(T^3) ~ 1e3..1e4, so truncation h^2/6*f''' ~ 1e-9 at
// h = 1e-6 (5e-8 at 1e-5). Roundoff is ~1e-16/2h for telescoped OIS and ~5e-14/2h ~ 4e-8 for the averaged futures
// (daily DF-ratio-minus-1 cancellation). Those rows are few and local.
constexpr double kFdStep = 1e-6;

b::Date eng_date(const ql::Date& d) {
  std::ostringstream os;
  os << ql::io::iso_date(d);
  return b::Date::from_iso(os.str());
}

enum Role { SOFR = 0, FF = 1, ESTR = 2, EUR6M = 3, NROLES = 4 };

struct Fixture {
  ql::Date today;
  ql::DayCounter dc = ql::Actual365Fixed();
  cal::BundleProblem prob;
  std::vector<std::function<double()>> ql_quote;  // QuantLib's quote for row i, off whatever the handles link to
  std::vector<ql::RelinkableHandle<ql::YieldTermStructure>> h{NROLES};
  Eigen::VectorXd x_true, x_seed;
  ql::ext::shared_ptr<ql::Sofr> sofr;
  ql::ext::shared_ptr<ql::FedFunds> ff;
  ql::ext::shared_ptr<ql::Estr> estr;
  ql::ext::shared_ptr<ql::Euribor6M> eur6m;
  std::vector<ql::ext::shared_ptr<ql::Instrument>> keep;
};

// OIS rows: our par_swap from the DB and the QuantLib OIS on the SAME conventions (as calibration_oracle_test).
// `knotted` rows contribute their last pay time as a curve knot; the others only over-determine.
template <class Index>
std::vector<double> add_ois(Fixture& f, const b::SwapConv& conv, const ql::ext::shared_ptr<Index>& idx, int fc,
                            int disc, const std::vector<std::string>& tenors) {
  std::vector<double> pays;
  for (const std::string& t : tenors) {
    const ql::ext::shared_ptr<ql::OvernightIndexedSwap> qls =
        ql::MakeOIS(ql::PeriodParser::parse(t), idx, 0.03)
            .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
            .withSettlementDays(conv.spot_lag)
            .withPaymentLag(conv.pay_lag)
            .withPaymentAdjustment(ql::ModifiedFollowing);
    f.keep.push_back(qls);
    f.prob.instruments.push_back(b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0));
    pays.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
    f.ql_quote.emplace_back([qls] { return qls->fairRate(); });
  }
  return pays;
}

std::vector<double> add_ibor(Fixture& f, const b::SwapConv& conv, int fc, int disc,
                             const std::vector<std::string>& tenors) {
  std::vector<double> pays;
  for (const std::string& t : tenors) {
    const ql::ext::shared_ptr<ql::VanillaSwap> qls =
        ql::MakeVanillaSwap(ql::PeriodParser::parse(t), f.eur6m, 0.03)
            .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
            .withSettlementDays(conv.spot_lag);
    f.keep.push_back(qls);
    f.prob.instruments.push_back(b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0));
    pays.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
    f.ql_quote.emplace_back([qls] { return qls->fairRate(); });
  }
  return pays;
}

// Monthly arithmetic-average Fed funds futures (RateAveraging::Simple), as calibration_oracle_test.
std::vector<double> add_averaged_futures(Fixture& f, int fc, int n) {
  std::vector<double> knots;
  const ql::Calendar qcal = f.ff->fixingCalendar();
  ql::Date s = ql::Date(1, static_cast<ql::Month>(f.today.month() + 1), f.today.year());
  for (int i = 0; i < n; ++i) {
    const ql::Date e = qcal.advance(s, 1, ql::Months, ql::Following);
    const ql::ext::shared_ptr<ql::OvernightIndexFuture> qlf = ql::ext::make_shared<ql::OvernightIndexFuture>(
        f.ff, s, e, ql::Handle<ql::Quote>(), ql::RateAveraging::Simple);
    f.keep.push_back(qlf);
    const std::string dcs = b::index_day_count("USD-FEDFUNDS"), cls = b::index_calendar("USD-FEDFUNDS");
    f.prob.instruments.push_back(b::rate_instrument(
        fc, b::observation(eng_date(f.today), eng_date(s), eng_date(e), "averaged", 0.0, dcs, cls), 0.0));
    knots.push_back(b::curve_time(eng_date(f.today), eng_date(e)));
    f.ql_quote.emplace_back([qlf] { return 1.0 - qlf->NPV() / 100.0; });
    s = e;
  }
  return knots;
}

// The four-curve bundle of calibration_oracle_test (SOFR; FF over SOFR; ESTR; EURIBOR-6M over ESTR), plus
// over-determining rows with no knot of their own and one EXACT duplicate (the SOFR 10Y twice).
void build(Fixture& f) {
  f.today = ql::Date(8, ql::July, 2026);
  ql::Settings::instance().evaluationDate() = f.today;
  f.sofr = ql::ext::make_shared<ql::Sofr>(f.h[SOFR]);
  f.ff = ql::ext::make_shared<ql::FedFunds>(f.h[FF]);
  f.estr = ql::ext::make_shared<ql::Estr>(f.h[ESTR]);
  f.eur6m = ql::ext::make_shared<ql::Euribor6M>(f.h[EUR6M]);

  const std::vector<std::string> knotted = {"1Y", "2Y", "3Y", "5Y", "10Y", "30Y"};
  const b::SwapConv usd_sofr = b::swap_conv("USD", "USD-SOFR");
  const b::SwapConv usd_ff = b::swap_conv("USD", "USD-FEDFUNDS");
  const b::SwapConv eur_estr = b::swap_conv("EUR", "EUR-ESTR");
  const b::SwapConv eur_6m = b::swap_conv("EUR", "EUR-EURIBOR-6M");

  const std::vector<double> k0 = add_ois(f, usd_sofr, f.sofr, SOFR, SOFR, knotted);
  std::vector<double> k1 = add_averaged_futures(f, FF, 3);
  const std::vector<double> k1b = add_ois(f, usd_ff, f.ff, FF, SOFR, knotted);
  k1.insert(k1.end(), k1b.begin(), k1b.end());
  const std::vector<double> k2 = add_ois(f, eur_estr, f.estr, ESTR, ESTR, knotted);
  const std::vector<double> k3 = add_ibor(f, eur_6m, EUR6M, ESTR, knotted);

  // Over-determining rows: they reach existing knots only.
  (void)add_ois(f, usd_sofr, f.sofr, SOFR, SOFR, {"4Y", "7Y", "15Y", "20Y"});
  (void)add_ois(f, eur_estr, f.estr, ESTR, ESTR, {"4Y", "7Y", "20Y"});
  (void)add_ibor(f, eur_6m, EUR6M, ESTR, {"7Y"});
  (void)add_ois(f, usd_sofr, f.sofr, SOFR, SOFR, {"10Y"});  // exact duplicate of knotted SOFR row 4

  f.prob.curves.resize(NROLES);
  f.prob.curves[SOFR] = px::CurveStructure{.base = -1, .currency = 0, .regions = cv::flat_hermite({}, k0)};
  f.prob.curves[FF] = px::CurveStructure{.base = SOFR, .currency = 0, .regions = cv::flat_hermite({}, k1)};
  f.prob.curves[ESTR] = px::CurveStructure{.base = -1, .currency = 1, .regions = cv::flat_hermite({}, k2)};
  f.prob.curves[EUR6M] = px::CurveStructure{.base = ESTR, .currency = 1, .regions = cv::flat_hermite({}, k3)};

  const double level[NROLES] = {0.030, 0.0008, 0.025, 0.0015};
  const double tilt[NROLES] = {4e-4, 2e-5, 3e-4, 1e-5};
  f.x_true.setZero(f.prob.n_knots());
  f.x_seed.setZero(f.prob.n_knots());
  for (int c = 0; c < NROLES; ++c) {
    const int o = f.prob.offset(c);
    for (int i = 0; i < f.prob.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i) {
      f.x_true[o + i] = level[c] + tilt[c] * i;
      f.x_seed[o + i] = level[c];
    }
  }
  // Self-consistent market (our model at x_true): the over-determined system still has zero residual.
  const Eigen::VectorXd q = f.prob.residuals<double>(f.x_true) + f.prob.market();
  for (int i = 0; i < f.prob.n_residuals(); ++i) f.prob.instruments[static_cast<std::size_t>(i)].market = q[i];
}

// ARM B: QuantLib quotes at our state x, and their central-difference Jacobian.
class QuantLibArm {
 public:
  explicit QuantLibArm(Fixture& f) : f_(&f) {}

  Eigen::VectorXd quotes(const Eigen::VectorXd& x) const {
    relink(x);
    Eigen::VectorXd q(static_cast<int>(f_->ql_quote.size()));
    for (int i = 0; i < q.size(); ++i) q[i] = f_->ql_quote[static_cast<std::size_t>(i)]();
    return q;
  }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, double eps) const {
    const int n = static_cast<int>(x.size()), m = static_cast<int>(f_->ql_quote.size());
    Eigen::MatrixXd J(m, n);
    for (int j = 0; j < n; ++j) {
      Eigen::VectorXd up = x, dn = x;
      up[j] += eps;
      dn[j] -= eps;
      J.col(j) = (quotes(up) - quotes(dn)) / (2.0 * eps);
    }
    return J;
  }

 private:
  void relink(const Eigen::VectorXd& x) const {
    const cal::BundleProblem& p = f_->prob;
    curves_ = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
    ts_.clear();
    for (std::size_t c = 0; c < curves_.size(); ++c) {
      ts_.push_back(ql::ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
          f_->today, f_->dc, curves_[c].get()));
      f_->h[c].linkTo(ts_.back());
    }
  }
  Fixture* f_;
  mutable std::vector<std::unique_ptr<cal::CurveHandle<double>>> curves_;
  mutable std::vector<ql::ext::shared_ptr<ql::YieldTermStructure>> ts_;
};

json::array to_json(const Eigen::VectorXd& v) {
  json::array a;
  for (Eigen::Index i = 0; i < v.size(); ++i) a.push_back(v[i]);
  return a;
}

// The verb's document AND the calibrated state it was computed at. The state comes from the verb's own library
// call on the same decoded bundle (the verb does not emit x). The test asserts the two agree on everything the
// verb does emit.
struct VerbRun {
  json::object doc;
  cal::CalibrationReport lib;
};

VerbRun run_verb(const Fixture& f) {
  const json::value bundle = api::bundle_to_json(f.prob);
  json::object payload;
  payload["bundle"] = bundle;
  payload["x0"] = to_json(f.x_seed);
  json::object req;
  req["calib_report"] = std::move(payload);

  VerbRun r;
  r.doc = json::parse(api::calib_report_json(req)).as_object();  // the object seam (E6.3)
  cal::CalibrationReportRequest lr;
  lr.bundle = api::bundle_from_json(bundle);
  lr.x0 = f.x_seed;
  r.lib = cal::calibration_report<api::BundleSession>(std::move(lr));
  return r;
}

double num(const json::value& v) { return v.to_number<double>(); }

}  // namespace

// Per quote: the verb's model / residual / target vs QuantLib's instrument at the verb's calibrated state.
TEST(CalibReportOracle, EveryQuoteDiagnosticMatchesQuantLibAtTheCalibratedState) {
  Fixture f;
  build(f);
  const VerbRun v = run_verb(f);
  const int m = f.prob.n_residuals();

  ASSERT_TRUE(v.doc.at("converged").as_bool()) << v.doc.at("status").as_string();
  ASSERT_EQ(v.doc.at("n").to_number<int>(), m);
  ASSERT_EQ(static_cast<int>(v.lib.quotes.size()), m);
  const Eigen::VectorXd& x = v.lib.calibration.x;
  ASSERT_LT((x - f.x_true).cwiseAbs().maxCoeff(), 1e-10) << "a self-consistent over-determined market recovers x_true";
  // The emitted document is the library report (only the JSON double round trip can separate them).
  EXPECT_NEAR(num(v.doc.at("rms_residual")), v.lib.calibration.rms_residual, 1e-15);

  const QuantLibArm qlarm(f);
  const Eigen::VectorXd qq = qlarm.quotes(x);
  const json::array& quotes = v.doc.at("quotes").as_array();
  double worst = 0.0;
  for (int i = 0; i < m; ++i) {
    const json::object& d = quotes[static_cast<std::size_t>(i)].as_object();
    const double market = f.prob.instruments[static_cast<std::size_t>(i)].market;
    EXPECT_NEAR(num(d.at("model")), qq[i], kQuoteTol) << "row " << i;
    EXPECT_NEAR(num(d.at("residual")), qq[i] - market, kQuoteTol) << "row " << i;
    EXPECT_DOUBLE_EQ(num(d.at("target")), market) << "row " << i;
    EXPECT_FALSE(d.at("soft").as_bool()) << "row " << i << " -- the fixture is hard pins only";
    EXPECT_EQ(num(d.at("weight")), 1.0) << "row " << i;
    worst = std::max(worst, std::abs(num(d.at("model")) - qq[i]));
  }
  std::cout << "  [calib_report] worst |model - QuantLib| over " << m << " rows = " << worst << "\n";
}

// The Jacobian-derived health numbers vs the same numbers recomputed from QuantLib's bump-and-reprice Jacobian.
TEST(CalibReportOracle, SpectrumConditionRankAndIdentifiabilityMatchAQuantLibJacobian) {
  Fixture f;
  build(f);
  const VerbRun v = run_verb(f);
  ASSERT_TRUE(v.doc.at("converged").as_bool()) << v.doc.at("status").as_string();
  const Eigen::VectorXd& x = v.lib.calibration.x;
  const int m = f.prob.n_residuals(), n = f.prob.n_knots();
  ASSERT_GT(m, n) << "the fixture must be over-determined or identifiability is identically 1";

  // ---- the reference Jacobian, with its FD error ASSERTED rather than assumed --------------------------------
  const QuantLibArm qlarm(f);
  const Eigen::MatrixXd Jq = qlarm.jacobian(x, kFdStep);
  const Eigen::MatrixXd Jq2 = qlarm.jacobian(x, 2.0 * kFdStep);
  // ||E||_2 estimate: truncation at h is 1/3 of |J(h) - J(2h)|; roundoff at h is of the same order as the difference.
  // The x4 covers both. Measured on QuantLib's numbers only. Capped by the a-priori Frobenius ceiling.
  const double diff2 = Eigen::JacobiSVD<Eigen::MatrixXd>(Jq - Jq2).singularValues()[0];
  const double e2 = 4.0 * diff2 + 1e-12;
  std::cout << "  [calib_report] FD ||J(h) - J(2h)||_2 = " << diff2 << " -> ||E||_2 bound " << e2 << "\n";

  // ---- the reference spectrum and projector: Eigen BDCSVD, not the verb's JacobiSVD + COD ----------------------
  Eigen::BDCSVD<Eigen::MatrixXd> svd(Jq, Eigen::ComputeThinU);
  svd.setThreshold(cal::kRankThreshold);
  const Eigen::VectorXd sq = svd.singularValues();
  ASSERT_EQ(sq.size(), n);
  const double smax = sq[0], smin = sq[n - 1];
  ASSERT_GT(smin, 0.0);

  const double kappa_rel_tol = e2 / smin + e2 / smax;
  const double ident_tol = 2.0 * e2 / smin;
  std::cout << "  [calib_report] QuantLib sigma_max=" << smax << " sigma_min=" << smin << " -> sigma tol " << e2
            << ", kappa rel tol " << kappa_rel_tol << ", identifiability tol " << ident_tol << "\n";
  ASSERT_LT(e2, kFdEntry * std::sqrt(static_cast<double>(m) * n)) << "FD noisier than the a-priori ceiling";
  ASSERT_LT(e2, tol::jacobian_rel * smax) << "the sigma bound is vacuous";
  ASSERT_LT(ident_tol, 1e-3) << "sigma_min too small for a meaningful projector comparison";

  // singular values (descending in both)
  const json::array& sv = v.doc.at("singular_values").as_array();
  ASSERT_EQ(static_cast<int>(sv.size()), n);
  double worst_sv = 0.0;
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(num(sv[static_cast<std::size_t>(i)]), sq[i], e2) << "sigma_" << i;
    worst_sv = std::max(worst_sv, std::abs(num(sv[static_cast<std::size_t>(i)]) - sq[i]));
  }

  // condition number
  const double kappa_q = smax / smin;
  const double kappa = num(v.doc.at("condition_number"));
  EXPECT_NEAR(kappa / kappa_q - 1.0, 0.0, kappa_rel_tol);

  // rank deficiency (full rank here: an EXTERNAL rank, not the engine's COD echoed back)
  EXPECT_EQ(v.doc.at("rank_deficiency").to_number<int>(), n - static_cast<int>(svd.rank()));

  // identifiability = diag of the projector onto range(J): h_i = ||U_r(i, .)||^2
  const Eigen::Index r = svd.rank();
  const Eigen::MatrixXd Ur = svd.matrixU().leftCols(r);
  const json::array& quotes = v.doc.at("quotes").as_array();
  Eigen::VectorXd hq(m);
  double worst_h = 0.0;
  for (int i = 0; i < m; ++i) {
    hq[i] = Ur.row(i).squaredNorm();
    const double hv = num(quotes[static_cast<std::size_t>(i)].as_object().at("identifiability"));
    EXPECT_NEAR(hv, hq[i], ident_tol) << "row " << i;
    worst_h = std::max(worst_h, std::abs(hv - hq[i]));
  }

  // The reference itself must be discriminating (a fixture sanity, all QuantLib-side):
  EXPECT_NEAR(hq.sum(), static_cast<double>(r), 1e-9) << "trace of a projector = rank";
  const int dup_a = 4, dup_b = m - 1;  // knotted SOFR 10Y and its exact duplicate (last row added in build())
  EXPECT_NEAR(hq[dup_a], hq[dup_b], ident_tol) << "identical rows carry identical leverage";
  EXPECT_LT(hq[dup_b], 0.9) << "a duplicated row cannot pin its pillar alone";

  std::cout << "  [calib_report] worst |sigma - QL| = " << worst_sv << ", worst |identifiability - QL| = " << worst_h
            << ", kappa = " << kappa << " vs QL " << kappa_q << "\n";
}
