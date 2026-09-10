// @oracle-test — QuantLib's OWN instrument objects define the residuals of a four-curve calibration and our
// LM solves it; the KNOT FORWARDS must match our own instruments'. DO NOT DELETE OR WEAKEN.
// E5 taxonomy: T1 oracle (engine number vs an independent number)
//
// See tests/ORACLE_TESTS.md before changing anything here.
//
// WHAT THIS PINS THAT NOTHING ELSE DID (2026-09-10). Every other QuantLib oracle here compares a QUOTE at a
// FIXED curve: given this curve, do we and QuantLib price this instrument the same? That checks the pricing
// kernel. It does not check what a client actually consumes -- the CURVE the calibration lands on -- and an
// error in the instrument assembly shows up there magnified by the inverse Jacobian.
//
// Both arms live in the SAME state space, which is what makes the comparison exact: both build OUR
// ModularCurve, with the same regions, from the same knot forwards x. The only difference is who turns that
// curve into a quote.
//   arm A  our builders (conventions DB -> build::par_swap / observation) + our templated kernel
//   arm B  QuantLib instrument objects (OvernightIndexedSwap, VanillaSwap, OvernightIndexFuture) repricing
//          off our curve through qlx::CurveTermStructure
// Both are solved by the SAME LM (calibrate_with), so nothing but the residual definition differs, and the
// answer is compared in knot forwards -- basis points of curve, not abstract NPV.
//
// The E2 averaged-weight bug would have failed this by ~1.4% of the FF curve.
//
// SCOPE, stated rather than implied: this covers the instruments QuantLib can express. A soft BAND (a quote
// resting off-market inside it is not a bootstrap at all), a TURN JUMP as a calibrated free variable, a
// BUTTERFLY as one combination residual and an MtM XCCY basis have no QuantLib counterpart, so they stay on
// cross-path parity and their own pins. See tests/ORACLE_TESTS.md.
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>

#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/observations.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;
namespace b = swaps::build;

namespace {

swaps::build::Date eng_date(const QuantLib::Date& d) {
  std::ostringstream os;
  os << QuantLib::io::iso_date(d);
  return swaps::build::Date::from_iso(os.str());
}

// Curve roles. FF is a SPREAD over SOFR and EURIBOR a spread over ESTR, so the bundle exercises the
// recursive base chain as well as two independent currencies.
enum Role { SOFR = 0, FF = 1, ESTR = 2, EUR6M = 3, NROLES = 4 };

struct Fixture {
  Date today;
  DayCounter dc = Actual365Fixed();
  cal::BundleProblem prob;                                  // arm A: our instruments
  std::vector<std::function<double()>> ql_quote;            // arm B: QuantLib's fair rate for the same trade
  std::vector<RelinkableHandle<YieldTermStructure>> h{NROLES};
  Eigen::VectorXd x_true, x0;
  // QuantLib state kept alive for the fixture's lifetime.
  ext::shared_ptr<Sofr> sofr;
  ext::shared_ptr<FedFunds> ff;
  ext::shared_ptr<Estr> estr;
  ext::shared_ptr<Euribor6M> eur6m;
  std::vector<ext::shared_ptr<Instrument>> keep;
};

}  // namespace

namespace {

const std::vector<std::string>& tenors() {
  static const std::vector<std::string> t = {"1Y", "2Y", "3Y", "5Y", "10Y", "30Y"};
  return t;
}

// One OIS block: our par_swap from the DB, and the QuantLib OvernightIndexedSwap that means the same trade,
// built to the SAME conventions (spot lag, payment lag, roll) rather than to MakeOIS's defaults.
template <class Index>
std::vector<double> add_ois(Fixture& f, const b::SwapConv& conv, const ext::shared_ptr<Index>& idx, int fc,
                            int disc) {
  std::vector<double> knots;
  for (const std::string& t : tenors()) {
    const ext::shared_ptr<OvernightIndexedSwap> qls =
        MakeOIS(PeriodParser::parse(t), idx, 0.03)
            .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
            .withSettlementDays(conv.spot_lag)
            .withPaymentLag(conv.pay_lag)
            .withPaymentAdjustment(ModifiedFollowing);
    f.keep.push_back(qls);
    f.prob.instruments.push_back(
        b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
    f.ql_quote.emplace_back([qls] { return qls->fairRate(); });
  }
  return knots;
}

// One IBOR block: fixed vs 6M EURIBOR, discounted on ESTR. QuantLib's VanillaSwap needs an explicit pricing
// engine for the discount curve; the forecast curve reaches it through the index handle.
std::vector<double> add_ibor(Fixture& f, const b::SwapConv& conv, int fc, int disc) {
  std::vector<double> knots;
  for (const std::string& t : tenors()) {
    const ext::shared_ptr<VanillaSwap> qls = MakeVanillaSwap(PeriodParser::parse(t), f.eur6m, 0.03)
                                                 .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
                                                 .withSettlementDays(conv.spot_lag);
    f.keep.push_back(qls);
    f.prob.instruments.push_back(
        b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
    f.ql_quote.emplace_back([qls] { return qls->fairRate(); });
  }
  return knots;
}

// Monthly ARITHMETIC-AVERAGE Fed funds futures on the front of the FF curve -- the observation shape the E2
// bug lived in, and the one the moment path approximates. QuantLib: RateAveraging::Simple.
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
    f.ql_quote.emplace_back([qlf] { return 1.0 - qlf->NPV() / 100.0; });
    s = e;
  }
  return knots;
}

}  // namespace

namespace {

// The residual engine of ARM B: QuantLib prices, our state vector. Every call rebuilds our curves at x,
// relinks them into the QuantLib handles, and asks each QuantLib instrument for its fair rate. The
// Jacobian is a central difference because QuantLib offers no AD -- which is a feature here, since it
// means arm B shares no derivative code with arm A either.
class QuantLibEngine {
 public:
  QuantLibEngine(Fixture& f, Eigen::VectorXd market) : f_(&f), q_(std::move(market)) {}

  Eigen::VectorXd residuals(const Eigen::VectorXd& x) const {
    relink(x);
    Eigen::VectorXd r(static_cast<int>(f_->ql_quote.size()));
    for (int i = 0; i < r.size(); ++i) r[i] = f_->ql_quote[static_cast<std::size_t>(i)]() - q_[i];
    return r;
  }
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {
    const int n = static_cast<int>(x.size()), m = static_cast<int>(f_->ql_quote.size());
    Eigen::MatrixXd J(m, n);
    const double eps = 1e-7;  // knot forwards are O(1e-2); 1e-7 keeps ~8 digits of the difference
    for (int j = 0; j < n; ++j) {
      Eigen::VectorXd up = x, dn = x;
      up[j] += eps;
      dn[j] -= eps;
      J.col(j) = (residuals(up) - residuals(dn)) / (2 * eps);
    }
    return J;
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
      f_->h[c].linkTo(ts_.back());  // notifies every instrument built off this handle
    }
  }
  Fixture* f_;
  Eigen::VectorXd q_;
  mutable std::vector<std::unique_ptr<cal::CurveHandle<double>>> curves_;
  mutable std::vector<ext::shared_ptr<YieldTermStructure>> ts_;
};

// The bundle: SOFR outright, FF as a spread over it (averaged futures on the front, OIS on the back),
// ESTR outright, 6M EURIBOR as a spread over ESTR. Square and hard -- one knot per instrument -- which is
// what makes "the same knot forwards" a well-posed question.
Fixture build() {
  Fixture f;
  f.today = Date(8, July, 2026);
  Settings::instance().evaluationDate() = f.today;
  f.sofr = ext::make_shared<Sofr>(f.h[SOFR]);
  f.ff = ext::make_shared<FedFunds>(f.h[FF]);
  f.estr = ext::make_shared<Estr>(f.h[ESTR]);
  f.eur6m = ext::make_shared<Euribor6M>(f.h[EUR6M]);

  const auto k0 = add_ois(f, b::swap_conv("USD", "USD-SOFR"), f.sofr, SOFR, SOFR);
  auto k1 = add_averaged_futures(f, FF, 3);
  const auto k1b = add_ois(f, b::swap_conv("USD", "USD-FEDFUNDS"), f.ff, FF, SOFR);
  k1.insert(k1.end(), k1b.begin(), k1b.end());
  const auto k2 = add_ois(f, b::swap_conv("EUR", "EUR-ESTR"), f.estr, ESTR, ESTR);
  const auto k3 = add_ibor(f, b::swap_conv("EUR", "EUR-EURIBOR-6M"), EUR6M, ESTR);

  f.prob.curves.resize(NROLES);
  f.prob.curves[SOFR] = px::CurveStructure{.base = -1, .currency = 0, .regions = cv::flat_hermite({}, k0)};
  f.prob.curves[FF] = px::CurveStructure{.base = SOFR, .currency = 0, .regions = cv::flat_hermite({}, k1)};
  f.prob.curves[ESTR] = px::CurveStructure{.base = -1, .currency = 1, .regions = cv::flat_hermite({}, k2)};
  f.prob.curves[EUR6M] = px::CurveStructure{.base = ESTR, .currency = 1, .regions = cv::flat_hermite({}, k3)};

  // A plausible state: 3% USD, +8 bp FF, 2.5% EUR, +15 bp EURIBOR, each tilted so no two knots coincide.
  const double level[NROLES] = {0.030, 0.0008, 0.025, 0.0015};
  const double tilt[NROLES] = {4e-4, 2e-5, 3e-4, 1e-5};
  f.x_true.setZero(f.prob.n_knots());
  f.x0.setZero(f.prob.n_knots());
  for (int c = 0; c < NROLES; ++c) {
    const int o = f.prob.offset(c);
    for (int i = 0; i < f.prob.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i) {
      f.x_true[o + i] = level[c] + tilt[c] * i;
      f.x0[o + i] = level[c];  // a flat cold seed, deliberately not the answer
    }
  }
  // The market is OUR model quote at x_true, so arm A returns x_true by construction and the whole question
  // is whether arm B -- QuantLib's instruments, same curve, same solver -- lands in the same place.
  const Eigen::VectorXd q = f.prob.residuals<double>(f.x_true) + f.prob.market();
  for (int i = 0; i < f.prob.n_residuals(); ++i) f.prob.instruments[static_cast<std::size_t>(i)].market = q[i];
  return f;
}

}  // namespace

// Diagnostic first: at the SAME curve, do the two definitions of each instrument agree on the quote? This is
// the quantity the solve amplifies, so when the calibration test below fails this says which row did it.
TEST(CalibrationOracle, EveryInstrumentQuoteAgreesWithQuantLibAtTheGeneratingCurve) {
  Fixture f = build();
  const QuantLibEngine ql(f, Eigen::VectorXd::Zero(f.prob.n_residuals()));
  const Eigen::VectorXd ql_q = ql.residuals(f.x_true);  // market is zero here, so this IS the QuantLib quote
  const Eigen::VectorXd our_q = f.prob.market();        // == our model quote at x_true, by construction
  double worst = 0.0;
  for (int i = 0; i < ql_q.size(); ++i) {
    if (std::abs(ql_q[i] - our_q[i]) > 5e-13)  // only the offenders, so a failure names its own row
      std::cout << "    row " << i << "  ours=" << our_q[i] << "  ql=" << ql_q[i]
                << "  diff_bp=" << (our_q[i] - ql_q[i]) * 1e4 << "\n";
    EXPECT_NEAR(ql_q[i], our_q[i], 5e-13) << "row " << i;
    worst = std::max(worst, std::abs(ql_q[i] - our_q[i]));
  }
  std::cout << "  [quote] worst |ours - QuantLib| over " << ql_q.size() << " rows = " << worst << "\n";
}

// THE TEST: solve the four-curve bundle with QuantLib defining every residual, and demand the knot forwards
// our own instruments imply. Same LM, same seed, same state space.
TEST(CalibrationOracle, CalibratingAgainstQuantLibInstrumentsRecoversTheSameKnotForwards) {
  Fixture f = build();
  const int n = f.prob.n_knots();
  ASSERT_EQ(n, f.prob.n_residuals()) << "the comparison needs a square, hard problem";

  // arm A -- our instruments, our kernel, our LM.
  const cal::CalibrationResult a = cal::calibrate(f.prob, f.x0);
  ASSERT_TRUE(a.converged) << a.status;
  EXPECT_LT((a.x - f.x_true).cwiseAbs().maxCoeff(), 1e-10) << "arm A must recover its own generating state";

  // arm B -- QuantLib's instruments, our curve, the SAME LM loop.
  const QuantLibEngine eng(f, f.prob.market());
  const cal::CalibrationResult bres = cal::calibrate_with(eng, n, f.prob.n_residuals(), f.x0);
  ASSERT_TRUE(bres.converged) << bres.status;

  const Eigen::VectorXd d = bres.x - a.x;
  const double worst_bp = d.cwiseAbs().maxCoeff() * 1e4;
  for (int c = 0; c < NROLES; ++c) {
    const int o = f.prob.offset(c);
    const int nk = f.prob.curves[static_cast<std::size_t>(c)].n_interp_knots();
    const double curve_bp = d.segment(o, nk).cwiseAbs().maxCoeff() * 1e4;
    EXPECT_LT(curve_bp, 1e-4) << "curve " << c << " knot forwards differ by " << curve_bp << " bp";
  }
  std::cout << "  [calibration] worst knot-forward difference over " << n
            << " knots, 4 curves = " << worst_bp << " bp\n";
}
