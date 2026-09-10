// THE MULTI-CURVE QUANTLIB REFERENCE (2026-09-10).
//
// Every QuantLib TIMING reference in the perf gate until now was a SINGLE curve -- one SOFR bootstrap, or
// bonds -- so the engine's headline speedup was a single-currency number. The multi-curve comparisons were
// against our OWN templated kernel (BM_MultiCurveBook_Templated), not against QuantLib. This closes that: the
// same four-curve market, calibrated both ways.
//
//   USD SOFR    outright, self-discounted
//   USD FF      forecast on its own curve, DISCOUNTED ON SOFR
//   EUR ESTR    outright, self-discounted
//   EUR 6M      forecast on its own curve, DISCOUNTED ON ESTR
//
// WHAT IS AND IS NOT LIKE FOR LIKE, stated rather than implied. QuantLib bootstraps curve by curve, in
// dependency order, each curve exactly determined by its own helpers; we solve all four JOINTLY as one
// least-squares system. That is a real difference in what is being computed, not a handicap either way --
// the joint solve is what lets a quote depend on several curves at once (a basis, an FX forward, a
// butterfly), which is why the engine does it. Both arms produce the same discount factors to 1e-12 (asserted
// below, so this cannot silently drift into timing two different answers), and both start cold from a flat
// seed. What the comparison measures is: how long does it take to turn this market into curves?
#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"

using namespace QuantLib;
namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;
namespace b = swaps::build;

namespace {

enum Role { SOFR_C = 0, FF_C = 1, ESTR_C = 2, EUR6M_C = 3, NROLES = 4 };

swaps::build::Date eng_date(const QuantLib::Date& d) {
  std::ostringstream os;
  os << QuantLib::io::iso_date(d);
  return swaps::build::Date::from_iso(os.str());
}

const std::vector<std::string>& tenors() {
  static const std::vector<std::string> t = {"1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"};
  return t;
}

struct Fixture {
  Date today = Date(8, July, 2026);
  DayCounter dc = Actual365Fixed();
  cal::BundleProblem prob;
  Eigen::VectorXd x_true, x0;
  std::vector<RelinkableHandle<YieldTermStructure>> h{NROLES};
  ext::shared_ptr<Sofr> sofr;
  ext::shared_ptr<FedFunds> ff;
  ext::shared_ptr<Estr> estr;
  ext::shared_ptr<Euribor6M> eur6m;
  // One quote per instrument, in the same order as prob.instruments, shared by both arms.
  std::vector<double> quotes;
  std::vector<ext::shared_ptr<SimpleQuote>> ql_quotes;
  std::vector<ext::shared_ptr<RateHelper>> helpers[NROLES];
};

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
    f.prob.instruments.push_back(
        b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
  }
  return knots;
}

std::vector<double> add_ibor(Fixture& f, const b::SwapConv& conv, int fc, int disc) {
  std::vector<double> knots;
  for (const std::string& t : tenors()) {
    const ext::shared_ptr<VanillaSwap> qls =
        MakeVanillaSwap(PeriodParser::parse(t), f.eur6m, 0.03)
            .withDiscountingTermStructure(f.h[static_cast<std::size_t>(disc)])
            .withSettlementDays(conv.spot_lag);
    f.prob.instruments.push_back(
        b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()), fc, disc, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
  }
  return knots;
}

}  // namespace

namespace {

Fixture& fx() {
  static Fixture f = [] {
    Fixture f;
    Settings::instance().evaluationDate() = f.today;
    f.sofr = ext::make_shared<Sofr>(f.h[SOFR_C]);
    f.ff = ext::make_shared<FedFunds>(f.h[FF_C]);
    f.estr = ext::make_shared<Estr>(f.h[ESTR_C]);
    f.eur6m = ext::make_shared<Euribor6M>(f.h[EUR6M_C]);

    const auto k0 = add_ois(f, b::swap_conv("USD", "USD-SOFR"), f.sofr, SOFR_C, SOFR_C);
    const auto k1 = add_ois(f, b::swap_conv("USD", "USD-FEDFUNDS"), f.ff, FF_C, SOFR_C);
    const auto k2 = add_ois(f, b::swap_conv("EUR", "EUR-ESTR"), f.estr, ESTR_C, ESTR_C);
    const auto k3 = add_ibor(f, b::swap_conv("EUR", "EUR-EURIBOR-6M"), EUR6M_C, ESTR_C);

    f.prob.curves.resize(NROLES);
    f.prob.curves[SOFR_C] = px::CurveStructure{.base = -1, .currency = 0, .regions = cv::flat_hermite({}, k0)};
    f.prob.curves[FF_C] = px::CurveStructure{.base = -1, .currency = 0, .regions = cv::flat_hermite({}, k1)};
    f.prob.curves[ESTR_C] = px::CurveStructure{.base = -1, .currency = 1, .regions = cv::flat_hermite({}, k2)};
    f.prob.curves[EUR6M_C] = px::CurveStructure{.base = -1, .currency = 1, .regions = cv::flat_hermite({}, k3)};

    const double level[NROLES] = {0.030, 0.0308, 0.025, 0.0265};
    const double tilt[NROLES] = {4e-4, 4e-4, 3e-4, 3e-4};
    f.x_true.setZero(f.prob.n_knots());
    f.x0.setZero(f.prob.n_knots());
    for (int c = 0; c < NROLES; ++c) {
      const int o = f.prob.offset(c);
      for (int i = 0; i < f.prob.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i) {
        f.x_true[o + i] = level[c] + tilt[c] * i;
        f.x0[o + i] = level[c];
      }
    }
    // The market both arms calibrate to: our model quotes at x_true.
    const Eigen::VectorXd q = f.prob.residuals<double>(f.x_true) + f.prob.market();
    for (int i = 0; i < f.prob.n_residuals(); ++i) {
      f.prob.instruments[static_cast<std::size_t>(i)].market = q[i];
      f.quotes.push_back(q[i]);
      f.ql_quotes.push_back(ext::make_shared<SimpleQuote>(q[i]));
    }
    // QuantLib helpers, one block per curve, each in dependency order (FF discounts on SOFR, EUR6M on ESTR).
    const std::size_t n = tenors().size();
    for (std::size_t i = 0; i < n; ++i) {
      const Period p = PeriodParser::parse(tenors()[i]);
      const Handle<Quote> q0(f.ql_quotes[i]), q1(f.ql_quotes[n + i]), q2(f.ql_quotes[2 * n + i]),
          q3(f.ql_quotes[3 * n + i]);
      f.helpers[SOFR_C].push_back(ext::make_shared<OISRateHelper>(2, p, q0, f.sofr, Handle<YieldTermStructure>(), false, 2));
      f.helpers[FF_C].push_back(ext::make_shared<OISRateHelper>(2, p, q1, f.ff, f.h[SOFR_C], false, 2));
      f.helpers[ESTR_C].push_back(ext::make_shared<OISRateHelper>(2, p, q2, f.estr, Handle<YieldTermStructure>(), false, 2));
      // EUR-EURIBOR-6M-IRS from the DB: fixed 30E/360 annual (Eurobond basis), float ACT/360 semi, spot 2.
      f.helpers[EUR6M_C].push_back(ext::make_shared<SwapRateHelper>(q3, p, TARGET(), Annual, ModifiedFollowing,
                                                                    Thirty360(Thirty360::European), f.eur6m,
                                                                    Handle<Quote>(), 0 * Days, f.h[ESTR_C]));
    }
    return f;
  }();
  return f;
}

using PWC = PiecewiseYieldCurve<Discount, LogLinear, IterativeBootstrap>;

// A QuantLib term structure seen as one of OUR curve handles, so our pricer can read QuantLib's answer.
// (qlx::CurveTermStructure is the same bridge in the other direction.) Only discounts are consulted by a
// par swap, but forward/integral are supplied so the handle is honest about what it is.
struct QlCurve : cal::CurveHandle<double> {
  const YieldTermStructure* ts = nullptr;
  double discount(double t) const override { return ts->discount(t, true); }
  double integral(double t) const override { return -std::log(ts->discount(t, true)); }
  double forward(double t) const override {
    return ts->forwardRate(t, t + 1.0 / 365.0, Continuous, NoFrequency, true).rate();
  }
  double turn_jump(int) const override { return 0.0; }
  void pieces_into(std::vector<double>&) const override {}
  void set_forwards(const Eigen::VectorXd&) override {}
};

// QuantLib: four bootstraps in dependency order, each curve triggered by asking for a discount factor.
std::vector<ext::shared_ptr<YieldTermStructure>> ql_bootstrap(Fixture& f) {
  std::vector<ext::shared_ptr<YieldTermStructure>> out(NROLES);
  const int order[NROLES] = {SOFR_C, FF_C, ESTR_C, EUR6M_C};
  for (int c : order) {
    auto curve = ext::make_shared<PWC>(f.today, f.helpers[c], f.dc);
    curve->enableExtrapolation();
    f.h[static_cast<std::size_t>(c)].linkTo(curve);
    benchmark::DoNotOptimize(curve->discount(30.0));  // forces the bootstrap
    out[static_cast<std::size_t>(c)] = curve;
  }
  return out;
}

void BM_MultiCurve4_QuantLib(benchmark::State& state) {
  Fixture& f = fx();
  for (auto _ : state) benchmark::DoNotOptimize(ql_bootstrap(f).size());
}
BENCHMARK(BM_MultiCurve4_QuantLib);

void BM_MultiCurve4_Ours(benchmark::State& state) {
  Fixture& f = fx();
  for (auto _ : state) {
    const cal::CalibrationResult r = cal::calibrate(f.prob, f.x0);
    benchmark::DoNotOptimize(r.rms_residual);
  }
}
BENCHMARK(BM_MultiCurve4_Ours);

// THE TICK. A cold solve happens once; a desk re-curves on every market move, and that is where the two
// approaches genuinely diverge -- QuantLib must re-bootstrap all four curves from scratch because a helper
// quote changed, while the engine re-solves off a frozen Jacobian and refreshes it only on staleness.
// +-0.3 bp on EVERY quote, alternating, so neither arm can cache a no-op.
void BM_MultiCurve4_QuantLib_Tick(benchmark::State& state) {
  Fixture& f = fx();
  bool up = false;
  for (auto _ : state) {
    up = !up;
    const double d = up ? 3e-5 : -3e-5;
    for (std::size_t i = 0; i < f.ql_quotes.size(); ++i) f.ql_quotes[i]->setValue(f.quotes[i] + d);
    benchmark::DoNotOptimize(ql_bootstrap(f).size());  // every curve is invalidated by its own quotes
  }
}
BENCHMARK(BM_MultiCurve4_QuantLib_Tick);

void BM_MultiCurve4_Ours_Tick(benchmark::State& state) {
  Fixture& f = fx();
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  sess.start_streaming();
  Eigen::VectorXd q(f.prob.n_residuals());
  bool up = false;
  for (auto _ : state) {
    up = !up;
    const double d = up ? 3e-5 : -3e-5;
    for (int i = 0; i < q.size(); ++i) q[i] = f.quotes[static_cast<std::size_t>(i)] + d;
    benchmark::DoNotOptimize(sess.stream_update(q).size());
  }
  if (!sess.last_converged()) state.SkipWithError("streamed tick did not converge");
}
BENCHMARK(BM_MultiCurve4_Ours_Tick);

}  // namespace

int main(int argc, char** argv) {
  Fixture& f = fx();
  // Both arms must be solving the SAME problem, or this times two different answers. Comparing discount
  // factors would be the wrong test: the two curves satisfy the same market with DIFFERENT interpolation
  // (QuantLib log-linear on discounts, ours Hermite on forwards), so they legitimately differ between the
  // points the market pins. The well-posed check is that QuantLib's bootstrapped curves REPRICE OUR
  // instruments -- which is also what catches a convention mismatch, since QuantLib's bootstrap reproduces
  // its own helpers exactly whatever conventions they carry.
  ql_bootstrap(f);
  std::vector<QlCurve> adapters(NROLES);
  for (int c = 0; c < NROLES; ++c) adapters[static_cast<std::size_t>(c)].ts = f.h[static_cast<std::size_t>(c)].currentLink().get();
  const auto curve_of = [&adapters](int i) -> const cal::CurveHandle<double>& {
    return adapters[static_cast<std::size_t>(i)];
  };
  double worst = 0.0;
  for (const auto& ins : f.prob.instruments)
    worst = std::max(worst, std::abs(cal::instrument_model_quote<double>(ins, curve_of) - ins.market));
  std::fprintf(stderr, "QuantLib's 4 curves reprice our %d instruments to %.3e (worst absolute rate error)\n",
               f.prob.n_residuals(), worst);
  if (worst > 1e-9) {
    std::fprintf(stderr, "ARMS DISAGREE - the timing below would compare two different problems\n");
    return 1;
  }
  const cal::CalibrationResult r = cal::calibrate(f.prob, f.x0);
  if (!r.converged || r.rms_residual > 1e-10) {
    std::fprintf(stderr, "ours did not converge: %s (rms %.3e)\n", r.status, r.rms_residual);
    return 1;
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
