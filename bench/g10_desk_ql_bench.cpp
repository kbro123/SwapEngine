// THE G10 DESK, BOTH WAYS (2026-09-21). QuantLib and this engine turn the SAME multi-currency market into
// curves, re-curve it on a tick, and reprice the same book off the result.
//
// Every QuantLib timing reference in the gate so far has been one curve (SOFR), four curves
// (multicurve_ql_bench) or a bond book. A rates desk is neither: it is ~8 curves across 7 currencies,
// re-solved on every market move, with a few thousand swaps hanging off it. This bench is that workload.
//
//   USD SOFR    outright, self-discounted            GBP SONIA   outright, self-discounted
//   EUR ESTR    outright, self-discounted            JPY TONA    outright, self-discounted
//   EUR 3M      forecast, DISCOUNTED ON ESTR         CHF SARON   outright, self-discounted
//   CAD CORRA   outright, self-discounted            AUD AONIA   outright, self-discounted
//
// Conventions come from the conventions DB on BOTH sides (spot lag per currency: GBP 0, CAD/AUD 1, else 2;
// the index carries its own calendar and day count), so neither arm is handicapped by a hand-typed date rule.
//
// WHICH QUANTLIB CONFIGURATION IS THE REFERENCE, and why it is the one that flatters us least. QuantLib's
// interpolator is a template argument, and the choice moves BOTH shape and speed:
//   LogLinear on discounts  -- local, fast, and piecewise-CONSTANT instantaneous forwards (a step of up to
//                              26 bp at a pillar; ours is C1, max daily move 0.12 bp)
//   MonotonicLogCubic       -- the comparable SHAPE to ours (max daily move 0.09 bp; the two smooth curves
//                              differ by 6.8 bp of forward between pillars, vs 18.8 bp for log-linear), but
//                              GLOBAL, so the bootstrap iterates: measured 41.0 ms cold / 44.4 ms per tick
//                              on this desk against log-linear's 13.1 / 14.8 ms -- 3.1x slower.
// Both are timed here (BM_G10_*_QuantLib_Smooth is the smooth arm). The GATED reference is LogLinear, the
// FASTER one, so the multiple we quote (4.2x cold, ~1,300x tick) is the conservative number; the
// like-for-like-on-shape comparison would be 13.0x and ~3,900x. Shape evidence: scratchpad/curve_shape.cpp.
//
// WHAT IS AND IS NOT LIKE FOR LIKE, stated rather than implied (as multicurve_ql_bench states it):
// QuantLib bootstraps curve by curve in dependency order, each exactly determined by its own helpers; we
// solve the whole set JOINTLY as one least-squares system. That difference is the point of the engine, not a
// handicap -- a joint solve is what lets a quote depend on several curves at once -- and both arms are held
// to the same answer: main() below refuses to time anything until QuantLib's bootstrapped curves reprice
// OUR instruments to 1e-9, and until the two portfolio arms agree on the book's NPV.
#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/ql/ql_term_structure.hpp"

using namespace QuantLib;
namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;
namespace b = swaps::build;
namespace pf = swaps::portfolio;

namespace {

// Curve roles. EUR_3M forecasts EURIBOR 3M and discounts on ESTR; every RFR curve is self-discounted.
enum Role { USD_SOFR = 0, EUR_ESTR, EUR_3M, GBP_SONIA, JPY_TONA, CHF_SARON, CAD_CORRA, AUD_AONIA, NROLES };
// Engine currency tags (engine-blind: only a builder reads them).
enum Ccy { USD = 0, EUR = 1, GBP = 2, JPY = 3, CHF = 4, CAD = 5, AUD = 6 };

b::Date eng_date(const QuantLib::Date& d) {
  std::ostringstream os;
  os << QuantLib::io::iso_date(d);
  return b::Date::from_iso(os.str());
}

const std::vector<std::string>& tenors() {
  static const std::vector<std::string> t = {"1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"};
  return t;
}

struct Leg {  // one curve's market: the DB convention, its QuantLib index, and where it discounts
  Role role;
  Ccy ccy;
  const char* ccy_code;
  const char* index;                         // conventions-DB INDEX id (swap_conv derives the product)
  ext::shared_ptr<OvernightIndex> on_index;  // null for the IBOR curve
  Role discount;
};

struct Fixture {
  Date today = Date(8, July, 2026);
  DayCounter dc = Actual365Fixed();
  cal::BundleProblem prob;
  Eigen::VectorXd x_true, x0;
  std::vector<RelinkableHandle<YieldTermStructure>> h{NROLES};
  std::vector<Leg> legs;
  ext::shared_ptr<Euribor3M> eur3m;
  std::vector<double> quotes;
  std::vector<ext::shared_ptr<SimpleQuote>> ql_quotes;
  std::vector<ext::shared_ptr<RateHelper>> helpers[NROLES];
  // The book, both ways.
  pf::MultiCurveBook book;
  std::vector<ext::shared_ptr<Swap>> ql_book;
  // OUR calibrated curves, wrapped so QuantLib can price off them (the book arms must share curves --
  // see link_ours below).
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> our_curves;
};

// One OIS strip: the engine instrument from the DB convention, the matching QuantLib helper, one knot each.
std::vector<double> add_ois(Fixture& f, const Leg& lg) {
  const b::SwapConv conv = b::swap_conv(lg.ccy_code, lg.index);
  std::vector<double> knots;
  for (const std::string& t : tenors()) {
    const ext::shared_ptr<OvernightIndexedSwap> qls =
        MakeOIS(PeriodParser::parse(t), lg.on_index, 0.03)
            .withDiscountingTermStructure(f.h[static_cast<std::size_t>(lg.discount)])
            .withSettlementDays(conv.spot_lag)
            .withPaymentLag(conv.pay_lag)
            .withPaymentAdjustment(ModifiedFollowing);
    f.prob.instruments.push_back(b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()),
                                             lg.role, lg.discount, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
  }
  return knots;
}

std::vector<double> add_ibor(Fixture& f, const Leg& lg) {
  const b::SwapConv conv = b::swap_conv(lg.ccy_code, lg.index);
  std::vector<double> knots;
  for (const std::string& t : tenors()) {
    const ext::shared_ptr<VanillaSwap> qls =
        MakeVanillaSwap(PeriodParser::parse(t), f.eur3m, 0.03)
            .withDiscountingTermStructure(f.h[static_cast<std::size_t>(lg.discount)])
            .withSettlementDays(conv.spot_lag);
    f.prob.instruments.push_back(b::par_swap(eng_date(f.today), conv, eng_date(qls->maturityDate()),
                                             lg.role, lg.discount, 0.0));
    knots.push_back(f.prob.instruments.back().fixed.coupons.back().pay);
  }
  return knots;
}

// The book: BOOK_N swaps spread over the curves and tenors, each a payer of a fixed rate near the market.
constexpr int kBookN = 2000;

void build_book(Fixture& f) {
  const char* tenor_cycle[] = {"2Y", "5Y", "10Y", "30Y"};
  for (int i = 0; i < kBookN; ++i) {
    const Leg& lg = f.legs[static_cast<std::size_t>(i % f.legs.size())];
    const std::string t = tenor_cycle[(i / static_cast<int>(f.legs.size())) % 4];
    const b::SwapConv conv = b::swap_conv(lg.ccy_code, lg.index);
    const double rate = 0.030 + 0.0001 * (i % 17);   // a spread of off-market contract rates
    const double notional = 1e7 * (1 + (i % 5));

    // MakeOIS/MakeVanillaSwap convert to their OWN pointer type; widen to Swap in a second step (the
    // compiler will not chain the conversion operator with a derived-to-base conversion).
    ext::shared_ptr<Swap> qls;
    QuantLib::Date maturity;
    if (lg.on_index) {
      const ext::shared_ptr<OvernightIndexedSwap> o =
          MakeOIS(PeriodParser::parse(t), lg.on_index, rate)
              .withDiscountingTermStructure(f.h[static_cast<std::size_t>(lg.discount)])
              .withSettlementDays(conv.spot_lag)
              .withPaymentLag(conv.pay_lag)
              .withPaymentAdjustment(ModifiedFollowing)
              .withNominal(notional);
      maturity = o->maturityDate();
      qls = o;
    } else {
      // The DB's EUR IRS fixed leg is ANNUAL 30E/360 on TARGET; MakeVanillaSwap's defaults are not, and the
      // calibration helper above says so explicitly too. Left implicit, the book's EUR swaps would be a
      // different instrument from ours -- which is exactly what gate 3 caught (rel 8.9e-5).
      const ext::shared_ptr<VanillaSwap> v =
          MakeVanillaSwap(PeriodParser::parse(t), f.eur3m, rate)
              .withDiscountingTermStructure(f.h[static_cast<std::size_t>(lg.discount)])
              .withSettlementDays(conv.spot_lag)
              .withFixedLegTenor(Period(1, Years))
              .withFixedLegDayCount(Thirty360(Thirty360::European))
              .withFixedLegCalendar(TARGET())
              .withFixedLegConvention(ModifiedFollowing)
              .withNominal(notional);
      maturity = v->maturityDate();
      qls = v;
    }
    f.ql_book.push_back(qls);

    // OURS: the same dated cashflows, from the same builder the calibration instruments come from.
    const cal::Instrument ins = b::par_swap(eng_date(f.today), conv, eng_date(maturity),
                                            lg.role, lg.discount, 0.0);
    pf::MultiCurveBook::Position pos;
    pos.kind = pf::MultiCurveBook::Kind::Swap;
    pos.notional = notional;
    pos.float_coupons = ins.fwd.coupons;
    pos.fwd_curve = lg.role;
    pos.disc_curve = lg.discount;
    pos.fixed_coupons = ins.fixed.coupons;
    pos.fixed_curve = lg.discount;
    pos.fixed_rate = rate;
    f.book.positions.push_back(std::move(pos));
  }
}

Fixture& fx() {
  static Fixture f = [] {
    Fixture f;
    Settings::instance().evaluationDate() = f.today;
    // CHF has no QuantLib index class; every other RFR does. A generic OvernightIndex carries the same
    // data the DB does (name, fixing days, calendar, day count), so the helper is as honest as the others.
    const auto saron = ext::make_shared<OvernightIndex>("SARON", 0, CHFCurrency(), Switzerland(),
                                                        Actual360(), f.h[CHF_SARON]);
    f.eur3m = ext::make_shared<Euribor3M>(f.h[EUR_3M]);
    f.legs = {
        {USD_SOFR, USD, "USD", "USD-SOFR", ext::make_shared<Sofr>(f.h[USD_SOFR]), USD_SOFR},
        {EUR_ESTR, EUR, "EUR", "EUR-ESTR", ext::make_shared<Estr>(f.h[EUR_ESTR]), EUR_ESTR},
        {EUR_3M, EUR, "EUR", "EUR-EURIBOR-3M", nullptr, EUR_ESTR},
        {GBP_SONIA, GBP, "GBP", "GBP-SONIA", ext::make_shared<Sonia>(f.h[GBP_SONIA]), GBP_SONIA},
        {JPY_TONA, JPY, "JPY", "JPY-TONA", ext::make_shared<Tona>(f.h[JPY_TONA]), JPY_TONA},
        {CHF_SARON, CHF, "CHF", "CHF-SARON", saron, CHF_SARON},
        {CAD_CORRA, CAD, "CAD", "CAD-CORRA", ext::make_shared<Corra>(f.h[CAD_CORRA]), CAD_CORRA},
        {AUD_AONIA, AUD, "AUD", "AUD-AONIA", ext::make_shared<Aonia>(f.h[AUD_AONIA]), AUD_AONIA},
    };

    f.prob.curves.resize(NROLES);
    for (const Leg& lg : f.legs) {
      const std::vector<double> knots = lg.on_index ? add_ois(f, lg) : add_ibor(f, lg);
      f.prob.curves[static_cast<std::size_t>(lg.role)] =
          px::CurveStructure{.base = -1, .currency = lg.ccy, .regions = cv::flat_hermite({}, knots)};
    }

    // A plausible level per currency, tilted so no two knots share a forward.
    const double level[NROLES] = {0.030, 0.025, 0.0265, 0.042, 0.004, 0.011, 0.028, 0.038};
    f.x_true.setZero(f.prob.n_knots());
    f.x0.setZero(f.prob.n_knots());
    for (int c = 0; c < NROLES; ++c) {
      const int o = f.prob.offset(c);
      for (int i = 0; i < f.prob.curves[static_cast<std::size_t>(c)].n_interp_knots(); ++i) {
        f.x_true[o + i] = level[c] + 3e-4 * i;
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
    // QuantLib helpers, one block per curve, reading the SAME DB conventions.
    const std::size_t n = tenors().size();
    for (std::size_t c = 0; c < f.legs.size(); ++c) {
      const Leg& lg = f.legs[c];
      const b::SwapConv conv = b::swap_conv(lg.ccy_code, lg.index);
      for (std::size_t i = 0; i < n; ++i) {
        const Handle<Quote> q_i(f.ql_quotes[c * n + i]);
        const Period p = PeriodParser::parse(tenors()[i]);
        if (lg.on_index) {
          const Handle<YieldTermStructure> disc =
              lg.discount == lg.role ? Handle<YieldTermStructure>() : f.h[static_cast<std::size_t>(lg.discount)];
          f.helpers[lg.role].push_back(ext::make_shared<OISRateHelper>(conv.spot_lag, p, q_i, lg.on_index,
                                                                       disc, false, conv.pay_lag));
        } else {  // EUR-EURIBOR-3M-IRS from the DB: fixed 30E/360 annual, float ACT/360 quarterly, spot 2
          f.helpers[lg.role].push_back(ext::make_shared<SwapRateHelper>(
              q_i, p, TARGET(), Annual, ModifiedFollowing, Thirty360(Thirty360::European), f.eur3m,
              Handle<Quote>(), 0 * Days, f.h[EUR_ESTR]));
        }
      }
    }
    build_book(f);
    return f;
  }();
  return f;
}

using PWC = PiecewiseYieldCurve<Discount, LogLinear, IterativeBootstrap>;
using OurTS = swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>;

// Link QuantLib's handles to OUR calibrated curves at state x. THE BOOK ARMS MUST SHARE CURVES: the two
// curve sets agree on the market to 2e-13 at the pillars, but a book's coupons fall BETWEEN pillars, where
// QuantLib's log-linear discounts and our Hermite forwards legitimately differ (measured: up to 1.2e-4 of
// NPV). Timing the book off different curves would be timing two different books -- which is what gate 3
// caught. Returns fresh term-structure objects each call, so a QuantLib re-price cannot serve a cached NPV.
std::vector<ext::shared_ptr<OurTS>> link_ours(Fixture& f, const Eigen::VectorXd& x) {
  f.our_curves = px::build_bundle_curves<double>(
      f.prob.curves, [&](int c, int i) { return x[f.prob.offset(c) + i]; });
  std::vector<ext::shared_ptr<OurTS>> ts(NROLES);
  for (int c = 0; c < NROLES; ++c) {
    ts[static_cast<std::size_t>(c)] =
        ext::make_shared<OurTS>(f.today, f.dc, f.our_curves[static_cast<std::size_t>(c)].get());
    f.h[static_cast<std::size_t>(c)].linkTo(ts[static_cast<std::size_t>(c)]);
  }
  return ts;
}

// A QuantLib term structure seen as one of OUR curve handles, so our pricer can read QuantLib's answer.
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

// QuantLib: one bootstrap per curve, in dependency order (EUR 3M discounts on ESTR, so ESTR goes first).
//
// TWO CONFIGURATIONS, because "like for like" cuts both ways. LogLinear on discounts is QuantLib's fast,
// LOCAL choice -- and it makes the instantaneous forward piecewise CONSTANT, stepping up to 26 bp at a
// pillar, where ours is C1 (scratchpad/curve_shape.cpp measured it). MonotonicLogCubic is the comparable
// SHAPE to ours (max daily move 0.09 bp vs our 0.12; the two smooth curves differ by 6.8 bp of forward
// between pillars, against 18.8 bp for log-linear) but it is GLOBAL, so its bootstrap iterates. Both are
// timed; the GATED reference is the FASTER one, so the multiple we quote is the conservative one.
template <class Interp>
void ql_bootstrap_with(Fixture& f, Interp interp) {
  for (const Leg& lg : f.legs) {
    auto curve = ext::make_shared<PiecewiseYieldCurve<Discount, Interp, IterativeBootstrap>>(
        f.today, f.helpers[lg.role], f.dc, interp);
    curve->enableExtrapolation();
    f.h[static_cast<std::size_t>(lg.role)].linkTo(curve);
    benchmark::DoNotOptimize(curve->discount(30.0));  // forces the bootstrap
  }
}
void ql_bootstrap(Fixture& f) { ql_bootstrap_with(f, LogLinear()); }
void ql_bootstrap_smooth(Fixture& f) { ql_bootstrap_with(f, MonotonicLogCubic()); }

// ---- 1. COLD CALIBRATION -------------------------------------------------------------------------
void BM_G10_Cold_QuantLib(benchmark::State& state) {
  Fixture& f = fx();
  for (auto _ : state) ql_bootstrap(f);
}
BENCHMARK(BM_G10_Cold_QuantLib);

void BM_G10_Cold_Ours(benchmark::State& state) {
  Fixture& f = fx();
  for (auto _ : state) {
    const cal::CalibrationResult r = cal::calibrate(f.prob, f.x0);
    benchmark::DoNotOptimize(r.rms_residual);
  }
}
BENCHMARK(BM_G10_Cold_Ours);

// The same cold solve against QuantLib's SMOOTH configuration -- the like-for-like curve shape.
void BM_G10_Cold_QuantLib_Smooth(benchmark::State& state) {
  Fixture& f = fx();
  for (auto _ : state) ql_bootstrap_smooth(f);
}
BENCHMARK(BM_G10_Cold_QuantLib_Smooth);

// ---- 2. THE TICK ---------------------------------------------------------------------------------
// A cold solve happens once a day; a desk re-curves on every move. QuantLib must re-bootstrap every curve
// whose helper quotes changed; the engine re-solves off a frozen Jacobian. +-0.3 bp on EVERY quote,
// alternating, so neither arm can cache a no-op.
void BM_G10_Tick_QuantLib(benchmark::State& state) {
  Fixture& f = fx();
  bool up = false;
  for (auto _ : state) {
    up = !up;
    const double d = up ? 3e-5 : -3e-5;
    for (std::size_t i = 0; i < f.ql_quotes.size(); ++i) f.ql_quotes[i]->setValue(f.quotes[i] + d);
    ql_bootstrap(f);
  }
}
BENCHMARK(BM_G10_Tick_QuantLib);

void BM_G10_Tick_QuantLib_Smooth(benchmark::State& state) {
  Fixture& f = fx();
  bool up = false;
  for (auto _ : state) {
    up = !up;
    const double d = up ? 3e-5 : -3e-5;
    for (std::size_t i = 0; i < f.ql_quotes.size(); ++i) f.ql_quotes[i]->setValue(f.quotes[i] + d);
    ql_bootstrap_smooth(f);
  }
}
BENCHMARK(BM_G10_Tick_QuantLib_Smooth);

void BM_G10_Tick_Ours(benchmark::State& state) {
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
BENCHMARK(BM_G10_Tick_Ours);

// ---- 3. THE BOOK ---------------------------------------------------------------------------------
// kBookN swaps across the eight curves, repriced off the calibrated curves. QuantLib walks every swap's
// cashflows; we reprice the whole book as matrix algebra off DF = exp(-Wx). The QuantLib arm must be forced
// to actually re-price -- relinking the curve handles invalidates its cached NPVs.
void BM_G10_Book_QuantLib(benchmark::State& state) {
  Fixture& f = fx();
  const cal::CalibrationResult r = cal::calibrate(f.prob, f.x0);
  for (auto _ : state) {
    link_ours(f, r.x);  // fresh term structures off the SAME calibrated curves: no cached NPV survives
    double total = 0.0;
    for (const auto& s : f.ql_book) total += s->NPV();
    benchmark::DoNotOptimize(total);
  }
}
BENCHMARK(BM_G10_Book_QuantLib);

void BM_G10_Book_Ours(benchmark::State& state) {
  Fixture& f = fx();
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  sess.bind_portfolio(f.book);  // the W-cache twin, built once (as a desk binds its book once)
  for (auto _ : state) benchmark::DoNotOptimize(sess.reprice_bound().npv);
}
BENCHMARK(BM_G10_Book_Ours);

}  // namespace

int main(int argc, char** argv) {
  Fixture& f = fx();
  std::fprintf(stderr, "G10 desk: %d curves, %d currencies, %d calibration instruments, %d book swaps\n",
               NROLES, 7, f.prob.n_residuals(), static_cast<int>(f.ql_book.size()));

  // GATE 1: both arms must be solving the SAME problem. Comparing discount factors would be the wrong test
  // (the two curves pin the same market with different interpolation), so the check is that QuantLib's
  // bootstrapped curves REPRICE OUR instruments -- which is also what catches a convention mismatch.
  ql_bootstrap(f);
  std::vector<QlCurve> adapters(NROLES);
  for (int c = 0; c < NROLES; ++c)
    adapters[static_cast<std::size_t>(c)].ts = f.h[static_cast<std::size_t>(c)].currentLink().get();
  const auto curve_of = [&adapters](int i) -> const cal::CurveHandle<double>& {
    return adapters[static_cast<std::size_t>(i)];
  };
  double worst = 0.0;
  for (const auto& ins : f.prob.instruments)
    worst = std::max(worst, std::abs(cal::instrument_model_quote<double>(ins, curve_of) - ins.market));
  std::fprintf(stderr, "QuantLib's %d curves reprice our %d instruments to %.3e (worst absolute rate error)\n",
               NROLES, f.prob.n_residuals(), worst);
  if (worst > 1e-9) {
    std::fprintf(stderr, "ARMS DISAGREE on the curves - the timing below would compare two different problems\n");
    return 1;
  }

  // GATE 2: ours calibrates this market.
  const cal::CalibrationResult r = cal::calibrate(f.prob, f.x0);
  if (!r.converged || r.rms_residual > 1e-9) {
    std::fprintf(stderr, "ours did not converge: %s (rms %.3e)\n", r.status, r.rms_residual);
    return 1;
  }

  // GATE 3: both arms must price the SAME book. Ours off our calibrated curves, QuantLib off its own -- the
  // two curve sets agree on the market (gate 1), so a book NPV mismatch is a booking mismatch, not curves.
  api::BundleSession sess(f.prob);
  const cal::CalibrationResult& cr = sess.calibrate(f.x0);
  const double ours_npv = sess.price_portfolio(f.book).npv;
  link_ours(f, cr.x);  // QuantLib prices the same book off OUR curves: the kernels, not the interpolations
  double ql_npv = 0.0;
  for (const auto& s : f.ql_book) ql_npv += s->NPV();
  const double rel = std::abs(ours_npv - ql_npv) / std::max(1.0, std::abs(ql_npv));
  std::fprintf(stderr, "book NPV: ours %.6e  QuantLib %.6e  rel %.3e\n", ours_npv, ql_npv, rel);
  if (rel > 1e-9) {
    std::fprintf(stderr, "ARMS DISAGREE on the book - the timing below would compare two different books\n");
    // WHICH positions? Price each one alone, both ways, and report the worst offenders by curve.
    const char* role_name[NROLES] = {"USD_SOFR", "EUR_ESTR", "EUR_3M", "GBP_SONIA", "JPY_TONA", "CHF_SARON", "CAD_CORRA", "AUD_AONIA"};
    double worst_by_role[NROLES] = {0};
    int n_bad_by_role[NROLES] = {0};
    int shown = 0;
    for (std::size_t i = 0; i < f.book.positions.size(); ++i) {
      pf::MultiCurveBook one;
      one.positions.push_back(f.book.positions[i]);
      const double a = sess.price_portfolio(one).npv, q = f.ql_book[i]->NPV();
      const double r1 = std::abs(a - q) / std::max(1.0, std::abs(q));
      const int role = f.book.positions[i].fwd_curve;
      if (r1 > 1e-9) {
        ++n_bad_by_role[role];
        if (r1 > worst_by_role[role]) worst_by_role[role] = r1;
        if (shown++ < 5)
          std::fprintf(stderr, "  pos %4zu %-10s ours %.6e ql %.6e rel %.2e (fixed %.4f%%, %zu float cpns)\n", i,
                       role_name[role], a, q, r1, 100 * f.book.positions[i].fixed_rate,
                       f.book.positions[i].float_coupons.size());
      }
    }
    for (int c = 0; c < NROLES; ++c)
      if (n_bad_by_role[c])
        std::fprintf(stderr, "  %-10s %4d positions disagree, worst rel %.2e\n", role_name[c], n_bad_by_role[c], worst_by_role[c]);
    return 1;
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
