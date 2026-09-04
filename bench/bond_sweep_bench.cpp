// Bond-universe baseline: our batched sweep vs QuantLib's per-bond loop (CLAUDE.md §3).
//
// Two paired metrics, both over the SAME universe of seasoned semiannual treasuries, both built from the
// SAME QuantLib (same compiler, same flags — perf-gate integrity, §3):
//
//   bond_sweep  YIELD space.  Ours: BondUniverse::yields_from_clean — ONE batched Newton across the whole
//               universe, Horner over the coupon polynomial (FMAs, one pow(v,w) per bond per iteration).
//               QuantLib: BondFunctions::yield bond-for-bond. Each of its NewtonSafe iterations walks the
//               Leg TWICE (CashFlows::npv for the residual, modifiedDuration for the derivative), and each
//               walk does, per cashflow, a virtual amount()/hasOccurred()/tradingExCoupon(), a
//               dynamic_pointer_cast<Coupon>, one-or-two DayCounter::yearFraction calls and a std::pow.
//
// TWO QuantLib yield baselines, because the default one is not a clean measurement of the object model:
//
//   BM_BondSweep_QuantLib       — the public API, BondFunctions::yield. This is what a user actually calls,
//                                 and it is the gate metric. But it is DOMINATED BY A SOLVER ARTIFACT:
//                                 CashFlows::IrrFinder::derivative returns modifiedDuration = −P'/P, while
//                                 its objective is npv−P(y), whose derivative is −P'(y) = P·modDur. In the
//                                 100-face basis QuantLib normalizes to, P ≈ 99, so NewtonSafe is handed a
//                                 derivative ~99× too small: every Newton step undershoots and the
//                                 safeguarded solver mostly bisects. MEASURED with a counting solver
//                                 substituted into the same CashFlows::yield<Solver> template: 34 npv walks
//                                 + 29 duration walks = 63 leg walks for ONE bond, ~11 µs each.
//   BM_BondSweep_QuantLibTuned  — the same QuantLib pricing (CashFlows::npv / CashFlows::duration over the
//                                 same Leg) driven by a CORRECTLY-SCALED Newton, converging in ~4
//                                 iterations. This isolates QuantLib's per-cashflow object-model cost from
//                                 its solver's stopping behaviour, and is the honest kernel-vs-kernel
//                                 number. Quote BOTH; never quote the first alone.
//
//   bond_book   CURVE space.  Ours: CompiledBondBook::dirty_prices(x) — DF = exp(-Wx), gathered flows,
//               sparse per-bond reduction. QuantLib: a per-bond Bond::dirtyPrice() off a
//               DiscountingBondEngine on the SAME discount factors (the handle is relinked each iteration
//               so QuantLib cannot serve cached NPVs).
//
// The benchmark ASSERTS agreement at fixture build (yields to 1e-11, dirty prices to 1e-9 relative) and
// aborts otherwise — a mis-set-up universe must never be able to quote a speedup.
//
// Ours is solved to a TIGHTER tolerance than QuantLib's (1e-13 on the dirty-price residual vs QuantLib's
// default 1e-10 accuracy on the yield step), so the comparison is conservative in QuantLib's favour: both
// sides run at the tolerance their library defaults to, and ours is the stricter of the two.
//
// Run on a quiesced machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/build/bond.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/portfolio/bond_universe.hpp"
#include "swaps/pricing/bond.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace bld = swaps::build;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;

namespace {

// The gate universe size, and the sizes swept by the (non-gate) scaling probe. kMax must be the largest.
constexpr int kGate = 5000;
constexpr int kMax = 10000;

QuantLib::Date qd(const bld::Date& d) {
  return QuantLib::Date(int(d.day()), Month(int(d.month())), d.year());
}

void die(const char* what, double got) {
  std::fprintf(stderr, "bond_sweep_bench: %s (%.3e) -- refusing to report a speedup\n", what, got);
  std::abort();
}

struct Fixture {
  // --- curve side: the reference market, calibrated, exposed to QuantLib as a term structure ---
  mutable RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  Eigen::VectorXd x = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
  swaps::curve::ModularCurve<double> curve =
      swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(prob.meeting_times, prob.back_times));
  using Ts = swaps::qlx::CurveTermStructure<swaps::curve::ModularCurve<double>>;
  mutable RelinkableHandle<YieldTermStructure> bh;  // the bond-pricing handle (relinked to force reprice)
  ext::shared_ptr<Ts> ts_a, ts_b;

  // --- the universe, in both representations ---
  std::vector<ext::shared_ptr<FixedRateBond>> ql;
  std::vector<DayCounter> dcs;
  std::vector<QuantLib::Date> settles;
  std::vector<double> clean;  // quoted clean prices, per 100 face
  std::vector<px::YieldBond> street;
  std::vector<px::Bond> curve_bonds;

  // Ours, per size: a BondUniverse (yield space) and a CompiledBondBook (curve space).
  std::vector<int> sizes{100, 1000, kGate, kMax};
  std::vector<std::unique_ptr<pf::BondUniverse>> universes;
  std::vector<std::unique_ptr<pf::CompiledBondBook>> books;
  std::vector<Eigen::VectorXd> clean_v;  // per-size clean-price vectors, per unit notional

  pf::BondUniverse& universe(int n) const {
    for (std::size_t i = 0; i < sizes.size(); ++i)
      if (sizes[i] == n) return *universes[i];
    std::abort();
  }
  pf::CompiledBondBook& book(int n) const {
    for (std::size_t i = 0; i < sizes.size(); ++i)
      if (sizes[i] == n) return *books[i];
    std::abort();
  }
  const Eigen::VectorXd& cleans(int n) const {
    for (std::size_t i = 0; i < sizes.size(); ++i)
      if (sizes[i] == n) return clean_v[i];
    std::abort();
  }

  Fixture() {
    curve.set_forwards(x);
    ts_a = ext::make_shared<Ts>(mk.today, mk.dc, &curve);
    ts_b = ext::make_shared<Ts>(mk.today, mk.dc, &curve);
    ts_a->enableExtrapolation();
    ts_b->enableExtrapolation();
    bh.linkTo(ts_a);

    // Value date == the reference market's evaluation date, so bond curve times (ACT/365F from the value
    // date) live on the SAME axis as the calibrated curve.
    const bld::Date value = bld::Date::ymd(mk.today.year(), unsigned(int(mk.today.month())),
                                           unsigned(mk.today.dayOfMonth()));

    auto engine = ext::make_shared<DiscountingBondEngine>(bh);
    for (int i = 0; i < kMax; ++i) {
      const unsigned m = (i % 2) ? 5u : 11u;
      const bld::Date issue = bld::Date::ymd(2024, m, 15);
      const bld::Date maturity = bld::Date::ymd(2028 + i % 25, m, 15);
      const double coupon = 0.01 + 0.0001 * double(i % 60);

      Schedule sched(qd(issue), qd(maturity), Period(Semiannual), NullCalendar(), Unadjusted, Unadjusted,
                     DateGeneration::Backward, false);
      DayCounter dc = ActualActual(ActualActual::ISMA, sched);
      auto b = ext::make_shared<FixedRateBond>(/*settlementDays=*/1, /*faceAmount=*/100.0, sched,
                                               std::vector<Rate>{coupon}, dc, Following, 100.0, qd(issue));
      b->setPricingEngine(engine);
      const QuantLib::Date s = b->settlementDate();
      const bld::Date settle = bld::Date::ymd(s.year(), unsigned(int(s.month())), unsigned(s.dayOfMonth()));

      const bld::BuiltBond ours = bld::us_treasury(value, settle, issue, maturity, coupon);
      ql.push_back(b);
      dcs.push_back(dc);
      settles.push_back(s);
      clean.push_back(90.0 + 0.002 * double(i % 5000));
      street.push_back(ours.yield);
      curve_bonds.push_back(ours.curve);
    }

    for (int n : sizes) {
      auto u = std::make_unique<pf::BondUniverse>();
      u->set(std::vector<px::YieldBond>(street.begin(), street.begin() + n));
      universes.push_back(std::move(u));
      books.push_back(std::make_unique<pf::CompiledBondBook>(
          prob.meeting_times, prob.back_times,
          std::vector<px::Bond>(curve_bonds.begin(), curve_bonds.begin() + n)));
      Eigen::VectorXd c(n);
      for (int i = 0; i < n; ++i) c[i] = clean[i] / 100.0;
      clean_v.push_back(c);
    }

    // ---- gate integrity: the two paths must agree before either timing means anything ----
    // NB the CORRECTNESS check drives QuantLib at a tightened accuracy (1e-14 on the yield step). At its
    // DEFAULT accuracy of 1e-10 the two differ by ~1e-10 on some bonds -- that is QuantLib's own stated
    // convergence tolerance, not a disagreement, and asserting against it would be asserting on its solver
    // stopping rule. The TIMED loop below uses the defaults both libraries ship with.
    const Eigen::VectorXd y_ours = universe(kGate).yields_from_clean(cleans(kGate));
    double dy = 0.0;
    for (int i = 0; i < kGate; ++i) {
      const double y_ql = BondFunctions::yield(*ql[i], clean[i], dcs[i], Compounded, Semiannual, settles[i],
                                               /*accuracy=*/1e-14, /*maxIterations=*/100);
      dy = std::max(dy, std::abs(y_ours[i] - y_ql));
    }
    if (!(dy < 1e-12)) die("batched yields disagree with a converged BondFunctions::yield", dy);

    const Eigen::VectorXd d_ours = book(kGate).dirty_prices(x);
    double dp = 0.0;
    for (int i = 0; i < kGate; ++i)
      dp = std::max(dp, std::abs(d_ours[i] * 100.0 - ql[i]->dirtyPrice()) / ql[i]->dirtyPrice());
    if (!(dp < 1e-9)) die("curve-space dirty prices disagree with DiscountingBondEngine", dp);
    max_dy_ = dy;
    max_dp_ = dp;
    y_ref_ = y_ours;
  }

  double max_dy_ = 0.0, max_dp_ = 0.0;
  mutable double max_dy_tuned_ = 0.0;
  Eigen::VectorXd y_ref_;  // our batched yields at kGate, for the tuned-baseline agreement check
};

double ql_yield_tuned(const Fixture& f, int i, double tol, int max_iter);

const Fixture& fx() {
  static const Fixture f;
  static const bool checked = [] {
    double d = 0.0;
    for (int i = 0; i < kGate; ++i)
      d = std::max(d, std::abs(f.y_ref_[i] - ql_yield_tuned(f, i, 1e-13, 100)));
    if (!(d < 1e-12)) die("tuned QuantLib baseline disagrees with the batched sweep", d);
    f.max_dy_tuned_ = d;
    return true;
  }();
  (void)checked;
  return f;
}

// ---- the two workloads, parameterized by universe size ----

void ql_yields(const Fixture& f, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i)
    acc += BondFunctions::yield(*f.ql[i], f.clean[i], f.dcs[i], Compounded, Semiannual, f.settles[i]);
  benchmark::DoNotOptimize(acc);
}

// QuantLib pricing, correctly-scaled Newton. The objective is P(y) − target on the DIRTY price, and its
// derivative is dP/dy = −modDur·P — the factor of P that IrrFinder::derivative omits. ~4 iterations.
double ql_yield_tuned(const Fixture& f, int i, double tol = 1e-11, int max_iter = 50) {
  const Leg& leg = f.ql[i]->cashflows();
  const QuantLib::Date s = f.settles[i];
  const double target = f.clean[i] + f.ql[i]->accruedAmount(s);
  double y = 0.05;
  for (int it = 0; it < max_iter; ++it) {
    const InterestRate ir(y, f.dcs[i], Compounded, Semiannual);
    const double P = CashFlows::npv(leg, ir, false, s, s);
    const double resid = P - target;
    if (std::abs(resid) < tol) break;
    const double md = CashFlows::duration(leg, ir, Duration::Modified, false, s, s);
    y -= resid / (-md * P);
  }
  return y;
}

void ql_yields_tuned(const Fixture& f, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) acc += ql_yield_tuned(f, i);
  benchmark::DoNotOptimize(acc);
}

void our_yields(const Fixture& f, int n) {
  const Eigen::VectorXd& y = f.universe(n).yields_from_clean(f.cleans(n));
  benchmark::DoNotOptimize(y.data());
  benchmark::ClobberMemory();
}

}  // namespace

// ================= gate metric: bond_sweep (yield space, B = kGate) =================

static void BM_BondSweep_QuantLib(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) ql_yields(f, kGate);
  state.counters["bonds"] = kGate;
  state.counters["max_abs_dy"] = f.max_dy_;
}
BENCHMARK(BM_BondSweep_QuantLib);

static void BM_BondSweep_Ours(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) our_yields(f, kGate);
  state.counters["bonds"] = kGate;
  state.counters["max_abs_dy"] = f.max_dy_;
}
BENCHMARK(BM_BondSweep_Ours);

// The harder QuantLib baseline (reported, NOT a gate metric): same QuantLib pricing, solver artifact
// removed. Its agreement with ours is asserted in the fixture.
static void BM_BondSweep_QuantLibTuned(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) ql_yields_tuned(f, kGate);
  state.counters["bonds"] = kGate;
  state.counters["max_abs_dy_tuned"] = f.max_dy_tuned_;
}
BENCHMARK(BM_BondSweep_QuantLibTuned);

// ================= gate metric: bond_book (curve space, B = kGate) =================

static void BM_BondBook_QuantLib(benchmark::State& state) {
  const auto& f = fx();
  bool flip = false;
  for (auto _ : state) {
    f.bh.linkTo(flip ? f.ts_a : f.ts_b);  // invalidate every bond's cached NPV
    flip = !flip;
    double acc = 0.0;
    for (int i = 0; i < kGate; ++i) acc += f.ql[i]->dirtyPrice();
    benchmark::DoNotOptimize(acc);
  }
  state.counters["bonds"] = kGate;
  state.counters["max_rel_dp"] = f.max_dp_;
}
BENCHMARK(BM_BondBook_QuantLib);

static void BM_BondBook_Ours(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    const Eigen::VectorXd& d = f.book(kGate).dirty_prices(f.x);
    benchmark::DoNotOptimize(d.data());
    benchmark::ClobberMemory();
  }
  state.counters["bonds"] = kGate;
  state.counters["max_rel_dp"] = f.max_dp_;
}
BENCHMARK(BM_BondBook_Ours);

// ================= scaling probe (NOT a gate metric, OPT-IN) =================
// The crossover matters: the batched Newton cannot arrest a converged bond, so it does relatively more
// work on a small universe. QuantLib's per-bond cost is flat, so this is really a probe of OUR fixed
// overhead. It is expensive (QuantLib alone is ~8 s per repetition across the four sizes), so it is
// registered only when SWAPS_BOND_SCALE is set in the environment — the perf gate runs the whole
// executable and should not pay for it:
//
//     SWAPS_BOND_SCALE=1 ./build/bench/bond_sweep_bench --benchmark_filter=Scale

const bool kScaleRegistered = [] {
  if (std::getenv("SWAPS_BOND_SCALE") == nullptr) return false;
  auto sizes = {100, 1000, kGate, kMax};
  auto* a = benchmark::RegisterBenchmark("BM_BondSweepScale_QuantLib", [](benchmark::State& st) {
    const auto& f = fx();
    for (auto _ : st) ql_yields(f, int(st.range(0)));
    st.counters["per_bond_ns"] = benchmark::Counter(
        double(st.range(0)), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
  });
  auto* b = benchmark::RegisterBenchmark("BM_BondSweepScale_Ours", [](benchmark::State& st) {
    const auto& f = fx();
    for (auto _ : st) our_yields(f, int(st.range(0)));
    st.counters["per_bond_ns"] = benchmark::Counter(
        double(st.range(0)), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
  });
  for (int n : sizes) { a->Arg(n); b->Arg(n); }
  return true;
}();

BENCHMARK_MAIN();
