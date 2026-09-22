// Desk-scale swaption PORTFOLIO reprice — thousands of random swaptions, ours vs QuantLib, swept over size.
//
// A market-making desk carries thousands of swaption positions across the standard expiry x tenor grid. This
// prices a random book of N of them (random standard expiry/tenor, random moneyness, random payer/receiver)
// and reports throughput (swaptions/sec), at N = 2k / 8k / 32k, for:
//   * BM_SwaptionPortfolio_Ours     — one native price_vol_cube over an N-cell VolCubeSpec. The per-cell
//                                     (forward, annuity) cache DEDUPS by (expiry, tenor): N positions across
//                                     the ~81-node grid cost 81 forward/annuity builds + N Bachelier evals.
//   * BM_SwaptionPortfolio_QuantLib — the standard per-instrument analytic path: each swaption forms its
//                                     forward/annuity from the QL term structure + bachelierBlackFormula.
// Both price off the SAME discounting (a QL DiscountCurve built from our calibrated fixture curve's DFs).
// Same closed form on both sides (QuantLib's bachelierBlackFormula IS our oracle) — a throughput comparison,
// not a correctness one (that is tests/vol_bachelier_ql_oracle.cpp). QuantLib-free math wins here come from
// batching + dedup, not the formula. Run quiesced.
#include <benchmark/benchmark.h>

#include <ql/quantlib.hpp>

#include <boost/json.hpp>

#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "swaps/api/bundle_api.hpp"

namespace api = swaps::api;
namespace QL = QuantLib;

namespace {

const std::string& bundle_json() {
  static const std::string js = [] {
    std::ifstream f(SOFR_BUNDLE_JSON);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }();
  return js;
}

const char* kExp[] = {"1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"};
const int kExpY[] = {0, 0, 0, 1, 2, 3, 5, 7, 10};       // integer pay-schedule years is 0 for sub-year expiries
const double kExpT[] = {1.0 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10};
const char* kTen[] = {"1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"};
const int kTenY[] = {1, 2, 3, 5, 7, 10, 15, 20, 30};
constexpr double kVol = 0.008;

api::BundleSession calibrated_session() {
  api::BundleSession sess(api::bundle_from_json(boost::json::parse(bundle_json())));
  api::RegSpec reg;
  reg.lambda = 0.02;
  reg.curves = {0};
  sess.calibrate(api::flat_x0(sess.problem()), reg);
  return sess;
}

// A deterministic random book of N swaptions: (expiry idx, tenor idx, moneyness bp, payer).
struct Trade {
  int e, t;
  double bp;
  bool payer;
};
std::vector<Trade> random_book(int n) {
  std::mt19937 rng(0xB00Cu);  // fixed seed: reproducible
  std::uniform_int_distribution<int> ei(0, 8), ti(0, 8);
  std::uniform_real_distribution<double> mo(-150.0, 150.0);
  std::bernoulli_distribution pay(0.5);
  std::vector<Trade> book(n);
  for (auto& tr : book) tr = {ei(rng), ti(rng), mo(rng), pay(rng)};
  return book;
}

api::VolCubeSpec ours_spec(const std::vector<Trade>& book) {
  api::VolCubeSpec s;
  s.value_date = "2026-07-08";
  s.index = "USD-SOFR";
  s.cells.reserve(book.size());
  for (const Trade& tr : book) {
    api::VolCubeCell c;
    c.expiry = kExp[tr.e];
    c.tenor = kTen[tr.t];
    c.normal_vol = kVol;
    c.moneyness_bp = {tr.bp};
    c.payer_set = true;
    c.payer = tr.payer;
    s.cells.push_back(std::move(c));
  }
  return s;
}

QL::ext::shared_ptr<QL::YieldTermStructure> ql_curve_from(const api::BundleSession& sess) {
  const QL::Date ref(8, QL::July, 2026);
  QL::Settings::instance().evaluationDate() = ref;
  std::vector<double> times;
  for (int i = 0; i <= 492; ++i) times.push_back(i / 12.0);
  const std::vector<api::CurveSample> cs = sess.sample(times);
  const std::vector<double>& df = cs.at(0).discount;
  std::vector<QL::Date> dates;
  std::vector<QL::DiscountFactor> dfs;
  for (std::size_t i = 0; i < times.size(); ++i) {
    dates.push_back(ref + static_cast<QL::Integer>(std::lround(times[i] * 365.0)));
    dfs.push_back(i == 0 ? 1.0 : df[i]);
  }
  auto curve = QL::ext::make_shared<QL::DiscountCurve>(dates, dfs, QL::Actual365Fixed());
  curve->enableExtrapolation();
  return curve;
}

// The QL-side per-trade schedule (dates precomputed once, like a desk caches schedules).
struct QLTrade {
  QL::Date start;
  std::vector<QL::Date> pay;
  int nyears;
  double expiry_T;
  double bp;
  QL::Option::Type type;
};
std::vector<QLTrade> ql_book(const std::vector<Trade>& book) {
  const QL::Date ref(8, QL::July, 2026);
  std::vector<QLTrade> out;
  out.reserve(book.size());
  for (const Trade& tr : book) {
    QLTrade q;
    const int ey = std::max(kExpY[tr.e], 0);
    q.start = ref + static_cast<QL::Integer>(std::lround(kExpT[tr.e] * 365.0));
    const int n = kTenY[tr.t];
    for (int k = 1; k <= n; ++k) q.pay.push_back(q.start + static_cast<QL::Integer>(std::lround(k * 365.0)));
    q.nyears = n;
    q.expiry_T = kExpT[tr.e];
    q.bp = tr.bp;
    q.type = tr.payer ? QL::Option::Call : QL::Option::Put;
    (void)ey;
    out.push_back(std::move(q));
  }
  return out;
}

}  // namespace

static void BM_SwaptionPortfolio_Ours(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  const api::BundleSession sess = calibrated_session();
  const api::VolCubeSpec spec = ours_spec(random_book(n));
  for (auto _ : state) {
    api::VolCube r = sess.price_vol_cube(spec);
    benchmark::DoNotOptimize(r.price.data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_SwaptionPortfolio_Ours)->Arg(2000)->Arg(8000)->Arg(32000);

static void BM_SwaptionPortfolio_QuantLib(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  const api::BundleSession sess = calibrated_session();
  const auto curve = ql_curve_from(sess);
  const std::vector<QLTrade> book = ql_book(random_book(n));
  for (auto _ : state) {
    double sink = 0.0;
    for (const QLTrade& q : book) {
      const double df_start = curve->discount(q.start);
      double annuity = 0.0;
      for (const QL::Date& p : q.pay) annuity += curve->discount(p);  // annual tau = 1
      const double fwd = (df_start - curve->discount(q.pay.back())) / annuity;
      const double K = fwd + q.bp / 1e4;
      sink += QL::bachelierBlackFormula(q.type, K, fwd, kVol * std::sqrt(q.expiry_T), annuity);
    }
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_SwaptionPortfolio_QuantLib)->Arg(2000)->Arg(8000)->Arg(32000);

BENCHMARK_MAIN();
