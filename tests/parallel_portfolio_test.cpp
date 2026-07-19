// Coherent parallel portfolio reprice (Q2, the data-parallel pricing side). A large book split across
// threads, ALL pricing one pinned curve, must be BIT-IDENTICAL to a serial full-book CompiledPortfolio
// (a position's NPV is independent of the rest of the book) -- so the parallel cut is deterministic and
// internally coherent. Also ties it to the LiveCurveFeed: ONE snapshot -> the coherent point-in-time cut.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <vector>

#include "swaps/calibration/live_curve.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/parallel.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace pf = swaps::portfolio;
namespace cal = swaps::calibration;

namespace {
const std::vector<double> kMeeting{0.5};
const std::vector<double> kBack{1, 2, 3, 5, 7, 10, 15, 20, 30};

pf::Portfolio make_book(int P) {
  pf::Portfolio book;
  for (int p = 0; p < P; ++p) {
    pf::Portfolio::Position pos;
    const double T = 1.0 + (p % 30);  // 1..30y annual swaps, varied
    double prev = 0.0;
    for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
      swaps::pricing::FloatCoupon c;
      c.obs.sub_start = {prev};
      c.obs.sub_end = {u};
      c.obs.tau_index = u - prev;
      c.pay = u;
      c.tau_pay = u - prev;
      pos.float_coupons.push_back(c);
      pos.fixed_coupons.push_back({u, u - prev});
      prev = u;
    }
    pos.fixed_rate = 0.03 + 0.0001 * (p % 20);
    pos.notional = 1.0e6 * (1 + p % 5) * ((p % 2) ? 1.0 : -1.0);  // mixed payer/receiver
    book.positions.push_back(pos);
  }
  return book;
}
Eigen::VectorXd forwards() {
  Eigen::VectorXd x(kMeeting.size() + kBack.size());
  for (int i = 0; i < x.size(); ++i) x[i] = 0.030 + 0.0008 * i;
  return x;
}
}  // namespace

TEST(ParallelPortfolio, CoherentSplitIsBitIdenticalToSerial) {
  const pf::Portfolio book = make_book(2000);
  const Eigen::VectorXd x = forwards();
  const pf::CompiledPortfolio serial(kMeeting, kBack, book);
  const pf::ParallelPortfolio parallel(kMeeting, kBack, book, 8);

  ASSERT_EQ(parallel.n_slices(), 8);
  ASSERT_EQ(parallel.n_swaps(), 2000);
  const Eigen::VectorXd s = serial.npv(x);
  const Eigen::VectorXd& p = parallel.reprice(x);
  const double diff = (s - p).cwiseAbs().maxCoeff();
  std::cout << "  [parallel-portfolio] swaps=2000 slices=8 |parallel - serial|=" << diff
            << " total=" << parallel.total_npv(x) << "\n";
  EXPECT_EQ(diff, 0.0) << "the coherent parallel split must equal the serial full-book reprice, bit-for-bit";
  EXPECT_EQ(parallel.total_npv(x), serial.total_npv(x));
}

TEST(ParallelPortfolio, PricesOffOnePinnedSnapshotFromTheFeed) {
  // The coherent cut in practice: the calibrator publishes curves; the pricer snapshots ONCE (pins a
  // version) and reprices the whole book off that single x -> every worker prices the SAME curve.
  const pf::Portfolio book = make_book(500);
  const pf::CompiledPortfolio serial(kMeeting, kBack, book);
  const pf::ParallelPortfolio parallel(kMeeting, kBack, book, 4);

  cal::LiveCurveFeed feed(static_cast<int>(kMeeting.size() + kBack.size()));
  // Publish a few distinct curves (as a calibrator would); the pricer pins the latest.
  Eigen::VectorXd x = forwards();
  for (int v = 0; v < 5; ++v) { x.array() += 1e-4; feed.publish(x); }

  Eigen::VectorXd pinned(feed.n_knots());
  const std::uint64_t ver = feed.snapshot(pinned);  // ONE snapshot = the coherent version for this cut
  const Eigen::VectorXd& npvs = parallel.reprice(pinned);
  EXPECT_EQ((npvs - serial.npv(pinned)).cwiseAbs().maxCoeff(), 0.0)
      << "the whole book must price off exactly the pinned snapshot, coherently";
  EXPECT_EQ(ver, 5u);
}
