// Gate for the VEGA LADDER (vol/vega_ladder.hpp): a swaption book's aggregate sensitivity to the vol surface's
// PARAMETERS, the vol analogue of the rates delta ladder. QuantLib-free: hand-build a flat-forward curve's
// discount factors, form each cell's forward/annuity via vol/swaption.hpp, then check
//   (a) a single ATM swaption's ladder vega == Bachelier vega · notional, exactly;
//   (b) the analytic ladder matches a central finite-difference of the book value wrt each vol parameter
//       (normal_vol, and SABR alpha/rho/nu), to tight relative tolerance;
//   (c) a two-swaption book on two different cells scatters vega to the right buckets (an empty cell = 0).
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/vol/swaption.hpp"     // forward_swap — build each cell's forward/annuity from hand-built DFs
#include "swaps/vol/vega_ladder.hpp"

namespace v = swaps::vol;

namespace {
// Flat forward -> DF(t) = exp(-f t). A cell = 1y-expiry into an n-year annual swap: start t=1, pays t=2..n+1.
constexpr double kFwd = 0.03;
double df(double t) { return std::exp(-kFwd * t); }

v::VegaCell make_cell(int n_years, double expiry = 1.0) {
  std::vector<double> df_pay, tau(n_years, 1.0);
  for (int i = 1; i <= n_years; ++i) df_pay.push_back(df(expiry + i));
  const v::ForwardSwap fs = v::forward_swap(df(expiry), df(expiry + n_years), df_pay, tau);
  v::VegaCell c;
  c.forward = fs.rate;
  c.annuity = fs.annuity;
  c.expiry_years = expiry;
  return c;
}
}  // namespace

// (a) A single ATM swaption's bucket equals the analytic Bachelier vega times its notional, exactly.
TEST(VegaLadder, SingleAtmSwaptionEqualsBachelierVega) {
  v::VegaCell c = make_cell(5);
  c.has_sabr = false;
  c.normal_vol = 0.0090;
  const double notional = 250.0;

  std::vector<v::VegaCell> cells{c};
  std::vector<v::VegaSwaption> book{v::VegaSwaption{/*cell*/ 0, /*strike*/ c.forward, /*payer*/ true, notional}};

  const v::VegaLadder L = v::vega_ladder(cells, book);
  const double expected =
      notional * v::bachelier_vega<double>(c.forward, c.forward, c.normal_vol, c.expiry_years, c.annuity);
  ASSERT_EQ(L.n_cells, 1);
  EXPECT_NEAR(L.d_normal_vol[0], expected, 1e-9);
  // SABR buckets untouched for a normal-vol cell.
  EXPECT_EQ(L.d_alpha[0], 0.0);
  EXPECT_EQ(L.d_rho[0], 0.0);
  EXPECT_EQ(L.d_nu[0], 0.0);
}

// (b1) Normal-vol ladder vs a central finite-difference of the book value wrt normal_vol.
TEST(VegaLadder, NormalVolMatchesFiniteDifference) {
  v::VegaCell c = make_cell(5);
  c.has_sabr = false;
  c.normal_vol = 0.0090;
  std::vector<v::VegaCell> cells{c};
  // A small OTM payer + receiver book, mixed notionals -> a non-trivial aggregate.
  std::vector<v::VegaSwaption> book{
      v::VegaSwaption{0, c.forward + 0.0025, true, 100.0},
      v::VegaSwaption{0, c.forward - 0.0040, false, -60.0},
      v::VegaSwaption{0, c.forward, true, 30.0}};

  const v::VegaLadder L = v::vega_ladder(cells, book);

  const double h = 1e-5;
  auto reprice = [&](double nv) {
    std::vector<v::VegaCell> cc = cells;
    cc[0].normal_vol = nv;
    return v::swaption_book_value(cc, book);
  };
  const double fd = (reprice(c.normal_vol + h) - reprice(c.normal_vol - h)) / (2.0 * h);
  EXPECT_NEAR(L.d_normal_vol[0], fd, std::abs(fd) * 1e-5 + 1e-9);
}

// (b2) SABR ladder vs a central finite-difference of the book value wrt alpha, rho, nu.
TEST(VegaLadder, SabrParamsMatchFiniteDifference) {
  v::VegaCell c = make_cell(5);
  c.has_sabr = true;
  c.sabr_alpha = 0.0088;
  c.sabr_rho = -0.30;
  c.sabr_nu = 0.45;
  std::vector<v::VegaCell> cells{c};
  std::vector<v::VegaSwaption> book{
      v::VegaSwaption{0, c.forward + 0.0050, true, 120.0},
      v::VegaSwaption{0, c.forward - 0.0050, false, 80.0},
      v::VegaSwaption{0, c.forward, true, 40.0}};

  const v::VegaLadder L = v::vega_ladder(cells, book);

  const double h = 1e-5;
  auto reprice = [&](double da, double dr, double dn) {
    std::vector<v::VegaCell> cc = cells;
    cc[0].sabr_alpha += da;
    cc[0].sabr_rho += dr;
    cc[0].sabr_nu += dn;
    return v::swaption_book_value(cc, book);
  };
  const double fd_alpha = (reprice(h, 0, 0) - reprice(-h, 0, 0)) / (2.0 * h);
  const double fd_rho = (reprice(0, h, 0) - reprice(0, -h, 0)) / (2.0 * h);
  const double fd_nu = (reprice(0, 0, h) - reprice(0, 0, -h)) / (2.0 * h);

  EXPECT_NEAR(L.d_alpha[0], fd_alpha, std::abs(fd_alpha) * 1e-5 + 1e-9);
  EXPECT_NEAR(L.d_rho[0], fd_rho, std::abs(fd_rho) * 1e-5 + 1e-9);
  EXPECT_NEAR(L.d_nu[0], fd_nu, std::abs(fd_nu) * 1e-5 + 1e-9);
  // Normal-vol bucket untouched for a SABR cell.
  EXPECT_EQ(L.d_normal_vol[0], 0.0);
}

// (c) A two-swaption book on two different cells scatters vega to the right buckets; a cell with no swaption
// stays exactly 0.
TEST(VegaLadder, ScattersToRightBucketsAndEmptyCellIsZero) {
  v::VegaCell c0 = make_cell(5);
  c0.has_sabr = false;
  c0.normal_vol = 0.0090;
  v::VegaCell c1 = make_cell(10);
  c1.has_sabr = false;
  c1.normal_vol = 0.0075;
  v::VegaCell c2 = make_cell(2);  // no swaption references this cell
  c2.has_sabr = false;
  c2.normal_vol = 0.0100;
  std::vector<v::VegaCell> cells{c0, c1, c2};

  std::vector<v::VegaSwaption> book{
      v::VegaSwaption{0, c0.forward, true, 100.0},
      v::VegaSwaption{1, c1.forward, true, 50.0}};

  const v::VegaLadder L = v::vega_ladder(cells, book);
  ASSERT_EQ(L.n_cells, 3);

  const double e0 =
      100.0 * v::bachelier_vega<double>(c0.forward, c0.forward, c0.normal_vol, c0.expiry_years, c0.annuity);
  const double e1 =
      50.0 * v::bachelier_vega<double>(c1.forward, c1.forward, c1.normal_vol, c1.expiry_years, c1.annuity);
  EXPECT_NEAR(L.d_normal_vol[0], e0, 1e-9);
  EXPECT_NEAR(L.d_normal_vol[1], e1, 1e-9);
  EXPECT_EQ(L.d_normal_vol[2], 0.0);  // empty cell -> exactly zero vega
  EXPECT_GT(L.d_normal_vol[0], 0.0);
  EXPECT_GT(L.d_normal_vol[1], 0.0);
}
