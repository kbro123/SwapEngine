// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// Gate for the batched swaption VOL CUBE (api::BundleSession::price_vol_cube_json). Loads a realistic SOFR
// bundle fixture, calibrates a session, and asserts the curve-AGNOSTIC identities that must hold whatever the
// exact forward is — the same robustness anchors as vol_swaption_test, but through the cube's JSON seam +
// batched SoA output, plus the strike-mode plumbing (absolute / moneyness / ATM) and the analytic Greeks.
#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "swaps/api/bundle_api.hpp"
#include "swaps/vol/normal.hpp"

namespace api = swaps::api;
namespace json = boost::json;

namespace {

std::string read_fixture() {
  std::ifstream f(SOFR_BUNDLE_JSON);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

api::BundleSession sofr_session() {
  api::BundleSession sess(api::bundle_from_json(json::parse(read_fixture())));
  api::RegSpec reg;
  reg.tension = true;
  reg.lambda = 0.02;
  reg.curves = {0};
  sess.calibrate(api::flat_x0(sess.problem()), reg);
  return sess;
}

constexpr const char* VD = "2026-07-08";

}  // namespace

TEST(VolCube, ForwardAnnuityPositiveAndShapeConsistent) {
  const api::BundleSession sess = sofr_session();
  const api::VolCube c = sess.price_vol_cube_json(
      std::string(R"({"value_date":")") + VD +
      R"(","index":"USD-SOFR","cells":[{"expiry":"1Y","tenor":"5Y","normal_vol":0.009,)"
      R"("moneyness_bp":[-50,0,50]}]})");
  ASSERT_EQ(c.n_cells, 1);
  ASSERT_EQ(c.n_points, 3);
  EXPECT_GT(c.cell_forward[0], 0.0);
  EXPECT_GT(c.cell_annuity[0], 0.0);
  EXPECT_GT(c.cell_expiry_years[0], 0.9);
  EXPECT_LT(c.cell_expiry_years[0], 1.1);  // ~1y expiry
  EXPECT_GE(c.price_us, 0.0);
  // moneyness echoes the request and strike == forward + offset.
  for (int i = 0; i < c.n_points; ++i) EXPECT_EQ(static_cast<int>(c.point_cell[i]), 0);
  EXPECT_NEAR(c.moneyness_bp[0], -50.0, 1e-9);
  EXPECT_NEAR(c.moneyness_bp[1], 0.0, 1e-9);
  EXPECT_NEAR(c.moneyness_bp[2], 50.0, 1e-9);
  for (int i = 0; i < c.n_points; ++i)
    EXPECT_NEAR(c.strike[i], c.cell_forward[0] + c.moneyness_bp[i] / 1e4, 1e-12);
}

TEST(VolCube, AtmClosedFormAndParity) {
  const api::BundleSession sess = sofr_session();
  // ATM payer and receiver at strike == forward: equal, and == A·σ√T·φ(0).
  const api::VolCube atm = sess.price_vol_cube_json(
      std::string(R"({"value_date":")") + VD +
      R"(","index":"USD-SOFR","cells":[)"
      R"({"expiry":"2Y","tenor":"5Y","normal_vol":0.0095,"atm":true,"payer":true},)"
      R"({"expiry":"2Y","tenor":"5Y","normal_vol":0.0095,"atm":true,"payer":false}]})");
  ASSERT_EQ(atm.n_points, 2);
  const double A = atm.cell_annuity[0], T = atm.cell_expiry_years[0];
  const double closed = A * 0.0095 * std::sqrt(T) * swaps::vol::kInvSqrt2Pi;
  EXPECT_NEAR(atm.price[0], closed, 1e-12);
  EXPECT_NEAR(atm.price[1], closed, 1e-12);  // payer == receiver ATM

  // Put-call parity at an off-ATM strike: payer - receiver == A·(F-K).
  const double F = atm.cell_forward[0], K = F - 0.004;
  const std::string body = std::string(R"({"value_date":")") + VD +
      R"(","index":"USD-SOFR","cells":[)"
      R"({"expiry":"2Y","tenor":"5Y","normal_vol":0.0095,"strikes":[)" + std::to_string(K) +
      R"(],"payer":true},)"
      R"({"expiry":"2Y","tenor":"5Y","normal_vol":0.0095,"strikes":[)" + std::to_string(K) +
      R"(],"payer":false}]})";
  const api::VolCube pk = sess.price_vol_cube_json(body);
  // Compare against the engine's OWN parsed strike (the JSON literal is truncated), so parity is exact.
  EXPECT_NEAR(pk.price[0] - pk.price[1], pk.cell_annuity[0] * (pk.cell_forward[0] - pk.strike[0]), 1e-12);
}

TEST(VolCube, VolSurfaceMatchesPriceVolCube) {
  const api::BundleSession sess = sofr_session();
  api::VolCubeSpec spec;
  spec.value_date = VD;
  spec.index = "USD-SOFR";
  {
    api::VolCubeCell c;
    c.expiry = "1Y"; c.tenor = "5Y"; c.has_sabr = true;
    c.sabr_alpha = 0.009; c.sabr_rho = -0.25; c.sabr_nu = 0.45;
    c.moneyness_bp = {-50, 0, 50};
    spec.cells.push_back(c);
  }
  {
    api::VolCubeCell c;
    c.expiry = "5Y"; c.tenor = "10Y"; c.normal_vol = 0.0095; c.atm = true;
    spec.cells.push_back(c);
  }
  {
    api::VolCubeCell c;
    c.expiry = "2Y"; c.tenor = "2Y"; c.normal_vol = 0.008;
    c.strikes = {0.03, 0.035}; c.payer_set = true; c.payer = false;
    spec.cells.push_back(c);
  }
  const api::VolCube ref = sess.price_vol_cube(spec);
  api::VolSurface surf(spec);
  const api::VolCube& got = surf.reprice(sess);
  ASSERT_EQ(got.n_points, ref.n_points);
  ASSERT_EQ(got.n_cells, ref.n_cells);
  for (int i = 0; i < ref.n_points; ++i) {
    EXPECT_NEAR(got.strike[i], ref.strike[i], 1e-14) << "pt " << i;
    EXPECT_NEAR(got.normal_vol[i], ref.normal_vol[i], 1e-14) << "pt " << i;
    EXPECT_NEAR(got.price[i], ref.price[i], 1e-14) << "pt " << i;
    EXPECT_NEAR(got.vega[i], ref.vega[i], 1e-14) << "pt " << i;
    EXPECT_NEAR(got.delta[i], ref.delta[i], 1e-14) << "pt " << i;
  }
  // A second reprice (warm, x unchanged) is bit-identical.
  const api::VolCube& got2 = surf.reprice(sess);
  for (int i = 0; i < ref.n_points; ++i) EXPECT_EQ(got2.price[i], got.price[i]);
}

TEST(VolCube, SabrNuZeroIsFlatAndGreeksSane) {
  const api::BundleSession sess = sofr_session();
  const api::VolCube c = sess.price_vol_cube_json(
      std::string(R"({"value_date":")") + VD +
      R"(","index":"USD-SOFR","cells":[{"expiry":"5Y","tenor":"10Y",)"
      R"("sabr":{"alpha":0.0088,"rho":-0.3,"nu":0.0},"moneyness_bp":[-100,0,100]}]})");
  ASSERT_EQ(c.n_points, 3);
  // nu = 0 -> flat smile at alpha, independent of strike.
  for (int i = 0; i < c.n_points; ++i) EXPECT_NEAR(c.normal_vol[i], 0.0088, 1e-12);
  // Greeks: vega > 0, ATM gamma > 0, ATM delta ~ +/-0.5 region (payer above fwd here since moneyness 0 -> payer).
  for (int i = 0; i < c.n_points; ++i) {
    EXPECT_GT(c.vega[i], 0.0);
    EXPECT_GT(c.gamma[i], 0.0);
  }
}
