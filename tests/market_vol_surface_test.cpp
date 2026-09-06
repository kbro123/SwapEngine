// market::Market's NAMED VOL-SURFACE store — the vol analogue of the named-curve store, added so vega/scenario
// can share ONE vol source that lives inside the Market snapshot instead of re-specifying cells inline. A
// stored VolSurface is a curve-INDEPENDENT grid of cells (normal vol or a vol::SabrParams smile). This gate
// proves: (1) a surface added by name is retrievable, with a normal cell and a SABR cell round-tripping;
// (2) clone() carries an INDEPENDENT copy — mutating the fork's surface never touches the parent (the
// Scenario-fork invariant); (3) missing names throw and re-keying overwrites, exactly like the curve store.
// Includes ONLY market/ + vol/ headers (no api/, no boost) -> belongs in swaps_tests.
#include <gtest/gtest.h>

#include "swaps/market/market.hpp"
#include "swaps/vol/sabr.hpp"

namespace mkt = swaps::market;

namespace {
// A small two-cell surface: one flat-normal cell and one SABR cell.
mkt::VolSurface make_surface() {
  mkt::VolSurface s;
  s.add_cell(mkt::VolCell{"1Y", "5Y", /*has_sabr=*/false, /*normal_vol=*/0.0090, {}});
  s.add_cell(mkt::VolCell{"5Y", "10Y", /*has_sabr=*/true, /*normal_vol=*/0.0,
                          swaps::vol::SabrParams{0.0075, -0.30, 0.45}});
  return s;
}
}  // namespace

TEST(MarketVolSurface, AddAndRetrieveByName) {
  mkt::Market m;
  EXPECT_FALSE(m.has_vol_surface("USD-SWPN"));
  m.add_vol_surface("USD-SWPN", make_surface());

  ASSERT_TRUE(m.has_vol_surface("USD-SWPN"));
  const mkt::VolSurface& s = m.vol_surface("USD-SWPN");
  ASSERT_EQ(s.n_cells(), 2);

  // Normal cell round-trips.
  EXPECT_EQ(s.cells()[0].expiry, "1Y");
  EXPECT_EQ(s.cells()[0].tenor, "5Y");
  EXPECT_FALSE(s.cells()[0].has_sabr);
  EXPECT_NEAR(s.cells()[0].normal_vol, 0.0090, 1e-12);

  // SABR cell round-trips (the existing vol::SabrParams value).
  EXPECT_TRUE(s.cells()[1].has_sabr);
  EXPECT_NEAR(s.cells()[1].sabr.alpha, 0.0075, 1e-12);
  EXPECT_NEAR(s.cells()[1].sabr.rho, -0.30, 1e-12);
  EXPECT_NEAR(s.cells()[1].sabr.nu, 0.45, 1e-12);

  // Names enumerate like the curve store.
  const std::vector<std::string> names = m.vol_surface_names();
  ASSERT_EQ(names.size(), 1u);
  EXPECT_EQ(names[0], "USD-SWPN");
}

TEST(MarketVolSurface, CloneCarriesAnIndependentCopy) {
  mkt::Market base;
  base.add_vol_surface("USD-SWPN", make_surface());

  mkt::Market fork = base.clone();
  ASSERT_TRUE(fork.has_vol_surface("USD-SWPN"));

  // Mutate the FORK's surface in place: bump the normal cell and shift the SABR level, add a third cell.
  mkt::VolSurface& fs = fork.vol_surface("USD-SWPN");
  fs.cells()[0].normal_vol = 0.0200;
  fs.cells()[1].sabr.alpha = 0.0500;
  fs.add_cell(mkt::VolCell{"10Y", "10Y", false, 0.0111, {}});

  // The fork sees the mutation...
  EXPECT_EQ(fork.vol_surface("USD-SWPN").n_cells(), 3);
  EXPECT_NEAR(fork.vol_surface("USD-SWPN").cells()[0].normal_vol, 0.0200, 1e-12);
  EXPECT_NEAR(fork.vol_surface("USD-SWPN").cells()[1].sabr.alpha, 0.0500, 1e-12);

  // ...the PARENT is untouched (the Scenario-fork invariant: an independent deep copy).
  const mkt::VolSurface& bs = base.vol_surface("USD-SWPN");
  ASSERT_EQ(bs.n_cells(), 2);
  EXPECT_NEAR(bs.cells()[0].normal_vol, 0.0090, 1e-12);
  EXPECT_NEAR(bs.cells()[1].sabr.alpha, 0.0075, 1e-12);
}

TEST(MarketVolSurface, MissingNameThrowsAndReKeyingOverwrites) {
  mkt::Market m;
  EXPECT_THROW(m.vol_surface("nope"), std::runtime_error);

  m.add_vol_surface("S", make_surface());
  EXPECT_EQ(m.vol_surface("S").n_cells(), 2);

  // Re-keying the same name overwrites (insert_or_assign), like add_curve/add_quote.
  mkt::VolSurface one;
  one.add_cell(mkt::VolCell{"2Y", "2Y", false, 0.0055, {}});
  m.add_vol_surface("S", std::move(one));
  ASSERT_EQ(m.vol_surface("S").n_cells(), 1);
  EXPECT_NEAR(m.vol_surface("S").cells()[0].normal_vol, 0.0055, 1e-12);
  EXPECT_EQ(m.vol_surface_names().size(), 1u);
}
