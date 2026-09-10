// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// Tenor is a typed wrapper over a tenor token: it parses "3M"/"1Y"/"2W"/... ONCE into (count, unit), and
// its resolve() delegates to the parity-tested schedule::resolve — so introducing the type changes no
// behaviour. These tests pin the parse, the approximate size, and (crucially) resolve() equality with the
// existing raw-string path for both normalized and special tokens.
#include <gtest/gtest.h>

#include "swaps/build/tenor.hpp"

namespace b = swaps::build;

TEST(Tenor, ParsesNormalizedTokens) {
  const b::Tenor t3m("3M");
  EXPECT_FALSE(t3m.is_special());
  EXPECT_EQ(t3m.count(), 3);
  EXPECT_EQ(t3m.unit(), b::Tenor::Unit::Month);

  const b::Tenor t6m("6M");
  EXPECT_EQ(t6m.count(), 6);
  EXPECT_EQ(t6m.unit(), b::Tenor::Unit::Month);

  const b::Tenor t1y("1Y");
  EXPECT_EQ(t1y.count(), 1);
  EXPECT_EQ(t1y.unit(), b::Tenor::Unit::Year);

  const b::Tenor t2w("2W");
  EXPECT_EQ(t2w.count(), 2);
  EXPECT_EQ(t2w.unit(), b::Tenor::Unit::Week);

  const b::Tenor t10d("10D");
  EXPECT_EQ(t10d.count(), 10);
  EXPECT_EQ(t10d.unit(), b::Tenor::Unit::Day);
}

TEST(Tenor, YearsIsSane) {
  EXPECT_NEAR(b::Tenor("3M").years(), 0.25, 1e-12);
  EXPECT_NEAR(b::Tenor("6M").years(), 0.5, 1e-12);
  EXPECT_NEAR(b::Tenor("1Y").years(), 1.0, 1e-12);
  EXPECT_NEAR(b::Tenor("2W").years(), 14.0 / 365.0, 1e-12);
  EXPECT_NEAR(b::Tenor("10D").years(), 10.0 / 365.0, 1e-12);
}

TEST(Tenor, RoundTripsToken) {
  EXPECT_EQ(b::Tenor("3M").to_string(), "3M");
  EXPECT_EQ(b::Tenor("1Y").to_string(), "1Y");
  EXPECT_EQ(b::Tenor(" 2W ").token(), "2W");  // trimmed
}

// The whole point: the typed wrapper must NOT change resolution. resolve() must land on the exact same
// serial as calling schedule::resolve on the raw token.
TEST(Tenor, ResolveMatchesFreeFunction) {
  const b::Date vd = b::Date::from_iso("2026-09-01");
  for (const std::string tok : {"3M", "1Y", "2W"}) {
    const b::Tenor t(tok);
    EXPECT_EQ(t.resolve(vd, "NONE", "Following", 0).serial(), b::resolve(tok, vd, "NONE", "Following", 0).serial()) << "tok=" << tok;
    // and unrolled behaviour matches too
    EXPECT_EQ(t.resolve(vd, "NONE", "Unadjusted", 0).serial(), b::resolve(tok, vd, "NONE", "Unadjusted", 0).serial()) << "tok=" << tok;
  }
}

// A special token (IMM code) is stored verbatim and round-trips through resolve() identically.
TEST(Tenor, SpecialTokenDelegatesIdentically) {
  const b::Date vd = b::Date::from_iso("2026-09-01");
  const b::Tenor imm("U27");
  EXPECT_TRUE(imm.is_special());
  EXPECT_EQ(imm.token(), "U27");
  EXPECT_EQ(imm.to_string(), "U27");
  EXPECT_EQ(imm.resolve(vd, "NONE", "Following", 0).serial(), b::resolve("U27", vd, "NONE", "Following", 0).serial());

  const b::Tenor on("ON");
  EXPECT_TRUE(on.is_special());
  EXPECT_EQ(on.resolve(vd, "NONE", "Following", 0).serial(), b::resolve("ON", vd, "NONE", "Following", 0).serial());
}

TEST(Tenor, EqualityAndOrderingNormalized) {
  EXPECT_EQ(b::Tenor("3M"), b::Tenor("3M"));
  EXPECT_NE(b::Tenor("3M"), b::Tenor("6M"));
  EXPECT_LT(b::Tenor("3M"), b::Tenor("6M"));      // same unit, smaller count
  EXPECT_LT(b::Tenor("10D"), b::Tenor("1Y"));     // Day unit sorts before Year unit
}
