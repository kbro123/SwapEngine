// QuantLib baseline smoke test: proves the QuantLib oracle/baseline links and runs.
// Builds a trivial flat-forward term structure and prints a discount factor.
// Compiled directly against Homebrew QuantLib (see tools/ql_smoke.sh); the shipped
// engine itself does NOT depend on QuantLib — this is oracle/baseline only.
#include <ql/quantlib.hpp>
#include <iostream>

using namespace QuantLib;

int main() {
  Calendar cal = TARGET();
  Date today(7, July, 2026);
  Settings::instance().evaluationDate() = today;
  DayCounter dc = Actual365Fixed();

  // Flat 3% continuously-compounded forward curve.
  Handle<YieldTermStructure> curve(
      ext::make_shared<FlatForward>(today, 0.03, dc, Continuous));

  Date in5y = cal.advance(today, Period(5, Years));
  Real df = curve->discount(in5y);
  Real t = dc.yearFraction(today, in5y);

  std::cout << "QuantLib " << QL_VERSION << " OK\n";
  std::cout << "t(5y)=" << t << "  DF=" << df
            << "  (expect ~exp(-0.03*t))\n";
  return 0;
}
