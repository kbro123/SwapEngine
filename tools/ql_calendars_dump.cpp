// ql_calendars_dump — dump QuantLib's holidays for the conventions-DB calendars that are TABULATED (lunar /
// Islamic / announced dates) so tools/calendars_from_ql.py can write them into conventions.json as per-year
// rules with a `source` field. QuantLib is the independent source (PRINCIPLES.md P8/P11); the engine never
// generates its own goldens. Prints JSON: {cal_id: {"coverage": Y, "closed": ["YYYY-MM-DD",...],
// "working_weekend": [...]}} for years [from, coverage].
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <ql/quantlib.hpp>

int main(int argc, char** argv) {
  using namespace QuantLib;
  const int from = argc > 1 ? std::atoi(argv[1]) : 2024;
  struct M { std::string id; Calendar cal; int coverage; };
  const std::vector<M> ms = {
      {"CNY", China(China::IB), 2024}, {"INR", India(), 2025}, {"IDR", Indonesia(Indonesia::IDX), 2025},
      {"KRW", SouthKorea(SouthKorea::Settlement), 2035}, {"SAR", SaudiArabia(SaudiArabia::Tadawul), 2029},
      {"TRY", Turkey(), 2034},
  };
  std::printf("{\n");
  for (std::size_t k = 0; k < ms.size(); ++k) {
    const auto& m = ms[k];
    std::printf("  \"%s\": {\"quantlib\": \"%s\", \"coverage\": %d, \"closed\": [", m.id.c_str(), m.cal.name().c_str(), m.coverage);
    bool first = true;
    std::vector<std::string> ww;
    for (Date d(1, January, from); d.year() <= m.coverage; d = d + 1) {
      const bool weekend = m.cal.isWeekend(d.weekday());
      const bool open = m.cal.isBusinessDay(d);
      char buf[16];
      std::snprintf(buf, sizeof buf, "%04d-%02d-%02d", int(d.year()), int(d.month()), int(d.dayOfMonth()));
      if (!weekend && !open) { std::printf("%s\"%s\"", first ? "" : ", ", buf); first = false; }
      if (weekend && open) ww.emplace_back(buf);
    }
    std::printf("], \"working_weekend\": [");
    for (std::size_t i = 0; i < ww.size(); ++i) std::printf("%s\"%s\"", i ? ", " : "", ww[i].c_str());
    std::printf("]}%s\n", k + 1 < ms.size() ? "," : "");
  }
  std::printf("}\n");
  return 0;
}
