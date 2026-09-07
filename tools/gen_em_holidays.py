"""Generate tabulated per-year EM holiday dates (lunar / Islamic / astronomical) into
conventions/conventions.json, replacing the fixed-Gregorian PLACEHOLDERS.

These holidays are not expressible as recurring Gregorian rules, so — exactly as QuantLib and vendor
calendars do for Asian markets — we tabulate the observed date PER YEAR (a "fixed" rule pinned with
from_year==to_year==Y). Dates are COMPUTED from authoritative calendars, not hand-typed:

  * China (Beijing UTC+8) lunisolar  -> LunarCalendar            (Spring Festival, Dragon Boat, Mid-Autumn)
  * Korea (KST UTC+9) lunisolar       -> korean_lunar_calendar    (Seollal, Chuseok, Buddha's Birthday)
  * Islamic (Umm al-Qura, official SA)-> hijridate                (Eid al-Fitr = 1 Shawwal, Eid al-Adha = 10 Dhu al-Hijjah)
  * Japan equinoxes                   -> astronomical formula (valid 1980-2099)
  * China Qingming (solar term)       -> 21st-century approximation

Window: 2025-2035 (the horizon requested). Islamic dates are moon-sighting in practice and vary ±1 day
and by country; the Umm al-Qura values are the standard estimate and are labelled as such. Beyond the
window these holidays simply don't fire until the table is regenerated (better than a wrong rule).

Build-time only — the calendar libraries are NOT a runtime dependency; the dates are baked into the JSON.
Run:  python3 tools/gen_em_holidays.py   (then re-run tools/gen_conventions_hpp.py)
"""
import datetime as dt
import json
import re
from pathlib import Path

from lunarcalendar import Converter, Lunar
from korean_lunar_calendar import KoreanLunarCalendar
from hijridate import Hijri

YEARS = range(2025, 2036)
JSON = Path(__file__).resolve().parent.parent / "conventions" / "conventions.json"


# ---- authoritative date computations ---------------------------------------------------------------
def cn(y, lm, ld):
    """China (Beijing) lunar month/day -> Gregorian date in Gregorian year `y`'s lunar year."""
    return Converter.Lunar2Solar(Lunar(y, lm, ld, isleap=False)).to_date()


_kcal = KoreanLunarCalendar()


def kr(y, lm, ld):
    """Korea (KST) lunar month/day -> Gregorian date."""
    _kcal.setLunarDate(y, lm, ld, False)
    return dt.date.fromisoformat(_kcal.SolarIsoFormat())


def jp_vernal(y):
    return dt.date(y, 3, int(20.8431 + 0.242194 * (y - 1980) - (y - 1980) // 4))


def jp_autumnal(y):
    return dt.date(y, 9, int(23.2488 + 0.242194 * (y - 1980) - (y - 1980) // 4))


def qingming(y):
    c = y % 100
    return dt.date(y, 4, int(c * 0.2422 + 4.81) - c // 4)


def span(d0, n):
    return [d0 + dt.timedelta(days=k) for k in range(n)]


def islamic_in_year(hmonth, hday, gy):
    """All Gregorian dates in year `gy` on which Hijri (hmonth, hday) falls (0, 1 or 2 as it drifts)."""
    out = []
    for hy in range(1446, 1462):
        g = Hijri(hy, hmonth, hday).to_gregorian()
        if g.year == gy:
            out.append(dt.date(g.year, g.month, g.day))
    return out


# ---- per-calendar festival tables ------------------------------------------------------------------
# Each entry: (label, list-of-dates-for-this-year). observance is "none" for exact-date lunar/Islamic
# holidays; JP equinoxes inherit the calendar default (sun_to_mon substitute) so we pass observance=None.
def cny_festivals(y):
    return [("Spring Festival", span(cn(y, 1, 1), 3)),
            ("Qingming / Tomb-Sweeping", [qingming(y)]),
            ("Dragon Boat Festival", [cn(y, 5, 5)]),
            ("Mid-Autumn Festival", [cn(y, 8, 15)])]


def krw_festivals(y):
    seollal = kr(y, 1, 1)
    chuseok = kr(y, 8, 15)
    return [("Seollal (Lunar New Year)", span(seollal - dt.timedelta(days=1), 3)),
            ("Buddha's Birthday", [kr(y, 4, 8)]),
            ("Chuseok", span(chuseok - dt.timedelta(days=1), 3))]


def islamic_festivals(y, fitr_days, adha_days, fitr_label, adha_label):
    out = []
    for d0 in islamic_in_year(10, 1, y):
        out.append((fitr_label, span(d0, fitr_days)))
    for d0 in islamic_in_year(12, 10, y):
        out.append((adha_label, span(d0, adha_days)))
    return out


FESTIVALS = {
    "CNY": (cny_festivals, "none"),
    "KRW": (krw_festivals, "none"),
    "SAR": (lambda y: islamic_festivals(y, 4, 4, "Eid al-Fitr", "Eid al-Adha"), "none"),
    "TRY": (lambda y: islamic_festivals(y, 3, 4, "Ramadan Feast (Eid al-Fitr)",
                                        "Sacrifice Feast (Eid al-Adha)"), "none"),
    "IDR": (lambda y: islamic_festivals(y, 2, 1, "Idul Fitri", "Idul Adha"), "none"),
    "JPY": (lambda y: [("Vernal Equinox Day", [jp_vernal(y)]),
                       ("Autumnal Equinox Day", [jp_autumnal(y)])], None),
}


def dated_rules(cid):
    """The per-year 'fixed' holiday rules for calendar `cid` over the window, as ordered dicts."""
    fest_fn, observance = FESTIVALS[cid]
    rules = []
    for y in YEARS:
        for label, dates in fest_fn(y):
            for d in dates:
                note = " (Islamic; Umm al-Qura estimate — moon-sighting)" if cid in ("SAR", "TRY", "IDR") else ""
                r = {"label": f"{label} {y}{note}", "rule": "fixed", "month": d.month, "day": d.day,
                     "from_year": y, "to_year": y}
                if observance is not None:
                    r["observance"] = observance
                rules.append(r)
    return rules


def main():
    raw = JSON.read_text()
    for cid, _ in FESTIVALS.items():
        block = re.search(rf'("{cid}": \{{.*?"holidays": \[\n)(.*?)(\n      \])', raw, re.S)
        if not block:
            raise SystemExit(f"could not locate holidays array for {cid}")
        # keep the existing NON-placeholder (fixed national) holidays, drop the placeholders
        kept = [ln for ln in block.group(2).split("\n")
                if ln.strip() and "placeholder" not in ln.lower()]
        kept = [ln.rstrip(",") for ln in kept]                      # normalise trailing commas
        new = ["        " + json.dumps(r, ensure_ascii=False) for r in dated_rules(cid)]
        inner = ",\n".join(kept + new)
        raw = raw[:block.start(2)] + inner + raw[block.end(2):]
    JSON.write_text(raw)
    # report
    d = json.loads(JSON.read_text())
    for cid in FESTIVALS:
        n = len(d["calendars"][cid]["holidays"])
        print(f"  {cid}: {n} holiday rules "
              f"({sum(1 for h in d['calendars'][cid]['holidays'] if h.get('to_year'))} dated)")
    print(f"wrote {JSON} (window {YEARS.start}-{YEARS.stop - 1})")


if __name__ == "__main__":
    main()
