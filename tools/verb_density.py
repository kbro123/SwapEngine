#!/usr/bin/env python3
"""verb_density.py — HOW MUCH BEHAVIOUR DOES THE VERB LAYER STILL CARRY?  (E7 stage 1; PRINCIPLES.md P14)

P14: a verb body is parse -> ONE call into include/swaps/** -> emit. Behaviour belongs in the library, where
the unit tests and the QuantLib oracles are; whatever is left in api/ sits outside every oracle's closure
(tools/oracle_coverage.py reports the api layer at 0 of 20 headers). This gate measures what is left, per
file, and locks it: a count may only go DOWN.

What is counted -- from clang's typed AST, not a regex, so a pointer `*` is not a multiplication, a `for` in
a comment is not a loop, and `jd(o, "dt_years", 0.0)` is a default however it is spelled:

  lookups   a read of the conventions DB: a conventions_db.hpp lookup or require_* (free, or a Registry
            member), a build/conventions.hpp resolver, period_years, or a method on a build/ref_data.hpp
            value object. A verb that resolves a convention has made a market decision.
  defaults  a ternary (`?:`), a scalar JSON accessor with a fallback (jd/ji/jb/js, get_d/get_i/get_b/get_s)
            or optional::value_or. Each is a value the verb invents for an absent field -- usually a second
            copy of a default the library struct already declares.
  loops     for / range-for / while / do, and the std:: algorithms that iterate. A loop in a verb is decode
            (belongs in a declared codec) or result-building (belongs in the library).
  compute   floating-point arithmetic (+ - * / and negation on double, Eigen operators), <cmath> calls and
            linear algebra (Eigen decompositions, dot/norm/solve/...). Integer/index arithmetic is NOT counted.
  branches  if-statements: validation, dispatch and shape checks.

Codec files (json_util.hpp, codec.hpp, codec.cpp) exist to decode, so their loops and branches are reported but not
locked; their lookups, defaults and compute are.

NOT measured: orchestration (a verb calling two library functions in sequence with nothing between them).
Clang's JSON AST does not name a callee's namespace, and a name-based guess would be a gate that can be
wrong in both directions. E7 stage 7 (driver parity) and review cover it.

TRUST. Every counted file is cross-checked against its own token stream (comments, literals and
preprocessor lines removed): `for`+`while` tokens must equal the loop statements found, `if` tokens the
if-statements, `?` tokens the ternaries; and every file in scope must be reached by some translation unit.
A mismatch -- an `#if 0` block, a macro that hides a loop, code outside namespace swaps::api -- FAILS rather
than reporting a number that might be wrong.

  (no flag)        print the table
  --by-function    also break each file down per function
  --check          FAIL if any locked count exceeds tests/verb_density.lock (a file not in the lock locks at 0)
  --update         rewrite the lock; refuses to RAISE a count unless --allow-increase (say why in the commit)
  --selftest       prove the gate can fail: inject one of each construct into a scratch copy of api/pnl.cpp and
                   assert every metric rises by exactly the injected amount and --check trips; then hide a loop
                   in `#if 0` and assert the cross-check refuses
"""
import argparse
import collections
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "tests", "verb_density.lock")
METRICS = ("lookups", "defaults", "loops", "compute", "branches")

CODEC = {"include/swaps/api/json_util.hpp", "include/swaps/api/codec.hpp", "api/codec.cpp"}
CODEC_UNLOCKED = {"loops", "branches"}
NOT_VERBS = {"api/swaps_capi.cpp", "api/swaps_api_cli.cpp"}  # the C-ABI transport shim and the CLI driver
PARKED = {"bundle_api", "compile"}  # the session and the spec compiler: staged separately (E7)
DEFERRED = {"credit", "inflation", "options", "vega", "fx_option", "ndf", "exposure", "var"}  # TASKS-DEFERRED-DOMAINS
TIERS = ("core", "codec", "parked", "deferred")

LOOP_STMTS = {"ForStmt", "CXXForRangeStmt", "WhileStmt", "DoStmt"}
TERNARIES = {"ConditionalOperator", "BinaryConditionalOperator"}
FUNC_KINDS = {"FunctionDecl", "CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl", "CXXConversionDecl"}
ARITH_OPS = {"+", "-", "*", "/", "+=", "-=", "*=", "/="}
FLOAT = re.compile(r"^(const\s+)?(volatile\s+)?(long\s+)?(double|float)(\s*&+)?$")
MATH = {"sqrt", "cbrt", "exp", "expm1", "log", "log1p", "log10", "log2", "pow", "abs", "fabs", "floor", "ceil",
        "round", "lround", "llround", "trunc", "erf", "erfc", "hypot", "isfinite", "isnan", "isinf"}
MATH_IF_FLOAT = {"max", "min", "clamp", "fmax", "fmin"}
EIGEN_METHODS = {"dot", "transpose", "adjoint", "inverse", "norm", "squaredNorm", "sum", "prod", "mean", "solve",
                 "singularValues", "cwiseProduct", "cwiseQuotient", "cwiseAbs", "determinant", "maxCoeff",
                 "minCoeff", "lpNorm"}
EIGEN_DECOMP = re.compile(r"\bEigen::(JacobiSVD|BDCSVD|LLT|LDLT|HouseholderQR|ColPivHouseholderQR|"
                          r"FullPivHouseholderQR|FullPivLU|PartialPivLU|SelfAdjointEigenSolver|EigenSolver|"
                          r"CompleteOrthogonalDecomposition)\b")
ALGOS = {"transform", "for_each", "accumulate", "reduce", "inner_product", "partial_sum", "adjacent_difference",
         "sort", "stable_sort", "copy_if", "remove_if", "count_if", "find_if", "any_of", "all_of", "none_of",
         "max_element", "min_element", "minmax_element", "iota", "generate", "unique", "partition",
         "nth_element", "lower_bound", "upper_bound", "binary_search", "equal_range"}
FALLBACK = {"jd", "ji", "jb", "js", "get_d", "get_i", "get_b", "get_s"}


# ---------------------------------------------------------------------------------------------------------
# vocabulary: what counts as a conventions lookup is READ from the headers that define the lookups, so a new
# require_* is covered the day it is added.
def vocabulary():
    rd = lambda p: open(os.path.join(ROOT, p), encoding="utf-8").read()
    db = set(re.findall(r"\b(\w+)\(std::string_view \w+\)\s*(?:const\s*)?\{", rd("include/swaps/conventions_db.hpp")))
    db -= {"intern"}
    build = set(re.findall(r"^inline [^(\n]*?\b(\w+)\(", rd("include/swaps/build/conventions.hpp"), re.M))
    build -= {"sv_str", "upper"}  # string helpers, not resolvers
    ref_types = set(re.findall(r"^struct (\w+)", rd("include/swaps/build/ref_data.hpp"), re.M))
    for must, got in (({"product", "index", "require_product", "require_bond"}, db),
                      ({"swap_conv", "conv_from_product"}, build), ({"Calendar", "Index"}, ref_types)):
        if not must <= got:
            raise SystemExit(f"verb_density: lookup vocabulary extraction broke (missing {sorted(must - got)}); "
                             "a header changed shape -- fix vocabulary() before trusting any count")
    return {"db": db, "free": db | build | {"period_years"},
            "ref": re.compile(r"\bswaps::build::(" + "|".join(sorted(ref_types)) + r")\b")}


# ---------------------------------------------------------------------------------------------------------
def scope_files():
    files = sorted("api/" + f for f in os.listdir(os.path.join(ROOT, "api")) if f.endswith(".cpp"))
    files = [f for f in files if f not in NOT_VERBS]
    files += sorted("include/swaps/api/" + f for f in os.listdir(os.path.join(ROOT, "include/swaps/api"))
                    if f.endswith(".hpp"))
    return files


def tier(rel):
    stem = os.path.splitext(os.path.basename(rel))[0]
    if rel in CODEC:
        return "codec"
    if stem in PARKED:
        return "parked"
    if stem in DEFERRED:
        return "deferred"
    return "core"


def compiler_for(first):
    for cand in (first, shutil.which("clang++")):
        if not cand:
            continue
        try:
            if "clang" in subprocess.run([cand, "--version"], capture_output=True, text=True).stdout:
                return cand
        except OSError:
            pass
    raise SystemExit("verb_density: FAIL — needs clang (for -ast-dump=json); neither the build's compiler nor "
                     "clang++ on PATH is clang. This gate does not skip.")


def ast_command(entry, source):
    args = entry["arguments"] if "arguments" in entry else shlex.split(entry["command"])
    out, skip = [compiler_for(args[0])], False
    for a in args[1:]:
        if skip:
            skip = False
            continue
        if a in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
            continue
        if a in ("-c", "-MD", "-MMD") or os.path.realpath(os.path.join(entry["directory"], a)) == \
                os.path.realpath(entry["file"]):
            continue
        out.append(a)
    return out + ["-fsyntax-only", "-Xclang", "-ast-dump=json", "-Xclang", "-ast-dump-filter=swaps::api", source]


# ---------------------------------------------------------------------------------------------------------
def typ(x):
    t = x.get("type") or {}
    return t.get("desugaredQualType") or t.get("qualType") or ""


def peel(c):
    while c.get("kind") in ("ImplicitCastExpr", "ParenExpr", "MaterializeTemporaryExpr") and c.get("inner"):
        c = c["inner"][0]
    return c


def callee(n):
    """(name, base type or None) of a call; base type is the object's type for a member call."""
    inner = n.get("inner") or []
    if not inner:
        return None, None
    c = peel(inner[0])
    if n["kind"] == "CXXMemberCallExpr":
        if c.get("kind") == "MemberExpr":
            return c.get("name"), typ((c.get("inner") or [{}])[0])
        return None, None
    if c.get("kind") == "DeclRefExpr":
        return (c.get("referencedDecl") or {}).get("name"), None
    if c.get("kind") == "UnresolvedLookupExpr":
        return c.get("name"), None
    return None, None


def classify(n, V):
    k = n.get("kind")
    if k in LOOP_STMTS:
        return "loops"
    if k == "IfStmt":
        return "branches"
    if k in TERNARIES:
        return "defaults"
    if k in ("BinaryOperator", "CompoundAssignOperator"):
        return "compute" if n.get("opcode") in ARITH_OPS and FLOAT.match(typ(n)) else None
    if k == "UnaryOperator":
        return "compute" if n.get("opcode") == "-" and FLOAT.match(typ(n)) else None
    if k in ("CXXConstructExpr", "CXXTemporaryObjectExpr"):
        return "compute" if EIGEN_DECOMP.search(typ(n)) else None
    if k == "CXXOperatorCallExpr":
        name, _ = callee(n)
        op = (name or "")[len("operator"):] if (name or "").startswith("operator") else None
        operands = [typ(x) for x in (n.get("inner") or [])[1:]]
        if op == "*" and len(operands) == 1:
            return None  # a DEREFERENCE (std::optional, an iterator), never a multiplication
        if op in ARITH_OPS:
            if FLOAT.match(typ(n)) or any("Eigen::" in t for t in [typ(n)] + operands):
                return "compute"
        return None
    if k in ("CallExpr", "CXXMemberCallExpr"):
        name, base = callee(n)
        if not name:
            return None
        if base is not None:  # member call
            if name in V["db"] and "swaps::conventions::Registry" in base:
                return "lookups"
            if V["ref"].search(base):
                return "lookups"
            if name == "value_or":
                return "defaults"
            if name in EIGEN_METHODS and "Eigen::" in base:
                return "compute"
            return None
        if name in V["free"]:
            return "lookups"
        if name in FALLBACK:
            return "defaults"
        if name in ALGOS:
            return "loops"
        if name in MATH or (name in MATH_IF_FLOAT and FLOAT.match(typ(n))):
            return "compute"
    return None


class Walker:
    """Walks clang's JSON AST in dump order. The dumper writes a node's `file` only when it differs from the
    last location it printed, so the current file is a running state that must see every location object in
    order -- including the ones nested in spellingLoc/expansionLoc, and never the ones under includedFrom."""

    def __init__(self, scope, V):
        self.scope, self.V = scope, V  # realpath -> repo-relative path
        self.file, self.real = None, {}
        self.seen = collections.defaultdict(set)
        self.tally = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
        self.stmts = collections.defaultdict(collections.Counter)  # for the token cross-check
        self.reached = set()

    def rel(self, f):
        if f not in self.real:
            self.real[f] = self.scope.get(os.path.realpath(f)) if f else None
        return self.real[f]

    def bare(self, d):
        if d and "file" in d:
            self.file = d["file"]
        return self.file, (d or {}).get("offset")

    def loc(self, d):
        if not isinstance(d, dict) or not d:
            return self.file, self.file, None
        if "spellingLoc" in d or "expansionLoc" in d:
            sp, _ = self.bare(d.get("spellingLoc"))
            ex, off = self.bare(d.get("expansionLoc"))
            return sp, ex, off
        f, off = self.bare(d)
        return f, f, off

    def node(self, n, fn):
        if "loc" in n:
            self.loc(n["loc"])
        sp = ex = self.file
        b = e = None
        r = n.get("range")
        if isinstance(r, dict):
            sp, ex, b = self.loc(r.get("begin"))
            _, _, e = self.loc(r.get("end"))
        kind = n.get("kind")
        if kind in FUNC_KINDS and n.get("name") != "operator()" and \
                any(c.get("kind") == "CompoundStmt" for c in n.get("inner") or []):
            fn = n["name"]
        rel = self.rel(ex)
        if rel and sp == ex:  # a node spelled in a macro defined elsewhere is not this file's code
            self.reached.add(rel)
            key = (kind, b, e)
            if key not in self.seen[rel]:  # a template, its instantiations and a lambda's two bodies share offsets
                self.seen[rel].add(key)
                m = classify(n, self.V)
                if m:
                    self.tally[rel][fn][m] += 1
                if kind in LOOP_STMTS:
                    self.stmts[rel]["loop"] += 1
                elif kind == "IfStmt":
                    self.stmts[rel]["if"] += 1
                elif kind in TERNARIES:
                    self.stmts[rel]["?"] += 1
        for key, v in n.items():
            if key not in ("loc", "range"):
                self.value(v, fn)

    def value(self, v, fn):
        if isinstance(v, dict):
            if "kind" in v and ("range" in v or "loc" in v or "inner" in v):
                self.node(v, fn)
            elif "offset" in v or "spellingLoc" in v or "expansionLoc" in v:
                self.loc(v)
            else:
                for key, x in v.items():
                    if key != "includedFrom":
                        self.value(x, fn)
        elif isinstance(v, list):
            for x in v:
                self.value(x, fn)

    def feed(self, text):
        self.file = None
        dec, i, n = json.JSONDecoder(), 0, len(text)
        while i < n:
            while i < n and text[i] in " \r\n\t":
                i += 1
            if i >= n:
                break
            obj, i = dec.raw_decode(text, i)
            self.value(obj, "<file scope>")


# ---------------------------------------------------------------------------------------------------------
RAW = re.compile(r'(?<![\w])(?:u8|u|U|L)?R"([^()\\\s]{0,16})\(')


def code_only(text):
    """The source with comments, string/char literals and preprocessor directives removed."""
    out, i, n, bol = [], 0, len(text), True
    while i < n:
        if bol:
            j = i
            while j < n and text[j] in " \t":
                j += 1
            bol = False
            if j < n and text[j] == "#":
                while True:
                    k = text.find("\n", j)
                    if k < 0:
                        i = n
                        break
                    if text[k - 1] == "\\":
                        j = k + 1
                        continue
                    i = k
                    break
                continue
        c = text[i]
        if c == "\n":
            out.append(c)
            i, bol = i + 1, True
        elif text.startswith("//", i):
            k = text.find("\n", i)
            i = n if k < 0 else k
        elif text.startswith("/*", i):
            k = text.find("*/", i + 2)
            out.append(" " + "\n" * text.count("\n", i, n if k < 0 else k))
            i = n if k < 0 else k + 2
        elif RAW.match(text, i):
            m = RAW.match(text, i)
            k = text.find(")" + m.group(1) + '"', m.end())
            i = n if k < 0 else k + len(m.group(1)) + 2
            out.append('""')
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append('""')
        elif c == "'":
            if i and text[i - 1].isdigit() and i + 1 < n and text[i + 1].isalnum():
                i += 1  # digit separator, 1'000
                continue
            i += 1
            while i < n and text[i] != "'":
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append("''")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def token_counts(path):
    t = code_only(open(path, encoding="utf-8", errors="ignore").read())
    return {"loop": len(re.findall(r"\b(?:for|while)\b", t)), "if": len(re.findall(r"\bif\b", t)),
            "?": t.count("?")}


# ---------------------------------------------------------------------------------------------------------
def measure(build, only=None, substitute=None):
    """Counts per file. `only` limits the translation units; `substitute` = {rel: path} measures a scratch copy
    in place of a real file (the selftest)."""
    substitute = substitute or {}
    V = vocabulary()
    files = scope_files() if only is None else list(only)
    real = {rel: substitute.get(rel, os.path.join(ROOT, rel)) for rel in files}
    scope = {os.path.realpath(p): rel for rel, p in real.items()}
    cc_path = os.path.join(build, "compile_commands.json")
    if not os.path.exists(cc_path):
        raise SystemExit(f"verb_density: FAIL — no {cc_path} (configure the build first)")
    cc = {os.path.realpath(e["file"]): e for e in json.load(open(cc_path))}
    tus = []
    for rel in files:
        if rel.endswith(".cpp"):
            e = cc.get(os.path.realpath(os.path.join(ROOT, rel)))
            if not e:
                raise SystemExit(f"verb_density: FAIL — {rel} is not in compile_commands.json")
            tus.append((rel, ast_command(e, real[rel]), e["directory"]))

    def run(tu):
        rel, cmd, cwd = tu
        p = subprocess.run(cmd, cwd=cwd, capture_output=True)
        if p.returncode:
            raise SystemExit(f"verb_density: FAIL — clang could not parse {rel}:\n{p.stderr.decode()[-2000:]}")
        return p.stdout.decode("utf-8", errors="replace")

    jobs = int(os.environ.get("SWAPS_BUILD_JOBS") or os.cpu_count() // 2 or 2)
    walker = Walker(scope, V)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for text in pool.map(run, tus):
            walker.feed(text)

    problems = []
    for rel in files:
        tok = token_counts(real[rel])
        got = walker.stmts[rel]
        if rel not in walker.reached and any(tok.values()):
            problems.append(f"{rel}: no translation unit reached it, but it has code to count")
        for key, label in (("loop", "for/while"), ("if", "if"), ("?", "?:")):
            if tok[key] != got[key]:
                problems.append(f"{rel}: {tok[key]} `{label}` tokens but {got[key]} in the AST")
    if problems:
        raise SystemExit("verb_density: FAIL — the AST and the source disagree, so no count can be trusted:\n  "
                         + "\n  ".join(problems) + "\n  (an #if 0 block, a macro hiding a loop, or code outside "
                         "namespace swaps::api)")
    return {rel: {fn: dict(c) for fn, c in walker.tally[rel].items()} for rel in files}


def totals(per_fn):
    t = collections.Counter()
    for c in per_fn.values():
        t.update(c)
    return {m: t.get(m, 0) for m in METRICS}


def locked(rel):
    return [m for m in METRICS if not (tier(rel) == "codec" and m in CODEC_UNLOCKED)]


def read_lock():
    lock = {}
    if os.path.exists(LOCK):
        for line in open(LOCK, encoding="utf-8"):
            line = line.split("#", 1)[0].split()
            if line:
                lock[line[0]] = {k: int(v) for k, v in (f.split("=") for f in line[1:])}
    return lock


def violations(counts, lock):
    out = []
    for rel, per_fn in counts.items():
        t, want = totals(per_fn), lock.get(rel, {})
        for m in locked(rel):
            if t[m] > want.get(m, 0):
                out.append((rel, m, want.get(m, 0), t[m]))
    return out


def table(counts, by_function):
    print("verb_density: behaviour left in the verb layer  (P14: parse -> one library call -> emit)")
    print(f"  {'tier':<9}{'file':<42}" + "".join(f"{m:>9}" for m in METRICS))
    grand = {t: collections.Counter() for t in TIERS}
    for tr in TIERS:
        rows = [(rel, totals(c)) for rel, c in counts.items() if tier(rel) == tr]
        for rel, t in sorted(rows, key=lambda r: (-sum(r[1].values()), r[0])):
            grand[tr].update(t)
            if not sum(t.values()) and not by_function:
                continue
            print(f"  {tr:<9}{rel:<42}" + "".join(f"{t[m]:>9}" for m in METRICS))
            if by_function:
                for fn, c in sorted(counts[rel].items(), key=lambda kv: -sum(kv[1].values())):
                    print(f"  {'':<11}{fn:<40}" + "".join(f"{c.get(m, 0):>9}" for m in METRICS))
    print()
    for tr in TIERS:
        n = sum(1 for rel in counts if tier(rel) == tr)
        clean = sum(1 for rel in counts if tier(rel) == tr and not sum(totals(counts[rel]).values()))
        print(f"  {tr:<9}{f'{n} files, {clean} at zero':<42}" + "".join(f"{grand[tr][m]:>9}" for m in METRICS))


def write_lock(counts):
    with open(LOCK, "w", encoding="utf-8") as f:
        f.write("# tools/verb_density.py lock (E7 stage 1, PRINCIPLES.md P14). One line per api file: the most\n"
                "# behaviour it may carry. Counts may only go DOWN; a file absent from this list locks at zero.\n"
                "# Codec files do not lock loops/branches (decode walks arrays and checks shape).\n")
        for rel in sorted(counts):
            t = totals(counts[rel])
            if any(t[m] for m in locked(rel)):
                f.write(f"{rel} " + " ".join(f"{m}={t[m]}" for m in locked(rel)) + f"   # {tier(rel)}\n")


SELFTEST_SNIPPET = """
#include <cmath>
#include <optional>
#include "swaps/conventions_data.hpp"
namespace swaps::api {
namespace {
[[maybe_unused]] double verb_density_selftest(const boost::json::object& o) {
  const std::optional<double> deref(1.0);
  double s = *deref;  // a DEREFERENCE of a double: must count nothing (it once counted as a multiplication)
  for (int i = 0; i < 2; ++i) s = s * 2.0;
  if (o.contains("k")) s = jd(o, "k", 3.0);
  const double t = o.contains("t") ? std::sqrt(s) : s;
  (void)swaps::conventions::require_currency(std::string_view("X"));
  return t;
}
}  // namespace
}  // namespace swaps::api
"""
SELFTEST_EXPECT = {"lookups": 1, "defaults": 2, "loops": 1, "compute": 2, "branches": 1}


def selftest(build):
    rel = "api/pnl.cpp"
    src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    with tempfile.TemporaryDirectory() as d:
        injected, hidden = os.path.join(d, "injected", "pnl.cpp"), os.path.join(d, "hidden", "pnl.cpp")
        for p, extra in ((injected, SELFTEST_SNIPPET),
                         (hidden, "\n#if 0\nvoid never() { for (;;) {} }\n#endif\n")):
            os.makedirs(os.path.dirname(p))
            open(p, "w", encoding="utf-8").write(src + extra)
        with ThreadPoolExecutor(max_workers=3) as pool:
            base_f = pool.submit(measure, build, [rel])
            inj_f = pool.submit(measure, build, [rel], {rel: injected})
            hid_f = pool.submit(measure, build, [rel], {rel: hidden})
            base, inj = totals(base_f.result()[rel]), inj_f.result()
            try:
                hid_f.result()
                refused = False
            except SystemExit as e:
                refused = "disagree" in str(e)
    delta = {m: totals(inj[rel])[m] - base[m] for m in METRICS}
    ok = True
    if delta != SELFTEST_EXPECT:
        print(f"verb_density selftest: FAIL — injected {SELFTEST_EXPECT}, measured {delta}", file=sys.stderr)
        ok = False
    if not violations(inj, {rel: base}):
        print("verb_density selftest: FAIL — --check did not trip on the injected file", file=sys.stderr)
        ok = False
    if not refused:
        print("verb_density selftest: FAIL — a loop hidden in #if 0 did not make the cross-check refuse",
              file=sys.stderr)
        ok = False
    if ok:
        print(f"verb_density selftest: OK — each injected construct counted exactly {delta}; --check trips; "
              "an #if 0 loop makes the cross-check refuse")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=os.path.join(ROOT, "build"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--allow-increase", action="store_true")
    ap.add_argument("--by-function", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.build)
    counts = measure(a.build)
    lock = read_lock()
    if a.update:
        up = violations(counts, lock) if os.path.exists(LOCK) else []
        if up and not a.allow_increase:
            for rel, m, was, now in up:
                print(f"  {rel}: {m} {was} -> {now}", file=sys.stderr)
            print("verb_density: refusing to RAISE the lock (a count may only go down). Remove the behaviour, "
                  "or pass --allow-increase and say why in the commit.", file=sys.stderr)
            return 1
        write_lock(counts)
        print(f"verb_density: locked {sum(1 for r in counts if any(totals(counts[r])[m] for m in locked(r)))} files")
        return 0
    if a.check:
        bad = violations(counts, lock)
        if bad:
            for rel, m, was, now in bad:
                print(f"  {rel}: {m} {now} > locked {was}", file=sys.stderr)
            print("verb_density: FAIL — behaviour was ADDED to the verb layer (P14). Put it in include/swaps/** "
                  "and have the verb call it.", file=sys.stderr)
            return 1
        lower = sum(1 for rel in counts for m in locked(rel) if totals(counts[rel])[m] < lock.get(rel, {}).get(m, 0))
        t = collections.Counter()
        for rel in counts:
            if tier(rel) == "core":
                t.update(totals(counts[rel]))
        print(f"verb_density: OK — no api file gained behaviour; core verbs carry "
              + ", ".join(f"{t[m]} {m}" for m in METRICS)
              + (f" ({lower} counts now below the lock: run --update to bank them)" if lower else ""))
        return 0
    table(counts, a.by_function)
    return 0


if __name__ == "__main__":
    sys.exit(main())
