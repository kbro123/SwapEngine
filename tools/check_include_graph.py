#!/usr/bin/env python3
"""check_include_graph.py — the layer DAG in ARCHITECTURE.md is the LAW, and this checks the code against it.

ARCHITECTURE.md opens with a ```mermaid graph TD``` block whose every `a --> b` arrow reads "layer a depends on
/ includes layer b". Until now that diagram was prose: nothing stopped a header from including upwards (or
sideways) and quietly turning the compile-time DAG into a cycle. This walks the real `#include "swaps/<layer>/…"`
edges of include/ and api/ and fails on any edge the diagram does not declare, plus any cycle.

  python3 tools/check_include_graph.py [--verbose]

Exit 0 = every actual edge is declared and the graph is acyclic. Exit 1 = violations (listed with the file and
line that introduced them). Declared-but-unused arrows are reported as STALE — the diagram claiming a
dependency the code dropped — and are a warning, not a failure.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(ROOT, "ARCHITECTURE.md")
INC_RE = re.compile(r'^\s*#include\s+"swaps/([a-z_]+)/')
ARROW_RE = re.compile(r"^\s*([a-z_]+)\s*-->\s*([a-z_]+)\s*$")


def declared_edges():
    """The arrows inside ARCHITECTURE.md's first mermaid block."""
    text = open(ARCH).read()
    start = text.index("```mermaid")
    end = text.index("```", start + 3)
    edges = set()
    for line in text[start:end].splitlines():
        m = ARROW_RE.match(line)
        if m:
            edges.add((m.group(1), m.group(2)))
    if not edges:
        sys.exit("check_include_graph: no arrows found in ARCHITECTURE.md's mermaid block")
    return edges


def actual_edges():
    """(layer -> layer, [(file, line), ...]) from the real includes of include/swaps and api/."""
    edges = {}
    roots = [(os.path.join(ROOT, "include", "swaps"), None), (os.path.join(ROOT, "api"), "api")]
    for base, forced in roots:
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith((".hpp", ".cpp", ".h", ".inc")):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, ROOT)
                if forced:
                    layer = forced
                else:
                    parts = os.path.relpath(path, base).split(os.sep)
                    layer = parts[0] if len(parts) > 1 else None  # include/swaps/*.hpp itself is layer-less
                    if layer is None:
                        continue
                for i, line in enumerate(open(path, errors="replace"), 1):
                    m = INC_RE.match(line)
                    if not m:
                        continue
                    dep = m.group(1)
                    if dep == layer:
                        continue  # intra-layer includes are free
                    edges.setdefault((layer, dep), []).append((rel, i))
    return edges


def cycles(edges):
    """Every SIMPLE cycle, each reported once (canonical: a cycle is recorded only from its smallest node)."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
    out = []

    def walk(start, node, path, on):
        for nxt in sorted(adj.get(node, ())):
            if nxt == start:
                out.append(path + [start])
            elif nxt not in on and nxt > start:  # only paths whose nodes all exceed the start => one per cycle
                walk(start, nxt, path + [nxt], on | {nxt})

    for n in sorted(adj):
        walk(n, n, [n], {n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="list every actual edge with its include sites")
    a = ap.parse_args()

    declared = declared_edges()
    actual = actual_edges()
    layers = sorted({l for e in actual for l in e} | {l for e in declared for l in e})

    undeclared = sorted(e for e in actual if e not in declared)
    stale = sorted(e for e in declared if e not in actual)
    cyc = cycles(set(actual))

    if a.verbose:
        print(f"layers: {', '.join(layers)}")
        for (src, dst), sites in sorted(actual.items()):
            mark = " " if (src, dst) in declared else "!"
            print(f" {mark} {src} --> {dst}  ({len(sites)} include(s), e.g. {sites[0][0]}:{sites[0][1]})")

    rc = 0
    for src, dst in undeclared:
        sites = actual[(src, dst)]
        print(f"  UNDECLARED  {src} --> {dst}  — {len(sites)} include(s); add the arrow to ARCHITECTURE.md or "
              f"break the dependency", file=sys.stderr)
        for f, ln in sites[:4]:
            print(f"              {f}:{ln}", file=sys.stderr)
        rc = 1
    for c in cyc:
        print(f"  CYCLE       {' --> '.join(c)}", file=sys.stderr)
        rc = 1
    for src, dst in stale:
        print(f"  stale arrow {src} --> {dst} is declared in ARCHITECTURE.md but no include uses it (warning)")

    if rc:
        print("check_include_graph: FAIL", file=sys.stderr)
        return 1
    print(f"check_include_graph: OK — {len(actual)} cross-layer edges over {len(layers)} layers, all declared, acyclic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
