#!/usr/bin/env python3
"""Retarget a Claude Code transcript to a new absolute project path so it resumes on another machine.

Rewrites the `cwd` field on every line, and prints the destination folder + filename to drop it into.
Claude encodes a project's session folder by replacing every non-alphanumeric char in the ABSOLUTE
path with '-'. Usage:

    python3 retarget_transcript.py <source.jsonl> "/Users/Shared/dev/SwapsEngine" [out.jsonl]
"""
import json, os, re, sys

def encode(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", path)

def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, new_cwd = sys.argv[1], os.path.normpath(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(src)[0] + ".retargeted.jsonl"
    session_id = os.path.splitext(os.path.basename(src))[0].replace(".retargeted", "")
    n = 0
    with open(src) as f, open(out, "w") as g:
        for ln in f:
            s = ln.strip()
            if s:
                try:
                    d = json.loads(s)
                    if isinstance(d, dict) and "cwd" in d:
                        d["cwd"] = new_cwd
                        n += 1
                    s = json.dumps(d, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass  # leave non-JSON lines verbatim
            g.write(s + "\n")
    folder = encode(new_cwd)
    print(f"rewrote cwd on {n} lines -> {new_cwd}")
    print(f"wrote: {out}")
    print("\nOn the target machine:")
    print(f"  mkdir -p ~/.claude/projects/{folder}")
    print(f"  cp '{os.path.basename(out)}' ~/.claude/projects/{folder}/{session_id}.jsonl")
    print(f"  cd '{new_cwd}' && claude --resume {session_id}")

if __name__ == "__main__":
    main()
