# scratchpad/ — not source code

This folder holds a **Claude Code chat transcript**, kept here only so it travels with the repo.
It is NOT part of the engine and is not built, tested, or referenced by any code.

## Contents
- `session-50236c63-beac-41c0-b1d6-a82cd1b1d709.jsonl` — full transcript of the session that landed
  `checkpoint-12` (the outright-vs-spread bundle parameterization). ~5 MB, one JSON object per line.
- `retarget_transcript.py` — rewrites the transcript's `cwd` to a new machine's absolute repo path.

## Using it on another machine

**Just want a fresh, oriented session?** You don't need this file at all — clone the repo, run
`./tools/bootstrap_deps.sh`, and open Claude Code in the repo; it auto-loads `CLAUDE.md`.

**Want a session to read this chat for context?** Point it at the `.jsonl` explicitly (it's large).

**Want to RESUME this exact session with full history?** Claude only resumes transcripts stored under
`~/.claude/projects/<encoded-path>/`, keyed by the repo's absolute path — so retarget first:

```bash
# from this folder, on the target machine (adjust the path to where you cloned the repo)
python3 retarget_transcript.py \
  session-50236c63-beac-41c0-b1d6-a82cd1b1d709.jsonl \
  "$(cd .. && pwd)"        # the repo's absolute path on THIS machine
# then follow the printed mkdir/cp/`claude --resume` commands
```

The transcript's original `cwd` was `/Users/kjb/Desktop/Claude Projects/SwapsEngine`; resume on a
machine with a different username/path requires the retarget step above.
