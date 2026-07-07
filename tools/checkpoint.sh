#!/usr/bin/env bash
# checkpoint.sh — the ONLY sanctioned path to a major checkpoint + web backup.
#   1. runs both gates (verify.sh) and refuses to proceed unless they pass
#   2. commits the working tree
#   3. tags an annotated checkpoint-NN
#   4. pushes to the web backup remote 'origin' (private GitHub repo) — only if it exists
#
# Usage: ./tools/checkpoint.sh "feat(curve): global LM calibration landing"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MSG="${1:-}"
if [ -z "${MSG}" ]; then echo "usage: checkpoint.sh \"commit message\"" >&2; exit 2; fi

echo ">> running gates before checkpoint"
if ! ./tools/verify.sh; then
  echo "!! gates failed — refusing to checkpoint. Fix correctness/perf first." >&2
  exit 1
fi

# Next checkpoint number.
last="$(git tag --list 'checkpoint-*' | sed 's/checkpoint-//' | sort -n | tail -1)"
next=$(printf '%02d' $(( ${last:-0} + 1 )))
tag="checkpoint-${next}"

git add -A
git commit -m "${MSG}

Verified: correctness+perf gates passing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" || echo "(nothing to commit)"
git tag -a "${tag}" -m "${MSG}"
echo ">> created ${tag}"

# Web backup — only if a remote is configured. Never auto-creates one.
if git remote get-url origin >/dev/null 2>&1; then
  echo ">> pushing ${tag} to web backup (origin)"
  git push origin main --tags
else
  echo ">> no 'origin' remote yet — skipping web backup."
  echo "   To enable: create the private GitHub repo and 'git remote add origin <url>'."
fi
