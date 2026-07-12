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
VERIFY_OUT="$(mktemp)"
if ! ./tools/verify.sh | tee "${VERIFY_OUT}"; then
  echo "!! gates failed — refusing to checkpoint. Fix correctness/perf first." >&2
  exit 1
fi

# Report the ACTUAL gate status in the trailer. Never claim "perf passing" when the
# perf gate merely skipped — a commit trailer that overstates verification is worse
# than none, because later work trusts it.
gate() { grep -E "^  $1 gate:" "${VERIFY_OUT}" | awk '{print $NF}'; }
C_STATUS="$(gate correctness)"; P_STATUS="$(gate performance)"
rm -f "${VERIFY_OUT}"
TRAILER="Verified: correctness=${C_STATUS:-UNKNOWN} perf=${P_STATUS:-UNKNOWN}"

if [ "${P_STATUS}" != "PASS" ]; then
  echo ">> NOTE: perf gate is '${P_STATUS}' — this checkpoint is not perf-verified."
fi

# Next checkpoint number.
last="$(git tag --list 'checkpoint-*' | sed 's/checkpoint-//' | sort -n | tail -1)"
next=$(printf '%02d' $(( 10#${last:-0} + 1 )))  # 10# forces base-10 (else 08/09 parse as octal)
tag="checkpoint-${next}"

git add -A
git commit -m "${MSG}

${TRAILER}

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
