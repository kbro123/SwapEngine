#!/usr/bin/env bash
# install_hooks.sh — point git at the committed hooks (.githooks/). One-time per clone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "hooks installed: core.hooksPath=.githooks ($(ls .githooks | tr '\n' ' '))"
echo "bypass for an emergency (and say why in the commit): SWAPS_SKIP_HOOKS=1 git push"
