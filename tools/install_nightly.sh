#!/usr/bin/env bash
# install_nightly.sh — register tools/nightly.sh as a user launchd agent (03:00 local, this machine only).
# Idempotent. Remove with:  launchctl bootout gui/$(id -u)/com.swapengine.nightly; rm ~/Library/LaunchAgents/com.swapengine.nightly.plist
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST="${HOME}/Library/LaunchAgents/com.swapengine.nightly.plist"
mkdir -p "${HOME}/Library/LaunchAgents" "${HOME}/Library/Logs/swapengine-nightly"
cat > "${PLIST}" <<EOT
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.swapengine.nightly</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>${ROOT}/tools/nightly.sh</string></array>
  <key>WorkingDirectory</key><string>${ROOT}</string>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>${HOME}/Library/Logs/swapengine-nightly/launchd.out</string>
  <key>StandardErrorPath</key><string>${HOME}/Library/Logs/swapengine-nightly/launchd.err</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
</dict></plist>
EOT
launchctl bootout "gui/$(id -u)/com.swapengine.nightly" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "${PLIST}"
launchctl print "gui/$(id -u)/com.swapengine.nightly" >/dev/null && echo "nightly installed: ${PLIST} (03:00 daily; logs in ~/Library/Logs/swapengine-nightly)"
