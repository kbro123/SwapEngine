#!/bin/bash
# Re-run the Part 3 branch probe ONLY after chain.sh has finished (its ./coarse binary holds on any process named branchprobe, and
# branchprobe held on ./coarse: a mutual hold, deadlocked 14:02-16:13). Timing inside still holds on the gate (util.hpp).
cd "$(dirname "$0")"
while pgrep -f "bash chain.sh" >/dev/null; do sleep 30; done
echo "=== branchprobe (after chain) start $(date +%T) load $(sysctl -n vm.loadavg)"
./branchprobe
echo "=== branchprobe end $(date +%T) exit $? load $(sysctl -n vm.loadavg)"
echo "BRANCH DONE"
