#!/bin/bash
# a77 = a46 (narrow, no generator, 30 rounds) with the depth set to the observed
# 2019->2020 fall of the block-3 floor instead of the curve extended to 2024.
set -u
LOG="${1:?usage: gate_a77_obsdepth.sh <log>}"
[ -s "$LOG" ] || { echo "A77_FAIL: no log $LOG"; exit 1; }
grep -q "\[DEPTH-OBSERVED\] CONTROL ON | block 3 fitted depth \$214.21 REPLACED by observed 2019->2020 fall \$23.97" "$LOG" \
  || { echo "A77_FAIL: observed-depth control did not fire with \$214.21 -> \$23.97"; exit 1; }
grep -q "\[TREND-AIM\].*-> block 3 (12-16h) centre=14.0h" "$LOG" \
  || { echo "A77_FAIL: aim is not block 3"; exit 1; }
grep -q "TREND-DEPTH=\$24 " "$LOG" \
  || { echo "A77_FAIL: the \$24 depth did not reach the edit strength"; exit 1; }
grep -q "\[NO-GEN\] diffusion never sampled and never trained" "$LOG" \
  || { echo "A77_FAIL: the generator is not off"; exit 1; }
grep -q "Head range on warm-start | \[0.0, 1.0\]" "$LOG" \
  || { echo "A77_FAIL: head range is not narrow"; exit 1; }
echo "A77_OK: block 3, depth \$23.97 observed (fit \$214.21 replaced), no generator, narrow head"
