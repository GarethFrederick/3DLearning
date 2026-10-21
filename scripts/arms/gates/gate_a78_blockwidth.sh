#!/bin/bash
# a78 gate: the block width reached the aim and the depth with the values the
# training data predicts, and the rest of a72's recipe is intact.
set -u
LOG="${1:?usage: gate_a78_blockwidth.sh <log> <width>}"; W="${2:?width}"
[ -s "$LOG" ] || { echo "A78_FAIL: no log $LOG"; exit 1; }
case "$W" in
  2) AIM='-> block 6 (12-14h) centre=13.0h'; DEP='\[TREND-DEPTH\] block 6 (12-14h).*DEPTH=\$216.17';;
  6) AIM='-> block 2 (12-18h) centre=15.0h'; DEP='\[TREND-DEPTH\] block 2 (12-18h).*DEPTH=\$213.48';;
  8) AIM='-> block 1 (08-16h) centre=12.0h'; DEP='\[TREND-DEPTH\] block 1 (08-16h).*DEPTH=\$224.31';;
  *) echo "A78_FAIL: unknown width $W"; exit 1;;
esac
grep -qF -- "$AIM" "$LOG" || { echo "A78_FAIL: aim line '$AIM' missing"; exit 1; }
grep -q "$DEP" "$LOG" || { echo "A78_FAIL: depth for width $W missing"; exit 1; }
grep -q "\[NO-GEN\] diffusion never sampled and never trained" "$LOG" || { echo "A78_FAIL: generator not off"; exit 1; }
grep -q "Head range on warm-start | \[-0.5, 1.5\] (env override)" "$LOG" || { echo "A78_FAIL: head range not wide"; exit 1; }
echo "A78_OK: width ${W}h, $AIM, depth matches, no generator, wide head"
