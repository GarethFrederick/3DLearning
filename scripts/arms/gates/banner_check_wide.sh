#!/bin/bash
# Gate a wide-head arm on the head range ACTUALLY widening, and on the arm's own
# feature set matching its narrow-head reference. Two checks because a forgotten
# PRED_NORM_* returns a byte-identical narrow run, which is one of the answers
# under test.
set -u
LOG="$1"; REF="$2"
[ -s "$LOG" ] || { echo "WIDE_GATE_FAIL: no log $LOG"; exit 1; }
grep -q "Head range on warm-start | \[-0.5, 1.5\] (env override)" "$LOG" \
  || { echo "WIDE_GATE_FAIL: head range not widened"; grep -m1 "Head range" "$LOG"; exit 1; }
scripts/arms/gates/banner_match.sh "$LOG" "$REF" || exit 1
echo "WIDE_BANNER_OK"
