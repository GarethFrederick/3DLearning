#!/bin/bash
# a74 replaces the trend-derived aim block with a random one. The gate asserts
# the control actually fired AND that it landed somewhere else. A knob that
# printed its banner and then reused block 3 would be the treatment arm wearing
# the control's name.
set -u
LOG="${1:?usage: gate_a74_randomaim.sh <log>}"
[ -s "$LOG" ] || { echo "A74_FAIL: no log $LOG"; exit 1; }
L=$(grep -m1 "\[RANDOM-AIM\] CONTROL ON" "$LOG") \
  || { echo "A74_FAIL: the random-aim control never fired"; exit 1; }
FROM=$(echo "$L" | grep -oE "trend block [0-9]+" | grep -oE "[0-9]+")
TO=$(echo "$L" | grep -oE "random block [0-9]+" | grep -oE "[0-9]+")
[ -n "$FROM" ] && [ -n "$TO" ] || { echo "A74_FAIL: cannot read the blocks from: $L"; exit 1; }
[ "$FROM" != "$TO" ] || { echo "A74_FAIL: the random draw returned the trend block $TO — no contrast"; exit 1; }
grep -q "\[NO-GEN\] diffusion never sampled and never trained" "$LOG" \
  || { echo "A74_FAIL: the generator is not off — a74 must match a72's recipe"; exit 1; }
grep -q "Head range on warm-start | \[-0.5, 1.5\] (env override)" "$LOG" \
  || { echo "A74_FAIL: head range not widened"; exit 1; }
echo "A74_OK: aim moved from block $FROM to block $TO, no generator, wide head"
