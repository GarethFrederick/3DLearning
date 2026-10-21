#!/bin/bash
# a76 forces the aim to block 4 (16-20h). The gate asserts the fixed-aim control
# fired, that the aim that reached training IS block 4 (not the trend block 3
# wearing the control's name), that the depth was derived for block 4 ($223.04,
# diag_block4_depth.sh), and that the rest of a72's recipe is intact.
set -u
LOG="${1:?usage: gate_a76_block4.sh <log>}"
[ -s "$LOG" ] || { echo "A76_FAIL: no log $LOG"; exit 1; }
grep -q "\[FIXED-AIM\] CONTROL ON | trend block 3 REPLACED by fixed block 4" "$LOG" \
  || { echo "A76_FAIL: the fixed-aim control never fired, or did not move 3 -> 4"; exit 1; }
grep -q "\[TREND-AIM\].*-> block 4 (16-20h) centre=18.0h" "$LOG" \
  || { echo "A76_FAIL: the aim that reached training is not block 4 at 18.0h"; exit 1; }
grep -q "\[TREND-DEPTH\] block 4 (16-20h).*DEPTH=\$223" "$LOG" \
  || { echo "A76_FAIL: depth not derived for block 4 (expected \$223)"; exit 1; }
grep -q "\[NO-GEN\] diffusion never sampled and never trained" "$LOG" \
  || { echo "A76_FAIL: the generator is not off"; exit 1; }
grep -q "Head range on warm-start | \[-0.5, 1.5\] (env override)" "$LOG" \
  || { echo "A76_FAIL: head range not widened"; exit 1; }
echo "A76_OK: aim forced to block 4 (18.0h), depth \$223, no generator, wide head"
