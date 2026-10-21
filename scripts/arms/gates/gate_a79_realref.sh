#!/bin/bash
# a79 gate: the real reference pool was built AND used in round 2, and the rest
# of a72's recipe (aim, depth, no generator, wide head) is unchanged.
set -u
LOG="${1:?usage: gate_a79_realref.sh <log>}"
[ -s "$LOG" ] || { echo "A79_FAIL: no log $LOG"; exit 1; }
grep -q "\[REF-POOL\] REAL reference pool: 105216 rows from .*, 512 midnight-start sequences drawn by RandomState(20261001) from 2189 start days .* -> 512 sequences of 192 (z0 NOT used for c_ref)" "$LOG" \
  || { echo "A79_FAIL: real pool not built with 105216 rows / 512 sequences"; exit 1; }
grep -q "\[REF-POOL\] round 2 c_ref from the REAL pool" "$LOG" \
  || { echo "A79_FAIL: round 2 did not use the real pool"; exit 1; }
grep -q "\[TREND-AIM\].*block 3 (12-16h) centre=14.0h" "$LOG" || { echo "A79_FAIL: aim is not block 3"; exit 1; }
grep -q "\[TREND-DEPTH\] block 3 (12-16h).*DEPTH=\$214.21" "$LOG" || { echo "A79_FAIL: depth is not \$214.21"; exit 1; }
grep -q "\[NO-GEN\] diffusion never sampled and never trained" "$LOG" || { echo "A79_FAIL: generator not off"; exit 1; }
grep -q "Head range on warm-start | \[-0.5, 1.5\] (env override)" "$LOG" || { echo "A79_FAIL: head range not wide"; exit 1; }
echo "A79_OK: real reference pool built and used, a72 recipe intact"
