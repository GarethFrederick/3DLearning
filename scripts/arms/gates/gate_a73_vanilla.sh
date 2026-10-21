#!/bin/bash
# a73 is 3D-Learning with the co-evolution wrapper OFF. The gate asserts the
# ABSENCE of the wrapper and the PRESENCE of the wide head. Absence is the hard
# half: a forgotten FOLD_IN=0 gives a run identical to a71, which is the arm a73
# exists to be compared against.
set -u
LOG="${1:?usage: gate_a73_vanilla.sh <log>}"
[ -s "$LOG" ] || { echo "A73_FAIL: no log $LOG"; exit 1; }
grep -q "\[FOLD-IN\] CO-EVOLUTION ON" "$LOG" \
  && { echo "A73_FAIL: the co-evolution wrapper is ON — this is a71, not a73"; exit 1; }
grep -q "Head range on warm-start | \[-0.5, 1.5\] (env override)" "$LOG" \
  || { echo "A73_FAIL: head range not widened"; grep -m1 "Head range" "$LOG"; exit 1; }
grep -q "3DLearning Training Start" "$LOG" \
  || { echo "A73_FAIL: training never started — an absence check on a dead log passes for the wrong reason"; exit 1; }
echo "A73_OK: no co-evolution wrapper, wide head, training started"
