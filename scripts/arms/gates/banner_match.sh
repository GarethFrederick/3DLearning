#!/bin/bash
# Prove a RECONSTRUCTED config reproduces an original run's feature set.
# Usage: banner_match.sh <smoke.out> <reference.out>
# Compares the feature banners only — the lines that say which mechanisms are
# ON. Per-epoch numbers differ (different seed, fewer rounds) and are excluded.
# A reconstructed recipe that silently drops a knob would otherwise launch a
# 48-hour fleet that answers a different question.
set -u
SM="$1"; REF="$2"
[ -s "$SM" ] || { echo "MATCH_FAIL: no smoke log $SM"; exit 1; }
[ -s "$REF" ] || { echo "MATCH_FAIL: no reference log $REF"; exit 1; }
pick() { grep -oE "\[FOLD-IN\](\[[A-Z_]+\])? (CO-EVOLUTION ON|MODE=[a-z_]+|ON \(A3[^|]*|INSTR-HEAD-AUG mode ON)" "$1" \
         | sed 's/ *$//' | sort -u; }
diff <(pick "$REF") <(pick "$SM") > /tmp/bm_$$.diff 2>&1
if [ -s /tmp/bm_$$.diff ]; then
  echo "MATCH_FAIL: feature banners differ from the reference run"
  echo "  < reference   > smoke"; cat /tmp/bm_$$.diff; rm -f /tmp/bm_$$.diff; exit 1
fi
rm -f /tmp/bm_$$.diff
echo "BANNER_MATCH_OK: same feature set as $REF"
