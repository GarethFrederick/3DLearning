#!/bin/bash
# Smoke-log gate for a25_recency, a control. A control that silently ran as the
# treatment is worse than none, so the gate proves the real pool shrank to recent
# years, the discount depth is zero, and the aug path still ran.
# Usage: SMOKE_LOG=<path> bash banner_check_a25.sh
set -u
L="$SMOKE_LOG"
fail() { echo "A25_BANNER_FAIL: $1"; exit 1; }

[ -s "$L" ] || fail "smoke log missing/empty: $L"

# 1. instructor typical mode + head aug configured (same skeleton as a19)
grep -Fq 'MODE=instructor_typical' "$L" || fail "instructor_typical mode banner missing"
grep -Fq '[FOLD-IN] INSTR-HEAD-AUG mode ON' "$L" || fail "INSTR-HEAD-AUG setup banner missing"
grep -Fq '[INSTR-HEAD-AUG] ON in ML inner loop' "$L" || fail "INSTR-HEAD-AUG train_ml banner missing"

# 2. head on REAL data, and the pool RESTRICTED — the row count must actually
#    fall, otherwise the recency knob was a no-op that looked active.
grep -Fq '[HEAD-REAL] loaded' "$L" || fail "[HEAD-REAL] missing — head not on real data"
grep -Fq '[HEAD-REAL-RECENCY] pool restricted' "$L" \
  || fail "[HEAD-REAL-RECENCY] missing — HEAD_REAL_FROM_YEAR did not engage"
grep -Pq '\[HEAD-REAL-RECENCY\] pool restricted to years >= \d+: (\d+) -> (\d+) rows' "$L" \
  || fail "recency line present but row counts unreadable"
awk '/HEAD-REAL-RECENCY/ {
       if (match($0, /: [0-9]+ -> [0-9]+ rows/)) {
         split(substr($0, RSTART+2, RLENGTH-8), a, " -> ");
         if (a[2]+0 >= a[1]+0) { print "NOSHRINK"; exit }
         if (a[2]+0 == 0)      { print "EMPTY";    exit }
       }
     }' "$L" | grep -q . && fail "recency filter did not shrink the pool (or emptied it)"

# 3. the discount depth is ZERO — this is the control's defining property
grep -Fq 'alpha_max=$0' "$L" || fail "alpha_max is not \$0 — this is NOT the zero-discount control"

# 4. the aug path still ran, so a19 vs a25 differs in strength, not in which
#    code executed.
grep -Pq '\[INSTR-HEAD-AUG\] ep\d+ applied to [1-9][0-9]* batches' "$L" \
  || fail "aug never ran — control would differ from a19 by more than the tested knobs"

# 5. generator-teaching severed (same as every head-only arm)
grep -Fq '[FOLD-IN] HEAD-ONLY mode ON' "$L" || fail "HEAD-ONLY banner missing"
grep -Fq 's0 replay loader' "$L" && fail "s0 replay fired — generator being taught"
grep -Fq '[CONSOL SESSION]' "$L" && fail "consolidation session fired — generator being taught"

# 6. completed
grep -Fq 'Restored best' "$L" || fail "smoke did not reach final restore"

echo "A25_BANNER_OK"
