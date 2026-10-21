#!/bin/bash
# Smoke-log gate for a23_zerodose, the all-years, zero-discount control. It must
# differ from a19 by the discount alone and from a25 by the year restriction
# alone, so the gate checks what a23 must not have as hard as what it must have.
# Usage: SMOKE_LOG=<path> bash banner_check_a23.sh
set -u
L="$SMOKE_LOG"
fail() { echo "A23_BANNER_FAIL: $1"; exit 1; }

[ -s "$L" ] || fail "smoke log missing/empty: $L"

# 1. the a19 skeleton
grep -Fq 'MODE=instructor_typical' "$L" || fail "instructor_typical mode banner missing"
grep -Fq '[FOLD-IN] INSTR-HEAD-AUG mode ON' "$L" || fail "INSTR-HEAD-AUG setup banner missing"
grep -Fq '[INSTR-HEAD-AUG] ON in ML inner loop' "$L" || fail "INSTR-HEAD-AUG train_ml banner missing"
grep -Fq '[HEAD-REAL] loaded' "$L" || fail "[HEAD-REAL] missing — head not on real data"
grep -Fq 'WIDTH-DISC=' "$L" || fail "WIDTH-DISC missing — width not discovered"

# 2. the discount depth is ZERO — the defining property of the zero-dose control
grep -Fq 'alpha_max=$0' "$L" || fail "alpha_max is not \$0 — this is NOT the zero-dose control"

# 3. no year restriction: if the recency filter fires here, a23 duplicates a25.
grep -Fq '[HEAD-REAL-RECENCY]' "$L" \
  && fail "[HEAD-REAL-RECENCY] present — HEAD_REAL_FROM_YEAR leaked in; a23 would duplicate a25"

# 4. the real pool is the full pool: every all-years arm loads 105216 rows. A
#    smaller count means another cut, which confounds the discount contrast.
_ROWS=$(grep -o '\[HEAD-REAL\] loaded [0-9]* real train rows' "$L" | grep -o '[0-9]*' | head -1)
[ -n "${_ROWS:-}" ] || fail "could not read the [HEAD-REAL] row count"
[ "$_ROWS" -eq 105216 ] \
  || fail "real pool is $_ROWS rows, expected the full 105216 — a23 must use ALL years"

# 5. the aug code path still RAN, so a19 vs a23 differs in the discount strength,
#    not in which code executed.
grep -Pq '\[INSTR-HEAD-AUG\] ep\d+ applied to [1-9][0-9]* batches' "$L" \
  || fail "aug never ran — control would differ from a19 by more than the discount"

# 6. generator teaching severed
grep -Fq '[FOLD-IN] HEAD-ONLY mode ON' "$L" || fail "HEAD-ONLY banner missing"
grep -Fq 's0 replay loader' "$L" && fail "s0 replay fired — generator being taught"
grep -Fq '[CONSOL SESSION]' "$L" && fail "consolidation session fired — generator being taught"

# 7. completed
grep -Fq 'Restored best' "$L" || fail "smoke did not reach final restore"

echo "A23_BANNER_OK (alpha_max=\$0, full $_ROWS-row pool, no year restriction)"
