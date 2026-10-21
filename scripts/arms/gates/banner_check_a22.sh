#!/bin/bash
# Smoke-log gate for a22_nogen_discdepth: proves the instructor discovers a
# direction, the broad head augmentation fires on real batches with it, and every
# generator-teaching path and the narrow E3 replay stay severed.
# Usage: SMOKE_LOG=<path> bash banner_check_a22.sh
set -u
L="$SMOKE_LOG"
fail() { echo "A22_BANNER_FAIL: $1"; exit 1; }

[ -s "$L" ] || fail "smoke log missing/empty: $L"

# 1. instructor TYPICAL mode live (discovers ONE mean-judged direction/round)
grep -Fq 'MODE=instructor_typical' "$L" || fail "instructor_typical mode banner missing"
grep -Fq '[INSTRUCTOR-TYP] direction=' "$L" || fail "typical instructor direction line missing"

# 2. head-aug configured (both setup banners)
grep -Fq '[FOLD-IN] INSTR-HEAD-AUG mode ON' "$L" || fail "INSTR-HEAD-AUG setup banner missing"
grep -Fq '[INSTR-HEAD-AUG] ON in ML inner loop' "$L" || fail "INSTR-HEAD-AUG train_ml banner missing"
grep -Fq 'width=WIDE' "$L" || fail "width=WIDE missing — a13 must run the WIDE bell, not narrow"
grep -Fq '[HEAD-REAL] loaded' "$L" || fail "[HEAD-REAL] missing — head not trained on REAL data (HEAD_TRAIN_REAL not engaged)"
grep -Fq 'WIDTH-DISC=' "$L" || fail "WIDTH-DISC missing — instructor did NOT discover the width (INSTR_HEAD_DISCOVER_WIDTH not engaged)"
grep -Fq '[INSTR-REAL-WIN]' "$L" || fail "[INSTR-REAL-WIN] missing — instructor NOT judging on real windows (INSTR_REAL_WINDOWS not engaged)"
grep -Fq 'DEPTH-DISC=' "$L" || fail "DEPTH-DISC missing — instructor did NOT discover the depth (INSTR_HEAD_DISCOVER_DEPTH not engaged)"

# 3. a direction was DISCOVERED and handed to the head aug
grep -Fq '[INSTR-HEAD-AUG] aim centre=' "$L" || fail "no discovered aim handed to head aug"

# 4. the aug fired on real batches (the effect, not the setting): a silent no-op
#    prints 'applied to 0 batches' or 'NO DIRECTION YET'.
grep -Pq '\[INSTR-HEAD-AUG\] ep\d+ applied to [1-9][0-9]* batches \| centre=' "$L" \
  || fail "head aug never fired on a real batch with a discovered block"
grep -Fq 'NO DIRECTION YET' "$L" && echo "  (note: some early epoch had no direction yet — expected in round 0)"

# 5. generator-teaching + narrow-replay severed
grep -Fq '[FOLD-IN] HEAD-ONLY mode ON' "$L" || fail "HEAD-ONLY banner missing (generator not severed)"
grep -Fq 's0 replay loader' "$L" && fail "s0 replay fold-in fired — generator being taught"
grep -Fq '[CONSOL SESSION]' "$L" && fail "consolidation session fired — generator being taught"
grep -Fq '[FOLD-IN->HEAD]' "$L" && fail "E3 narrow buffer->head replay fired — should be OFF under INSTR_HEAD_AUG"

# 6. run completed (not a crash mid-way)
grep -Fq 'Restored best' "$L" || fail "smoke did not reach final restore"

echo "A22_BANNER_OK"
