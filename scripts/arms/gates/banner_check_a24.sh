#!/bin/bash
# Smoke-log gate for a24_trendaim. The trend aim must pick block 3 (centre 14.0h)
# as diag_train_only_aim_robust.py does; any other block means the code and the
# diagnostic disagree and the arm is void. Also checks the head aug fired at that
# centre and generator teaching stays severed.
# Usage: SMOKE_LOG=<path> bash banner_check_a24.sh
set -u
L="$SMOKE_LOG"
fail() { echo "A24_BANNER_FAIL: $1"; exit 1; }

[ -s "$L" ] || fail "smoke log missing/empty: $L"

# 1. instructor TYPICAL mode live (discovers ONE mean-judged direction/round)
grep -Fq 'MODE=instructor_typical' "$L" || fail "instructor_typical mode banner missing"
grep -Fq '[INSTRUCTOR-TYP] direction=' "$L" || fail "typical instructor direction line missing"

# 2. head-aug configured on the a19 base (wide bell, real data, discovered width)
grep -Fq '[FOLD-IN] INSTR-HEAD-AUG mode ON' "$L" || fail "INSTR-HEAD-AUG setup banner missing"
grep -Fq '[INSTR-HEAD-AUG] ON in ML inner loop' "$L" || fail "INSTR-HEAD-AUG train_ml banner missing"
grep -Fq 'width=WIDE' "$L" || fail "width=WIDE missing — must run the WIDE bell, not narrow"
grep -Fq '[HEAD-REAL] loaded' "$L" || fail "[HEAD-REAL] missing — head not trained on REAL data (HEAD_TRAIN_REAL not engaged)"
grep -Fq 'WIDTH-DISC=' "$L" || fail "WIDTH-DISC missing — width not discovered (INSTR_HEAD_DISCOVER_WIDTH not engaged)"

# 3. the TREND aim engaged, is train-only, and reproduces the registered block
grep -Fq '[TREND-AIM]' "$L" || fail "[TREND-AIM] missing — trend aim not engaged (INSTR_HEAD_AIM_TREND)"
grep -Fq 'NO OOD data' "$L" || fail "trend-aim provenance line missing (train-only derivation not asserted)"
grep -Eq '\[TREND-AIM\].*-> block 3 \(12-16h\) centre=14\.0h' "$L" \
  || fail "TREND-AIM did not pick block 3 / centre 14.0h — code disagrees with diag_train_only_aim_robust.py; arm VOID until reconciled"

# 4. the head aug received and applied the trend centre, not the divergence
#    argmax; a silent no-op prints 'applied to 0 batches' or 'NO DIRECTION YET'.
grep -Fq '[INSTR-HEAD-AUG] aim centre=14.0h' "$L" \
  || fail "trend centre 14.0h was NOT the aim handed to the head aug"
grep -Pq '\[INSTR-HEAD-AUG\] ep\d+ applied to [1-9][0-9]* batches \| centre=14\.0h' "$L" \
  || fail "head aug never fired on a real batch at the trend centre 14.0h"

# 5. generator-teaching + narrow-replay severed
grep -Fq '[FOLD-IN] HEAD-ONLY mode ON' "$L" || fail "HEAD-ONLY banner missing (generator not severed)"
grep -Fq 's0 replay loader' "$L" && fail "s0 replay fold-in fired — generator being taught"
grep -Fq '[CONSOL SESSION]' "$L" && fail "consolidation session fired — generator being taught"
grep -Fq '[FOLD-IN->HEAD]' "$L" && fail "E3 narrow buffer->head replay fired — should be OFF under INSTR_HEAD_AUG"

# 6. run completed (not a crash mid-way)
grep -Fq 'Restored best' "$L" || fail "smoke did not reach final restore"

echo "A24_BANNER_OK"
