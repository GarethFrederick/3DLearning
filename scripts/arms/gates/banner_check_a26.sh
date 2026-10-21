#!/bin/bash
# Smoke-log gate for a26_aimbank. A bank that never grows past one entry is a
# silent no-op that still prints an active banner, so the gate checks the bank
# size and its distinct centres, not just that the line exists.
# Usage: SMOKE_LOG=<path> bash banner_check_a26.sh
set -u
L="$SMOKE_LOG"
fail() { echo "A26_BANNER_FAIL: $1"; exit 1; }

[ -s "$L" ] || fail "smoke log missing/empty: $L"

# 1. the a19 skeleton
grep -Fq 'MODE=instructor_typical' "$L" || fail "instructor_typical mode banner missing"
grep -Fq '[FOLD-IN] INSTR-HEAD-AUG mode ON' "$L" || fail "INSTR-HEAD-AUG setup banner missing"
grep -Fq '[INSTR-HEAD-AUG] ON in ML inner loop' "$L" || fail "INSTR-HEAD-AUG train_ml banner missing"
grep -Fq '[HEAD-REAL] loaded' "$L" || fail "[HEAD-REAL] missing — head not on real data"
grep -Fq 'WIDTH-DISC=' "$L" || fail "WIDTH-DISC missing — width not discovered"

# 2. the bank exists
grep -Fq '[AIM-BANK]' "$L" || fail "[AIM-BANK] missing — INSTR_HEAD_ACCUM did not engage"

# 3. the bank GROWS past one entry. A smoke runs 2 rounds, so size must reach >=2
#    unless both rounds discovered the identical aim — in which case the arm
#    cannot demonstrate its mechanism and is VOID for its stated purpose.
_MAX=$(grep -o '\[AIM-BANK\] size=[0-9]*' "$L" | grep -o '[0-9]*$' | sort -n | tail -1)
[ -n "${_MAX:-}" ] || fail "could not read any bank size"
[ "$_MAX" -ge 2 ] || fail "bank never grew past size 1 (max=$_MAX) — every round found the same aim, so the accumulate mechanism is untested; VOID for its purpose"

# 4. the bank holds more than one DISTINCT centre. Size could in principle grow on
#    a differing width alone, which would not test aim memory.
_NCENTRES=$(grep -o '\[AIM-BANK\].*centres=\[[^]]*\]' "$L" | tail -1 \
            | sed -E 's/.*centres=\[([^]]*)\].*/\1/' | tr ',' '\n' | sort -u | grep -c .)
[ "${_NCENTRES:-0}" -ge 2 ] \
  || fail "bank holds only $_NCENTRES distinct centre(s) — the aim did not change, so aim memory is untested"

# 5. the aug fired on real batches
grep -Pq '\[INSTR-HEAD-AUG\] ep\d+ applied to [1-9][0-9]* batches' "$L" \
  || fail "head aug never fired on a real batch"

# 6. generator teaching severed
grep -Fq '[FOLD-IN] HEAD-ONLY mode ON' "$L" || fail "HEAD-ONLY banner missing"
grep -Fq 's0 replay loader' "$L" && fail "s0 replay fired — generator being taught"
grep -Fq '[CONSOL SESSION]' "$L" && fail "consolidation session fired — generator being taught"

# 7. completed
grep -Fq 'Restored best' "$L" || fail "smoke did not reach final restore"

echo "A26_BANNER_OK (bank reached size $_MAX, $_NCENTRES distinct centres)"
