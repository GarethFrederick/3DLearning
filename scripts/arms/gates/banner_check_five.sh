#!/bin/bash
# Gate each arm on the EFFECT of its knob, not on the knob being set.
# Usage: banner_check_five.sh <arm> <smoke-log>
set -u
A="${1:?arm}"; LOG="${2:?log}"
[ -f "$LOG" ] || { echo "GATE_FAIL: no log $LOG"; exit 1; }
c(){ grep -c -e "$1" "$LOG" || true; }
FAIL=0
# Helper: every fixed-width arm must show the width AND prove discovery is off.
fixed_width () {   # $1 expected half-width in hours
  w=$(c "width=WIDE(±${1}h)"); d=$(c 'WIDTH-DISC=')
  echo "  | width=WIDE(±${1}h) lines=$w  WIDTH-DISC lines=$d (must be 0) |"
  grep -m1 -e 'ON in ML inner loop' "$LOG"
  [ "$w" -ge 1 ] || { echo "GATE_FAIL: the bell is not ±${1}h"; FAIL=1; }
  [ "$d" -eq 0 ] || { echo "GATE_FAIL: width discovery still ran — the width is not fixed"; FAIL=1; }
}
case "$A" in
  a55_twolobe)
     n=$(c 'ADDITIVE\]')
     k=$(grep -o 'ADDITIVE\] [0-9]* distinct aims' "$LOG" | grep -oE '[0-9]+' | head -1)
     up=$(c 'ADDITIVE\].*up'); dn=$(c 'ADDITIVE\].*down')
     echo "  | ADDITIVE lines=$n  aims=${k:-0}  lines-with-up=$up  lines-with-down=$dn |"
     grep -m1 -e 'ADDITIVE\]' "$LOG"
     [ "${k:-0}" -ge 2 ] || { echo "GATE_FAIL: only ${k:-0} aim(s)"; FAIL=1; }
     # The whole point is a CONTRAST. Two lobes pointing the same way is a34 twice.
     [ "$up" -ge 1 ] && [ "$dn" -ge 1 ] || { echo "GATE_FAIL: aims are not opposed (need one up and one down)"; FAIL=1; }
     fixed_width 4 ;;
  a56_width04) fixed_width 4 ;;
  a57_width08) fixed_width 8 ;;
  a58_width12) fixed_width 12 ;;
  a59_drospo)
     n=$(c 'warm-start | source=WassersteinDRO_SPO')
     echo "  | DRO_SPO warm-start lines=$n |"; grep -m1 -e 'ML-head warm-start' "$LOG"
     [ "$n" -ge 1 ] || { echo "GATE_FAIL: did not warm-start from WassersteinDRO_SPO"; FAIL=1; }
     [ "$(c 'Cold-start ML head')" -eq 0 ] || { echo "GATE_FAIL: fell back to a cold start"; FAIL=1; } ;;
  *) echo "GATE_FAIL: unknown arm $A"; FAIL=1 ;;
esac
# every arm must still be a34's recipe underneath
for pat in 'INSTR-REAL-WIN' 'TREND-AIM\]'; do
  [ "$(c "$pat")" -ge 1 ] || { echo "GATE_FAIL: missing $pat — not a34's recipe"; FAIL=1; }
done
# the checker's FAIL branch must be able to fire
T=$(mktemp); echo "nothing here" > "$T"
[ "$(grep -c -e 'INSTR-REAL-WIN' "$T")" -eq 0 ] || { echo "GATE_FAIL: checker inert"; FAIL=1; }
rm -f "$T"
[ "$FAIL" -eq 0 ] || exit 1
echo "GATE_OK $A"
