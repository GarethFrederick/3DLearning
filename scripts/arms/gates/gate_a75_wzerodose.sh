#!/bin/bash
# Gate for a75_wzerodose. Usage: gate_a75_wzerodose.sh <smoke.log> <a72-ref.log>
# Check 1 asserts the dose directly: banner_recon_check.sh replaces every digit
# with '#', so "alpha_max=$214.215" and "alpha_max=$0" look the same to it, and
# the obvious recipe for this arm is a silent no-op.
# Only the [INSTR-HEAD-AUG] banner may differ from a72: it carries the dose, aim
# and width, and at zero dose the width label is arbitrary. Any other differing
# line is a real recipe divergence and fails.
set -u
SM="${1:?usage: gate_a75_wzerodose.sh <smoke.log> <ref.log>}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
REF="${2:?usage: gate_a75_wzerodose.sh <smoke.log> <ref.log>}"
cd "$REPO_ROOT" || exit 1
[ -s "$SM" ] || { echo "A75_GATE_FAIL: smoke log $SM missing or empty"; exit 1; }
[ -s "$REF" ] || { echo "A75_GATE_FAIL: reference log $REF missing"; exit 1; }
FAIL=0

# 1. The dose is zero.
if grep -q 'alpha_max=\$0 ' "$SM"; then
  echo "  A75 dose: alpha_max=\$0  OK"
else
  echo "  A75_GATE_FAIL: the discount is NOT zero. The arm is a duplicate of a72."
  grep -m1 -oE 'alpha_max=\$[0-9.]+' "$SM" || echo "    (no alpha_max line at all)"
  FAIL=1
fi

# 2. THE HEAD IS STILL WIDE.
if grep -q 'Head range on warm-start | \[-0.5, 1.5\] (env override)' "$SM"; then
  echo "  A75 head range: [-0.5, 1.5]  OK"
else
  echo "  A75_GATE_FAIL: the wide head is missing."
  grep -m1 'Head range on warm-start' "$SM" || echo "    (no head range line)"
  FAIL=1
fi

# 3. EVERY OTHER MECHANISM MATCHES a72.
OUT=$(scripts/arms/gates/banner_recon_check.sh "$SM" "$REF" 2>&1); RC=$?
if [ $RC -eq 0 ]; then
  echo "  A75 mechanism: identical to $(basename "$REF")  OK"
else
  BAD=$(echo "$OUT" | grep -E '^[<>]' | grep -vc 'INSTR-HEAD-AUG' || true)
  if [ "${BAD:-1}" -eq 0 ]; then
    echo "  A75 mechanism: differs ONLY on the [INSTR-HEAD-AUG] banner  OK (allowed, §8.74 am. 1)"
    echo "$OUT" | grep -E '^[<>]' | sed 's/^/      /'
  else
    echo "  A75_GATE_FAIL: the recipe differs from a72 beyond the augmentation banner."
    echo "$OUT" | sed 's/^/      /'
    FAIL=1
  fi
fi

[ $FAIL -eq 0 ] && { echo "######## A75_GATE_PASS ########"; exit 0; }
echo "######## A75_GATE_FAIL ########"; exit 1
