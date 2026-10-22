#!/bin/bash
# Gate for a80_fixed120. Usage: gate_a80_fixed120.sh <smoke.log> <a72-ref.log>
# banner_recon_check.sh masks digits, so the cap is asserted directly: the
# per-round augmentation line must read alpha<=$120 with no TREND-DEPTH, or the
# arm is a silent copy of a72. Only [INSTR-HEAD-AUG] lines may differ from a72.
set -u
SM="${1:?usage: gate_a80_fixed120.sh <smoke.log> <ref.log>}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
REF="${2:?usage: gate_a80_fixed120.sh <smoke.log> <ref.log>}"
cd "$REPO_ROOT" || exit 1
[ -s "$SM" ] || { echo "A80_GATE_FAIL: smoke log $SM missing or empty"; exit 1; }
[ -s "$REF" ] || { echo "A80_GATE_FAIL: reference log $REF missing"; exit 1; }
FAIL=0

# 1. The cap is $120 in every round and the trend depth never ran.
N_AUG=$(grep -c '\[INSTR-HEAD-AUG\] aim centre=' "$SM")
N_120=$(grep '\[INSTR-HEAD-AUG\] aim centre=' "$SM" | grep -cE 'alpha<=\$120(\.0+)? ')
if [ "$N_AUG" -gt 0 ] && [ "$N_AUG" -eq "$N_120" ] && ! grep -q 'TREND-DEPTH' "$SM"; then
  echo "  A80 cap: alpha<=\$120 in $N_AUG of $N_AUG rounds, no TREND-DEPTH  OK"
else
  echo "  A80_GATE_FAIL: cap lines $N_120 of $N_AUG at \$120, or TREND-DEPTH present."
  grep -m2 -oE 'alpha<=\$[0-9.]+|TREND-DEPTH[^|]*' "$SM"
  FAIL=1
fi

# 2. The aim is block 3 (12-16 h).
if grep '\[INSTR-HEAD-AUG\] aim centre=' "$SM" | grep -qv 'aim centre=14.0h'; then
  echo "  A80_GATE_FAIL: an aim is not 14.0h"; FAIL=1
else
  echo "  A80 aim: 14.0h every round  OK"
fi

# 3. The head is still wide.
if grep -q 'Head range on warm-start | \[-0.5, 1.5\] (env override)' "$SM"; then
  echo "  A80 head range: [-0.5, 1.5]  OK"
else
  echo "  A80_GATE_FAIL: the wide head is missing."; FAIL=1
fi

# 4. Every other mechanism matches a72.
OUT=$(scripts/arms/gates/banner_recon_check.sh "$SM" "$REF" 2>&1); RC=$?
if [ $RC -eq 0 ]; then
  echo "  A80 mechanism: identical to $(basename "$REF")  OK"
else
  BAD=$(echo "$OUT" | grep -E '^[<>]' | grep -v 'INSTR-HEAD-AUG' | grep -vc 'TREND-DEPTH' || true)
  if [ "${BAD:-1}" -eq 0 ]; then
    echo "  A80 mechanism: differs ONLY on the augmentation / trend-depth lines  OK"
    echo "$OUT" | grep -E '^[<>]' | sed 's/^/      /'
  else
    echo "  A80_GATE_FAIL: the recipe differs from a72 beyond the augmentation lines."
    echo "$OUT" | sed 's/^/      /'
    FAIL=1
  fi
fi

[ $FAIL -eq 0 ] && { echo "######## A80_GATE_PASS ########"; exit 0; }
echo "######## A80_GATE_FAIL ########"; exit 1
