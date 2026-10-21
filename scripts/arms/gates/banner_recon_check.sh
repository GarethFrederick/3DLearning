#!/bin/bash
# Gates a reconstructed wide-head recipe against the original run's own log.
# banner_match.sh reads only "[FOLD-IN]" lines, so it passes a rebuild that drops
# INSTR_HEAD_AIM_TREND; this gate compares the full set of mechanism tags.
# Digits are stripped before the compare: row counts, dollars, gains and block
# indices move with seed and round count, and a 2-round smoke would false-fail.
# It cannot check MV_DRO_ITERS (a smoke runs 2 by design; check the dose in the
# seed job's log) or the tuned aim values, which it prints but does not assert.
set -u
SM="${1:?usage: banner_recon_check.sh <smoke.log> <reference.log>}"
REF="${2:?usage: banner_recon_check.sh <smoke.log> <reference.log>}"
[ -s "$SM" ]  || { echo "RECON_FAIL: no smoke log $SM"; exit 1; }
[ -s "$REF" ] || { echo "RECON_FAIL: no reference log $REF"; exit 1; }

fingerprint() {
  # Three lines are excluded:
  #  [FOLD-IN][SMOKE] - printed only by a smoke run.
  #  [CONSOL]         - its ON/OFF text was a print-only bug for a34-family arms,
  #                     fixed mid-campaign; runs either side of the fix disagree
  #                     on the text while running identical code.
  #  aim centre=      - the instructor derives it from data, and a 2-round smoke
  #                     can land on another block. It is an outcome, not a knob.
  grep -ohE '\[(FOLD-IN|NO-GEN|HEAD-REAL|INSTR-REAL-WIN|HEAD-ROWS|INSTR-HEAD-AUG)\](\[[A-Z_]+\])?[^|]*' "$1" \
    | grep -vE '\[INSTR-HEAD-AUG\] (ep[0-9]|aim centre=)' \
    | grep -vE '\[FOLD-IN\]\[(SMOKE|CONSOL)\]' \
    | sed -E 's/[0-9]+\.[0-9]+/#/g; s/[0-9]+/#/g; s/ +/ /g; s/ $//' \
    | sort -u
}
# The [HEAD-REAL] row count is compared exactly, before the digit strip: it names
# the CSV the head trained on and does not move with seed or round count. Without
# it, a rebuild that trained an oracle arm on past data passed this gate.
hr() { grep -m1 -oE '\[HEAD-REAL\] loaded [0-9]+ real' "$1" | grep -oE '[0-9]+'; }
HR_REF=$(hr "$REF"); HR_SM=$(hr "$SM")
if [ -n "$HR_REF" ] || [ -n "$HR_SM" ]; then
  [ "$HR_REF" = "$HR_SM" ] || { echo "RECON_FAIL: the head trained on a DIFFERENT data file"; \
    echo "  reference loaded ${HR_REF:-<no line>} rows, smoke loaded ${HR_SM:-<no line>} rows"; exit 1; }
fi

diff <(fingerprint "$REF") <(fingerprint "$SM") > "${TMPDIR:-/tmp}/recon_$$.diff" 2>&1
if [ -s "${TMPDIR:-/tmp}/recon_$$.diff" ]; then
  echo "RECON_FAIL: the mechanism set differs from $REF"
  echo "  < reference-only   > smoke-only"
  cat "${TMPDIR:-/tmp}/recon_$$.diff"; rm -f "${TMPDIR:-/tmp}/recon_$$.diff"; exit 1
fi
rm -f "${TMPDIR:-/tmp}/recon_$$.diff"
echo "RECON_MECHANISM_OK: same mechanism set as $REF"
echo "--- tuned aim values, NOT asserted, read them yourself ---"
echo "  reference: $(grep -m1 'INSTR-HEAD-AUG\] ON in ML inner loop' "$REF" | sed 's/^ *| *//')"
echo "  smoke    : $(grep -m1 'INSTR-HEAD-AUG\] ON in ML inner loop' "$SM"  | sed 's/^ *| *//')"
