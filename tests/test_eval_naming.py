"""Dry-run the eval tooling's run naming on EXISTING artifacts.

poc_eval_ood.find_selected_epoch matches a run by searching each slurm log for
'_gt_<arm>_s<seed>'. A void arm and its _r2 replacement must never match each
other's logs or checkpoints, or the void run is scored under the registered name.

Calls the REAL poc_eval_ood functions, never a copy. No solver, no checkpoints
loaded. Exits 1 on any failed check.
"""
import os
import sys
from pathlib import Path

ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[1])
os.chdir(ROOT)
os.environ['POC_RUN_PREFIX'] = 'gt'
sys.path.insert(0, ROOT)

import poc_eval_ood as P

fails = []


def check(label, got, want):
    ok = got == want
    print(f'  | {"PASS" if ok else "FAIL"}  {label}: got {got!r}', flush=True)
    if not ok:
        fails.append(f'{label}: got {got!r}, want {want!r}')


print(f'  | RUN_PREFIX={P.RUN_PREFIX}  arms resolved={len(P.ARMS)}', flush=True)

# 1. every registered _r2 arm is in the sweep list. A missing arm is a SILENT
#    omission: the sweep exits 0 and the row never appears.
print('\n  | 1. the new arms are in the sweep list', flush=True)
for a in ('a22_nogendepth_r2', 'a23_zerodose_r2', 'a24_trendaim_r2',
          'a25_recency_r2', 'a26_aimbank_r2', 'a27_trenddepth_r2'):
    check(f'{a} present', a in P.ARMS, True)

# 2. basenames are distinct between the void run and its replacement.
print('\n  | 2. void and replacement basenames differ', flush=True)
for a in ('a22_nogendepth', 'a24_trendaim', 'a25_recency', 'a26_aimbank'):
    b_void = P.run_basename(a, 1)
    b_r2 = P.run_basename(a + '_r2', 1)
    check(f'{a}: void != r2', b_void != b_r2, True)
    # The dangerous direction: the void arm's search string must NOT sit inside
    # the replacement's. '_gt_a24_trendaim_s1' inside '_gt_a24_trendaim_r2_s1'
    # would make the void log answer for the registered arm.
    check(f'{a}: void suffix not a substring of r2 suffix',
          f'_gt_{a}_s1' in f'_gt_{a}_r2_s1', False)
    check(f'{a}: r2 suffix not a substring of void suffix',
          f'_gt_{a}_r2_s1' in f'_gt_{a}_s1', False)

# 3. the checkpoint glob still finds a19, which has 9 saved epochs on disk. If
#    this returns [] the whole eval silently skips every arm.
print('\n  | 3. the checkpoint finder still works on a19', flush=True)
eps = P.saved_ml_epochs('a19_discwidth', 1)
print(f'  | a19_discwidth s1 saved epochs = {eps}', flush=True)
check('a19 s1 has saved epochs', len(eps) > 0, True)

# 4. the selected-epoch resolver reads a19 s1's log and returns epoch 1, which is
#    the epoch its log actually prints ("Restored best ... from outer epoch 1").
print('\n  | 4. the selected-epoch resolver agrees with the log', flush=True)
sel, src = P.find_selected_epoch('a19_discwidth', 1, eps)
print(f'  | a19_discwidth s1 selected epoch = {sel}  source = {src}', flush=True)
check('a19 s1 selected epoch is 1', sel, 1)
check('a19 s1 resolved from the log, not the fallback', 'fallback' not in src, True)

# 5. The finder must not borrow another arm's checkpoints for a name that
#    never ran; a99_neverran can never exist, so it tests that forever.
print('\n  | 5. an arm that never ran returns empty rather than borrowing', flush=True)
check('a99_neverran returns no checkpoints', P.saved_ml_epochs('a99_neverran', 1), [])
check('a23_zerodose_r2 HAS run and returns its own checkpoints',
      len(P.saved_ml_epochs('a23_zerodose_r2', 1)) > 0, True)

# 6. The void name and the registered _r2 name are SEPARATE name spaces; a
#    prefix-matching resolver would return the same list for both.
print('\n  | 6. the void name and the registered name stay separate', flush=True)
void_eps = P.saved_ml_epochs('a24_trendaim', 1)
r2_eps = P.saved_ml_epochs('a24_trendaim_r2', 1)
print(f'  | a24_trendaim (void) epochs = {void_eps}   '
      f'a24_trendaim_r2 epochs = {r2_eps}', flush=True)
check('the two names do not resolve to the same list', void_eps == r2_eps, False)
check('the registered _r2 arm has its own checkpoints', len(r2_eps) > 0, True)

print('\n' + '=' * 70, flush=True)
if fails:
    print(f'  | {len(fails)} CHECK(S) FAILED — fix before the runs land:', flush=True)
    for f in fails:
        print(f'  |   {f}', flush=True)
    sys.exit(1)
print('  | ALL CHECKS PASSED — the eval tooling is ready for the _r2 runs', flush=True)
