"""Write results/thesis/tables/t-seed-stats.tex: per configuration and era,
n/min/max/median/mean/SD of norm_regret from results/scores/poc_eval_measure/
*_selected.json, rows named and ordered as results/thesis/tables/t2-main.tex.
It asserts each row's seed count and, for the REF rows, the seed mean (to 5e-5)
against the t2-main.tex values, so this table cannot drift from the headline one.
Also prints results/analysis/schedule_table.json for the schedule.tex check."""
import glob, json
import os, sys
from pathlib import Path
import numpy as np
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
EVAL = f'{ROOT}/results/scores/poc_eval_measure'
ERAS = [('oodtest_2023_2025_renewables', 'Renewables'), ('oodtest_2021_2022_gas', 'Gas')]
ROWS = [('Wide output range', None, None),
        (r'Trend-aim', 'a72_nogenwide', 20),
        (r'\quad generator on, 8 rounds', 'a70_a34wide', 3),
        (r'\quad block width 2\,h (12--14\,h)', 'a78_bw2', 3),
        (r'\quad block width 6\,h (12--18\,h)', 'a78_bw6', 3),
        (r'\quad block width 8\,h (08--16\,h)', 'a78_bw8', 3),
        (r'\quad real-window reference', 'a79_realref', 3),
        (r'3D-Learning', 'a71_a1wide', 5),
        (r'\quad wrapper off', 'a73_vanilla3dl', 3),
        (r'Zero discount', 'a75_wzerodose', 5),
        (r'Discount aimed at a random block', 'a74_randomaim', 5),
        (r'Discount aimed at 16--20\,h', 'a76_block4', 1),
        ('Narrow output range', None, None),
        (r'Trend-aim', 'a46_nogen30', 3),
        (r'\quad generator on', 'a44_long30', 3),
        (r'\quad generator on, 8 rounds', 'a34_honestdose', 5),
        (r'\quad as above, WDRO warm start', 'a59_drospo', 3),
        (r'\quad observed depth (\$24)', 'a77_obsdepth', 3),
        (r'3D-Learning', 'a1_stall', 5),
        (r'Divergence-aimed discount, \$120', 'a19_discwidth', 3),
        (r'No discount, all training years', 'a23_zerodose_r2', 3),
        (r'No discount, 2019--2020 only', 'a25_recency_r2', 3)]
REF = {('a72_nogenwide', 0): 0.4438, ('a72_nogenwide', 1): 0.3760, ('a71_a1wide', 0): 0.8884,
       ('a71_a1wide', 1): 0.6112, ('a75_wzerodose', 0): 0.7125, ('a75_wzerodose', 1): 0.4893,
       ('a46_nogen30', 0): 0.5495, ('a46_nogen30', 1): 0.4296, ('a1_stall', 0): 0.9383,
       ('a1_stall', 1): 0.6606}


def load(arm, era):
    return [float(json.load(open(f))['norm_regret'])
            for f in sorted(glob.glob(f'{EVAL}/{arm}_s*_{era}_selected.json'))]


L = [r'\begin{table}[H]', r'\centering\footnotesize\setstretch{1}', r'\setlength{\tabcolsep}{3pt}',
     r'\caption{Normalised regret over seeds.}', r'\label{tab:seedstats}',
     r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.2cm} l c r r r r r@{}}', r'\toprule',
     r'Configuration & Era & Seeds & Min & Max & Median & Mean & SD \\', r'\midrule']
first = True
for name, arm, nexp in ROWS:
    if arm is None:
        if not first:
            L.append(r'\addlinespace')
        L.append(r'\multicolumn{8}{@{}l}{\textit{' + name + r'}}\\')
        first = False
        continue
    for k, (era, en) in enumerate(ERAS):
        v = np.array(load(arm, era))
        assert len(v) == nexp, f'{arm} {era}: {len(v)} seeds, t2-main says {nexp}'
        if (arm, k) in REF:
            assert abs(v.mean() - REF[(arm, k)]) < 5e-5, f'REF MISMATCH {arm} {era} {v.mean():.4f}'
        if len(v) == 1:
            cells = f'\\multicolumn{{4}}{{c}}{{{v[0]:.4f} (single run)}} & ---'
        else:
            cells = (f'{v.min():.4f} & {v.max():.4f} & {np.median(v):.4f} & {v.mean():.4f} & '
                     f'{v.std(ddof=1):.4f}')
        L.append(f'{name if k == 0 else ""} & {en} & {len(v)} & {cells} \\\\')
        print(f'  {name:42s} {en:10s} n={len(v):2d} min={v.min():.4f} max={v.max():.4f} '
              f'median={np.median(v):.4f} mean={v.mean():.4f} '
              f'sd={(v.std(ddof=1) if len(v) > 1 else float("nan")):.4f}')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{ROOT}/results/thesis/tables/t-seed-stats.tex', 'w').write('\n'.join(L) + '\n')
print('  wrote results/thesis/tables/t-seed-stats.tex (reference values asserted)')

print('\n== schedule_table.json (renewables part B) ==')
S = json.load(open(f'{ROOT}/results/analysis/schedule_table.json'))
for reg, d in S.items():
    B = d['B']
    print(f'  {reg}: start {B["start"]} window {B["window_index"]}')
    print('    true price  ' + ' '.join(f'{x:8.2f}' for x in B['true_price_block']))
    for lab, v in B['forecast_block'].items():
        print(f'    fc {lab:9s}' + ' '.join(f'{x:8.2f}' for x in v))
    for lab, m in B['mip'].items():
        print(f'    MWh {lab:9s}' + ' '.join(f'{x:8.0f}' for x in m['energy_block'])
              + f'   cost ${m["cost_true"]:,.0f}' + (f'  regret ${m["regret"]:,.0f}' if 'regret' in m else ''))
