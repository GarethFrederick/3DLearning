"""Print-size gate for thesis figures.

finalize() saves a figure at EXACTLY its build size (no tight bbox, which trims
the page and so changes the printed size) and refuses to save when:
  - the figure is not the requested w x h cm,
  - any drawn, non-empty text is below MIN_PT points,
  - any drawn text reaches outside the page,
  - a text named in `inside` is not inside its box (schematics),
and after saving re-reads the PDF MediaBox and checks the page size.
The PDF is 1:1 with print, so the pt sizes checked here are the printed sizes
when LaTeX includes the PDF at its natural width (15 cm).

`python figcheck.py` runs the negative tests: each failure branch must fire.
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text

MIN_PT = 9.0
CM = 1 / 2.54


def _drawn_texts(fig):
    """Texts the renderer will actually draw: visible, non-empty, and for tick
    labels only those on ticks inside the view (matplotlib keeps spare ones)."""
    fig.canvas.draw()
    out = []
    ticklabels = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                for lab in (tick.label1, tick.label2):
                    ticklabels.add(id(lab))
                    loc = tick.get_loc()
                    if (loc is not None and lab.get_visible() and lab.get_text().strip()
                            and lo - 1e-9 <= loc <= hi + 1e-9):
                        out.append(lab)
    for t in fig.findobj(Text):
        if id(t) in ticklabels:
            continue
        if t.get_visible() and t.get_text().strip():
            out.append(t)
    return out


def check(fig, w_cm, h_cm, inside=()):
    """-> list of problems (empty = pass)."""
    probs = []
    W, H = fig.get_size_inches()
    if abs(W - w_cm * CM) > 1e-6 or abs(H - h_cm * CM) > 1e-6:
        probs.append(f'size {W / CM:.2f} x {H / CM:.2f} cm, want {w_cm} x {h_cm}')
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    for t in _drawn_texts(fig):
        fs = t.get_fontsize()
        if fs < MIN_PT - 1e-9:
            probs.append(f'{fs:.2f} pt < {MIN_PT}: {t.get_text()[:40]!r}')
        e = t.get_window_extent(r)
        if (e.x0 < fb.x0 - 0.5 or e.y0 < fb.y0 - 0.5 or e.x1 > fb.x1 + 0.5
                or e.y1 > fb.y1 + 0.5):
            probs.append(f'off the page: {t.get_text()[:40]!r}')
    for t, patch in inside:
        e, b = t.get_window_extent(r), patch.get_window_extent(r)
        if e.x0 < b.x0 or e.x1 > b.x1 or e.y0 < b.y0 or e.y1 > b.y1:
            probs.append(f'text overflows its box: {t.get_text()[:40]!r}')
    return probs


def pdf_size_cm(path):
    raw = open(path, 'rb').read().decode('latin-1')
    m = re.search(r'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]', raw)
    if not m:
        raise AssertionError(f'no MediaBox in {path}')
    x0, y0, x1, y1 = map(float, m.groups())
    return (x1 - x0) / 72 * 2.54, (y1 - y0) / 72 * 2.54


def finalize(fig, path_noext, w_cm, h_cm, inside=()):
    probs = check(fig, w_cm, h_cm, inside)
    if probs:
        raise AssertionError(f'{path_noext}: ' + '; '.join(probs))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_noext}.{ext}', bbox_inches=None, dpi=300)
    pw, ph = pdf_size_cm(f'{path_noext}.pdf')
    if abs(pw - w_cm) > 0.01 or abs(ph - h_cm) > 0.01:
        raise AssertionError(f'{path_noext}.pdf page is {pw:.2f} x {ph:.2f} cm, want {w_cm} x {h_cm}')
    plt.close(fig)
    print(f'  wrote {path_noext}.pdf  page {pw:.2f} x {ph:.2f} cm  text >= {MIN_PT} pt  OK',
          flush=True)


if __name__ == '__main__':
    # Negative tests: every failure branch must fire on a known-bad figure.
    import tempfile
    from matplotlib.patches import Rectangle

    def fires(make, what):
        fig = make()
        p = check(fig, 15, 6)
        plt.close(fig)
        assert any(what in s for s in p), f'checker did NOT fire on {what!r}: {p}'
        print(f'  negative test fired: {what!r}')

    def small():
        f, a = plt.subplots(figsize=(15 * CM, 6 * CM))
        a.text(0.5, 0.5, 'tiny', fontsize=8)
        return f

    def off():
        f, a = plt.subplots(figsize=(15 * CM, 6 * CM))
        a.text(1.3, 0.5, 'outside', fontsize=9, transform=a.transAxes)
        return f

    def size():
        f, a = plt.subplots(figsize=(16 * CM, 6 * CM))
        return f

    def tick():
        f, a = plt.subplots(figsize=(15 * CM, 6 * CM))
        a.tick_params(labelsize=7)
        return f
    plt.rcParams.update({'font.size': 9})
    fires(small, '< 9.0')
    fires(off, 'off the page')
    fires(size, 'size')
    fires(tick, '< 9.0')
    f, a = plt.subplots(figsize=(15 * CM, 6 * CM))
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    box = Rectangle((0.4, 0.4), 0.05, 0.05)
    a.add_patch(box)
    t = a.text(0.42, 0.42, 'a long text that cannot fit', fontsize=9)
    p = check(f, 15, 6, inside=[(t, box)])
    assert any('overflows' in s for s in p), p
    print("  negative test fired: 'overflows'")
    plt.close(f)
    # and a clean figure must pass, and finalize must write an exact page
    f, a = plt.subplots(figsize=(15 * CM, 6 * CM), constrained_layout=True)
    a.plot([0, 1], [0, 1])
    a.set_xlabel('x')
    p = check(f, 15, 6)
    assert not p, p
    d = tempfile.mkdtemp()
    finalize(f, f'{d}/clean', 15, 6)
    print('FIGCHECK_SELFTEST_OK')
