"""The n = 300 comparison figure of the paper, drawn from the stored runs.

    (a) good proposals per run W, one dot per seed and a line at the mean;
    (b) W' = W x D, good proposals with near-duplicates counted once (D = the share of a run's visits kept
        by the greedy near-duplicate pass);
    (c) seed-paired differences in W at depth L = 127, first-named method minus second, in per cent of the
        second's mean: per-seed differences, the mean and its 95 % paired t interval.

Rows are grouped by how the sampled errors are seeded (one stream per code, or common random numbers). The
script prints every mean and interval it draws, then writes results/fig-main300.pdf. Reads data/n300 only;
no oracle call; about 82 s on a laptop. Run from the repo root:
    python -m readouts.fig_main300
"""
import numpy as np

import matplotlib
matplotlib.use("pdf")              # vector output only, never a window
# LaTeX fonts (Computer Modern, the paper's text font) for every label
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
                            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}'})

# matplotlib 3.11.0's Type1Font.subset overwrites encoding slot 0, which is Computer Modern's minus sign under
# usetex; this copy keeps slot 0.
import matplotlib._type1font as _type1font


def _subset_keep_slot0(self, characters, name_prefix):
    characters = frozenset(characters)
    encoding = {code: glyph for code, glyph in self.prop['Encoding'].items() if code in characters}
    encoding.setdefault(0, '.notdef')                  # the fix (was: encoding[0] = '.notdef')
    todo = set(encoding.values()) | {'.notdef'}        # .notdef must stay in the font program
    done = set()
    seen_subrs = {0, 1, 2, 3}
    while todo:
        glyph = todo.pop()
        called_glyphs, called_subrs = _type1font._CharstringSimulator(self).run(glyph)
        todo.update(called_glyphs - done)
        seen_subrs.update(called_subrs)
        done.add(glyph)
    charstrings = self._subset_charstrings(done)
    subrs = self._subset_subrs(seen_subrs)
    newparts = self._replace(
        [(x, f'/FontName /{name_prefix}{self.prop["FontName"]} def') for x in self._pos['FontName']]
        + [(self._pos['CharStrings'][0], charstrings),
           (self._pos['Subrs'][0], subrs),
           (self._pos['Encoding'][0], self._postscript_encoding(encoding))]
        + [(x, '') for x in self._pos.get('UniqueID', [])])
    return type(self)((newparts[0], self._encrypt(newparts[1], 'eexec'), self.parts[2]))


_type1font.Type1Font.subset = _subset_keep_slot0

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms        # headers placed in figure x, data y

from gflowqec import counts, stats
from gflowqec.paths import OUT, ROOT

# ----------------------------------------------------------------------------------------------- style
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
# Okabe-Ito, one colour per method family in every figure
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GREY = "#999999"                   # neutral grey for the two methods outside the eight colours
COLUMN = 3.4                       # REVTeX column width, inches


def offsets(n, width):
    """Evenly spread offsets so that n points on one row or column do not overlap."""
    return [(i - (n - 1) / 2) * width / max(n - 1, 1) for i in range(n)]


def save(fig, name):
    path = OUT / name
    OUT.mkdir(exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


MODE_NAME = {"code": "one stream per code", "crn": "common random numbers"}

# (cell of counts.CELLS, row name, colour, seeding mode)
MAIN300 = [
    ("d127", "GFlowQEC-online, $L = 127$", VERMILLION, "code"),
    ("u127", "Local search alone, $L = 127$", ORANGE, "code"),
    ("pt", "Tuned PT", BLUE, "code"),
    ("d31", "GFlowQEC-online, $L = 31$", VERMILLION, "code"),
    ("me", "MAP-Elites", GREEN, "code"),
    ("ppo1", "PPO, entropy weight $0.1$", YELLOW, "code"),
    ("ss300", "Supervised screen", PURPLE, "code"),
    ("rnd", "Uniform sampling", BLACK, "code"),
    ("c127", "GFlowQEC-online, $L = 127$", VERMILLION, "crn"),
    ("ptc", "Tuned PT", BLUE, "crn"),
    ("bo40k", "Bayesian optimisation", GREY, "crn"),
]
# The comparisons of panel (c), each the first-named method minus the second, both at L = 127 and in the
# same seeding mode.
MAIN300_PAIRS = [("d127", "pt"), ("d127", "ss300"), ("d127", "u127"), ("u127", "pt"), ("c127", "ptc")]
# Short names for the panel (c) labels; every comparison is at L = 127, which the caption says once.
SHORT_300 = {"d127": "GFlowQEC-online", "c127": "GFlowQEC-online", "u127": "local search alone", "pt": "tuned PT",
             "ptc": "tuned PT", "ss300": "supervised screen"}


def plain(label):
    """A row name without the LaTeX dollar signs, for the printed lines."""
    return label.replace("$", "")


def fig_main300():
    # The five reference cells fix y* at each seed, so they are loaded with the eleven drawn cells.
    cells = list(counts.REFERENCE_CELLS) + [r[0] for r in MAIN300 if r[0] not in counts.REFERENCE_CELLS]
    runs = counts.load_visited(cells)              # runs[cell][seed] = the visited file of that run
    q = {s: counts.y_star(runs, s) for s in counts.SEEDS}
    drawn = [r[0] for r in MAIN300]
    # W per drawn cell and seed, D from one greedy near-duplicate pass over the run's visits
    Y = {c: np.array([counts.good_count(runs[c][s], q[s]) for s in counts.SEEDS]) for c in drawn}
    D = {c: np.array([counts.d_visit(counts.source_path(runs[c][s])) for s in counts.SEEDS]) for c in drawn}
    Yp = {c: Y[c] * D[c] for c in drawn}           # good proposals with near-duplicates counted once, per seed
    # (c): the seed-paired comparisons; a cell's seeding mode is fixed by its name (MAIN300)
    mode_of = {cell: mode for cell, _, _, mode in MAIN300}
    stats_of = {}
    for a, b in MAIN300_PAIRS:
        pct, lo, hi, per_seed, _, _ = stats.paired_t(Y[a], Y[b])
        stats_of[(a, b)] = (mode_of[a], pct, lo, hi, per_seed)

    # The printed values: (a) and (b) means in the order the figure draws them, then the (c) intervals
    print("Fig. E.2, n = 300, means over seeds 0-5: (a) good proposals per run | (b) near-duplicates counted once")
    for mode in ("code", "crn"):
        print(MODE_NAME[mode])
        for cell, label, _, _ in sorted([r for r in MAIN300 if r[3] == mode], key=lambda r: -Y[r[0]].mean()):
            print(f"{plain(label)} | {Y[cell].mean():,.0f} | {Yp[cell].mean():,.0f}")
    print("(c) first minus second at L = 127, per cent of the second's mean, 95 per cent paired t interval")
    for mode in ("code", "crn"):
        print(MODE_NAME[mode])
        for a, b in MAIN300_PAIRS:
            m_, m, l, h, _ = stats_of[(a, b)]
            if m_ == mode:
                name = f"{SHORT_300[a]} - {SHORT_300[b]}"
                print(f"{name[0].upper() + name[1:]} | {m:.1f} [{l:.1f}, {h:.1f}]")
    # Results at n = 300: the depth L = 127 was set after GFlowQEC-online at L = 31 trailed tuned PT on good proposals;
    # the count of seeds where it trailed, from the same per-seed W as panel (a)
    trailed = int(np.sum(Y["d31"] < Y["pt"]))
    print(f"basis: per-seed W, L = 31 {Y['d31'].tolist()} against tuned PT {Y['pt'].tolist()}")
    print(f"App. E.4: GFlowQEC-online, L = 31, trailed tuned PT on good proposals in {trailed} of {len(Y['pt'])} seeds")

    # Layout in inches. The row names take a left gutter; (a) and (b) sit side by side above (c) and share the
    # rows and the x scale, so a method's drop from (a) to (b) reads directly.
    W, H = COLUMN, 4.47
    gutter, right, gap = 1.68, 0.07, 0.10
    half = (W - gutter - right - gap) / 2          # width of (a) and of (b), 0.775 in
    top_bottom, top_height = 2.60, 1.67            # (a) and (b): thirteen rows of 0.12 in (two headers, eleven methods)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((gutter / W, top_bottom / H, half / W, top_height / H))
    bx = fig.add_axes(((gutter + half + gap) / W, top_bottom / H, half / W, top_height / H), sharey=ax)
    cx = fig.add_axes((gutter / W, 0.36 / H, (W - gutter - right) / W, 1.50 / H))

    def header(axes, y, mode):
        """The seeding-mode header, in italics at the left edge of the figure, on its own row."""
        trans = matplotlib.transforms.blended_transform_factory(fig.transFigure, axes.transData)
        axes.text(0.03 / W, y, r"\textit{" + MODE_NAME[mode] + "}", transform=trans, va="center", ha="left")

    def tag(axes, text, x_inch):
        """A bold panel tag just above the axes, at a fixed distance from the figure's left edge."""
        trans = matplotlib.transforms.blended_transform_factory(fig.transFigure, axes.transAxes)
        axes.text(x_inch / W, 1.0, r"\textbf{" + text + "}", transform=trans, ha="left", va="bottom")

    # (a) and (b): a header row per seeding mode, then its methods sorted by mean good proposals, highest first
    ticks, labels, y = [], [], 0.0
    for mode in ("code", "crn"):
        rows = sorted([r for r in MAIN300 if r[3] == mode], key=lambda r: -Y[r[0]].mean())
        header(ax, y, mode)
        y -= 1
        for cell, label, colour, _ in rows:
            edge = BLACK if colour == YELLOW else colour       # yellow dots get a black rim
            for axes, values in ((ax, Y[cell]), (bx, Yp[cell])):
                for s, dy in enumerate(offsets(6, 0.5)):       # one dot per seed, spread so none hides another
                    axes.plot(values[s], y + dy, "o", ms=2.6, mfc=colour, mec=edge, mew=0.4, alpha=0.9, zorder=3)
                axes.plot([values.mean()] * 2, [y - 0.38, y + 0.38], color=BLACK, lw=1.1, zorder=4)   # the mean
            ticks.append(y)
            labels.append(label)
            y -= 1
        y -= 0.3
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y + 0.6, 0.6)
    bx.tick_params(axis="y", labelleft=False)
    for axes in (ax, bx):
        axes.tick_params(axis="y", length=0)
        axes.set_xlim(0, 26500)
        axes.set_xticks([0, 20000])                    # two labels only: three collide at 0.79 in
        axes.set_xticks([5000, 10000, 15000, 25000], minor=True)
        axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axes.grid(axis="x", which="both", lw=0.3, color="#dddddd", zorder=0)
    ax.set_xlabel("Good proposals\nper run", labelpad=2)
    bx.set_xlabel("Good proposals,\nnear-duplicates\ncounted once", labelpad=2)
    tag(ax, "(a)", 0.03)
    tag(bx, "(b)", gutter + half + gap)

    # (c) per comparison: the six per-seed differences (circles) above, the mean (diamond) on its
    # 95 % paired t interval (bar) below, in per cent of the second method's mean
    ticks, labels, y = [], [], 0.0
    for mode in ("code", "crn"):
        header(cx, y, mode)
        y -= 1
        for a, b in MAIN300_PAIRS:
            m_, m, l, h, per_seed = stats_of[(a, b)]
            if m_ != mode:
                continue
            colour = ORANGE if a == "u127" else VERMILLION
            two_line = (a, b) in (("d127", "u127"), ("u127", "pt"), ("d127", "ss300"))   # the names wider than the gutter
            if two_line:
                y -= 0.5
            for s, dy in enumerate(offsets(6, 0.24)):
                cx.plot(per_seed[s], y + 0.2 + dy, "o", ms=2.2, mfc="white", mec=colour, mew=0.6, zorder=3)
            cx.plot([l, h], [y - 0.25] * 2, color=colour, lw=1.3, zorder=3)
            cx.plot(m, y - 0.25, "D", ms=3.6, mfc=colour, mec=BLACK, mew=0.4, zorder=4)
            ticks.append(y)
            name = f"{SHORT_300[a]} − {SHORT_300[b]}"      # "first minus second", a true minus sign
            if two_line:
                name = name.replace(" − ", " −\n")
            labels.append(name[0].upper() + name[1:])
            y -= 1.5 if two_line else 1
        y -= 0.3
    cx.axvline(0, color=BLACK, lw=0.6, ls="--", zorder=1)
    cx.set_yticks(ticks, labels)
    cx.tick_params(axis="y", length=0)
    cx.set_ylim(y + 0.6, 0.6)
    cx.set_xlim(-8, 20)
    cx.set_xticks([-5, 0, 5, 10, 15, 20])
    cx.set_xlabel("Difference in good proposals (\\% of the second method's mean)", ha="right", x=1.0, labelpad=2)
    cx.grid(axis="x", lw=0.3, color="#dddddd", zorder=0)
    tag(cx, "(c)", 0.03)
    save(fig, "fig-main300.pdf")


if __name__ == "__main__":
    fig_main300()
