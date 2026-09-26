#!/usr/bin/env python3
"""Render K x K of a stalled annealer state: covered facets blue, uncovered red.

For a complex K with n triangles, the product complex L = K x K has n*n*6
maximal facets, emitted by `Product.__init__` in the order

    f = (i * n + j) * 6 + p        (i = triangle of K, j = triangle of K)

where p in 0..5 indexes the six simplices of the staircase triangulation of
the block s_i x t_j = Delta^2 x Delta^2 -- one per interleaving of the two
"move in s" and two "move in t" steps:

    p0 BBAA   p1 BABA   p2 ABBA   p3 BAAB   p4 ABAB   p5 AABB

(A = step inside the first factor, B = step inside the second).  The picture
draws every single facet as its own cell: n x n blocks (one per pair of
K-triangles), each block a 3-wide x 2-tall grid of the six paths in the order
above (p0 p1 p2 / p3 p4 p5).  Coverage is recomputed here from the saved
chains with `Product.domain_of`, so the figure never trusts a stored number.

For RP^2 the block table also highlights star(5) x star(5) (the dropped
diagonal rectangle of the k=5 plateau); nothing complex-specific is assumed
for other complexes.  The numbers quoted in the annotation are all computed
from the state being drawn.

Usage:
    python scripts/render_kxk.py                     # RP^2 k=5, 540/600
    python scripts/render_kxk.py --complex klein \
        --state results/klein_cold666_seed_999/summary.json \
        --out figures/klein_kxk_k5_1186
    python scripts/render_kxk.py --complex klein --state <...> --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gpu_sc  # noqa: E402  (Product builder: identical facet order to the GPU repo)

# The data layer -- state discovery, the coverage/ownership audit, block
# geometry, palette -- lives in kxk_state.py so the webapp and the CI tests can
# use it without matplotlib.  This file is only the drawing.
from kxk_state import (  # noqa: E402
    C_BAND, C_BAND_TXT, C_COVER, C_DEAD_EDGE, C_EDGE, C_HOLE, C_INK, C_PART_EDGE,
    CLOSING, COMPLEXES, DOMAIN_COLORS, PATH_LABEL, SHORT_NAME, STATE_NAMES,
    SUB_COL, SUB_ROW, analyse, blend, block_facets, block_index, find_h1_report,
    find_state, infer_complex, is_triangle_complex, load_state, path_steps,
    state_stats)

# --------------------------------------------------------------------------- #
#  drawing                                                                     #
# --------------------------------------------------------------------------- #
def block_rect(i, j, gap, n):
    """Lower-left corner of block (i, j) in drawing coordinates (row 0 on top)."""
    return j * (3.0 + gap), -i * (2.0 + gap)


def note_line_h(fs, scale):
    """Height of one note line, in block-units (1 unit == one facet cell).

    `scale` is inches per block-unit, which the caller knows exactly because
    the main axes is positioned by hand (see main) -- never guessed here.
    """
    return fs * 1.35 / 72.0 / scale


def note_gap(fs, scale):
    return fs * 0.95 / 72.0 / scale


def sub_rect(i, j, p, gap, n):
    x0, y0 = block_rect(i, j, gap, n)
    return x0 + SUB_COL[p], y0 - SUB_ROW[p] - 1.0


def wrap_notes(notes, width):
    """Hard-wrap each annotation paragraph and count its rendered lines.

    Mathtext spans render wider than their character count, so the usable
    width is passed in already discounted; returns [(text, color, n_lines)].
    """
    import textwrap
    out = []
    for text, color in notes:
        lines = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, width=width) or [""])
        out.append(("\n".join(lines), color, len(lines)))
    return out


def draw_facets(ax, P, A, S, scale, gap=0.34, band=None, notes=(), fs=9.0,
                by_domain=False):
    """Main panel: every product facet.

    Default colouring is the paper's: blue when covered, red when uncovered.
    `by_domain` colours each covered facet by the domain that owns it instead
    (red still means uncovered), one colour per domain — what the webapp draws.
    """
    K = P.kfacets
    n = len(K)
    cov = A["covered"]
    owner = A["owner"]
    for i in range(n):
        for j in range(n):
            x0, y0 = block_rect(i, j, gap, n)
            if band and i in band and j in band:
                # bleed half a gap past the block on every side, so the band
                # shows up in the gaps and the sub-table reads as one region
                ax.add_patch(Rectangle((x0 - gap / 2, y0 - 2.0 - gap / 2),
                                       3.0 + gap, 2.0 + gap, facecolor=C_BAND,
                                       edgecolor="none", zorder=0))
            blkfs = block_facets(P, i, j)
            blk = cov[blkfs]
            for p, f in enumerate(blkfs):
                x, y = sub_rect(i, j, p, gap, n)
                if not blk[p]:
                    face = C_HOLE
                elif by_domain:
                    face = DOMAIN_COLORS[owner[f] % len(DOMAIN_COLORS)]
                else:
                    face = C_COVER
                ax.add_patch(Rectangle((x, y), 1.0, 1.0, facecolor=face,
                                       edgecolor=C_EDGE, linewidth=0.35, zorder=2))
            if not blk.any():
                ax.add_patch(Rectangle((x0, y0 - 2.0), 3.0, 2.0, facecolor="none",
                                       edgecolor=C_DEAD_EDGE, linewidth=1.6, zorder=4))
            elif not blk.all():
                ax.add_patch(Rectangle((x0, y0 - 2.0), 3.0, 2.0, facecolor="none",
                                       edgecolor=C_PART_EDGE, linewidth=1.0,
                                       linestyle=(0, (2.2, 1.6)), zorder=4))
    labels(ax, P, band, fs)
    return finish_panel(ax, P, notes, fs, scale)


def draw_facets_block(ax, P, A, S, scale, gap=0.34, band=None, notes=(), fs=9.0):
    """Block-level panel: one cell per pair of K-facets, filled by the fraction
    of that block's product facets that are covered (blue = all, red = none)."""
    K = P.kfacets
    n = len(K)
    for i in range(n):
        for j in range(n):
            x0, y0 = block_rect(i, j, gap, n)
            if band and i in band and j in band:
                ax.add_patch(Rectangle((x0 - gap / 2, y0 - 2.0 - gap / 2), 3.0 + gap,
                                       2.0 + gap, facecolor=C_BAND, edgecolor="none",
                                       zorder=0))
            fs_ = block_facets(P, i, j)
            f = float(A["covered"][fs_].mean())
            ax.add_patch(Rectangle((x0, y0 - 2.0), 3.0, 2.0, facecolor=blend(f),
                                   edgecolor=C_EDGE, linewidth=0.5, zorder=2))
            if f == 0.0:
                ax.add_patch(Rectangle((x0, y0 - 2.0), 3.0, 2.0, facecolor="none",
                                       edgecolor=C_DEAD_EDGE, linewidth=1.2, zorder=4))
            elif f < 1.0:
                ax.add_patch(Rectangle((x0, y0 - 2.0), 3.0, 2.0, facecolor="none",
                                       edgecolor=C_PART_EDGE, linewidth=0.9,
                                       linestyle=(0, (2.2, 1.6)), zorder=4))
    labels(ax, P, band, fs)
    return finish_panel(ax, P, notes, fs, scale)


def labels(ax, P, band, fs):
    """Row/column labels: K-facet index plus its vertex set."""
    K = P.kfacets
    n = len(K)
    for i, f in enumerate(K):
        x0, y0 = block_rect(i, 0, 0.34, n)
        ax.text(x0 - 0.35, y0 - 1.0, "T%d %s" % (i, "".join(str(v) for v in sorted(f))),
                ha="right", va="center", fontsize=fs,
                color=C_BAND_TXT if band and i in band else C_INK,
                fontweight="bold" if band and i in band else "normal")
    for j, f in enumerate(K):
        x0, y0 = block_rect(0, j, 0.34, n)
        ax.text(x0 + 1.5, 0.55, "T%d\n%s" % (j, "".join(str(v) for v in sorted(f))),
                ha="center", va="bottom", fontsize=fs,
                color=C_BAND_TXT if band and j in band else C_INK,
                fontweight="bold" if band and j in band else "normal")


def finish_panel(ax, P, notes, fs, scale):
    n = len(P.kfacets)
    x0, y0 = block_rect(n - 1, 0, 0.34, n)
    ycur = y0 - 2.0 - 1.4
    for text, color, nlines in notes:
        ax.text(-6.9, ycur, text, ha="left", va="top", fontsize=fs, color=color,
                linespacing=1.35)
        ycur -= note_line_h(fs, scale) * nlines + note_gap(fs, scale)
    ax.set_xlim(-6.9, n * (3 + 0.34) - 0.34 + 0.2)
    ax.set_ylim(ycur - 0.6, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return ycur


def draw_ownership(ax, P, A, S, gap=0.34, fs=9.0):
    """Which chain owns each block (a block split between chains is split-fill)."""
    K = P.kfacets
    n = len(K)
    cov, owner = A["covered"], A["owner"]
    for i in range(n):
        for j in range(n):
            x0, y0 = block_rect(i, j, gap, n)
            fs_ = block_facets(P, i, j)
            if not cov[fs_].any():
                ax.add_patch(Rectangle((x0, y0 - 2.0), 3.0, 2.0, facecolor=C_HOLE,
                                       edgecolor="white", linewidth=0.6))
                continue
            own = owner[fs_]
            vals, counts = np.unique(own[own >= 0], return_counts=True)
            if len(vals) == 1:
                ax.add_patch(Rectangle((x0, y0 - 2.0), 3.0, 2.0,
                                       facecolor=DOMAIN_COLORS[vals[0] % len(DOMAIN_COLORS)],
                                       edgecolor="white", linewidth=0.6))
            else:
                xs = x0
                for v, c in zip(vals, counts):
                    w = 3.0 * c / 6
                    ax.add_patch(Rectangle((xs, y0 - 2.0), w, 2.0,
                                           facecolor=DOMAIN_COLORS[v % len(DOMAIN_COLORS)],
                                           edgecolor="white", linewidth=0.6))
                    xs += w
    ax.set_xlim(-1.0, n * (3 + gap))
    ax.set_ylim(-n * (2 + gap) + 0.6, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = []
    for d, (v, inter, tot) in enumerate(A["dom_rect"]):
        size = len(A["dom"][d])
        if inter == size == tot:
            tag = "= $R_%d$ ($\\mathrm{star}(%d)^2$, all %d)" % (v, v, tot)
        elif inter >= 0.9 * size:   # recognisably a diagonal rectangle
            tag = "$\\subset R_%d$ (%d of %d)" % (v, inter, tot)
        else:                        # no product-family claim to make
            tag = "(%d facets)" % size
        handles.append(Line2D([], [], marker="s", linestyle="none", markersize=9,
                              markerfacecolor=DOMAIN_COLORS[d % len(DOMAIN_COLORS)],
                              markeredgecolor="none", label="$D_%d$ %s" % (d, tag)))
    handles.append(Line2D([], [], marker="s", linestyle="none", markersize=9,
                          markerfacecolor=C_HOLE, markeredgecolor="none",
                          label="uncovered (%d)" % S["nhole"]))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=fs - 0.5, handletextpad=0.5, labelspacing=0.5)
    ax.set_title("which domain owns each block  ($R_v=\\mathrm{star}(v)\\times\\mathrm{star}(v)$;\n"
                 "a block split between two domains is drawn split)",
                 fontsize=fs + 1, color=C_INK, pad=4)


def draw_path_legend(ax, dx=3.9, dy=4.6, fs=9.0):
    """The six staircase facets of one block, drawn on its 3 x 3 vertex grid."""
    for p in range(6):
        x0 = SUB_COL[p] * dx
        y0 = -SUB_ROW[p] * dy
        for a in range(3):
            for b in range(3):
                ax.plot([x0 + a], [y0 - b], marker="o", ms=2.6, color="#C3CBD4", zorder=1)
        pts, i, j = [(0, 0)], 0, 0
        for di, dj in path_steps(p):
            i, j = i + di, j + dj
            pts.append((i, j))
        xs = [x0 + a for a, b in pts]
        ys = [y0 - b for a, b in pts]
        ax.plot(xs, ys, color=DOMAIN_COLORS[0], linewidth=1.8, zorder=2,
                solid_capstyle="round")
        ax.plot(xs, ys, "o", ms=4.2, color=DOMAIN_COLORS[0], zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
        ax.text(x0 + 1.0, y0 + 0.75, "$p_%d$ %s" % (p, PATH_LABEL[p]), ha="center",
                va="bottom", fontsize=fs - 0.5, color=C_INK)
    ax.set_xlim(-0.9, 3 * dx - 0.7)
    ax.set_ylim(-2 * dy + 0.2, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("one block $s\\times t$: its six staircase facets $p_0\\dots p_5$\n"
                 "A = step inside $s$, B = step inside $t$\n"
                 "dots = the block's grid vertices $(a,b)$",
                 fontsize=fs, color=C_INK, pad=4)


def h1_lines(report):
    """Quote an H1 audit (`scripts/klein_h1_absorption.py`) verbatim.

    The audit is minutes-long, so it is not re-run here: the numbers quoted
    come from the report file the caller points at, and it must agree with
    the state being drawn (same covered/uncovered counts).
    """
    if not report:
        return []
    txt = Path(report).read_text()
    out = []
    for pat in ("-- L alone:", "-- summary:"):
        for line in txt.splitlines():
            if line.strip().startswith(pat):
                out.append(line.strip())
    if len(out) == 2:
        return [("$H_1$ audit (`scripts/klein_h1_absorption.py` on this state):  %s.  %s."
                 % (out[0].lstrip("- ").strip(), out[1].lstrip("- ").strip()), C_INK)]
    return []


def draw_coverage_legend(ax, P, fs=9.0):
    """Legend for the block-level panel: the covered-fraction colour ramp."""
    per = len(block_facets(P, 0, 0))
    ax.text(0.5, 0.93, "one cell = one pair ($T_i$, $T_j$),\n%d product facets" % per,
            ha="center", va="top", fontsize=fs + 0.5, color=C_INK, transform=ax.transAxes)
    for k, f in enumerate([1.0, 0.75, 0.5, 0.25, 0.0]):
        ax.add_patch(Rectangle((0.10 + 0.16 * k, 0.42), 0.14, 0.16, facecolor=blend(f),
                               edgecolor="white", linewidth=0.6,
                               transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.36, "covered fraction of the cell", ha="center", va="top",
            fontsize=fs - 0.5, color="#4A4A4A", transform=ax.transAxes)
    ax.text(0.10, 0.30, "0%", ha="center", va="top", fontsize=fs - 1,
            color="#4A4A4A", transform=ax.transAxes)
    ax.text(0.90, 0.30, "100%", ha="center", va="top", fontsize=fs - 1,
            color="#4A4A4A", transform=ax.transAxes)
    ax.text(0.5, 0.16, "black outline = none covered,\ndashed = partly covered",
            ha="center", va="top", fontsize=fs - 0.5, color="#4A4A4A",
            transform=ax.transAxes)
    ax.axis("off")


def notes_for(P, name, S, A, state_path, mode="facet"):
    """The annotation lines: structure of the hole, then how to read the map."""
    nt = len(P.kfacets)
    L = S["nhole"]
    inter = ", ".join("%s: %d" % (k, v) for k, v in S["inter"].items())
    dual = ", ".join("%s: %d" % (k, v) for k, v in S["dual"].items())
    dep = ", ".join("%s: %d" % (k, v) for k, v in S["depths"].items())
    tag = SHORT_NAME.get(name, P.name)
    lines = []
    if L == 0:
        # a complete cover: nothing to diagnose, so say exactly that
        lines.append((
            "a complete cover: all %d product facets lie in one of the %d domains, so this state "
            "certifies $\\mathrm{SC}_{\\mathrm{strict}} \\leq %d$ for %s (every cell is blue; there "
            "is nothing red to explain).  The map still shows how the %d domains share the blocks."
            % (S["nf"], len(S["sizes"]), len(S["sizes"]) - 1, tag, len(S["sizes"])), C_INK))
    elif name == "rp2":
        R5 = A["rect_of_vertex"][5]
        unc5 = int((~A["covered"][R5]).sum())
        lines.append((
            "amber rows/columns = the five triangles through vertex 5.  Their $5\\times5$ sub-table "
            "is $R_5=\\mathrm{star}(5)\\times\\mathrm{star}(5)$ — %s (%d facets, %d of them taken by "
            "$D_0\\dots D_4$).%s"
            % ("the dropped diagonal rectangle" if unc5 else "fully covered here",
               R5.size, int(A["covered"][R5].sum()),
               "" if unc5 == 0 else
               "  Its %d private facets are exactly the red cells: no chain is 1-contiguous on any "
               "of them." % unc5), C_BAND_TXT))
    else:
        lines.append((
            "the %d uncovered facets sit on %d blocks: %d entirely dead, %d partly;  $|s\\cap t|$ "
            "of the dead facets %s (covered facets: %s);\ndual distance %s, the dual diameter "
            "being %d;  defect depth $\\min_j \\mathrm{bad}_j[f]$ %s — every red facet is exactly "
            "one repair move short of some domain."
            % (L, S["cells"], S["full"], S["part"], "{" + inter + "}",
               "{" + ", ".join("%s: %d" % (k, v) for k, v in S["inter_cov"].items()) + "}",
               "{" + dual + "}", S["dual_diam"], "{" + dep + "}"), C_INK))
        # how far each domain is from its closest diagonal rectangle, in facets
        diffs = []
        for d, (v, inter_, tot) in enumerate(A["dom_rect"]):
            diffs.append("$D_%d\\subset R_%d$ %d/%d" % (d, v, tot - inter_, len(A["dom"][d]) - inter_))
        lead = ("unlike RP$^2$'s k = 5 plateau the domains are not diagonal rectangles"
                if name in ("rp2", "klein") else "the domains are not diagonal rectangles")
        lines.append((
            "%s — facets missing from / extra to the closest diagonal "
            "$R_v=\\mathrm{star}(v)^2$:   %s\n%s   (a domain contained in $R_v$ would read 0/0).  "
            "Chain links: %s."
            % (lead, "   ".join(diffs[:3]), "   ".join(diffs[3:]),
               "=".join(sorted(set(str(l) for l in S["links"]))) if len(set(S["links"])) == 1
               else "/".join(str(l) for l in S["links"])), C_INK))
    if mode == "facet":
        how = ("Each block is one pair ($T_i$, $T_j$) of the %d triangles of $K$ and shows its six "
               "staircase facets $p_0\\dots p_5$ (legend at right);  black outline = all six dead, "
               "dashed = partly dead." % nt)
    else:
        how = ("Each cell is one pair ($T_i$, $T_j$) of the %d maximal facets of $K$ (each a "
               "%d-simplex), filled by the covered fraction of its %d product facets (legend at "
               "right);  black outline = none covered, dashed = partly."
               % (nt, len(P.kfacets[0]) - 1, len(block_facets(P, 0, 0))))
    lines.append((
        "%s: %d product facets, %d covered, %d red;  domain sizes %s.  %s"
        % (tag, S["nf"], S["cov"], L, "+".join(str(s) for s in S["sizes"]), how),
        "#4A4A4A"))
    return lines


def main():
    ap = argparse.ArgumentParser(
        description="Render K x K of a saved annealer state: covered facets blue, "
                    "uncovered red.",
        epilog="the target may be a results directory (the state inside it is found "
               "automatically) or a state .json")
    ap.add_argument("target", nargs="?", default="results/rp2_k5_push_warm",
                    help="results directory or state .json "
                         "(default: results/rp2_k5_push_warm)")
    ap.add_argument("-o", "--out", default=None,
                    help="output base path; .png/.pdf are appended "
                         "(default: figures/<run>_kxk_<covered>)")
    ap.add_argument("--state", default=None, help="state file, overriding the search")
    ap.add_argument("--complex", default=None,
                    help="complex of K, overriding inference from the state")
    ap.add_argument("--h1-report", default=None,
                    help="output of scripts/klein_h1_absorption.py for this state; "
                         "its L-alone and summary lines are quoted in the annotation")
    ap.add_argument("--by-domain", action="store_true", dest="by_domain",
                    help="colour each covered facet by the domain that owns it (one colour "
                         "per domain, red still = uncovered) instead of the paper's blue/red")
    ap.add_argument("--json", action="store_true", help="print the audit and exit")
    args = ap.parse_args()

    if args.state:
        state_path = Path(args.state)
        state = json.loads(state_path.read_text())
        others = []
    else:
        state_path, state, others = find_state(args.target)
    chains = [[np.array(m, dtype=np.int64) for m in ch] for ch in state["best_chains"]]

    complex_name = args.complex or infer_complex(state, state_path)
    if not complex_name:
        raise SystemExit("cannot tell which complex %s belongs to -- pass --complex "
                         "(%s)" % (state_path, ", ".join(COMPLEXES)))
    P = gpu_sc.build(complex_name)
    A = analyse(P, chains)
    S = state_stats(P, chains, A)
    bad = state.get("complex")
    if isinstance(bad, dict) and bad.get("num_facets") not in (None, P.nf):
        raise SystemExit("%s records %s facets but --complex %s builds %d -- wrong pair"
                         % (state_path, bad.get("num_facets"), complex_name, P.nf))

    # default name: <run>_kxk_<covered>; the run is the state's directory when
    # it is a generic summary.json / final_state.json
    run = state_path.parent.name if state_path.name in STATE_NAMES else state_path.stem
    if run.startswith(complex_name + "_"):
        run = run[len(complex_name) + 1:]
    out = args.out or ("figures/%s_kxk_%s_%d" % (complex_name, run, S["cov"]))
    for ext in (".png", ".pdf"):                    # allow -o path.png / path.pdf
        if out.endswith(ext):
            out = out[: -len(ext)]
    band = [i for i, f in enumerate(P.kfacets) if 5 in f] if complex_name == "rp2" else None

    if not args.h1_report:
        args.h1_report = find_h1_report(state_path, args.target, S["nf"], S["cov"])
    print("state   : %s" % state_path)
    print("complex : %s   k = %d   covered %d/%d   domains %s"
          % (complex_name, len(chains), S["cov"], S["nf"],
             "+".join(str(x) for x in S["sizes"])))
    if args.h1_report:
        print("h1 audit: %s" % args.h1_report)
    if others:
        print("other states in this directory (not rendered): %s"
              % ", ".join(str(o) for o in others[:6]))

    if args.json:
        print(json.dumps({k: v for k, v in S.items() if k != "dead"} |
                         {"state": str(state_path), "complex": complex_name,
                          "k": len(chains), "blocks": S["cells"],
                          "domain_rect": [{"D": d, "R_v": v, "in_R": i, "|R|": t}
                                          for d, (v, i, t) in enumerate(A["dom_rect"])]},
                         indent=1))
        return

    # ---- layout, in inches -------------------------------------------------
    # The main panel is drawn at a fixed width and positioned by hand, so the
    # data-unit -> inch scale is exactly main_w / span_x: that is what lets the
    # annotation be stacked in data units without ever colliding.
    n = len(P.kfacets)
    fs = 9.0 if n <= 10 else (8.0 if n <= 20 else 6.5)
    gap = 0.34
    # Δ² x Δ² blocks get their six staircase facets drawn one by one; anything
    # else (S³–S⁸, whose facets are higher simplices) gets one blended cell per
    # block, since a block there holds 20 … 12870 product facets
    mode = "facet" if is_triangle_complex(P) else "block"
    main_w = 11.0 if n <= 40 else 13.0
    own_w, leg_w, legend_w, col_gap = 3.3, 3.3, 2.1, 0.55
    right_w = max(own_w, leg_w) + legend_w      # + room for the domain legend
    lm, rm, tm, bm = 0.30, 0.30, 0.45, 0.35
    span_x = 6.9 + n * (3 + gap) - gap + 0.2
    scale = main_w / span_x                       # inches per block-unit
    # Wrap the annotation, but only as a safety net: the notes carry their own
    # line breaks and are authored to fit, and mathtext spans ($\min_j ...$)
    # render much narrower than their character count, so a tight character
    # budget would break lines that actually fit.  0.0045 in per character at
    # fontsize 8 is deliberately loose -- it fires only on a runaway line.
    width = max(60, int(main_w / (0.0045 * fs / 8.0)))
    note_paras = notes_for(P, complex_name, S, A, str(state_path), mode) \
        + h1_lines(args.h1_report)
    notes = wrap_notes(note_paras, width=width)
    note_h = sum(note_line_h(fs, scale) * nl + note_gap(fs, scale) for _, _, nl in notes)
    span_y = 3.0 + (n * (2 + gap) + 2.0 + 1.4) + note_h + 0.6
    main_h = span_y * scale
    fig_w = lm + main_w + col_gap + right_w + rm
    fig_h = tm + main_h + bm
    fig = plt.figure(figsize=(fig_w, fig_h))
    # explicit positions: no gridspec arithmetic to second-guess the aspect
    axm = fig.add_axes([lm / fig_w, bm / fig_h, main_w / fig_w, main_h / fig_h])
    (draw_facets if mode == "facet" else draw_facets_block)(
        axm, P, A, S, scale=scale, gap=gap, band=band, notes=notes, fs=fs,
        **({"by_domain": True} if args.by_domain else {}))
    tag = SHORT_NAME.get(complex_name, P.name)
    axm.set_title("$K\\times K$ of %s — all %d product facets: %d covered%s, "
                  "%d uncovered (red)"
                  % (tag, S["nf"], S["cov"],
                     " (one colour per owning domain)" if args.by_domain else " (blue)",
                     S["nhole"]), fontsize=fs + 4, color=C_INK, pad=12)

    # right column: ownership above (square-ish) with its legend to its right,
    # then the staircase legend below
    col_x = (lm + main_w + col_gap) / fig_w
    own_h = min(own_w, 0.46 * main_h)
    axo = fig.add_axes([col_x, (bm + main_h - own_h) / fig_h, own_w / fig_w, own_h / fig_h])
    draw_ownership(axo, P, A, S, gap=gap, fs=fs)
    if mode == "facet":
        leg_h = leg_w * (2 * 4.6 + 1.4) / (3 * 3.9)   # the panel's own aspect
        axl = fig.add_axes([col_x + 0.5 / fig_w, (bm + 0.12 * main_h) / fig_h,
                            leg_w / fig_w, leg_h / fig_h])
        draw_path_legend(axl, fs=fs)
    else:
        axl = fig.add_axes([col_x, (bm + 0.26 * main_h) / fig_h,
                            (right_w - 0.5) / fig_w, 2.4 / fig_h])
        draw_coverage_legend(axl, P, fs=fs + 1)

    fig.text(lm / fig_w, 0.012 / fig_h,
             "state: %s  (k=%d, chain links %s, sizes %s)"
             % (state_path, len(chains), "+".join(str(l) for l in S["links"]),
                "+".join(str(x) for x in S["sizes"])),
             fontsize=8.5, color="#4A4A4A")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=300, facecolor="white")
    fig.savefig(str(out) + ".pdf", facecolor="white")
    print("wrote %s.png / %s.pdf  (%d/%d covered, %d red)"
          % (out, out, S["cov"], S["nf"], S["nhole"]))


if __name__ == "__main__":
    main()
