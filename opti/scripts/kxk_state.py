#!/usr/bin/env python3
"""kxk_state.py — the data layer behind the K x K pictures, without matplotlib.

`render_kxk.py` draws a saved annealer state; the local webapp browses, compares
and follows the same states live.  Both need the same facts computed the same
way, so the fact-finding lives here and the drawing lives there: this module
imports numpy and `gpu_sc` only, which is what lets it run under CI's
requirements.txt (numpy/numba/scipy, no matplotlib, no GPU) and inside a
long-lived server process.

What it provides:
  * state discovery        find_state / infer_complex / load_state   (results/ layout)
  * the audit              analyse / state_stats                     (coverage, ownership, defect depth)
  * block geometry         block_index / block_facets / is_triangle_complex
  * the webapp payload     state_payload                             (packed, JSON-ready)
  * palette + path labels  DOMAIN_COLORS / PATH_LABEL / blend

Facet ids come out of `gpu_sc.Product` in the order
    f = block(i, j) offset + p      (i, j = facets of K;  p = staircase simplex)
and blocks are contiguous runs of facets, six per block when K's facets are
triangles (the p0 BBAA … p5 AABB picture) and binom(p+q, q) otherwise (S^3
blocks hold 20 product facets, S^4 70, … S^8 12870).
"""
from __future__ import annotations

import base64
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gpu_sc  # noqa: E402  (Product builder: identical facet order to the GPU repo)

# ---- palette (shared by the paper figures and the webapp) ------------------
def hex_rgb(h):
    """'#RRGGBB' -> (r, g, b) floats in [0, 1].  Same values matplotlib's
    colors.to_rgb returns, without importing matplotlib."""
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

C_COVER = "#2E6FA7"      # covered facet, when not colouring by domain
C_HOLE = "#D0402F"       # uncovered facet
C_EDGE = "#FFFFFF"       # hairlines between facets
C_DEAD_EDGE = "#141414"  # outline of a fully dead block
C_PART_EDGE = "#4A4A4A"  # outline of a partially dead block
C_BAND = "#F7E9BE"       # star(5) x star(5) band highlight (RP^2 only)
C_BAND_TXT = "#9A7413"   # ... and its labels
C_INK = "#1A1A1A"
DOMAIN_COLORS = ["#6A51A3", "#2F8F5B", "#E08A1E", "#7B5E3B", "#2E8B9B"]


def _hsv_hex(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def _color_dist(a, b):
    """RGB distance on the usual 0..255 scale (0 = identical, 441 = opposite)."""
    return sum((255.0 * (x - y)) ** 2 for x, y in zip(hex_rgb(a), hex_rgb(b))) ** 0.5


def _extra_domain_colors(n):
    """n further domain colours, each visibly different from all the others.

    The first five are fixed (the paper figures are built on them); beyond that
    the recorded runs already go to k = 11 (the RP^2 ladder) and k = 9 (Klein),
    so k domains must get k genuinely different colours -- repeating them, or
    handing out two shades nobody can tell apart, would make the "one colour
    per domain" map lie.  Hue walks by the golden ratio (so the first few picks
    are ~222 deg apart) alternating dark and light, and every candidate must
    clear a minimum RGB distance from the colours already in use, the hole red
    included.  The threshold relaxes only if the pool cannot supply enough.
    """
    reserved = list(DOMAIN_COLORS) + [C_HOLE]
    for sep in (70, 60, 55, 50, 45, 40, 35, 30, 25, 20):
        out = []
        for i in range(400):
            hue = (0.61803398875 * i + 0.137) % 1.0
            c = _hsv_hex(hue, 0.68, 0.78 if i % 2 == 0 else 0.58)
            if min(_color_dist(c, x) for x in reserved + out) >= sep:
                out.append(c)
            if len(out) >= n:
                return out
    return out


DOMAIN_COLORS += _extra_domain_colors(19)      # 24 colours, all distinct

SHORT_NAME = {"rp2": "RP$^2$", "klein": "the Klein bottle", "torus": "the torus",
              "mobius": "the Möbius band", "s1xw": "$S^1\\times(S^1\\vee S^1)$"}

COMPLEXES = ["circle", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "wedge",
             "bary", "cube", "dodeca", "rp2", "klein", "torus", "mobius", "s1xw"]
STATE_NAMES = ("summary.json", "final_state.json", "live_state.json")

# sub-cell position of staircase path p inside a block: col = p % 3, row = p // 3
SUB_COL = {p: p % 3 for p in range(6)}
SUB_ROW = {p: p // 3 for p in range(6)}
PATH_LABEL = {0: "BBAA", 1: "BABA", 2: "ABBA", 3: "BAAB", 4: "ABAB", 5: "AABB"}
CLOSING = [3, 5, 6, 9, 10, 12]   # masks of the six paths, in emission order


def blend(f):
    """Blue -> red fill for a covered fraction f (1 = all covered)."""
    a = np.array(hex_rgb(C_COVER))
    b = np.array(hex_rgb(C_HOLE))
    return tuple(a * f + b * (1 - f))


def path_steps(p):
    """Lattice path of staircase simplex p: list of (di, dj) steps.

    Only meaningful when K's facets are triangles; general complexes use the
    masks from `masks_for` directly.
    """
    m = CLOSING[p]
    return [(1, 0) if not (m >> b) & 1 else (0, 1) for b in range(4)]


def masks_for(p, q):
    """The staircase masks of a block Delta^p x Delta^q, in emission order:
    every mask with popcount == q, by increasing value.  A mask's bit b says
    the b-th step moves inside the second factor."""
    return [m for m in range(1 << (p + q)) if bin(m).count("1") == q]


def facet_grid_vertices(P, i, j, p):
    """Grid ids (a * nv + b) of facet p of block (i, j), in emission order.

    Mirrors `Product.__init__`'s staircase walk, so a client can label any
    facet without shipping the (up to 13-vertex, 1.3M-facet) vertex tables.
    """
    nv = P.nv
    s0, s1 = P.kfacets[i], P.kfacets[j]
    m = masks_for(len(s0) - 1, len(s1) - 1)[p]
    out = [s0[0] * nv + s1[0]]
    pi = pj = 0
    for b in range(len(s0) - 1 + len(s1) - 1):
        if not (m >> b) & 1:
            pi += 1
        else:
            pj += 1
        out.append(s0[pi] * nv + s1[pj])
    return out


# --------------------------------------------------------------------------- #
#  state discovery                                                             #
# --------------------------------------------------------------------------- #
def load_state(path):
    """(state dict, list of chains as int64 arrays) for a saved state file."""
    s = json.loads(Path(path).read_text())
    chains = [[np.array(m, dtype=np.int64) for m in ch] for ch in s["best_chains"]]
    return s, chains


def infer_complex(state, path):
    """Complex of a saved state: the file's own record first, then the name."""
    v = state.get("complex")
    if isinstance(v, str) and v in COMPLEXES:
        return v
    v = (state.get("params") or {}).get("complex")
    if isinstance(v, str) and v in COMPLEXES:
        return v
    for part in [Path(path).parent.name, Path(path).stem]:
        for c in sorted(COMPLEXES, key=len, reverse=True):   # longest match wins
            if part == c or part.startswith(c + "_"):
                return c
    return None


def find_state(target):
    """Locate the state to use in a results directory (or take a file as given).

    Priority: summary.json, final_state.json, live_state.json, phase*/summary.json
    (latest phase first), then any *.json under the directory that carries
    best_chains (largest stored coverage wins).  Returns (path, state, others).
    """
    t = Path(target)
    if t.is_file():
        return t, json.loads(t.read_text()), []
    if not t.is_dir():
        raise SystemExit("no such results directory or state file: %s" % target)
    cands = [t / n for n in STATE_NAMES if (t / n).is_file()]
    cands += sorted(t.glob("phase*/summary.json"), key=lambda q: q.parent.name, reverse=True)
    seen, uniq = set(), []
    for c in cands + sorted(t.rglob("*.json")):
        if c.is_file() and c not in seen:
            seen.add(c)
            uniq.append(c)
    loaded = []
    for c in uniq:
        try:
            s = json.loads(c.read_text())
        except Exception:
            continue
        if isinstance(s, dict) and s.get("best_chains"):
            loaded.append((c, s))
    if not loaded:
        raise SystemExit("no state with best_chains under %s" % target)
    if len(loaded) > 1:                     # only when the priority names missed
        top = [c for c in loaded if c[0].name in STATE_NAMES]
        loaded = top or loaded
        loaded.sort(key=lambda cs: (-(cs[1].get("coverage") or 0), str(cs[0])))
    path, state = loaded[0]
    return path, state, [c for c, _ in loaded[1:]]


def list_runs(results_dir="results"):
    """Cheap index of every result directory holding a state, for a browser.

    Reads JSON only (no coverage recompute: on S^6 that costs 12 s), so the
    numbers here are the ones the run itself recorded.
    """
    out = []
    root = Path(results_dir)
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        try:
            path, state, others = find_state(d)
        except SystemExit:
            continue
        params = state.get("params") or {}
        out.append(dict(
            run=d.name, dir=str(d), state=str(path), state_name=path.name,
            complex=infer_complex(state, path),
            k=len(state.get("best_chains") or []) or params.get("k"),
            coverage=state.get("coverage"),
            nf=(state.get("complex") or {}).get("num_facets") if isinstance(state.get("complex"), dict) else None,
            sizes=state.get("chain_domain_sizes"),
            status=state.get("status") or ("complete" if state.get("all_domains_verified") else None),
            complete=bool(state.get("all_domains_verified")),
            live=path.name == "live_state.json",
            phase=path.parent.name if path.parent.name.startswith("phase") else None,
            seed=params.get("seed"), init=params.get("init"),
            params={kk: vv for kk, vv in params.items() if kk != "outdir"},
            mtime=path.stat().st_mtime,
            other_states=[str(o) for o in others],
        ))
    out.sort(key=lambda r: (-(r["coverage"] or 0), r["run"]))
    return out


def find_h1_report(state_path, target, nf, cov):
    """An H1 audit lying next to the state, if it describes *this* state."""
    state_path = Path(state_path)
    t = Path(target)
    roots = [state_path.parent] + ([t] if t.is_dir() else [])
    seen = set()
    for r in roots:
        for f in sorted(r.glob("h1_absorption*.txt")) + sorted(r.glob("h1*.txt")):
            if f in seen:
                continue
            seen.add(f)
            txt = f.read_text(errors="replace")
            m = re.search(r"\|F_L\|\s*=\s*(\d+).*?covered\s*=\s*(\d+)", txt)
            head = txt.splitlines()[0] if txt.splitlines() else ""
            if not m or int(m.group(1)) != nf or int(m.group(2)) != cov:
                continue
            if state_path.name not in head and state_path.parent.name not in head:
                continue
            return f
    return None


# --------------------------------------------------------------------------- #
#  block geometry                                                              #
# --------------------------------------------------------------------------- #
def block_index(P):
    """Facet id -> block id, plus the block boundaries.

    `Product.__init__` emits the facets block by block (facet i of K, facet j of
    K, then every staircase simplex of that pair), so facet ids of a block are
    contiguous.  A block's facet count is binom(p+q, q) for facets of dimension
    p and q -- 6 for two triangles, 20 for two 3-simplices, 70, 252, … -- so the
    boundaries have to be computed, not divided by 6.
    """
    K = P.kfacets
    n = len(K)
    offs = [0]
    for i in range(n):
        for j in range(n):
            offs.append(offs[-1] + math.comb(len(K[i]) - 1 + len(K[j]) - 1, len(K[j]) - 1))
    assert offs[-1] == P.nf, (offs[-1], P.nf)
    blk = np.zeros(P.nf, dtype=np.int64)
    for b in range(n * n):
        blk[offs[b]:offs[b + 1]] = b
    return blk, np.array(offs, dtype=np.int64)


def block_facets(P, i, j):
    """Facet ids of the product block (i, j) -- 6 when K's facets are triangles."""
    n = len(P.kfacets)
    if is_triangle_complex(P):
        return list(range((i * n + j) * 6, (i * n + j) * 6 + 6))
    p, q = len(P.kfacets[i]) - 1, len(P.kfacets[j]) - 1
    per = math.comb(p + q, q)
    return list(range((i * n + j) * per, (i * n + j) * per + per))


def is_triangle_complex(P):
    """True when every maximal facet of K is a triangle (so each block s x t is
    Delta^2 x Delta^2 with its six staircase facets and the path picture works).
    False for the higher spheres, whose facets are (k-1)-simplices: S^3 blocks
    hold 20 product facets, S^4 70, ... S^8 12870 -- one cell each, not six."""
    return all(len(f) == 3 for f in P.kfacets)


# --------------------------------------------------------------------------- #
#  the audit                                                                   #
# --------------------------------------------------------------------------- #
def analyse(P, chains):
    """Coverage, ownership, domain geometry and defect depth for one state."""
    K, nf, nt = P.kfacets, P.nf, len(P.kfacets)
    # One `chain_bad_counts` per chain: `domain_of(ch)` is exactly
    # flatnonzero(chain_bad_counts(ch) == 0) (Product.domain_of), so calling it
    # once serves both the domains and the defect-depth minimum -- the whole
    # cost of this function (S^6: ~9 s instead of ~18 s).
    bad = [P.chain_bad_counts(ch) for ch in chains]
    dom = [set(int(f) for f in np.flatnonzero(b == 0)) for b in bad]
    covered = np.zeros(nf, bool)
    owner = np.full(nf, -1, dtype=int)
    for j, d in enumerate(dom):
        for f in sorted(d):
            if not covered[f]:
                covered[f] = True
                owner[f] = j
    # defect depth of each facet: min_j (number of non-contiguous links of
    # chain j on that facet) -- 0 means some domain contains it.
    depth = np.minimum.reduce(bad) if bad else np.full(nf, 1 << 30, dtype=int)
    # the diagonal rectangles R_v = star(v) x star(v), v a vertex of K
    blk, _ = block_index(P)
    bi, bj = blk // nt, blk % nt
    rect_of_vertex = {}
    for v in range(P.nv):
        inK = np.array([v in f for f in K], dtype=bool)
        rect_of_vertex[v] = np.flatnonzero(inK[bi] & inK[bj])
    dom_rect = []
    for d in dom:
        dd = np.array(sorted(d), dtype=int)
        best, bestv = -1, 0
        for v in range(P.nv):
            inter = int(np.isin(dd, rect_of_vertex[v]).sum())
            if inter > best:
                best, bestv = inter, v
        dom_rect.append((bestv, best, len(rect_of_vertex[bestv])))
    return dict(covered=covered, owner=owner, dom=dom, depth=depth,
                dom_rect=dom_rect, rect_of_vertex=rect_of_vertex)


def state_stats(P, chains, A):
    """The numbers the annotation quotes, all recomputed from this state."""
    K, nf, nt = P.kfacets, P.nf, len(P.kfacets)
    cov = A["covered"]
    dead = [int(f) for f in np.flatnonzero(~cov)]
    blk, _ = block_index(P)
    per_block = int(6 if is_triangle_complex(P) else math.comb(
        2 * (len(K[0]) - 1), len(K[0]) - 1))
    cells = {}
    for f in dead:
        cells.setdefault(int(blk[f]), []).append(f)
    full = sum(1 for ps in cells.values() if len(ps) == per_block)
    part = len(cells) - full
    tris = [set(map(int, f)) for f in K]

    def by_inter(fs):
        c = {}
        for f in fs:
            i, j = divmod(int(blk[f]), nt)
            kk = len(tris[i] & tris[j])
            c[kk] = c.get(kk, 0) + 1
        return dict(sorted(c.items()))

    # dual distance (triangles adjacent iff they share an edge)
    adj = [[b for b in range(nt) if b != a and len(tris[a] & tris[b]) == 2] for a in range(nt)]

    def dd(a, b):
        if a == b:
            return 0
        seen, frontier, d = {a}, [a], 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
                        if v == b:
                            return d
            frontier = nxt
        return -1

    dual = {}
    for f in dead:
        i, j = divmod(int(blk[f]), nt)
        kk = dd(i, j)
        dual[kk] = dual.get(kk, 0) + 1
    dual_diam = max(dd(a, b) for a in range(nt) for b in range(nt))
    depths = {}
    for f in dead:
        d = int(A["depth"][f])
        depths[d] = depths.get(d, 0) + 1
    return dict(nf=nf, cov=int(cov.sum()), nhole=nf - int(cov.sum()), dead=dead,
                cells=len(cells), full=full, part=part, dual_diam=dual_diam,
                per_block=per_block, inter=by_inter(dead),
                inter_cov=by_inter([f for f in range(nf) if cov[f]]),
                dual=dict(sorted(dual.items())), depths=dict(sorted(depths.items())),
                sizes=[len(d) for d in A["dom"]],
                links=[len(c) - 1 for c in chains])


def domain_membership(P, A, k=None):
    """bool[nf, k]: is facet f in domain j?  (domains overlap, so this is a
    matrix, not a function -- `owner` is only its first-true column.)"""
    k = k if k is not None else len(A["dom"])
    memb = np.zeros((P.nf, k), bool)
    for j, d in enumerate(A["dom"]):
        if d:
            memb[sorted(d), j] = True
    return memb


# --------------------------------------------------------------------------- #
#  the webapp payload                                                          #
# --------------------------------------------------------------------------- #
def _b64(bits, pack=True):
    """Base64 of a boolean array packed to bits, or of a byte array as-is.

    pack=False for integer values (the domain owner): `np.packbits` coerces its
    input to bits, so packing an owner index would silently collapse 0..k-1.
    """
    a = np.ascontiguousarray(bits)
    buf = np.packbits(a).tobytes() if pack else a.astype(np.uint8, copy=False).tobytes()
    return base64.b64encode(buf).decode()


def state_payload(P, chains, A=None, S=None, complex_name=None, state_path=None):
    """Everything the browser needs to draw this state at any zoom, as JSON.

    Packed so the payload stays small (600 facets -> 0.1 KB, 1536 -> 0.2 KB,
    S^6's 59136 -> 7.2 KB): the client can therefore hold *every* facet and
    render any zoom level without asking the server again.  Facet geometry is
    not shipped either -- the client rebuilds each facet's grid vertices from
    the block's two K-facets plus the staircase masks (see `masks_for`).
    """
    A = A if A is not None else analyse(P, chains)
    S = S if S is not None else state_stats(P, chains, A)
    k = len(chains)
    blk, offs = block_index(P)
    memb = domain_membership(P, A, k)
    depth = np.minimum(A["depth"], 255).astype(np.uint8)
    pq = sorted({(len(P.kfacets[i]) - 1, len(P.kfacets[j]) - 1)
                 for i in range(len(P.kfacets)) for j in range(len(P.kfacets))})
    import json as _json
    return _json.loads(_json.dumps(dict(
        state=str(state_path) if state_path else None,
        complex=complex_name, name=P.name, nv=P.nv, ngrid=P.ngrid,
        nf=P.nf, nblocks=len(P.kfacets) ** 2, k=k,
        triangles=[list(f) for f in P.kfacets],
        masks={"%d,%d" % (p, q): masks_for(p, q) for p, q in pq},
        block_offsets=offs[:-1].tolist(),
        block_counts=np.diff(offs).tolist(),
        domain_colors=DOMAIN_COLORS[:max(k, 1)],
        hole_color=C_HOLE,
        covered=_b64(A["covered"]),
        owner=_b64(A["owner"] + 1, pack=False),   # one byte per facet, -1 -> 0
        memb=_b64(memb),
        depth=base64.b64encode(depth.tobytes()).decode(),
        path_labels=PATH_LABEL, sub_col=SUB_COL, sub_row=SUB_ROW,
        triangle_complex=bool(is_triangle_complex(P)),
        stats={kk: vv for kk, vv in S.items() if kk != "dead"},
        domain_rect=[{"D": d, "R_v": v, "in_R": i, "rect_size": t}
                     for d, (v, i, t) in enumerate(A["dom_rect"])],
    )))
