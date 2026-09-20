#!/usr/bin/env python3
"""gpu_sc_equiv.py — the chain-first annealer of gpu_sc.py restricted to
DIAGONALLY EQUIVARIANT chains.

If the cyclic shift  v -> v+1 (mod nv)  is a simplicial automorphism of K
(true for dDelta^n and for the Csaszar torus, whose vertex set is Z7 acting
by translation), we look for chains of maps of the form

        phi_i(a, b) = a + psi_i(b - a)        (mod nv),   psi_i : Z_nv -> Z_nv,

i.e. phi_i(a+1, b+1) = phi_i(a, b) + 1.  pi1 is psi = 0, pi2 is psi = id.
The state of a map is nv values (one per difference) instead of nv^2, and
one move changes nv grid entries coherently.  (The shift is an automorphism
of K but not of the staircase product, whose facets depend on the vertex
order, so the facets are NOT reduced: every facet of K x K is evaluated.)

Why this matters (optimized_algorithms.md §16): the product domains
star(v) x star(w) that the unconstrained annealer converges to are NOT
invariant under the diagonal shift, so equivariance forces the non-product
domains that a 3-domain torus cover (or a 2-domain S^3 cover) would need,
exactly as the continuous motion planners on a group
{(a, b) : b - a in V_i} are translation invariant.

Everything produced here is expanded to ordinary chains on K x K and
verified with gpu_sc.verify_system (the plain Definition 2.3 check) before
a certificate is written.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
from numba import cuda, int8, uint8, int32, float32, float64, uint64

import gpu_sc
from gpu_sc import Product, sm_next_dev, sm_u01_dev


# --------------------------------------------------------------------------- #
#  Quotient structure                                                          #
# --------------------------------------------------------------------------- #
class Equivariant:
    """P = K x K with maps parametrised by differences.

    Reduced grid vertex d in 0..nv-1 stands for all grid vertices (a, a+d); a
    grid vertex (a, b) has d = (b - a) mod nv and shift a.  For facet f,
    facet_verts[f, q] is the difference d_q of its q-th vertex and
    facet_shift[f, q] the shift a_q, so the image of that vertex under psi is
    (psi[d_q] + a_q) mod nv.  A facet may contain several vertices with the
    same difference.
    """

    def __init__(self, P: Product):
        nv = P.nv
        self.P, self.nv = P, nv
        self.name = P.name + " / diagonal Z" + str(nv)
        # check that the shift is an automorphism of K
        kf = {frozenset(f) for f in P.kfacets}
        for f in P.kfacets:
            assert frozenset((v + 1) % nv for v in f) in kf, "v -> v+1 is not an automorphism of K"
        # NOTE: the diagonal shift is an automorphism of K but NOT of the
        # staircase product (a monotone path wraps around), so there are no
        # facet orbits; the ansatz only shrinks the STATE (nv values per map)
        # and every facet of K x K is evaluated individually.
        reps = list(range(P.nf))
        members = [[f] for f in range(P.nf)]
        self.orbit_of = np.arange(P.nf, dtype=np.int32)
        self.reps, self.members = reps, members
        self.nf = len(reps)
        self.ngrid = nv
        self.maxfv = P.maxfv
        fv = np.zeros((self.nf, P.maxfv), np.int8)
        fs = np.zeros((self.nf, P.maxfv), np.int8)
        fnv = np.zeros(self.nf, np.int8)
        for r, f in enumerate(reps):
            vs = P.facets[f]
            fnv[r] = len(vs)
            for q, g in enumerate(vs):
                a, b = g // nv, g % nv
                fv[r, q] = (b - a) % nv
                fs[r, q] = a
        self.facet_verts, self.facet_shift, self.facet_nv = fv, fs, fnv
        # incidence: reduced vertex d -> reduced facets containing d (once each)
        inc = [[] for _ in range(nv)]
        for r in range(self.nf):
            for d in sorted(set(int(x) for x in fv[r, : fnv[r]])):
                inc[d].append(r)
        off = np.zeros(nv + 1, np.int32)
        for d in range(nv):
            off[d + 1] = off[d] + len(inc[d])
        self.incid_off = off
        self.incid_fac = np.array([r for d in range(nv) for r in inc[d]], np.int32)
        self.starmax = max(len(x) for x in inc)
        self.is_simplex = P.is_simplex
        self.pi1 = np.zeros(nv, np.int8)
        self.pi2 = np.arange(nv, dtype=np.int8)
        self.kfacets = P.kfacets

    # ---- host helpers --------------------------------------------------------
    def expand(self, chain_red):
        """reduced chain [c+1, nv] -> full chain [c+1, ngrid] on K x K."""
        nv = self.nv
        ch = np.asarray(chain_red, np.int64)
        a = np.arange(nv)[:, None]
        b = np.arange(nv)[None, :]
        d = (b - a) % nv
        full = (ch[:, d] + a[None]) % nv                # [c+1, nv, nv]
        return full.reshape(ch.shape[0], nv * nv).astype(np.int8)

    def chain_bad_counts(self, chain_red):
        ch = np.asarray(chain_red, np.int64)
        fv = self.facet_verts.astype(np.int64)
        fs = self.facet_shift.astype(np.int64)
        valid = self.facet_nv[:, None] > np.arange(self.maxfv)[None, :]
        img = (ch[:, fv] + fs[None]) % self.nv             # [c+1, nf, maxfv]
        m = np.zeros(img.shape[:2], np.int64)
        for q in range(self.maxfv):
            m |= np.where(valid[None, :, q], 1 << img[:, :, q], 0)
        link = m[:-1] | m[1:]
        return (self.is_simplex[link] == 0).sum(0)

    def domain_of(self, chain_red):
        return np.flatnonzero(self.chain_bad_counts(chain_red) == 0)


# --------------------------------------------------------------------------- #
#  Kernel: the annealer on the quotient (values carry per-vertex shifts)       #
# --------------------------------------------------------------------------- #
def make_equiv_kernel(KCH, C1, NGRID, NF, NV, STARMAX, LMAX=8):
    """Same energy and moves as gpu_sc.make_anneal_kernel:
        E = sum_f w_f * pen(min_j bad_j[f]) + gamma * sum_{j,f} bad_j[f]
    with the image of facet vertex q under map i being
        (ch[j, i, facet_verts[f, q]] + facet_shift[f, q]) mod NV.
    A reduced vertex may occur several times in one facet (with different
    shifts), so masks are always rebuilt per map (no single-occurrence fast
    path)."""
    CMAX = C1 - 1
    L1 = LMAX + 2

    @cuda.jit
    def kernel(chains, best, bad, score, seeds,
               facet_verts, facet_shift, facet_nv, incid_off, incid_fac, is_simplex,
               weights, iters, T0, T1, beta, gamma, pmove_copy, ptrack):
        s = cuda.grid(1)
        if s >= chains.shape[0]:
            return
        ch = cuda.local.array((KCH, C1, NGRID), int8)
        bd = cuda.local.array((KCH, NF), uint8)
        dl = cuda.local.array(STARMAX, int8)
        mo_ = cuda.local.array(L1, int32)
        mn_ = cuda.local.array(L1, int32)

        for j in range(KCH):
            for i in range(C1):
                for g in range(NGRID):
                    ch[j, i, g] = chains[s, j, i, g]
        for j in range(KCH):
            for f in range(NF):
                cnt = 0
                mprev = int32(0)
                for q in range(facet_nv[f]):
                    x = int32(ch[j, 0, facet_verts[f, q]]) + int32(facet_shift[f, q])
                    if x >= NV:
                        x -= NV
                    mprev |= int32(1) << x
                for i in range(1, C1):
                    mcur = int32(0)
                    for q in range(facet_nv[f]):
                        x = int32(ch[j, i, facet_verts[f, q]]) + int32(facet_shift[f, q])
                        if x >= NV:
                            x -= NV
                        mcur |= int32(1) << x
                    if is_simplex[mprev | mcur] == 0:
                        cnt += 1
                    mprev = mcur
                bd[j, f] = uint8(cnt)
        cov = int32(0)
        for f in range(NF):
            b = bd[0, f]
            for j in range(1, KCH):
                if bd[j, f] < b:
                    b = bd[j, f]
            if b == 0:
                cov += 1
        bestcov = cov
        for j in range(KCH):
            for i in range(C1):
                for g in range(NGRID):
                    best[s, j, i, g] = ch[j, i, g]

        state = seeds[s]
        state, x = sm_next_dev(state)
        lnr = math.log(T1 / T0)
        for it in range(iters):
            T = T0 * math.exp(lnr * (float64(it) / float64(iters)))
            state, x = sm_next_dev(state)
            j = int32(x % uint64(KCH))
            state, x = sm_next_dev(state)
            w = int32(x % uint64(NGRID))
            state, x = sm_next_dev(state)
            L = int32(1)
            if sm_u01_dev(x) < ptrack:
                state, x = sm_next_dev(state)
                L = int32(2) + int32(x % uint64(LMAX - 1))
            if L > CMAX - 1:
                L = CMAX - 1
            state, x = sm_next_dev(state)
            i1 = int32(1) + int32(x % uint64(CMAX - L))
            i2 = i1 + L - 1
            old = ch[j, i1, w]
            state, x = sm_next_dev(state)
            pm = sm_u01_dev(x)
            state, x = sm_next_dev(state)
            if pm < pmove_copy:
                if (x & uint64(1)) == uint64(0):
                    u = ch[j, i1 - 1, w]
                else:
                    u = ch[j, i2 + 1, w]
                if u == old:
                    state, x = sm_next_dev(state)
                    kk = int32(x % uint64(NV - 1))
                    u = kk if kk < old else kk + 1
            else:
                kk = int32(x % uint64(NV - 1))
                u = kk if kk < old else kk + 1

            dE = float64(0.0)
            dcov = int32(0)
            idx = 0
            for ii in range(incid_off[w], incid_off[w + 1]):
                f = incid_fac[ii]
                ob = int32(0)
                nb = int32(0)
                for l in range(L + 2):
                    i = i1 - 1 + l
                    m = int32(0)
                    mw = int32(0)          # contribution of the vertex w (old value)
                    mwn = int32(0)         # contribution of the vertex w (new value)
                    for q in range(facet_nv[f]):
                        v = facet_verts[f, q]
                        sh = int32(facet_shift[f, q])
                        if v != w:
                            xo = int32(ch[j, i, v]) + sh
                            if xo >= NV:
                                xo -= NV
                            m |= int32(1) << xo
                        else:
                            xo = int32(ch[j, i, w]) + sh
                            if xo >= NV:
                                xo -= NV
                            mw |= int32(1) << xo
                            xn = int32(u) + sh
                            if xn >= NV:
                                xn -= NV
                            mwn |= int32(1) << xn
                    mo_[l] = m | mw
                    if l == 0 or l == L + 1:
                        mn_[l] = m | mw
                    else:
                        mn_[l] = m | mwn
                for l in range(L + 1):
                    if is_simplex[mo_[l] | mo_[l + 1]] == 0:
                        ob += 1
                    if is_simplex[mn_[l] | mn_[l + 1]] == 0:
                        nb += 1
                d = nb - ob
                dl[idx] = int8(d)
                idx += 1
                if d != 0:
                    dE += gamma * float64(d)
                    bo = int32(bd[j, f])
                    bn = bo + d
                    mo = bo
                    mn = bn
                    for jj in range(KCH):
                        if jj != j:
                            bb = int32(bd[jj, f])
                            if bb < mo:
                                mo = bb
                            if bb < mn:
                                mn = bb
                    po = float64(0.0) if mo == 0 else 1.0 + beta * float64(mo)
                    pn = float64(0.0) if mn == 0 else 1.0 + beta * float64(mn)
                    dE += weights[f] * (pn - po)
                    if mo == 0 and mn != 0:
                        dcov -= 1
                    elif mo != 0 and mn == 0:
                        dcov += 1
            accept = dE <= 0.0
            if not accept:
                state, x = sm_next_dev(state)
                if sm_u01_dev(x) < math.exp(-dE / T):
                    accept = True
            if accept:
                for i in range(i1, i2 + 1):
                    ch[j, i, w] = u
                idx = 0
                for ii in range(incid_off[w], incid_off[w + 1]):
                    f = incid_fac[ii]
                    d = dl[idx]
                    idx += 1
                    if d != 0:
                        bd[j, f] = uint8(int32(bd[j, f]) + int32(d))
                cov += dcov
                if cov > bestcov:
                    bestcov = cov
                    for jj in range(KCH):
                        for i2_ in range(C1):
                            for g in range(NGRID):
                                best[s, jj, i2_, g] = ch[jj, i2_, g]
        for j in range(KCH):
            for i in range(C1):
                for g in range(NGRID):
                    chains[s, j, i, g] = ch[j, i, g]
            for f in range(NF):
                bad[s, j, f] = bd[j, f]
        score[s, 0] = float32(cov)
        score[s, 1] = float32(bestcov)

    return kernel


class ChainAnnealerE(gpu_sc.ChainAnnealer):
    """gpu_sc.ChainAnnealer on the quotient structure (E: Equivariant)."""

    def __init__(self, E: Equivariant, k: int, cmax: int, S: int, seed: int = 0, weights=None, threads=64):
        self.P, self.k, self.cmax, self.S = E, k, cmax, S
        self.C1 = cmax + 1
        self.threads = threads
        self.rng = np.random.default_rng(seed)
        self.kernel = make_equiv_kernel(k, self.C1, E.ngrid, E.nf, E.nv, E.starmax)
        self.weights = np.ones(E.nf, np.float32) if weights is None else np.asarray(weights, np.float32)
        self.d_fv = cuda.to_device(E.facet_verts)
        self.d_fs = cuda.to_device(E.facet_shift)
        self.d_fnv = cuda.to_device(E.facet_nv)
        self.d_off = cuda.to_device(E.incid_off)
        self.d_inc = cuda.to_device(E.incid_fac)
        self.d_iss = cuda.to_device(E.is_simplex)
        self.d_w = cuda.to_device(self.weights)
        self.chains = self.random_init(S)
        self.best = self.chains.copy()
        self.bestscore = np.full(S, -1.0, np.float32)
        self.kernel_time = 0.0
        self.moves = 0

    def epoch(self, iters, T0, T1, beta=0.05, pmove_copy=0.5, ptrack=0.3, gamma=0.0):
        S = self.S
        d_ch = cuda.to_device(self.chains)
        d_best = cuda.to_device(self.chains)
        d_bad = cuda.device_array((S, self.k, self.P.nf), np.uint8)
        d_score = cuda.device_array((S, 2), np.float32)
        seeds = self.rng.integers(0, 2**63 - 1, size=S, dtype=np.int64).astype(np.uint64)
        d_seeds = cuda.to_device(seeds)
        blocks = (S + self.threads - 1) // self.threads
        t0 = time.perf_counter()
        self.kernel[blocks, self.threads](
            d_ch, d_best, d_bad, d_score, d_seeds,
            self.d_fv, self.d_fs, self.d_fnv, self.d_off, self.d_inc, self.d_iss, self.d_w,
            np.int32(iters), np.float64(T0), np.float64(T1), np.float64(beta), np.float64(gamma),
            np.float64(pmove_copy), np.float64(ptrack))
        cuda.synchronize()
        self.kernel_time += time.perf_counter() - t0
        self.moves += S * iters
        self.chains = d_ch.copy_to_host()
        score = d_score.copy_to_host()
        best = d_best.copy_to_host()
        improved = score[:, 1] > self.bestscore
        self.best[improved] = best[improved]
        self.bestscore[improved] = score[improved, 1]
        return score


def anneal_cover_equiv(E: Equivariant, k, cmax=20, S=4096, iters=20000, epochs=50, T0=0.6, T1=0.03,
                       beta=5.0, gamma=0.5, pcopy=0.5, ptrack=0.3, seed=0, time_limit=None, log=print,
                       stop_at_full=True, tournament_every=2, tournament_frac=0.25, guide=0.0, guide_max=4.0,
                       init_chains=None):
    ann = ChainAnnealerE(E, k, cmax, S, seed=seed)
    if init_chains is not None:
        n = min(len(init_chains), S) // 2
        ann.chains[:n] = init_chains[:n]
    t_start = time.perf_counter()
    hist, best_chains, best_cov = [], None, -1.0
    for ep in range(1, epochs + 1):
        score = ann.epoch(iters, T0, T1, beta, pcopy, ptrack, gamma)
        chains, cov = ann.top()
        if guide > 0:
            _, covered = ann.coverage_of(chains)
            w = 1.0 + (ann.weights - 1.0) * 0.7
            w[~covered] += guide
            ann.set_weights(np.minimum(w, guide_max))
        el = time.perf_counter() - t_start
        srt = np.sort(ann.bestscore)[::-1]
        log(f"  epoch {ep:3d}: best {cov:6.1f}/{E.nf}  "
            f"top10 {srt[:10].astype(int).tolist()}  median {np.median(srt):.0f}  "
            f"{el:7.1f}s  {ann.moves / ann.kernel_time / 1e6:8.1f} Mmoves/s")
        hist.append(dict(epoch=ep, best=cov, median=float(np.median(srt)), elapsed=el))
        if cov > best_cov:
            best_cov, best_chains = cov, chains.copy()
        if stop_at_full and cov >= E.nf - 1e-6:
            log(f"  full coverage reached at epoch {ep}")
            break
        if time_limit is not None and el > time_limit:
            log("  time limit reached")
            break
        if ep % tournament_every == 0:
            ann.tournament(frac=tournament_frac, noise=max(1, S // 32))
    return best_chains, best_cov, hist, ann


def selfcheck(E: Equivariant, rng, n=20):
    """Bad counts computed through the difference parametrisation must equal
    the full-product bad counts of the expanded chain, facet by facet."""
    for _ in range(n):
        c = rng.integers(0, E.nv, size=(6, E.nv)).astype(np.int8)
        c[0] = E.pi1
        c[-1] = E.pi2
        red = E.chain_bad_counts(c)
        full = E.P.chain_bad_counts(E.expand(c))
        for r in range(E.nf):
            for f in E.members[r]:
                assert full[f] == red[r], (r, f, full[f], red[r])
    return True


def cli():
    ap = argparse.ArgumentParser(description="diagonally equivariant chain-first annealer")
    ap.add_argument("--complex", default="torus", choices=["circle", "s2", "s3", "torus"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cmax", type=int, default=20)
    ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--T0", type=float, default=0.6)
    ap.add_argument("--T1", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--pcopy", type=float, default=0.5)
    ap.add_argument("--ptrack", type=float, default=0.3)
    ap.add_argument("--guide", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--selfcheck", action="store_true", help="only run the reduced-vs-full consistency check")
    a = ap.parse_args()
    P = gpu_sc.build(a.complex)
    E = Equivariant(P)
    print(f"=== equivariant-ansatz chain-first annealer ===  K={P.name}  |F_L|={P.nf}  "
          f"state: grid {P.ngrid} -> {E.ngrid} differences per map  k={a.k} cmax={a.cmax} S={a.S} iters={a.iters} "
          f"beta={a.beta} gamma={a.gamma} ptrack={a.ptrack}")
    if a.selfcheck:
        selfcheck(E, np.random.default_rng(a.seed))
        print("selfcheck OK: parametrised bad counts == full bad counts on all facets")
        return
    t0 = time.perf_counter()
    chains, cov, hist, ann = anneal_cover_equiv(E, a.k, a.cmax, a.S, a.iters, a.epochs, a.T0, a.T1, a.beta, a.gamma,
                                                a.pcopy, a.ptrack, a.seed, a.time_limit, stop_at_full=not a.no_stop,
                                                guide=a.guide)
    dt = time.perf_counter() - t0
    full_chains = [E.expand(chains[j]) for j in range(a.k)]
    doms = [P.domain_of(c) for c in full_chains]
    print(f"best coverage {int(cov)}/{P.nf}  domain sizes {[len(d) for d in doms]}  time {dt:.1f}s  "
          f"kernel {ann.kernel_time:.1f}s  moves {ann.moves:.3g}")
    outdir = Path(a.outdir) if a.outdir else Path(f"results/{a.complex}_equiv_k{a.k}")
    meta = dict(params=vars(a), method="diagonally equivariant chain-first annealer (gpu_sc_equiv.py)",
                elapsed_s=dt, kernel_time_s=ann.kernel_time, moves=ann.moves, history=hist,
                best_chains_reduced=chains.tolist())
    parts = gpu_sc.partition_from_chains(P, full_chains)
    if parts is not None:
        gpu_sc.verify_system(P, parts)
        summ = gpu_sc.write_certificate(P, parts, outdir, meta)
        print(f"VERIFIED: {len(parts)} domains {summ['domain_sizes']}, chain lengths {summ['chain_lengths']}  "
              f"=> SC_strict({P.name}) <= {len(parts) - 1}; written to {outdir}")
    else:
        covered = np.zeros(P.nf, bool)
        for d in doms:
            covered[d] = True
        print(f"NOT a covering: {int((~covered).sum())} facets uncovered")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "summary.json").write_text(json.dumps(dict(
            complex=dict(name=P.name, nv=P.nv, num_facets=P.nf), coverage=int(covered.sum()),
            chain_domain_sizes=[len(d) for d in doms], best_chains=[c.tolist() for c in full_chains], **meta),
            indent=2, default=str))
        print(f"wrote {outdir}/summary.json (partial)")


if __name__ == "__main__":
    cli()
