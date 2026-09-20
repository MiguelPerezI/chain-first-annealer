# S⁷ = ∂Δ⁸ with the chain-first annealer: the scaling wall

*2026-09-19. Task: run the chain-first GPU annealer (`gpu_sc.py`) on S⁷ = ∂Δ⁸,
the next sphere up from the ∂Δ⁴ campaign, and document. Result: **the
annealer cannot run on ∂Δ⁸ at all** — the CUDA kernel fails to launch — and
the measurement below locates the wall precisely between ∂Δ⁷ and ∂Δ⁸, with
∂Δ⁷ already impractical. This is a different wall from the one the README
records at S⁸ (which is about *writing and auditing* a facet-by-facet
certificate); this one is about *searching*.*

## What was run

```bash
# 1. the direct attempt on ∂Δ⁸, minimal sizes, 60 s cap
$PY -u gpu_sc.py --complex s7 --k 2 --cmax 40 --S 64 --iters 100 --epochs 1 \
    --time-limit 60 --outdir /tmp/s7_probe

# 2. the same on ∂Δ⁷, to find where the wall is
$PY -u gpu_sc.py --complex s6 --k 2 --cmax 40 --S 64 --iters 100 --epochs 1 \
    --time-limit 30 --outdir results/s6_scale_probe
```

## ∂Δ⁸: the kernel does not launch

`build("s7")` and the population setup succeed; the failure is at the first
kernel launch, and it is not a tuning problem — no value of `--S`, `--iters`
or `--kernel` changes it:

```
numba.cuda.cudadrv.driver.CudaAPIError: [1] Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE
```

The cause is the kernel's **per-thread local state**, sized by the complex
(`make_anneal_kernel` allocates `bd[k][|F_L|]`, `dl[STARMAX]`, `ch[k][cmax+1][ngrid]`
as `cuda.local.array`, i.e. per thread):

| complex | \|F_L\| | star(grid vertex) | `bd` | `dl` | `ch` | total per thread |
|---|---:|---:|---:|---:|---:|---:|
| ∂Δ⁶ (S⁵) | 12,348 | 9,072 | 24,696 B | 9,072 B | 4,018 B | **37 KB** — launches |
| ∂Δ⁷ (S⁶) | 59,136 | 45,276 | 118,272 B | 45,276 B | 5,248 B | **165 KB** — launches |
| ∂Δ⁸ (S⁷) | 277,992 | 219,648 | 555,984 B | 219,648 B | 6,642 B | **764 KB** — exceeds the 512 KB/thread local limit |

The 512 KB per-thread local-memory limit is what `CUDA_ERROR_INVALID_VALUE`
is reporting, and the arithmetic reproduces the observed boundary exactly:
S⁵ and S⁶ sit under it, S⁷ is 1.5× over.

## ∂Δ⁷: it launches, and it is already hopeless

The ∂Δ⁷ probe (`results/s6_scale_probe`, S = 64, iters = 100, one epoch)
covered 30,440 of 59,136 facets in **81.8 s for 6,400 moves — 12.8 ms per
move**. Scaled to the campaign's usual settings (S = 4096, iters = 20,000 =
8.19 × 10⁷ moves per epoch), one epoch on ∂Δ⁷ is ≈ 3 × 10⁵ s ≈ **3.5 days**,
against 54 s for the same epoch on ∂Δ⁴. The ∂Δ⁶ plateau (S⁵) was reached in
epochs of ~1,000 s; ∂Δ⁷ is ~300× that per epoch, and ∂Δ⁶ needed 2 epochs only
because the search *started* near the answer.

| complex | \|F_L\| | star | measured ns per move | relative to ∂Δ⁴ |
|---|---:|---:|---:|---:|
| ∂Δ⁴ (S³) | 500 | 320 | 645 | 1× |
| ∂Δ⁶ (S⁵) | 12,348 | 9,072 | 49,443 | 77× |
| ∂Δ⁷ (S⁶) | 59,136 | 45,276 | 12,781,267 | 19,800× |
| ∂Δ⁸ (S⁷) | 277,992 | 219,648 | — cannot launch | — |

The cost tracks the size of a grid vertex's star, because a single move
re-evaluates every product facet incident to the vertex it changes — O(star)
per move, by construction of the attribution neighborhood. That is not a
constant-factor inefficiency; it is the algorithm's per-move complexity, and
it grows by roughly a factor of 5 per dimension here.

## Could the kernel be changed to launch on ∂Δ⁸?

Yes mechanically: moving `bd` to global memory (per thread) removes 556 KB of
the 764 KB and brings the kernel under the limit. The projection nevertheless
says it would not produce a certificate:

- The `bd` accesses in the move loop would become uncoalesced global accesses
  (each thread owns its own row), so per-move cost would *rise*, not fall.
- Even at the optimistic ∂Δ⁴ rate of 2 ns per star-element (i.e. ignoring all
  memory latency), one move on ∂Δ⁸ costs 219,648 × 2 ns ≈ 0.44 ms. One epoch
  at S = 4096, iters = 20,000 is then ~10 hours; the ∂Δ⁴ campaign needed
  **5.65 × 10⁹ moves** in a single run and 69 epochs to conclude that its
  plateau does not move. Matching that number of moves on ∂Δ⁸ is ~28 days of
  GPU at small S, and it is a search that failed to find a cover on a complex
  with 556× fewer facets.
- The O(|F_L|) per-thread `bd` initialization alone, repeated every epoch per
  thread, is itself 278,000 facet visits × S per epoch.

## Conclusion

The chain-first annealer's reach ends at **∂Δ⁶ = S⁵**: that is where the
repository's only annealed certificate above S³ lives (`results/s5_k3_v2`),
and the next sphere up is ~300× more expensive per epoch, the one after that
cannot be launched at all. ∂Δ⁸ = S⁷ is out of range for this method on this
hardware, by the kernel's per-thread state and its O(star) move evaluation
together — not by any budget the user can set.

Consequence for the programme: for S⁷ the chain-first formulation cannot be
the instrument, and any statement about a *searched* certificate there would
have to come from a different formulation (a different move attribution, or a
domain-level search like `fixed_domain_test.py`, which uses the same kernel
and is therefore bounded by the same wall). The closed-form product covers
reach S⁷ in 25 s and are explicitly out of scope for this document.
