# chain-first-annealer

Source code, run records and lower-bound tools for

> M. Perez and J. Flores, *Explicit covers and rigid obstructions for
> simplicial complexity: from spheres to surfaces* (arXiv preprint, 2026).

Gonzalez's strict simplicial complexity `SC_strict(K)` at subdivision level 0
is one less than the least number of contiguity domains covering the ordered
product `K × K`. This repository holds both halves of the paper:

- **Upper bounds (explicit covers).** The star cover (`opti/src/star_cover.py`,
  `opti/src/rect_cover.py`), the chain-first GPU annealer
  (`opti/src/gpu_sc.py`, `opti/src/annealer_stretch.py`) and the exact SAT
  completion of stalled states (`opti/scripts/rigid_analysis/domain_chain_sat.py`,
  `complete_stall.py`, `cegar_cover.py`).
- **Lower bounds (rigid subcomplexes).** `opti/scripts/rigid_analysis/rigid.py`
  builds the rigid hollow-triangle, closed-walk and graph families and decides
  the avoidance instances (`rigid.py bound`), or audits a stalled state
  (`rigid.py audit`); `opti/scripts/relaxation/` holds the H¹-cycle relaxation
  and the rigid-sphere tools.
- **Auditor.** `opti/src/independent_check.py` is a pure-Python checker that
  rebuilds each triangulation from its facet list and shares no code with the
  search.

Every cover the paper reports is archived and independently re-verified in
[simplicial-complexity-certificates](https://github.com/MiguelPerezI/simplicial-complexity-certificates)
(`python3 audit/verify_all.py --check`). The version described in the arXiv
paper carries the tag `arxiv-v1` in both repositories.

## Layout

| Path | Contents |
|---|---|
| `opti/paper-overleaf/arxiv/` | arXiv source of the paper (`main.tex` + the three `algorithmicx` style files) |
| `opti/paper-overleaf/paper/` | working copy of the paper, compiled PDF, CP-SAT model `scs_sat.py` |
| `opti/src/` | annealer, star cover, auditor, certificate writers |
| `opti/scripts/rigid_analysis/` | rigid-subcomplex bounds, stall audits, exact domain decisions; `certificates/` holds the bound logs, maximal-rectangle witnesses and learned non-domain sets |
| `opti/scripts/relaxation/` | H¹-cycle relaxation (`h1cegar.py`), rigid spheres, slice families |
| `opti/results/` | every run: covers (`cover.txt`, `planners.txt`, `summary.json`), stalled states, ladders |
| `opti/logs/`, `opti/experiments/` | run logs and lab notebooks |
| `opti/webapp/` | local viewer for `K × K` covers |

`opti/README.md` documents the first-phase sphere certificates in detail.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                 # star cover, auditor, CPU tools
pip install -r opti/requirements-sat.txt        # CP-SAT and python-sat for the exact tools

cd opti
python3 src/independent_check.py results/icosa_k4_exact      # audit a cover (no dependencies)
python3 scripts/rigid_analysis/rigid.py bound mobius 3        # SC_strict(Moebius band) >= 2
```

The annealer needs an NVIDIA GPU (`pip install -r requirements-cuda.txt`);
`opti/reproduce_all.sh` lists the CPU and GPU reproduction recipes.

## License

MIT (see `LICENSE`).
