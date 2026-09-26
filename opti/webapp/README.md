# K × K webapp

A local page for the annealing campaign: browse every saved state, see K × K the
way `scripts/render_kxk.py` draws it — but zoomable to the individual facet —
pick a state to continue from (or start cold), launch the run on the GPU host,
and follow it live.

```bash
bash webapp/serve.sh                 # http://127.0.0.1:8770
curl -s localhost:8770/api/runs | python -m json.tool | head
```

Needs a python with numba (for `gpu_sc`) **and** matplotlib (for the export
button): `serve.sh` defaults to `/home/mike666/whisper_env/bin/python`. No web
framework is involved — stdlib `http.server` plus one HTML page and Canvas 2D.

## What it shows

* **Red = uncovered. Every covered facet takes the colour of the domain that
  owns it**, one colour per domain (up to 24 distinct; the runs go to k = 11).
  Facets in more than one domain are marked (second owner's colour as an inner
  outline, a count in the tooltip, and a `shared by ≥2 domains` figure in the
  legend) — overlaps are never hidden. In the recorded states they are common:
  32 % of RP²'s covered facets, 18 % of Klein's.
* **Granular zoom.** The payload holds *every* facet (packed: 0.1 KB for RP²'s
  600, 7 KB for S⁶'s 59136), and the renderer never aggregates. Zoomed out it
  rasterises per-facet cells into an offscreen canvas and scales it; past ~8 px
  per cell it draws rects directly, culled to the viewport. Hit-testing is
  facet-exact at every zoom, and the tooltip reports the facet id, its block
  (`T_i` × `T_j` with both vertex triples), its staircase label (`p0 BBAA …`),
  the grid vertices `(a,b)` it spans, its owning domains and its defect depth.
* **Legend and notes** mirror the PNG's annotation: domain sizes, `D_j ⊂ R_v`,
  ρ(K) where this repo records it (unknown otherwise, never guessed), and the
  H₁ audit file when one sits next to the state.

## Launching

Configure the GPU host first — it is elastically addressed, so nothing is
hardcoded:

```bash
cp webapp/runner.example.json webapp/runner.json   # then set "host"
```

**No GPU? Set `"host": "local"`** and the *same kernels* run on this machine
through numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`) — the switch
`src/test_stretch_cpu.py` and `src/test_contracts.py` already use. It is what
the launch → live → stop loop can be exercised (and tested) with:

```bash
cp webapp/runner.example.json webapp/runner.json
python - <<'EOF'
import json, pathlib
p = pathlib.Path("webapp/runner.json"); d = json.loads(p.read_text())
d["host"] = "local"; p.write_text(json.dumps(d, indent=2))
EOF
```

The badge then reads `runner: local simulation (CUDASIM)` and the launch form
pre-fills a tiny preset (`k=3, S=32, iters=100, epochs=5`), because CUDASIM is
~1000× slower than the GPU: it is for exercising the pipeline on a small
complex (circle, S², the Möbius band), not for searching. A simulated
`circle` run really does finish and write a verified `cover.txt` /
`planners.txt`; a stopped one keeps its partial `summary.json`.

`Launch` tab → *continue from the selected state* (passes `--init <file>`, and
warns if k, cmax or the complex disagree with the seed) or *start cold*.

* **Preview command** composes the exact ssh command and shows it, together
  with warnings computed from the state: `k < ρ` (no product cover exists — so
  the run can only plateau), a complex/facet-count mismatch with the seed
  (`init_from_summary` does *not* check the complex), and a `cmax` shorter than
  the seed's reduced chains (which makes it raise at startup).
* **Launch** sends it only if the confirmation token still matches the body
  that was previewed. Runs auto-name to `results/<complex>_k<k>_<utc>/`, or into
  `<run>/phase<N>/` when continuing an existing run, and are launched with
  `PYTHONUNBUFFERED=1 nohup … --jsonl <outdir>/epochs.jsonl --checkpoint-every 1
  --graceful-stop`, so they are observable and stoppable.
* **Stop** sends SIGTERM; with `--graceful-stop` the run stops at the next epoch
  boundary and *keeps* its partial `summary.json`. A run started by hand without
  that flag will die without saving anything — the dialog says so.

Nothing reaches a shell unquoted: numbers go through the CLI's own
`validation.validate_parameters`, `--complex` through the whitelist, run names
through a character whitelist, and the argv list never uses `shell=True`.

## Following a run

The **Live** tab shows the epoch curve, the log tail, and the incumbent's K × K
map from `<outdir>/live_state.json` (written atomically every N epochs), polled
every 5 s. Runs it did not start are followed by pointing the app at their log
and checkpoints; if the run predates `--checkpoint-every`, the curve and the log
still work, only the live map needs the flag.

## The flags this relies on (both opt-in, default off)

`src/annealer_stretch.py` and `src/gpu_sc.py` gained three, all inert when
unused and byte-identical to before:

| flag | what it does |
|---|---|
| `--jsonl PATH` | append one JSON object per epoch (machine-readable progress) |
| `--checkpoint-every N` | atomically write `<outdir>/live_state.json` (the incumbent's `best_chains`, same keys as the partial summary, so `--init` and the renderer read it unchanged) |
| `--graceful-stop` / `--stop-file PATH` | stop at the next epoch boundary and keep the partial `summary.json`; without it a signal still kills the run and saves nothing, which is the long-standing behaviour scripts rely on |

## API (all JSON; POSTs need the `X-OptiMap: 1` header)

| endpoint | |
|---|---|
| `GET /api/config` | repo, python, complexes, ρ table, runner config state |
| `GET /api/runs` | every result dir with a state: complex, k, coverage/nf, sizes, status, ρ, `k<ρ` flag |
| `GET /api/map?state=` | the facet-level payload (covered/owner/membership packed, K facets, masks, palette, stats) + ETag |
| `POST /api/plan` | dry run: composed command + warnings + confirmation token |
| `POST /api/launch` | launch (token required) |
| `GET /api/jobs`, `/api/jobs/<id>` | registry; `…/history`, `…/log` for the curve and tail |
| `POST /api/jobs/stop` | SIGTERM the run |
| `GET /api/export?state=` | run `render_kxk.py` and return the PNG/PDF |

## Tests

```bash
python -B scripts/test_webapp_payload.py      # payload contract, live parser, telemetry hooks
```

It checks the payload round-trips exactly (against `kxk_state.analyse`), that the
**client's** facet-vertex reconstruction equals the Python's for every facet of
RP² and Klein (via node, skipped if absent), that the RP² k=5 hole is the 60
private facets of R₅, and that the telemetry hooks are inert when unused. It
runs in CI (`.github/workflows/contracts.yml`) with numpy/numba only — which is
why the data layer lives in `scripts/kxk_state.py` rather than in the renderer.
