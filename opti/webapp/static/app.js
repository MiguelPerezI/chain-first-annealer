/* app.js — wiring: runs list, the map, the launch form, the live tab. */
'use strict';

const $ = (id) => document.getElementById(id);
const api = {
  async get(path) { const r = await fetch(path); return r.json(); },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-OptiMap': '1' },
      body: JSON.stringify(body),
    });
    return r.json();
  },
};

const STATE = { runs: [], selected: null, payload: null, plan: null, config: null, warm: null };

/* ── the map ─────────────────────────────────────────────────────────── */
const map = new KxkMap($('map'), { onHover: showTip });

function fmtPct(cov, nf) { return nf ? (100 * cov / nf).toFixed(1) + '%' : '—'; }

async function selectRun(run) {
  STATE.selected = run;
  $('map-status').hidden = false;
  $('map-status').textContent = 'computing the audit of ' + run.state + ' …';
  const res = await api.get('/api/map?state=' + encodeURIComponent(run.state));
  if (res.status === 'computing') {                 // cold S^6: poll
    setTimeout(() => selectRun(run), 1000);
    return;
  }
  if (res.ok === false) { $('map-status').textContent = 'error: ' + res.error; return; }
  STATE.payload = decodePayload(res);
  map.setPayload(STATE.payload);
  $('map-status').hidden = true;
  $('map-title').textContent = `${res.name} — k=${res.k}, ${res.stats.cov}/${res.nf} covered (${fmtPct(res.stats.cov, res.nf)})`;
  $('btn-export').disabled = false;
  renderLegend(res);
  renderNotes(res);
  renderWarmInfo(run, res);
  $('subtitle').textContent = `${res.state}  ·  k=${res.k}  ·  ${res.stats.cov}/${res.nf}`;
}

function renderLegend(p) {
  const rows = [];
  for (let j = 0; j < p.k; j++) {
    const col = p.domain_colors[j % p.domain_colors.length];
    const size = p.stats.sizes[j];
    const dr = (p.domain_rect || [])[j];
    const tag = dr ? ` ⊂ R${dr.R_v} ${dr.in_R}/${dr.rect_size}` : '';
    rows.push(`<span class="sw"><i style="background:${col}"></i>D<sub>${j}</sub> · ${size} facets${tag}</span>`);
  }
  rows.push(`<span class="sw"><i style="background:${p.hole_color}"></i>uncovered · ${p.stats.nhole}</span>`);
  const cov = p.stats.cov || 0;
  const multi = p.stats.inter_cov ? Object.entries(p.stats.inter_cov)
    .filter(([n]) => +n > 1).reduce((a, [, c]) => a + c, 0) : 0;
  rows.push(`<span class="sw muted">shared by ≥2 domains: <b>${multi}</b>` +
            (cov ? ` (${(100 * multi / cov).toFixed(0)}% of covered)` : '') + `</span>`);
  $('legend').innerHTML = rows.join('');
}

function renderNotes(p) {
  const s = p.stats;
  const inter = Object.entries(s.inter || {}).map(([k, v]) => `${k}: ${v}`).join(', ');
  const dep = Object.entries(s.depths || {}).map(([k, v]) => `${k}: ${v}`).join(', ');
  const lines = [
    `<b>${s.cov}</b> covered, <b>${s.nhole}</b> red on <b>${s.cells}</b> blocks ` +
    `(${s.full} entirely dead, ${s.part} partly).`,
    s.nhole ? `dead facets by |s∩t| {${inter}} · defect depth {${dep}}` : '',
    p.rho ? `ρ(${p.complex}) = ${p.rho}` : `ρ(${p.complex}) unknown`,
    p.h1_report ? `H1 audit: ${p.h1_report}` : '',
    `computed in ${p.compute_s}s`,
  ].filter(Boolean);
  $('notes').innerHTML = lines.map((l) => `<div>${l}</div>`).join('');
}

function showTip(h, mx, my) {
  const tip = $('tip');
  if (!h || h.p === null || !STATE.payload) { tip.hidden = true; return; }
  const p = STATE.payload, f = h.facet;
  const tri = (x) => p.triangles[x].join('');
  const verts = facetGridVertices(p, h.i, h.j, h.p).map(([a, b]) => `(${a},${b})`).join(' ');
  const mem = p.members(f);
  const own = mem.length ? mem.map((j) => `D${j}`).join(', ') : 'none';
  const cov = p.covered[f] ? 'covered' : 'UNCOVERED';
  tip.innerHTML =
    `<b>facet ${f}</b> · ${cov}<br>` +
    `block (T${h.i} ${tri(h.i)} × T${h.j} ${tri(h.j)}) · p<sub>${h.p}</sub> ${pathLabel(p, h.i, h.j, h.p)}<br>` +
    `owning domains: <b>${own}</b>${mem.length > 1 ? ` (shared by ${mem.length})` : ''}<br>` +
    `grid vertices ${verts} · defect depth ${p.depth[f]}`;
  tip.hidden = false;
  const wrap = $('map-wrap').getBoundingClientRect();
  tip.style.left = Math.min(mx + 12, wrap.width - 320) + 'px';
  tip.style.top = Math.min(my + 12, wrap.height - 90) + 'px';
}

/* ── runs list ───────────────────────────────────────────────────────── */
function renderRuns() {
  const q = $('filter').value.toLowerCase();
  const rows = STATE.runs.filter((r) =>
    !q || (r.run + ' ' + (r.complex || '')).toLowerCase().includes(q));
  $('run-count').textContent = `${rows.length}/${STATE.runs.length}`;
  $('runs').innerHTML = rows.map((r) => {
    const pct = fmtPct(r.coverage, r.nf);
    const badges = [
      r.complete ? '<span class="b ok">cover</span>' : '',
      r.live ? '<span class="b live">live</span>' : '',
      r.phase ? `<span class="b">${r.phase}</span>` : '',
      r.k_below_rho ? `<span class="b warn" title="k < ρ: no product cover exists">k&lt;ρ</span>` : '',
    ].join(' ');
    return `<div class="run" data-run="${r.run}">
      <div class="r1"><b>${r.run}</b> ${badges}</div>
      <div class="r2"><span>${r.complex} k=${r.k}</span><span>${r.coverage}/${r.nf} · ${pct}</span></div>
      <div class="bar"><i style="width:${r.nf ? 100 * r.coverage / r.nf : 0}%"></i></div>
      <div class="r3">${(r.sizes || []).join('+') || ''}</div>
    </div>`;
  }).join('');
  document.querySelectorAll('#runs .run').forEach((el) => {
    el.onclick = () => {
      document.querySelectorAll('#runs .run').forEach((x) => x.classList.remove('sel'));
      el.classList.add('sel');
      selectRun(STATE.runs.find((r) => r.run === el.dataset.run));
    };
  });
}

/* ── launch form ─────────────────────────────────────────────────────── */
const FIELDS = [
  ['k', 5], ['cmax', 40], ['S', 4096], ['iters', 5000], ['epochs', 100], ['seed', 0],
  ['T0', 0.9], ['T1', 0.03], ['beta', 5], ['gamma', 0.5], ['mu', 0], ['guide', 3],
  ['guide_max', 11], ['ptrack', 0.3], ['pstretch', 0], ['pcollapse', 0], ['time_limit', 3600],
];

function renderParams() {
  $('params').innerHTML = FIELDS.map(([k, v]) =>
    `<label>${k}<input id="p-${k}" type="number" step="any" value="${v}"></label>`).join('');
}

function readParams() {
  const out = {};
  for (const [k] of FIELDS) out[k] = parseFloat($('p-' + k).value);
  return out;
}

const SIM_PRESET = { k: 3, S: 32, iters: 100, epochs: 5, time_limit: 300, cmax: 8 };

function renderWarmInfo(run, payload) {
  STATE.warm = { run, payload };
  const local = STATE.config && STATE.config.runner && STATE.config.runner.local;
  if (local) {                    // CUDASIM is ~1000x slower: keep it tiny
    for (const [k, v] of Object.entries(SIM_PRESET)) if ($('p-' + k)) $('p-' + k).value = v;
  }
  const warm = document.querySelector('input[name=mode]:checked').value === 'warm';
  $('warm-info').innerHTML = (warm
    ? `seeding from <code>${run.state}</code> (${payload.stats.cov}/${payload.nf}, k=${payload.k})`
    : 'cold start (no seed)') +
    (local ? '<br><b>local simulation</b>: the same kernels through numba\'s CUDA simulator, ' +
             'so keep S/iters/epochs tiny (' + JSON.stringify(SIM_PRESET) + ') — it is for ' +
             'exercising launch → live → stop, not for searching.' : '');
  // pre-fill the knobs from the state's own run, so "continue" means "carry on"
  const p = run.params || {};
  for (const [k] of FIELDS) {
    const v = k === 'k' ? run.k : p[k];
    if (v !== undefined && v !== null && $('p-' + k)) $('p-' + k).value = v;
  }
}

async function plan() {
  const mode = document.querySelector('input[name=mode]:checked').value;
  const body = {
    complex: STATE.selected ? STATE.selected.complex : $('c-complex') ? $('c-complex').value : 'rp2',
    params: readParams(),
    warm_from: mode === 'warm' && STATE.selected ? STATE.selected.state : 'cold',
    continue_run: $('opt-cont').checked && STATE.selected ? STATE.selected.run : null,
    checkpoint_every: 1,
  };
  const res = await api.post('/api/plan', body);
  $('plan-out').innerHTML = res.ok
    ? `<div class="muted">would write <code>${res.outdir}</code>, log <code>${res.log}</code></div>
       <pre class="cmd">${res.command.replace(/ /g, '\n  ')}</pre>` +
      (res.warnings || []).map((w) => `<div class="warnbox ${w.level}">${w.msg}</div>`).join('')
    : `<div class="warnbox error">${res.error}</div>`;
  STATE.plan = res.ok ? { body, token: res.token } : null;
  $('btn-launch').disabled = !res.ok;
  $('launch-msg').textContent = '';
}

async function launch() {
  if (!STATE.plan) return;
  const res = await api.post('/api/launch', Object.assign({}, STATE.plan.body, { token: STATE.plan.token }));
  $('launch-msg').textContent = res.ok
    ? `launched ${res.job.run} (pid ${res.job.pid}) — see the Live tab`
    : `not launched: ${res.error || (res.output || '').slice(-300)}`;
  if (res.ok) { loadJobs(); document.querySelector('.tab[data-tab=live]').click(); }
  $('btn-launch').disabled = true;
}

/* ── live tab ────────────────────────────────────────────────────────── */
let liveTimer = null;

function renderJobs(jobs) {
  if (!jobs.length) { $('jobs').innerHTML = '<div class="muted">no jobs launched from here yet. ' +
    'The Live tab also follows any run whose log you point it at.</div>'; return; }
  $('jobs').innerHTML = jobs.map((j) => `
    <div class="job">
      <div><b>${j.run}</b> <span class="b ${j.status}">${j.status}</span>
        ${j.live_coverage != null ? `<span class="muted">${j.live_coverage} covered</span>` : ''}</div>
      <div class="muted">${j.outdir}${j.init ? ' · init ' + j.init : ' · cold'}</div>
      <div class="row"><button class="ghost" data-attach="${j.id}">watch</button>
        <button class="ghost danger" data-stop="${j.id}">stop</button></div>
    </div>`).join('');
  document.querySelectorAll('[data-attach]').forEach((b) => b.onclick = () => attach(b.dataset.attach));
  document.querySelectorAll('[data-stop]').forEach((b) => b.onclick = async () => {
    if (!confirm('Stop this run? It stops at the next epoch boundary and keeps its partial summary.')) return;
    const res = await api.post('/api/jobs/stop', { id: b.dataset.stop });
    $('launch-msg').textContent = res.ok ? 'stop signalled: ' + (res.output || '') : res.error;
    loadJobs();
  });
}

async function attach(jid) {
  const job = await api.get('/api/jobs/' + jid);
  $('live-body').hidden = false;
  $('live-cov').textContent = job.live_coverage != null ? `${job.live_coverage} covered` : job.status;
  drawChart(job.history || []);
  $('logtail').textContent = (job.log || []).map((l) => l.text).join('\n');
  $('logtail').scrollTop = $('logtail').scrollHeight;
  if (job.live_state) {
    const res = await api.get('/api/map?state=' + encodeURIComponent(job.live_state));
    if (!res.status) {
      STATE.payload = decodePayload(res);
      liveMap.setPayload(STATE.payload);
      renderLegend(res);
    }
  }
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => attach(jid), 5000);
}

function drawChart(hist) {
  const cv = $('chart'), dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr; cv.height = 120 * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cv.clientWidth, H = 120;
  ctx.clearRect(0, 0, W, H);
  if (!hist.length) return;
  const xs = hist.map((h) => h.elapsed || 0), ys = hist.map((h) => h.best || 0);
  const x0 = 0, x1 = Math.max(...xs, 1), y0 = 0, y1 = Math.max(...ys, 1) * 1.02;
  const X = (v) => 40 + (W - 50) * (v - x0) / (x1 - x0), Y = (v) => H - 18 - (H - 30) * (v - y0) / (y1 - y0);
  ctx.strokeStyle = '#cfd6de'; ctx.beginPath();
  ctx.moveTo(40, Y(y0)); ctx.lineTo(W - 10, Y(y0)); ctx.stroke();
  ctx.strokeStyle = '#2E6FA7'; ctx.lineWidth = 2; ctx.beginPath();
  hist.forEach((h, i) => { const x = X(h.elapsed || 0), y = Y(h.best || 0); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = '#4A4A4A'; ctx.font = '11px system-ui';
  ctx.fillText('best', 6, Y(y1) + 10);
  ctx.fillText(String(y1.toFixed(0)), 4, Y(y1) + 22);
  ctx.fillText(String(y0), 4, Y(y0));
  ctx.fillText((x1 / 60).toFixed(0) + ' min', W - 60, H - 4);
}

async function loadJobs() { renderJobs((await api.get('/api/jobs')).jobs || []); }

/* ── boot ────────────────────────────────────────────────────────────── */
(async function boot() {
  renderParams();
  STATE.config = await api.get('/api/config');
  const R = STATE.config.runner;
  $('runner-badge').textContent = R.local
    ? 'runner: local simulation (CUDASIM)'
    : (R.configured ? 'runner: ' + R.host : 'runner: not configured (browse/export only)');
  $('runner-badge').classList.toggle('warn', !R.configured);
  STATE.runs = (await api.get('/api/runs')).runs || [];
  renderRuns();
  loadJobs();
  $('filter').oninput = renderRuns;
  $('btn-fit').onclick = () => map.fit();
  $('opt-shared').onchange = (e) => { map.markShared = e.target.checked; map.draw(); };
  $('opt-dead').onchange = (e) => { map.outlineDead = e.target.checked; map.draw(); };
  document.querySelectorAll('input[name=mode]').forEach((r) => r.onchange = () => {
    if (STATE.selected) renderWarmInfo(STATE.selected, STATE.payload);
  });
  $('btn-plan').onclick = plan;
  $('btn-launch').onclick = launch;
  $('btn-export').onclick = async () => {
    if (!STATE.selected) return;
    $('export-link').textContent = 'rendering…';
    const res = await api.get('/api/export?state=' + encodeURIComponent(STATE.selected.state));
    $('export-link').innerHTML = res.ok
      ? (res.urls || []).map((u) => `<a href="${u}" target="_blank">${u.split('/').pop()}</a>`).join(' ')
      : 'export failed: ' + res.error;
  };
  document.querySelectorAll('.tab').forEach((t) => t.onclick = () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
    t.classList.add('active');
    $('tab-launch').hidden = t.dataset.tab !== 'launch';
    $('tab-live').hidden = t.dataset.tab !== 'live';
    if (t.dataset.tab === 'live') loadJobs();
  });
})();

const liveMap = (window.liveMap = new KxkMap($('live-map'), { onHover: () => {} }));
