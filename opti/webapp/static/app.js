/* app.js — K×K webapp: runs list, map, launch form, live view, target mode */
'use strict';

const $ = (id) => document.getElementById(id);
const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok && r.status !== 202) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-OptiMap': '1' },
      body: JSON.stringify(body),
    });
    return r.json();
  },
};

const STATE = {
  runs: [], selected: null, payload: null, plan: null, config: null,
  targetMode: false, selectedDomain: -1, targetQueue: [],
  liveJobId: null, liveTimer: null,
};

/* ── map setup ───────────────────────────────────────────────────────── */
const map = new KxkMap($('map'), {
  onHover: showTip,
  onClick: onMapClick,
});
const liveMap = new KxkMap($('live-map'), { onHover: () => {} });
// also create a map instance for the target tab's live canvas (if present)
const targetLiveMap = $('target-live-map') ? new KxkMap($('target-live-map'), { onHover: () => {} }) : null;

function fmtPct(cov, nf) { return nf ? (100 * cov / nf).toFixed(1) + '%' : '—'; }

function showOverlay(text, spin) {
  const el = $('map-status');
  $('status-text').textContent = text;
  $('spinner').classList.toggle('hidden', !spin);
  el.hidden = false;
  el.classList.remove('hidden');
}
function hideOverlay() {
  const el = $('map-status');
  el.hidden = true;
  el.classList.add('hidden');
}

/* ── select a run and load its map ───────────────────────────────────── */
async function selectRun(run) {
  STATE.selected = run;
  showOverlay('loading ' + run.run + '…', true);
  try {
    const res = await api.get('/api/map?state=' + encodeURIComponent(run.state));
    if (res.status === 'computing') {
      setTimeout(() => selectRun(run), 1000);
      return;
    }
    if (res.ok === false) { showOverlay('error: ' + res.error, false); return; }
    STATE.payload = decodePayload(res);
    map.setPayload(STATE.payload);
    map.fit();
    hideOverlay();
    $('map-title').textContent = `${res.name} — k=${res.k}, ${res.stats.cov}/${res.nf} (${fmtPct(res.stats.cov, res.nf)})`;
    $('btn-export').disabled = false;
    $('btn-target').disabled = false;
    renderLegend(res);
    renderNotes(res);
    renderWarmInfo(run, res);
    $('subtitle').textContent = `${run.run}  ·  k=${res.k}  ·  ${res.stats.cov}/${res.nf}`;
    // update target tab state
    $('target-state').textContent = `Loaded: ${run.run} (${res.stats.cov}/${res.nf} covered, ${res.stats.nhole} red)`;
    $('target-state').className = res.stats.nhole > 0 ? 'info-box warn' : 'info-box success';
  } catch (err) {
    showOverlay('error: ' + err.message, false);
  }
}

/* ── legend with clickable domain swatches ───────────────────────────── */
function renderLegend(p) {
  const rows = [];
  for (let j = 0; j < p.k; j++) {
    const col = p.domain_colors[j % p.domain_colors.length];
    const size = p.stats.sizes[j];
    const dr = (p.domain_rect || [])[j];
    const tag = dr ? ` ⊂R${dr.R_v} ${dr.in_R}/${dr.rect_size}` : '';
    const sel = STATE.selectedDomain === j ? ' selected' : '';
    const dim = STATE.selectedDomain >= 0 && STATE.selectedDomain !== j ? ' dim' : '';
    rows.push(`<span class="sw${sel}${dim}" data-domain="${j}"><i style="background:${col}"></i>D<sub>${j}</sub> ${size}${tag}</span>`);
  }
  rows.push(`<span class="sw" data-domain="-1"><i style="background:${p.hole_color}"></i>uncovered ${p.stats.nhole}</span>`);
  const cov = p.stats.cov || 0;
  const multi = p.stats.inter_cov ? Object.entries(p.stats.inter_cov)
    .filter(([n]) => +n > 1).reduce((a, [, c]) => a + c, 0) : 0;
  if (multi > 0) {
    rows.push(`<span class="sw muted">shared: ${multi} (${(100 * multi / cov).toFixed(0)}%)</span>`);
  }
  $('legend').innerHTML = rows.join('');
  // click a domain swatch to highlight
  document.querySelectorAll('#legend .sw[data-domain]').forEach(el => {
    el.onclick = () => {
      const d = parseInt(el.dataset.domain);
      STATE.selectedDomain = STATE.selectedDomain === d ? -1 : d;
      map.highlightDomain = STATE.selectedDomain;
      map.draw();
      renderLegend(STATE.payload); // re-render legend with sel/dim classes
    };
  });
}

function renderNotes(p) {
  const s = p.stats;
  const lines = [];
  lines.push(`<strong>${s.cov}</strong> covered, <strong>${s.nhole}</strong> red on <strong>${s.cells}</strong> blocks (${s.full} dead, ${s.part} partial)`);
  if (s.nhole) {
    const dep = Object.entries(s.depths || {}).map(([k, v]) => `${k}:${v}`).join(' ');
    lines.push(`defect depth {${dep}}`);
  }
  if (p.rho) lines.push(`ρ(${p.complex}) = ${p.rho}`);
  $('notes').innerHTML = lines.map(l => `<div>${l}</div>`).join('');
}

/* ── tooltip ─────────────────────────────────────────────────────────── */
function showTip(h, mx, my) {
  const tip = $('tip');
  if (!h || h.facet == null || !STATE.payload) { tip.classList.remove('show'); return; }
  const p = STATE.payload, f = h.facet;
  const tri = x => p.triangles[x].join('');
  const verts = facetGridVertices(p, h.i, h.j, h.p).map(([a, b]) => `(${a},${b})`).join(' ');
  const mem = p.members(f);
  const cov = p.covered[f];
  const own = mem.length ? mem.map(j =>
    `<span class="tip-domain" style="background:${p.domain_colors[j]}"></span>D${j}`
  ).join(' ') : '—';
  tip.innerHTML =
    `<b>facet ${f}</b> · ${cov ? 'covered' : '<span style="color:#fca5a5">UNCOVERED</span>'}<br>` +
    `T${h.i} [${tri(h.i)}] × T${h.j} [${tri(h.j)}] · p${h.p} ${pathLabel(p, h.i, h.j, h.p)}<br>` +
    `domains: ${own}${mem.length > 1 ? ` (shared ×${mem.length})` : ''}<br>` +
    `vertices: ${verts} · depth: ${p.depth[f]}`;
  tip.classList.add('show');
  const wrap = $('map-wrap').getBoundingClientRect();
  tip.style.left = Math.min(mx + 12, wrap.width - 350) + 'px';
  tip.style.top = Math.min(my + 12, wrap.height - 80) + 'px';
}

/* ── map click ────────────────────────────────────────────────────────── */
function onMapClick(h) {
  console.log('map click', h, 'targetMode', STATE.targetMode, 'selectedDomain', STATE.selectedDomain);
  if (!STATE.payload || h.facet == null) return;
  const p = STATE.payload, f = h.facet;

  // In target mode: domain selection + facet targeting
  if (STATE.targetMode) {
    if (p.covered[f]) {
      // clicking a covered facet: select its domain
      STATE.selectedDomain = p.owner[f];
      map.highlightDomain = STATE.selectedDomain;
      map.draw();
      renderLegend(p);
      setTargetInfo(`Selected domain: <strong>D${p.owner[f]}</strong> (${p.stats.sizes[p.owner[f]]} facets). Now click a red facet to target.`);
      return;
    }
    // clicking a red facet
    if (STATE.selectedDomain < 0) {
      setTargetInfo('⚠️ Select a domain first — click a coloured facet or a legend swatch');
      return;
    }
    setTargetInfo(`Checking H₁ obstruction for facet ${f} → D${STATE.selectedDomain}…`);
    checkH1(f, STATE.selectedDomain);
    return;
  }

  // Outside target mode: clicking a covered facet highlights its domain
  if (p.covered[f]) {
    STATE.selectedDomain = STATE.selectedDomain === p.owner[f] ? -1 : p.owner[f];
    map.highlightDomain = STATE.selectedDomain;
    map.draw();
    renderLegend(p);
  }
}

/** Show feedback in the target-bar (always visible in target mode) */
function setTargetInfo(html) {
  $('target-info').innerHTML = html;
  // also update the target tab panel
  $('target-result').hidden = false;
  $('target-result').innerHTML = html;
}

async function checkH1(facet, chain) {
  if (!STATE.selected) return;
  try {
    const res = await api.post('/api/h1-check', {
      state: STATE.selected.state, facet, chain
    });
    if (res.absorbable) {
      const msg = `✅ <strong>Absorbable!</strong> f=${facet} → D${chain}: H₁ rank=0, depth=${res.defect_depth}` +
        (res.bad_links ? `, bad links [${res.bad_links.join(',')}]` : '');
      setTargetInfo(msg);
      $('target-result').className = 'info-box success';
      addTarget(facet, chain);
    } else {
      const msg = `❌ <strong>Obstructed.</strong> f=${facet} → D${chain}: H₁ rank=${res.h1_rank}. ` +
        (res.reason || 'This domain would wind — no chain can cover it.');
      setTargetInfo(msg);
      $('target-result').className = 'info-box error';
    }
  } catch (err) {
    setTargetInfo('Error: ' + err.message);
    $('target-result').className = 'info-box error';
  }
}

function addTarget(facet, chain) {
  STATE.targetQueue.push({ facet, chain });
  map.targetFacets.add(facet);
  map.draw();
  renderTargetQueue();
  $('btn-launch-target').disabled = false;
}

function renderTargetQueue() {
  if (!STATE.targetQueue.length) { $('target-queue').hidden = true; return; }
  $('target-queue').hidden = false;
  const p = STATE.payload;
  $('target-list').innerHTML = STATE.targetQueue.map((t, idx) =>
    `<div class="target-item">
      <span class="facet-swatch" style="background:${p.domain_colors[t.chain]}"></span>
      f=${t.facet} → D${t.chain}
      <button onclick="removeTarget(${idx})" style="margin-left:auto;cursor:pointer;border:none;background:none;color:var(--muted)">✕</button>
    </div>`
  ).join('');
}
window.removeTarget = function(idx) {
  const t = STATE.targetQueue.splice(idx, 1)[0];
  map.targetFacets.delete(t.facet);
  map.draw();
  renderTargetQueue();
  if (!STATE.targetQueue.length) $('btn-launch-target').disabled = true;
};

/* ── target mode toggle ──────────────────────────────────────────────── */
function toggleTargetMode(on) {
  STATE.targetMode = on;
  map.targetMode = on;
  $('btn-target').classList.toggle('active', on);
  $('target-bar').hidden = !on;
  if (!on) {
    map.targetFacets.clear();
    map.auditDoors = null;
    STATE.targetQueue = [];
    STATE.selectedDomain = -1;
    map.highlightDomain = -1;
  }
  map.draw();
  if (STATE.payload) renderLegend(STATE.payload);
}

async function runStallAudit() {
  if (!STATE.selected) return;
  $('btn-audit').textContent = 'auditing…';
  $('btn-audit').disabled = true;
  try {
    const res = await api.post('/api/stall-audit', { state: STATE.selected.state });
    if (res.doors) {
      const doors = new Map();
      for (const d of res.doors) {
        if (!doors.has(d.facet)) doors.set(d.facet, []);
        doors.get(d.facet).push(d.chain);
      }
      map.auditDoors = doors;
      map.draw();
      const total = res.doors.length;
      const facets = doors.size;
      $('target-info').textContent = `🎯 ${facets} facets have H₁ doors (${total} pairs). Green border = absorbable. Click a domain, then a green-bordered facet.`;
    } else {
      $('target-info').textContent = '🎯 Audit complete: ' + (res.error || 'no doors found');
    }
  } catch (err) {
    $('target-info').textContent = 'Audit error: ' + err.message;
  }
  $('btn-audit').textContent = 'Run stall audit';
  $('btn-audit').disabled = false;
}

/* ── runs list ───────────────────────────────────────────────────────── */
function renderRuns() {
  const q = $('filter').value.toLowerCase();
  const rows = STATE.runs.filter(r =>
    !q || (r.run + ' ' + (r.complex || '')).toLowerCase().includes(q));
  $('run-count').textContent = rows.length;
  $('runs').innerHTML = rows.map(r => {
    const pct = r.nf ? (100 * r.coverage / r.nf) : 0;
    const barClass = pct >= 100 ? 'full' : pct >= 70 ? 'partial' : 'low';
    const badges = [
      r.complete ? '<span class="b ok">✓ cover</span>' : '',
      r.live ? '<span class="b live">● live</span>' : '',
      r.k_below_rho ? '<span class="b warn">k&lt;ρ</span>' : '',
    ].filter(Boolean).join(' ');
    return `<div class="run" data-run="${r.run}">
      <div class="r1"><b>${r.run}</b> ${badges}</div>
      <div class="r2"><span>${r.complex || '?'} k=${r.k}</span><span>${r.coverage}/${r.nf} · ${fmtPct(r.coverage, r.nf)}</span></div>
      <div class="bar"><i class="${barClass}" style="width:${pct}%"></i></div>
      <div class="r3">${(r.sizes || []).join('+')}</div>
    </div>`;
  }).join('');
  document.querySelectorAll('#runs .run').forEach(el => {
    el.onclick = () => {
      document.querySelectorAll('#runs .run').forEach(x => x.classList.remove('sel'));
      el.classList.add('sel');
      selectRun(STATE.runs.find(r => r.run === el.dataset.run));
    };
  });
}

/* ── launch form ─────────────────────────────────────────────────────── */
const FIELDS = [
  ['complex', 'rp2'], ['k', 5], ['cmax', 40], ['S', 4096], ['iters', 5000], ['epochs', 100],
  ['seed', 0], ['T0', 0.9], ['T1', 0.03], ['beta', 5], ['gamma', 0.5], ['mu', 0],
  ['guide', 3], ['guide_max', 11], ['ptrack', 0.3], ['pstretch', 0], ['pcollapse', 0],
  ['time_limit', 3600],
];

function renderParams() {
  $('params').innerHTML = FIELDS.map(([k, v]) => {
    if (k === 'complex') return ''; // handled separately by run selection
    return `<label>${k}<input id="p-${k}" type="number" step="any" value="${v}"></label>`;
  }).join('');
}

function readParams() {
  const out = {};
  for (const [k] of FIELDS) {
    if (k === 'complex') continue;
    out[k] = parseFloat($('p-' + k).value);
  }
  return out;
}

function renderWarmInfo(run, payload) {
  const warm = document.querySelector('input[name=mode]:checked').value === 'warm';
  const local = STATE.config && STATE.config.runner && STATE.config.runner.local;
  $('warm-info').innerHTML = warm
    ? `<div class="info-box">Seeding from <code>${run.run}</code> (${payload.stats.cov}/${payload.nf}, k=${payload.k})</div>`
    : '<div class="info-box">Cold start — new random population</div>';
  // pre-fill knobs from the run's params
  const p = run.params || {};
  for (const [k] of FIELDS) {
    if (k === 'complex') continue;
    const v = k === 'k' ? run.k : p[k];
    if (v !== undefined && v !== null && $('p-' + k)) $('p-' + k).value = v;
  }
  if (local) {
    // CUDASIM preset: tiny values
    for (const [k, v] of [['S', 32], ['iters', 100], ['epochs', 5], ['time_limit', 300], ['cmax', 8]]) {
      if ($('p-' + k)) $('p-' + k).value = v;
    }
  }
}

async function plan() {
  const mode = document.querySelector('input[name=mode]:checked').value;
  const body = {
    complex: STATE.selected ? STATE.selected.complex : 'rp2',
    params: readParams(),
    warm_from: mode === 'warm' && STATE.selected ? STATE.selected.state : 'cold',
    continue_run: $('opt-cont').checked && STATE.selected ? STATE.selected.run : null,
    checkpoint_every: 1,
  };
  $('btn-plan').textContent = 'checking…';
  try {
    const res = await api.post('/api/plan', body);
    $('plan-out').innerHTML = res.ok
      ? `<div class="info-box">→ <code>${res.outdir}</code></div>
         <pre class="cmd">${(res.command || '').replace(/ --/g, ' \\\n  --')}</pre>` +
        (res.warnings || []).map(w => `<div class="warnbox ${w.level}">${w.msg}</div>`).join('')
      : `<div class="warnbox error">${res.error}</div>`;
    STATE.plan = res.ok ? { body, token: res.token } : null;
    $('btn-launch').disabled = !res.ok;
  } catch (err) {
    $('plan-out').innerHTML = `<div class="warnbox error">${err.message}</div>`;
  }
  $('btn-plan').textContent = 'Preview command';
  $('launch-msg').textContent = '';
}

async function launch() {
  if (!STATE.plan) return;
  $('btn-launch').disabled = true;
  $('btn-launch').textContent = 'launching…';
  try {
    const res = await api.post('/api/launch', { ...STATE.plan.body, token: STATE.plan.token });
    $('launch-msg').textContent = res.ok
      ? `✓ Launched ${res.job.run} (pid ${res.job.pid})`
      : `✗ ${res.error || (res.output || '').slice(-300)}`;
    if (res.ok) {
      loadJobs();
      document.querySelector('.tab[data-tab=live]').click();
      attach(res.job.id);
    }
  } catch (err) {
    $('launch-msg').textContent = '✗ ' + err.message;
  }
  $('btn-launch').textContent = 'Launch on GPU';
}

async function launchTargeted() {
  if (!STATE.targetQueue.length || !STATE.selected) return;
  const targets = STATE.targetQueue.map(t => t.chain >= 0 ? `${t.facet}:${t.chain}` : `${t.facet}`).join(',');
  const run = STATE.selected;

  // switch to Target tab
  document.querySelector('.tab[data-tab=target]').click();
  const status = $('target-attack-status');
  const log = $('target-log');

  // build the command the user would paste on the GPU server
  const p = run.params || {};
  const k = run.k || p.k || 5;
  const cmax = p.cmax || 40;
  const seed = Math.floor(Math.random() * 10000);
  const outdir = `results/${run.run}/phase_attack_${Date.now()}`;
  const initPath = run.state.replace(/.*\/opti\//, '');  // repo-relative

  const cmd = [
    `cd ~/chain-first-annealer/opti`,
    ``,
    `~/miniforge3/bin/python3 -u src/annealer_stretch.py \\`,
    `  --complex ${run.complex} --k ${k} --cmax ${cmax} \\`,
    `  --S 4096 --iters 5000 --epochs 20 --seed ${seed} \\`,
    `  --T0 0.6 --T1 0.03 --beta 5 --gamma 0.5 --mu 0 \\`,
    `  --guide 8 --guide-max 15 --ptrack 0.3 \\`,
    `  --pstretch 0.5 --pcollapse 0.3 \\`,
    `  --init ${initPath} \\`,
    `  --outdir ${outdir} \\`,
    `  --jsonl ${outdir}/epochs.jsonl \\`,
    `  --checkpoint-every 1 --graceful-stop \\`,
    `  --time-limit 300`,
  ].join('\n');

  status.innerHTML =
    `<div class="info-box success" style="margin-bottom:8px">` +
    `🎯 <strong>Targeted attack command ready</strong><br>` +
    `Facets: <code>${targets}</code> · Warm start from: <code>${run.run}</code></div>` +
    `<div style="font-size:11px;color:var(--muted);margin-bottom:4px">` +
    `Copy this command and paste it on the GPU server (${STATE.config?.runner?.host || 'mike666@136.108.172.74'}):</div>`;

  log.textContent = cmd;
  log.style.userSelect = 'all';  // easy to select all

  // add a copy button
  const copyBtn = document.createElement('button');
  copyBtn.className = 'toolbar-btn';
  copyBtn.textContent = '📋 Copy to clipboard';
  copyBtn.style.marginTop = '6px';
  copyBtn.onclick = () => {
    navigator.clipboard.writeText(cmd).then(() => {
      copyBtn.textContent = '✅ Copied!';
      setTimeout(() => { copyBtn.textContent = '📋 Copy to clipboard'; }, 2000);
    });
  };
  status.appendChild(copyBtn);

  // also explain what the flags do for this specific attack
  const explain = document.createElement('div');
  explain.className = 'info-box';
  explain.style.marginTop = '8px';
  explain.innerHTML =
    `<strong>What this does:</strong><br>` +
    `• Warm-starts from <code>${run.run}</code> (${run.coverage}/${run.nf} covered)<br>` +
    `• <code>--pstretch 0.5 --pcollapse 0.3</code>: the stretch/collapse compound moves that target uncovered facets<br>` +
    `• <code>--guide 8 --guide-max 15</code>: high adaptive weight on the uncovered facets to concentrate effort there<br>` +
    `• <code>--checkpoint-every 1</code>: writes live_state.json each epoch so you can watch or resume<br>` +
    `• <code>--graceful-stop</code>: Ctrl+C saves the partial state instead of losing it<br>` +
    `• <code>--time-limit 300</code>: 5 minutes (adjust as needed)<br>` +
    `<br><strong>To target specific facets</strong>, add to the command:<br>` +
    `<code style="background:#1e293b;padding:2px 6px;border-radius:3px;color:#94a3b8">` +
    `# not yet a CLI flag — the stretch move targets ALL uncovered facets;</code><br>` +
    `<code style="background:#1e293b;padding:2px 6px;border-radius:3px;color:#94a3b8">` +
    `# the guide weight concentrates effort on the hardest ones</code>`;
  status.appendChild(explain);
}

/* ── live tab ────────────────────────────────────────────────────────── */
async function loadJobs() {
  try {
    const res = await api.get('/api/jobs');
    renderJobs(res.jobs || []);
  } catch (err) {
    $('jobs').innerHTML = `<div class="info-box error">Error loading jobs: ${err.message}</div>`;
  }
}

function renderJobs(jobs) {
  if (!jobs.length) {
    $('jobs').innerHTML = '<div class="muted" style="padding:12px;font-size:12px">No jobs yet. Launch a run from the Launch tab.</div>';
    return;
  }
  $('jobs').innerHTML = jobs.map(j => `
    <div class="job">
      <div class="job-head"><b>${j.run}</b>
        <span class="b ${j.status}">${j.status}</span>
        ${j.live_coverage != null ? `<span class="muted">${j.live_coverage} covered</span>` : ''}
      </div>
      <div class="muted" style="font-size:10px">${j.outdir}</div>
      <div class="action-row" style="margin-top:4px">
        <button class="toolbar-btn" onclick="attach('${j.id}')">📡 Watch</button>
        ${j.status === 'running' ? `<button class="toolbar-btn danger-btn" onclick="stopJob('${j.id}')" style="font-size:11px;color:white">⏹ Stop</button>` : ''}
      </div>
    </div>`).join('');
}

async function attach(jid) {
  STATE.liveJobId = jid;
  $('live-body').hidden = false;
  $('btn-stop').disabled = false;
  clearTimeout(STATE.liveTimer);
  try {
    const job = await api.get('/api/jobs/' + jid);
    $('live-cov').textContent = job.live_coverage != null
      ? `${job.live_coverage} / ${job.params?.nf || '?'}`
      : job.status;
    drawChart(job.history || []);
    renderLog(job.log || []);
    // live map
    if (job.live_state) {
      try {
        const mp = await api.get('/api/map?state=' + encodeURIComponent(job.live_state));
        if (mp.nf) {
          liveMap.setPayload(decodePayload(mp));
          m.fit();
        }
      } catch (_) {}
    }
    $('live-dot').classList.toggle('pulse', job.status === 'running');
    $('btn-stop').disabled = job.status !== 'running';
  } catch (err) {
    $('live-cov').textContent = 'error: ' + err.message;
  }
  // poll
  if (STATE.liveJobId === jid) {
    STATE.liveTimer = setTimeout(() => attach(jid), 5000);
  }
}
window.attach = attach;

async function stopJob(jid) {
  if (!confirm('Stop this run? (Stops at the next epoch boundary, keeps partial summary)')) return;
  try {
    const res = await api.post('/api/jobs/stop', { id: jid });
    $('launch-msg').textContent = '⏹ ' + (res.output || 'stop signal sent');
    loadJobs();
  } catch (err) {
    $('launch-msg').textContent = '✗ ' + err.message;
  }
}
window.stopJob = stopJob;

function renderLog(lines) {
  const el = $('logtail');
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  el.innerHTML = lines.map(l => {
    const cls = l.kind === 'epoch' ? 'l-epoch' : l.kind === 'done' ? 'l-done' :
                l.kind === 'header' ? 'l-header' : l.kind === 'result' ? 'l-result' : '';
    return cls ? `<span class="${cls}">${escHtml(l.text)}</span>` : escHtml(l.text);
  }).join('\n');
  if (atBottom) el.scrollTop = el.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function drawChart(hist) {
  const cv = $('chart'), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = 140;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!hist.length) { ctx.fillStyle = '#7a7e8a'; ctx.font = '11px system-ui'; ctx.fillText('waiting for first epoch…', 20, 70); return; }
  const pad = { l: 44, r: 12, t: 16, b: 24 };
  const xs = hist.map(h => h.elapsed || 0), ys = hist.map(h => h.best || 0);
  const x1 = Math.max(...xs, 1), y0 = Math.min(...ys) * 0.95, y1 = Math.max(...ys) * 1.02;
  const X = v => pad.l + (W - pad.l - pad.r) * (v / x1);
  const Y = v => pad.t + (H - pad.t - pad.b) * (1 - (v - y0) / (y1 - y0 || 1));
  // grid
  ctx.strokeStyle = '#e2e5eb'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yy = pad.t + (H - pad.t - pad.b) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke();
  }
  // fill under curve
  ctx.beginPath();
  ctx.moveTo(X(xs[0]), Y(y0));
  hist.forEach(h => ctx.lineTo(X(h.elapsed), Y(h.best)));
  ctx.lineTo(X(xs[xs.length - 1]), Y(y0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(37,99,235,0.08)';
  ctx.fill();
  // curve
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2;
  ctx.beginPath();
  hist.forEach((h, i) => { const x = X(h.elapsed), y = Y(h.best); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  // dots on last few points
  const show = hist.slice(-10);
  show.forEach(h => {
    ctx.beginPath();
    ctx.arc(X(h.elapsed), Y(h.best), 3, 0, Math.PI * 2);
    ctx.fillStyle = '#2563eb';
    ctx.fill();
  });
  // labels
  ctx.fillStyle = '#7a7e8a'; ctx.font = '10px system-ui';
  ctx.textAlign = 'right';
  ctx.fillText(Math.round(y1).toString(), pad.l - 4, pad.t + 10);
  ctx.fillText(Math.round(y0).toString(), pad.l - 4, H - pad.b);
  ctx.textAlign = 'center';
  ctx.fillText((x1 / 60).toFixed(0) + ' min', W - pad.r, H - 4);
  ctx.fillText('best coverage', W / 2, 10);
}

/* ── boot ────────────────────────────────────────────────────────────── */
(async function boot() {
  renderParams();
  try {
    STATE.config = await api.get('/api/config');
    const R = STATE.config.runner;
    const badge = $('runner-badge');
    if (R.local) {
      badge.classList.add('local');
      $('runner-text').textContent = 'local simulation';
    } else if (R.configured) {
      badge.classList.add('online');
      $('runner-text').textContent = R.host;
    } else {
      badge.classList.add('offline');
      $('runner-text').textContent = 'not configured';
    }
    STATE.runs = (await api.get('/api/runs')).runs || [];
    renderRuns();
    loadJobs();
  } catch (err) {
    showOverlay('Cannot reach the server: ' + err.message, false);
  }

  // event wiring
  $('filter').oninput = renderRuns;
  $('btn-fit').onclick = () => map.fit();
  $('opt-shared').onchange = e => { map.markShared = e.target.checked; map.draw(); };
  $('opt-dead').onchange = e => { map.outlineDead = e.target.checked; map.draw(); };
  $('btn-target').onclick = () => toggleTargetMode(!STATE.targetMode);
  $('btn-cancel-target').onclick = () => toggleTargetMode(false);
  $('btn-audit').onclick = runStallAudit;
  $('btn-launch-target').onclick = launchTargeted;
  document.querySelectorAll('input[name=mode]').forEach(r => r.onchange = () => {
    if (STATE.selected && STATE.payload) renderWarmInfo(STATE.selected, STATE.payload);
  });
  $('btn-plan').onclick = plan;
  $('btn-launch').onclick = launch;
  $('btn-stop').onclick = () => { if (STATE.liveJobId) stopJob(STATE.liveJobId); };
  $('btn-export').onclick = async () => {
    if (!STATE.selected) return;
    $('btn-export').textContent = '⏳';
    try {
      const res = await api.get('/api/export?state=' + encodeURIComponent(STATE.selected.state));
      $('btn-export').textContent = res.ok ? '✓' : '✗';
      if (res.urls && res.urls.length) window.open(res.urls[0], '_blank');
    } catch (err) { $('btn-export').textContent = '✗'; }
    setTimeout(() => { $('btn-export').textContent = '📥'; }, 2000);
  };
  document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    for (const pane of ['launch', 'live', 'target']) {
      $('tab-' + pane).hidden = t.dataset.tab !== pane;
    }
    if (t.dataset.tab === 'live') loadJobs();
  });
})();
