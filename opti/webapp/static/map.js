/* map.js — the K x K canvas: every facet drawn, at every zoom level.
 *
 * The map is rendered at a guaranteed minimum cell size (MIN_CELL px) so every
 * facet is always visible and crisp.  For large complexes (dodecahedron 36x36
 * blocks, torus 14x14) the canvas is wider/taller than the viewport and the
 * container scrolls — no quality is ever lost.
 *
 * Zoom (mouse wheel) changes the cell size; the scroll position is preserved
 * around the cursor.  Hit-testing is facet-exact at every size.
 *
 * Colours: red = uncovered; every covered facet takes the colour of the domain
 * that owns it.  Facets in more than one domain get the second owner's colour
 * as an inner outline.  "Domain highlight" mode dims all but the selected domain.
 */
'use strict';

const GAP_RATIO = 0.12;                // gap between blocks, as fraction of cell size
const MIN_CELL = 8;                     // minimum px per facet cell
const MAX_CELL = 64;
const LABEL_FONT_RATIO = 1.2;          // label font size relative to cell size

class KxkMap {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.wrap = canvas.parentElement;   // scrollable container
    this.payload = null;
    this.cellSize = 12;                 // px per facet cell
    this.hover = null;
    this.onHover = opts.onHover || (() => {});
    this.onClick = opts.onClick || (() => {});
    this.markShared = true;
    this.outlineDead = false;
    this.highlightDomain = -1;          // -1 = none
    this.targetMode = false;
    this.targetFacets = new Set();      // facets marked as targets (pulsing border)
    this.auditDoors = null;             // Map<facet, [chain, ...]> from stall audit
    this._prev = null;                  // previous coverage for flash animation
    this._bind();
  }

  setPayload(p) {
    const prev = this.payload;
    this.payload = p;
    this.nt = p.triangles.length;
    this.layout = blockLayout(p.block_counts[0]);
    if (prev) this._prev = prev.covered;
    this.resize();
    this.draw();
  }

  /* ── geometry ──────────────────────────────────────────────────────── */
  gap() { return Math.max(1, Math.round(this.cellSize * GAP_RATIO)); }

  blockX(j) {
    const { cols } = this.layout, g = this.gap();
    return j * (cols * this.cellSize + g);
  }
  blockY(i) {
    const { rows } = this.layout, g = this.gap();
    return i * (rows * this.cellSize + g);
  }
  cellX(j, p) { return this.blockX(j) + (p % this.layout.cols) * this.cellSize; }
  cellY(i, p) { return this.blockY(i) + Math.floor(p / this.layout.cols) * this.cellSize; }

  totalW() {
    const { cols } = this.layout, g = this.gap();
    return this.nt * (cols * this.cellSize + g) - g;
  }
  totalH() {
    const { rows } = this.layout, g = this.gap();
    return this.nt * (rows * this.cellSize + g) - g;
  }

  /* Fit the map: choose the largest cell size that fits the viewport */
  fit() {
    if (!this.payload) return;
    const vw = this.wrap.clientWidth - 20, vh = this.wrap.clientHeight - 20;
    const { cols, rows } = this.layout;
    const g = GAP_RATIO;
    // solve: nt * (cols * s + s*g) - s*g ≤ vw  →  s ≤ vw / (nt * cols + (nt-1)*g*cols)
    const sw = vw / (this.nt * cols * (1 + g) - cols * g);
    const sh = vh / (this.nt * rows * (1 + g) - rows * g);
    this.cellSize = Math.max(MIN_CELL, Math.min(MAX_CELL, Math.floor(Math.min(sw, sh))));
    this.resize();
    this.draw();
    // centre
    this.wrap.scrollLeft = Math.max(0, (this.totalW() - this.wrap.clientWidth) / 2);
    this.wrap.scrollTop = Math.max(0, (this.totalH() - this.wrap.clientHeight) / 2);
  }

  /* screen coords (relative to scroll container) -> {i, j, p, facet} or null */
  hit(sx, sy) {
    if (!this.payload) return null;
    const x = sx + this.wrap.scrollLeft, y = sy + this.wrap.scrollTop;
    const { cols, rows } = this.layout, cs = this.cellSize, g = this.gap();
    const bw = cols * cs + g, bh = rows * cs + g;
    const j = Math.floor(x / bw), i = Math.floor(y / bh);
    if (i < 0 || j < 0 || i >= this.nt || j >= this.nt) return null;
    const lx = x - j * bw, ly = y - i * bh;
    if (lx >= cols * cs || ly >= rows * cs) return { i, j, p: null }; // in the gap
    const cx = Math.floor(lx / cs), cy = Math.floor(ly / cs);
    const p = cy * cols + cx;
    if (p >= this.payload.block_counts[i * this.nt + j]) return { i, j, p: null };
    return { i, j, p, facet: this.payload.block_offsets[i * this.nt + j] + p };
  }

  facetColor(f) {
    const p = this.payload;
    if (!p.covered[f]) return p.hole_color;
    const o = p.owner[f];
    return p.domain_colors[((o % p.domain_colors.length) + p.domain_colors.length) % p.domain_colors.length];
  }

  /* ── rendering ─────────────────────────────────────────────────────── */
  resize() {
    if (!this.payload) return;
    const dpr = window.devicePixelRatio || 1;
    const w = this.totalW() + 20, h = this.totalH() + 20; // 20px padding
    this.cv.style.width = w + 'px';
    this.cv.style.height = h + 'px';
    this.cv.width = Math.round(w * dpr);
    this.cv.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw() {
    if (!this.payload) return;
    const ctx = this.ctx, p = this.payload, cs = this.cellSize;
    const w = parseFloat(this.cv.style.width), h = parseFloat(this.cv.style.height);
    ctx.fillStyle = '#f4f5f8';
    ctx.fillRect(0, 0, w, h);
    const { cols, rows } = this.layout, g = this.gap();
    const dimming = this.highlightDomain >= 0;
    const showHair = cs >= 6;
    const showLabels = cs >= 10;
    const prev = this._prev;

    // visible viewport (for culling)
    const sl = this.wrap.scrollLeft, st = this.wrap.scrollTop;
    const vw = this.wrap.clientWidth, vh = this.wrap.clientHeight;

    for (let i = 0; i < this.nt; i++) {
      const by = this.blockY(i);
      if (by + rows * cs < st - 10 || by > st + vh + 10) continue;
      for (let j = 0; j < this.nt; j++) {
        const bx = this.blockX(j);
        if (bx + cols * cs < sl - 10 || bx > sl + vw + 10) continue;
        const b = i * this.nt + j;
        const first = p.block_offsets[b], count = p.block_counts[b];
        for (let idx = 0; idx < count; idx++) {
          const f = first + idx;
          const fx = bx + (idx % cols) * cs, fy = by + Math.floor(idx / cols) * cs;
          let color = this.facetColor(f);
          // domain dimming
          if (dimming && p.covered[f] && p.owner[f] !== this.highlightDomain) {
            color = this._dimColor(color);
          }
          ctx.fillStyle = color;
          ctx.fillRect(fx, fy, cs, cs);
          // hairlines
          if (showHair) {
            ctx.strokeStyle = 'rgba(255,255,255,0.6)';
            ctx.lineWidth = 1;
            ctx.strokeRect(fx + 0.5, fy + 0.5, cs - 1, cs - 1);
          }
          // shared facet: inner outline
          if (this.markShared && p.covered[f] && p.shared(f)) {
            const mem = p.members(f);
            if (mem.length > 1) {
              ctx.strokeStyle = p.domain_colors[mem[1] % p.domain_colors.length];
              ctx.lineWidth = Math.max(1, cs * 0.1);
              ctx.strokeRect(fx + 2, fy + 2, cs - 4, cs - 4);
            }
          }
          // target mode: absorbable facets glow
          if (this.targetMode && !p.covered[f] && this.auditDoors) {
            const doors = this.auditDoors.get(f);
            if (doors && doors.length > 0) {
              ctx.strokeStyle = '#16a34a';
              ctx.lineWidth = 2;
              ctx.strokeRect(fx + 1, fy + 1, cs - 2, cs - 2);
            }
          }
          // targeted facets: pulsing
          if (this.targetFacets.has(f)) {
            ctx.strokeStyle = '#d97706';
            ctx.lineWidth = 3;
            ctx.strokeRect(fx, fy, cs, cs);
          }
          // flash changed cells
          if (prev && prev[f] !== p.covered[f]) {
            ctx.fillStyle = p.covered[f] ? 'rgba(34,197,94,0.4)' : 'rgba(220,38,38,0.3)';
            ctx.fillRect(fx, fy, cs, cs);
          }
        }
        // dead block outline
        if (this.outlineDead) {
          let allDead = true, anyDead = false;
          for (let idx = 0; idx < count; idx++) {
            if (p.covered[first + idx]) allDead = false; else anyDead = true;
          }
          if (allDead) {
            ctx.strokeStyle = '#1a1c20';
            ctx.lineWidth = 2;
            ctx.strokeRect(bx, by, cols * cs, rows * cs);
          } else if (anyDead) {
            ctx.strokeStyle = '#7a7e8a';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 2]);
            ctx.strokeRect(bx + 0.5, by + 0.5, cols * cs - 1, rows * cs - 1);
            ctx.setLineDash([]);
          }
        }
      }
    }
    // row/col labels
    if (showLabels) {
      const fs = Math.max(8, Math.min(12, cs * LABEL_FONT_RATIO));
      ctx.font = `${fs}px system-ui`;
      ctx.fillStyle = '#7a7e8a';
      ctx.textAlign = 'center';
      for (let j = 0; j < this.nt; j++) {
        const bx = this.blockX(j) + cols * cs / 2;
        ctx.fillText('T' + j, bx, this.blockY(0) - 4);
      }
      ctx.textAlign = 'right';
      for (let i = 0; i < this.nt; i++) {
        const by = this.blockY(i) + rows * cs / 2;
        ctx.fillText('T' + i, this.blockX(0) - 4, by + fs / 3);
      }
    }
    // hover highlight
    if (this.hover && this.hover.facet != null) {
      const { i, j, p: pp } = this.hover;
      const fx = this.cellX(j, pp), fy = this.cellY(i, pp);
      ctx.strokeStyle = '#1a1c20';
      ctx.lineWidth = 2;
      ctx.strokeRect(fx, fy, cs, cs);
    }
    this._prev = null;
  }

  _dimColor(hex) {
    // parse and lerp toward white at 75%
    const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
    const f = 0.75;
    return `rgb(${Math.round(r + (255 - r) * f)}, ${Math.round(g + (255 - g) * f)}, ${Math.round(b + (255 - b) * f)})`;
  }

  zoomToBlock(i, j) {
    const bx = this.blockX(j), by = this.blockY(i);
    this.cellSize = Math.min(MAX_CELL, 32);
    this.resize();
    this.draw();
    this.wrap.scrollLeft = bx - this.wrap.clientWidth / 2 + this.layout.cols * this.cellSize / 2;
    this.wrap.scrollTop = by - this.wrap.clientHeight / 2 + this.layout.rows * this.cellSize / 2;
  }

  /* ── interaction ───────────────────────────────────────────────────── */
  _bind() {
    const cv = this.cv;
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = this.wrap.getBoundingClientRect();
      const mx = e.clientX - rect.left + this.wrap.scrollLeft;
      const my = e.clientY - rect.top + this.wrap.scrollTop;
      // what map-unit is under the mouse before zoom
      const ux = mx / (this.cellSize || 1), uy = my / (this.cellSize || 1);
      const old = this.cellSize;
      if (e.deltaY < 0) this.cellSize = Math.min(MAX_CELL, this.cellSize + Math.max(1, Math.floor(this.cellSize * 0.2)));
      else this.cellSize = Math.max(MIN_CELL, this.cellSize - Math.max(1, Math.floor(this.cellSize * 0.2)));
      if (this.cellSize !== old) {
        this.resize();
        this.draw();
        // keep the same map-unit under the mouse
        this.wrap.scrollLeft = ux * this.cellSize - (e.clientX - rect.left);
        this.wrap.scrollTop = uy * this.cellSize - (e.clientY - rect.top);
      }
    }, { passive: false });

    cv.addEventListener('mousemove', (e) => {
      const rect = this.wrap.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const h = this.hit(mx, my);
      const changed = JSON.stringify(h) !== JSON.stringify(this.hover);
      this.hover = h;
      if (changed) this.draw();
      this.onHover(h, e.clientX - rect.left, e.clientY - rect.top);
    });

    cv.addEventListener('mouseleave', () => {
      this.hover = null; this.draw();
      this.onHover(null);
    });

    cv.addEventListener('click', (e) => {
      const rect = this.wrap.getBoundingClientRect();
      const h = this.hit(e.clientX - rect.left, e.clientY - rect.top);
      if (h && h.facet != null) this.onClick(h);
    });

    cv.addEventListener('dblclick', () => this.fit());
  }
}

if (typeof module !== 'undefined') module.exports = { KxkMap };
