/* map.js — the K x K canvas: every facet drawn, at every zoom level.
 *
 * Layout: the whole complex is a uniform grid of blocks; block b = (i, j) of
 * the K-facet table occupies a `cols x rows` sub-grid (3 x 2 for the six
 * staircase facets of two triangles — the paper figure's layout).  Facet p of
 * a block sits at (p % cols, floor(p / cols)).
 *
 * Granularity: the data model always holds every facet, and the renderer never
 * aggregates.  When cells are smaller than a few pixels it rasterises *cells*
 * into an offscreen canvas (one rect per facet) and scales it down — that is
 * still per-facet data, just box-filtered by the browser — and once cells pass
 * ~8 px it draws rects directly, viewport-culled by block.  Zooming in always
 * reveals facets, never a coarser object; hit-testing is facet-exact at any
 * zoom.
 *
 * Colours: red = uncovered; every covered facet takes the colour of the domain
 * that owns it (one colour per domain).  Facets in more than one domain get a
 * corner marker and, with "mark shared facets" on, the second owner's colour
 * as an inner outline — overlaps are never hidden.
 */
'use strict';

const GAP = 0.34 / 3;                  // gap between blocks, in cell units

class KxkMap {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.payload = null;
    this.scale = 4;                    // px per cell
    this.tx = 0; this.ty = 0;          // pan, in css px
    this.hover = null;
    this.onHover = opts.onHover || (() => {});
    this.markShared = true;
    this.outlineDead = false;
    this._off = document.createElement('canvas');
    this._bind();
    window.addEventListener('resize', () => this.draw());
  }

  setPayload(p) {
    this.payload = p;
    const nt = p.triangles.length;
    const per = p.block_counts[0];
    this.layout = blockLayout(per);
    this.nt = nt;
    this.fit();
  }

  /* ── geometry ──────────────────────────────────────────────────────── */
  blockRect(i, j) {
    const { cols, rows } = this.layout;
    const w = cols, h = rows, g = GAP * cols;   // one gap unit per block edge
    return { x: j * (w + g), y: i * (h + g), w, h };
  }

  cellRect(i, j, p) {
    const b = this.blockRect(i, j), { cols } = this.layout;
    return { x: b.x + (p % cols), y: b.y + Math.floor(p / cols), w: 1, h: 1 };
  }

  fit() {
    if (!this.payload) return;
    const { cols, rows } = this.layout, nt = this.nt, g = GAP * cols;
    const W = nt * (cols + g), H = nt * (rows + g);
    const cssW = this.cv.clientWidth || 800, cssH = this.cv.clientHeight || 600;
    const s = Math.max(0.05, Math.min(cssW / W, cssH / H) * 0.96);
    this.scale = s;
    this.tx = (cssW - W * s) / 2;
    this.ty = (cssH - H * s) / 2;
    this.draw();
  }

  toScreen(x, y) { return [this.tx + x * this.scale, this.ty + y * this.scale]; }

  /* screen px -> {block, facet} or null */
  hit(cssX, cssY) {
    if (!this.payload) return null;
    const x = (cssX - this.tx) / this.scale, y = (cssY - this.ty) / this.scale;
    const { cols, rows } = this.layout, g = GAP * cols, nt = this.nt;
    const j = Math.floor(x / (cols + g)), i = Math.floor(y / (rows + g));
    if (i < 0 || j < 0 || i >= nt || j >= nt) return null;
    const b = this.blockRect(i, j);
    const cx = Math.floor(x - b.x), cy = Math.floor(y - b.y);
    if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) return { i, j, p: null };
    const p = cy * cols + cx;
    if (p >= this.payload.block_counts[i * nt + j]) return { i, j, p: null };
    return { i, j, p, facet: this.payload.block_offsets[i * nt + j] + p };
  }

  facetColor(f) {
    const p = this.payload;
    if (!p.covered[f]) return p.hole_color;
    const o = p.owner[f];
    return p.domain_colors[((o % p.domain_colors.length) + p.domain_colors.length) % p.domain_colors.length];
  }

  /* ── drawing ───────────────────────────────────────────────────────── */
  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.cv.clientWidth, h = this.cv.clientHeight;
    if (this.cv.width !== Math.round(w * dpr) || this.cv.height !== Math.round(h * dpr)) {
      this.cv.width = Math.round(w * dpr);
      this.cv.height = Math.round(h * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw() {
    if (!this.payload) return;
    this.resize();
    const ctx = this.ctx, cssW = this.cv.clientWidth, cssH = this.cv.clientHeight;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, cssW, cssH);
    if (this.scale < 4) this._drawRaster(ctx, cssW, cssH);
    else this._drawVector(ctx, cssW, cssH);
  }

  /* small scale: one rect per facet into an offscreen canvas, then scale.
     Still per-facet data — the browser box-filters it, we never average. */
  _drawRaster(ctx, cssW, cssH) {
    const nt = this.nt, { cols, rows } = this.layout, g = GAP * cols;
    const W = nt * (cols + g), H = nt * (rows + g);
    const unit = Math.max(1, Math.min(8, Math.round(1 / this.scale)));
    const off = this._off;
    off.width = Math.max(1, Math.ceil(W * unit));
    off.height = Math.max(1, Math.ceil(H * unit));
    const octx = off.getContext('2d');
    octx.fillStyle = '#ffffff';
    octx.fillRect(0, 0, off.width, off.height);
    const p = this.payload;
    for (let f = 0; f < p.nf; f++) {
      const b = this._blockOf(f);
      const i = Math.floor(b / nt), j = b % nt;
      const idx = f - p.block_offsets[b];
      const cx = j * (cols + g) + (idx % cols), cy = i * (rows + g) + Math.floor(idx / cols);
      octx.fillStyle = this.facetColor(f);
      octx.fillRect(cx * unit, cy * unit, unit, unit);
    }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, this.tx, this.ty, W * this.scale, H * this.scale);
  }

  _blockOf(f) {                          // binary search over block offsets
    const p = this.payload, offs = p.block_offsets;
    let lo = 0, hi = offs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (offs[mid] <= f) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  /* large scale: explicit rects, culled to the viewport, hairlines at size */
  _drawVector(ctx, cssW, cssH) {
    const p = this.payload, nt = this.nt, { cols, rows } = this.layout, g = GAP * cols;
    const x0 = (0 - this.tx) / this.scale, y0 = (0 - this.ty) / this.scale;
    const x1 = (cssW - this.tx) / this.scale, y1 = (cssH - this.ty) / this.scale;
    const jLo = Math.max(0, Math.floor(x0 / (cols + g))), jHi = Math.min(nt - 1, Math.floor(x1 / (cols + g)));
    const iLo = Math.max(0, Math.floor(y0 / (rows + g))), iHi = Math.min(nt - 1, Math.floor(y1 / (rows + g)));
    const hair = this.scale >= 6;
    for (let i = iLo; i <= iHi; i++) {
      for (let j = jLo; j <= jHi; j++) {
        const b = i * nt + j, first = p.block_offsets[b], count = p.block_counts[b];
        const bx = j * (cols + g), by = i * (rows + g);
        for (let idx = 0; idx < count; idx++) {
          const f = first + idx;
          const cx = bx + (idx % cols), cy = by + Math.floor(idx / cols);
          const [sx, sy] = this.toScreen(cx, cy);
          const sz = this.scale;
          ctx.fillStyle = this.facetColor(f);
          ctx.fillRect(sx, sy, Math.ceil(sz), Math.ceil(sz));
          if (hair) {
            ctx.strokeStyle = 'rgba(255,255,255,0.85)';
            ctx.lineWidth = 1;
            ctx.strokeRect(sx + 0.5, sy + 0.5, sz - 1, sz - 1);
          }
          if (this.markShared && p.covered[f] && p.shared(f)) {
            const mem = p.members(f);
            ctx.strokeStyle = p.domain_colors[mem[1] % p.domain_colors.length];
            ctx.lineWidth = Math.max(1, sz * 0.12);
            ctx.strokeRect(sx + ctx.lineWidth / 2, sy + ctx.lineWidth / 2,
                           sz - ctx.lineWidth, sz - ctx.lineWidth);
          }
        }
        if (this.outlineDead && p.block_counts[b] && !p.covered[first]) {
          const [sx, sy] = this.toScreen(bx, by);
          ctx.strokeStyle = '#141414';
          ctx.lineWidth = 1.6;
          ctx.strokeRect(sx, sy, cols * this.scale, rows * this.scale);
        }
      }
    }
    if (this.hover && this.hover.p !== null) {
      const { i, j, p: pp } = this.hover;
      const c = this.cellRect(i, j, pp);
      const [sx, sy] = this.toScreen(c.x, c.y);
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 2;
      ctx.strokeRect(sx, sy, this.scale, this.scale);
    }
  }

  /* ── interaction ───────────────────────────────────────────────────── */
  _bind() {
    const cv = this.cv;
    let drag = null;
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const before = [(mx - this.tx) / this.scale, (my - this.ty) / this.scale];
      const k = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      this.scale = Math.max(0.05, Math.min(120, this.scale * k));
      this.tx = mx - before[0] * this.scale;
      this.ty = my - before[1] * this.scale;
      this.draw();
    }, { passive: false });
    cv.addEventListener('mousedown', (e) => {
      drag = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty };
    });
    window.addEventListener('mouseup', () => { drag = null; });
    cv.addEventListener('mousemove', (e) => {
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      if (drag) {
        this.tx = drag.tx + (e.clientX - drag.x);
        this.ty = drag.ty + (e.clientY - drag.y);
        this.draw();
        return;
      }
      const h = this.hit(mx, my);
      const changed = JSON.stringify(h) !== JSON.stringify(this.hover);
      this.hover = h;
      if (changed) this.draw();
      this.onHover(h, mx, my);
    });
    cv.addEventListener('mouseleave', () => { this.hover = null; this.draw(); this.onHover(null); });
    cv.addEventListener('dblclick', () => this.fit());
  }

  zoomToBlock(i, j, scale = 14) {
    const b = this.blockRect(i, j);
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    this.scale = scale;
    this.tx = this.cv.clientWidth / 2 - cx * scale;
    this.ty = this.cv.clientHeight / 2 - cy * scale;
    this.draw();
  }
}

if (typeof module !== 'undefined') module.exports = { KxkMap };
