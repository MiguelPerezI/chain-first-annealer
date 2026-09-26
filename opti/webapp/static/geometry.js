/* geometry.js — the client's model of K x K.
 *
 * The server ships a *compact* payload (packed bits, no facet tables): one
 * entry per facet for covered/owner/membership, the K facet list, and the
 * staircase masks.  Everything geometric is rebuilt here, with the same rules
 * `gpu_sc.Product` uses when it emits the facets, so the page and the PNG
 * agree facet for facet.  The rule is checked against the Python by
 * `scripts/test_webapp_payload.py` (facet vertex reconstruction).
 */
'use strict';

function b64bytes(b64) {
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function unpackBits(b64, n) {          // np.packbits order: big-endian within a byte
  const bytes = b64bytes(b64), out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

/** Decode a /api/map payload into the model the renderer draws from. */
function decodePayload(p) {
  const nf = p.nf, k = p.k;
  const covered = unpackBits(p.covered, nf);
  const ownerBytes = b64bytes(p.owner);
  const owner = new Int8Array(nf);
  for (let f = 0; f < nf; f++) owner[f] = ownerBytes[f] - 1;    // -1 = uncovered
  const depth = b64bytes(p.depth);
  // membership: np.packbits(memb) packs each facet's k bits along the last axis
  const membBytes = b64bytes(p.memb), perFacet = Math.ceil(k / 8);
  const memb = new Uint8Array(nf * k);
  for (let f = 0; f < nf; f++)
    for (let j = 0; j < k; j++)
      memb[f * k + j] = (membBytes[f * perFacet + (j >> 3)] >> (7 - (j & 7))) & 1;

  return Object.assign({}, p, {
    covered, owner, depth, memb,
    shared: (f) => {                       // covered by more than one domain?
      let c = 0;
      for (let j = 0; j < k; j++) c += memb[f * k + j];
      return c > 1;
    },
    members: (f) => {
      const out = [];
      for (let j = 0; j < k; j++) if (memb[f * k + j]) out.push(j);
      return out;
    },
  });
}

/** The grid-vertex walk of facet p of block (i, j) — mirrors
 *  kxk_state.facet_grid_vertices: step through the staircase mask, advancing
 *  the cursor in the first or the second K-facet, and record the grid vertex
 *  (a, b) each step lands on.  Emitted in the same order as Product.facets. */
function facetGridVertices(payload, i, j, p) {
  const nv = payload.nv;
  const s = payload.triangles[i], t = payload.triangles[j];
  const mask = payload.masks[(s.length - 1) + ',' + (t.length - 1)][p];
  const out = [[s[0], t[0]]];
  let a = 0, b = 0;
  const steps = s.length - 1 + t.length - 1;
  for (let bit = 0; bit < steps; bit++) {
    if (((mask >> bit) & 1) === 0) a++; else b++;
    out.push([s[a], t[b]]);
  }
  return out;
}

/** Staircase label of facet p of a block: which steps move inside s (A) and
 *  which inside t (B).  For triangle complexes this reads p0 BBAA … p5 AABB. */
function pathLabel(payload, i, j, p) {
  const s = payload.triangles[i], t = payload.triangles[j];
  const mask = payload.masks[(s.length - 1) + ',' + (t.length - 1)][p];
  let out = '';
  for (let bit = 0; bit < s.length - 1 + t.length - 1; bit++)
    out += ((mask >> bit) & 1) === 0 ? 'A' : 'B';
  return out;
}

/** Per-block facet count and the sub-grid the facets are laid out on.
 *  Six facets (two triangles) become the paper figure's 3-wide x 2-tall grid;
 *  a higher-simplex block is laid out on the smallest grid that fits. */
function blockLayout(nfacets) {
  if (nfacets === 6) return { cols: 3, rows: 2 };
  const cols = Math.ceil(Math.sqrt(nfacets));
  return { cols, rows: Math.ceil(nfacets / cols) };
}

if (typeof module !== 'undefined') module.exports = {
  decodePayload, unpackBits, facetGridVertices, pathLabel, blockLayout, b64bytes,
};
