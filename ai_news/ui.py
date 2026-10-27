"""Gradio Blocks UI: a reading desk for AI research.

Design: serif (Newsreader) for what you read, sans (Hanken Grotesk) for what you operate. The one
expressive element is the sieve, which shows how the collection narrows to the few items worth reading.
Teal means "kept", ochre means "filtered out"; no other accent colors.
"""

from __future__ import annotations

import html
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import gradio as gr

from ai_news.atlas import collection_stats, research_map
from ai_news.fetchers.common import parse_iso
from ai_news.rag.state import Candidate
from ai_news.schema import NewsItem
from ai_news.search import DATE_WINDOWS, highlight, keyword_search
from ai_news.store import CorpusStore

TYPE_CHOICES = [("All", "any"), ("Papers", "paper"), ("Blog posts", "blog")]
EXAMPLES = [
    "What are the latest approaches to scaling test-time compute for reasoning models?",
    "Which new open-weight multimodal models were released this month?",
    "Recent papers on making RAG more faithful with citations",
    "Efficient fine-tuning of LLMs: LoRA and its newer variants",
    "What's new in AI agents that use tools and browse the web?",
]
ABOUT_MD = Path(__file__).with_name("about.md").read_text(encoding="utf-8")
ASK_PLACEHOLDER = "What do you want to read about?"
ASK_EMPTY = ("Ask a question above. The assistant searches the collection, keeps only what is relevant, "
             "and gives you a short reading list with the reason for each pick.")

FONTS_HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
              '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
              '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;'
              '0,6..72,500;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">')

# Gradio 6.30 sizes the Spaces iframe itself but stops growing it after a few quick grows (streaming,
# font loads), so the page could not scroll. This reports the real content height to the parent instead.
# It measures content-sized elements: main, .wrap and the columns stretch to the iframe and would feed back.
FIT_JS = """
(() => {
  if (window.__nvFit) return; window.__nvFit = true;
  const measure = () => {
    const root = document.querySelector('.gradio-container');
    if (!root) return 0;
    let bottom = 0;
    for (const el of root.querySelectorAll('main .block, main button, main [role=tablist]')) {
      if (el.offsetParent) bottom = Math.max(bottom, el.getBoundingClientRect().bottom);  // skip hidden tabs
    }
    const foot = root.querySelector('footer');
    return Math.ceil(bottom - root.getBoundingClientRect().top + (foot ? foot.offsetHeight : 0) + 48);
  };
  let raf = 0;
  const report = () => {
    raf = 0;
    const h = measure();
    if (h && window.parentIFrame && Math.abs(h - window.innerHeight) > 2) window.parentIFrame.size(h);
  };
  const schedule = () => { if (!raf) raf = requestAnimationFrame(report); };
  const start = () => {
    const main = document.querySelector('.gradio-container main');
    if (!main) return setTimeout(start, 300);
    new ResizeObserver(schedule).observe(main);
    new MutationObserver(schedule).observe(main, {childList: true, subtree: true, attributes: true});
    window.addEventListener('message', schedule);  // the parent's iframe-resizer handshake
    schedule();
  };
  start();
})();
"""

# The research map: 3d-force-graph (three.js inside) from a CDN, pinned. The page asks the server for the map
# with server.map_data() once it has mounted, so the ~0.5 MB of nodes stays out of the page config and the API.
MAP_HEAD = '<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.1/dist/3d-force-graph.min.js"></script>'
MAP_HTML = """
<section class="nv-map" aria-label="3D map of the papers in the collection">
  <div class="nv-map-canvas"></div>
  <div class="nv-map-labels" aria-hidden="true"></div>
  <div class="nv-map-bar">
    <input class="nv-map-find" type="search" list="nv-map-titles" placeholder="Find a paper by title"
           aria-label="Find a paper by title"><datalist id="nv-map-titles"></datalist>
    <button type="button" class="nv-map-reset">Reset view</button>
  </div>
  <p class="nv-map-hint">Drag to rotate · scroll to zoom · right-drag to pan · click a paper or a topic</p>
  <aside class="nv-map-card" hidden aria-live="polite"></aside>
  <p class="nv-map-msg">Loading the map…</p>
</section>
"""
LANDING_JS = "() => {{ document.querySelector('.nv-landing')?.classList.{op}('nv-off'); }}"
MAP_JS = r"""
(async () => {
  const root = element.querySelector('.nv-map');
  if (!root || root.dataset.ready) return;
  root.dataset.ready = '1';
  const box = root.querySelector('.nv-map-canvas'), labelsEl = root.querySelector('.nv-map-labels');
  const card = root.querySelector('.nv-map-card'), msg = root.querySelector('.nv-map-msg');
  const find = root.querySelector('.nv-map-find');
  const say = t => { msg.textContent = t; msg.hidden = !t; };
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  if (typeof ForceGraph3D === 'undefined') return say('The map library could not be loaded. Check your connection and reload.');
  const probe = document.createElement('canvas');
  if (!(probe.getContext('webgl2') || probe.getContext('webgl'))) return say('This browser has WebGL turned off, so the 3D map cannot be shown.');
  let data;
  try { data = await server.map_data(); } catch (e) { console.error(e); return say('The map could not be loaded. Reload the page to try again.'); }
  if (!data) return say('The map could not be loaded. Reload the page to try again.');
  const T = data.topics || [], P = data.papers || [];
  if (!P.length) return say('The map appears once the collection has papers.');
  say('');

  // nodes: topic hubs first (ids "t0".."tN"), then papers (ids 0..n-1); every paper links to its hub
  const hubs = T.map((t, g) => ({id: 't' + g, hub: true, g, label: t.label, size: t.size, x: t.pos[0], y: t.pos[1], z: t.pos[2]}));
  const papers = P.map((p, i) => ({id: i, i, ...p, x: p.pos[0], y: p.pos[1], z: p.pos[2]}));
  const links = papers.map(p => ({source: p.i, target: 't' + p.g, hub: true}))
    .concat((data.links || []).map(([a, b, w]) => ({source: a, target: b, w})));
  const near = P.map(() => []);
  for (const [a, b, w] of data.links || []) { near[a].push([b, w]); near[b].push([a, w]); }
  near.forEach(l => l.sort((x, y) => y[1] - x[1]));
  const members = T.map(() => []);
  papers.forEach(p => members[p.g] && members[p.g].push(p.i));
  members.forEach(m => m.sort((a, b) => (P[b].v - P[a].v) || P[b].d.localeCompare(P[a].d)));
  const list = root.querySelector('#nv-map-titles');
  list.innerHTML = P.map(p => `<option value="${esc(p.t)}">`).join('');

  // colour: papers on a teal ramp by recency (newest strongest), hubs in ink; a selection dims the rest
  const days = P.map(p => Date.parse(p.d) || 0).filter(Boolean);
  const d0 = Math.min(...days), d1 = Math.max(...days);
  let pal = {};
  const mix = (a, b, t) => {
    const h = s => [1, 3, 5].map(k => parseInt(s.slice(k, k + 2), 16));
    const [x, y] = [h(a), h(b)];
    return '#' + x.map((v, k) => Math.round(v + (y[k] - v) * t).toString(16).padStart(2, '0')).join('');
  };
  const readPalette = () => {
    const css = n => getComputedStyle(root).getPropertyValue(n).trim();
    pal = {bg: css('--paper') || '#F5F7F9', ink: css('--ink') || '#1B2440', old: css('--map-old') || '#A9C9C3',
           fresh: css('--kept') || '#0F7B6C', link: css('--map-link') || '#C9D4DD', dim: css('--map-dim') || '#DDE3EA'};
  };
  readPalette();
  const age = p => d1 > d0 ? ((Date.parse(p.d) || d0) - d0) / (d1 - d0) : 1;
  let sel = null, lit = null;  // the selected node and the set of highlighted node ids
  const nodeColor = n => {
    if (lit && !lit.has(n.id)) return pal.dim;
    if (n.hub) return pal.ink;
    if (sel === n) return pal.ink;
    return mix(pal.old, pal.fresh, Math.pow(age(n), 1.5));
  };
  const nodeVal = n => (lit && !lit.has(n.id) ? 0.35 : 1) *
    (n.hub ? 10 + Math.sqrt(n.size) * 2 : 1 + Math.log1p(n.v || 0) * 1.6);
  const linkTouches = l => sel && (l.source === sel || l.target === sel);

  const Graph = new ForceGraph3D(box, {controlType: 'orbit'})
    .backgroundColor(pal.bg)
    .showNavInfo(false)
    .graphData({nodes: hubs.concat(papers), links})
    .nodeRelSize(3.2)
    .nodeResolution(8)
    .nodeOpacity(0.92)
    .nodeVal(nodeVal)
    .nodeColor(nodeColor)
    .nodeLabel(n => n.hub ? `<b>${esc(n.label)}</b><br>${n.size} papers · click to explore`
                          : `<b>${esc(n.t)}</b><br>${esc(n.s)} · ${esc(n.d)}${n.v ? ' · ' + n.v + ' upvotes' : ''}`)
    .linkColor(l => linkTouches(l) ? pal.fresh : pal.link)
    .linkOpacity(0.35)
    .linkWidth(l => linkTouches(l) ? 1.2 : 0)
    .linkVisibility(l => !l.hub || linkTouches(l) || !sel)
    .enableNodeDrag(false)
    .warmupTicks(10)
    .cooldownTicks(120)
    .onNodeClick(n => n.hub ? openTopic(n.g) : openPaper(n.i))
    .onBackgroundClick(() => clear());
  root.graph = () => ({graph: Graph, selected: sel});  // for tests and the browser console
  Graph.d3Force('link').distance(l => l.hub ? 70 : 22).strength(l => l.hub ? 0.06 : 0.25 + (l.w || 0) * 0.3);
  Graph.d3Force('charge').strength(-14);
  Graph.cameraPosition({x: 0, y: 0, z: 820});
  let fitted = false;
  Graph.onEngineStop(() => { if (!fitted) { fitted = true; Graph.zoomToFit(900, 30); } });
  const repaint = () => { Graph.nodeVal(nodeVal).nodeColor(nodeColor).linkColor(Graph.linkColor()).linkWidth(Graph.linkWidth()).linkVisibility(Graph.linkVisibility()); };

  // fly to a node; with the card open, slide the view so the node sits in the part the card leaves free
  // (beside the card on wide screens, above it on phones, where the card fills the lower half)
  const side = window.matchMedia('(min-width: 641px)');
  const fly = n => {
    const d = n.hub ? 340 : 170, r = Math.hypot(n.x, n.y, n.z);
    const to = r > 1 ? {x: n.x * (1 + d / r), y: n.y * (1 + d / r), z: n.z * (1 + d / r)} : {x: n.x, y: n.y, z: n.z + d};
    const at = {x: n.x, y: n.y, z: n.z};
    if (!card.hidden) {
      const W = box.clientWidth, H = box.clientHeight, cam = Graph.camera();
      const f = [n.x - to.x, n.y - to.y, n.z - to.z].map(v => v / d);
      let rt = [-f[2], 0, f[0]];  // right = forward × world up
      const rl = Math.hypot(...rt) || 1;
      rt = rt.map(v => v / rl);
      const up = [rt[1] * f[2] - rt[2] * f[1], rt[2] * f[0] - rt[0] * f[2], rt[0] * f[1] - rt[1] * f[0]];
      const halfH = d * Math.tan((cam.fov || 40) * Math.PI / 360), halfW = halfH * W / H;
      const [dir, amount] = side.matches ? [rt, halfW * (card.offsetWidth + 20) / W]
                                         : [up.map(v => -v), halfH * (card.offsetHeight + 10) / H];
      for (const p of [to, at]) { p.x += dir[0] * amount; p.y += dir[1] * amount; p.z += dir[2] * amount; }
    }
    Graph.cameraPosition(to, at, 1100);
  };
  const paperBtn = i => `<li><button type="button" data-i="${i}">${esc(P[i].t)}</button><span>${esc(P[i].s)} · ${esc(P[i].d)}</span></li>`;
  const show = html => { card.innerHTML = `<button type="button" class="nv-map-x" aria-label="Close">×</button>${html}`; card.hidden = false; };
  function openPaper(i) {
    const p = papers[i];
    sel = p; lit = new Set([p.id, 't' + p.g, ...near[i].map(x => x[0])]);
    repaint();
    const nb = near[i].slice(0, 8);
    show(`<button type="button" class="nv-map-kicker" data-g="${p.g}">${esc(T[p.g] ? T[p.g].label : '')}</button>
      <a class="nv-map-title" href="${esc(p.u)}" target="_blank" rel="noopener">${esc(p.t)}</a>
      <p class="nv-map-meta">${esc(p.s)} · ${esc(p.d)}${p.v ? ' · ' + p.v + ' upvotes' : ''}</p>
      <p class="nv-map-sub">${nb.length ? 'Nearby papers' : 'No paper sits close to this one'}</p>
      <ol>${nb.map(x => paperBtn(x[0])).join('')}</ol>`);
    fly(p);
  }
  function openTopic(g) {
    if (!T[g]) return;
    const h = hubs[g];
    sel = h; lit = new Set([h.id, ...members[g]]);
    repaint();
    show(`<p class="nv-map-kicker">Topic</p><p class="nv-map-title">${esc(T[g].label)}</p>
      <p class="nv-map-meta">${T[g].size} papers · most upvoted and newest first</p>
      <ol>${members[g].slice(0, 10).map(paperBtn).join('')}</ol>`);
    fly(h);
  }
  function clear() { sel = null; lit = null; card.hidden = true; repaint(); }
  card.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.classList.contains('nv-map-x')) return clear();
    if (b.dataset.i) openPaper(+b.dataset.i);
    else if (b.dataset.g) openTopic(+b.dataset.g);
  });
  root.querySelector('.nv-map-reset').addEventListener('click', () => { clear(); Graph.zoomToFit(900, 30); });
  const go = () => {
    const q = find.value.trim().toLowerCase();
    if (!q) return;
    const i = P.findIndex(p => p.t.toLowerCase() === q);
    const j = i >= 0 ? i : P.findIndex(p => p.t.toLowerCase().includes(q));
    if (j >= 0) openPaper(j); else show('<p class="nv-map-meta">No paper title contains that.</p>');
  };
  find.addEventListener('change', go);
  find.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
  // topic chips in the stats panel above fly the map to that topic
  document.addEventListener('click', e => {
    const b = e.target.closest('[data-nv-topic]');
    if (!b) return;
    openTopic(+b.dataset.nvTopic);
    root.scrollIntoView({behavior: 'smooth', block: 'center'});
  });

  // topic names float over their hubs as HTML, so they stay crisp and clickable
  const rank = T.map((t, g) => g).sort((a, b) => T[b].size - T[a].size);
  const labelEls = T.map((t, g) => {
    const el = document.createElement('button');
    el.type = 'button'; el.className = 'nv-map-label'; el.textContent = t.label; el.tabIndex = -1;
    el.addEventListener('click', () => openTopic(g));
    labelsEl.appendChild(el);
    return el;
  });
  const top = new Set(rank.slice(0, 14));
  const place = () => {
    const cam = Graph.camera(), f = cam.position.clone();
    cam.getWorldDirection(f);
    hubs.forEach((h, g) => {
      const el = labelEls[g], dx = h.x - cam.position.x, dy = h.y - cam.position.y, dz = h.z - cam.position.z;
      const dist = Math.hypot(dx, dy, dz);
      const keep = (dx * f.x + dy * f.y + dz * f.z) > 0 && (top.has(g) || dist < 520 || sel === h) && !(lit && !lit.has(h.id));
      if (!keep) { el.style.display = 'none'; return; }
      const s = Graph.graph2ScreenCoords(h.x, h.y, h.z);
      el.style.display = '';
      el.style.transform = `translate(${s.x}px, ${s.y}px) translate(-50%, -150%)`;
      el.style.opacity = Math.max(0.45, Math.min(1, 1.5 - dist / 1800)).toFixed(2);
    });
  };
  let running = true;
  const loop = () => { if (running) place(); requestAnimationFrame(loop); };
  loop();

  // size to the box; pause drawing while the map is hidden or scrolled away
  const fit = () => { if (box.clientWidth) Graph.width(box.clientWidth).height(box.clientHeight); };
  new ResizeObserver(fit).observe(box); fit();
  new IntersectionObserver(([e]) => {
    running = e.isIntersecting;
    running ? Graph.resumeAnimation() : Graph.pauseAnimation();
  }).observe(root);
  new MutationObserver(() => { readPalette(); Graph.backgroundColor(pal.bg); repaint(); })
    .observe(document.body, {attributes: true, attributeFilter: ['class']});
})();
"""

CSS = """
:root {
  --paper:#F5F7F9; --ink:#1B2440; --graphite:#5B6478; --rule:#D7DCE5; --field:#FFFFFF;
  --kept:#0F7B6C; --dropped:#B7791F;
  --map-old:#A9C9C3; --map-link:#C3CCD8; --map-dim:#DFE4EB;
  --serif:"Newsreader", Georgia, "Times New Roman", serif;
}
.dark {
  --paper:#141B2D; --ink:#E6EAF2; --graphite:#9AA4B8; --rule:#2A3350; --field:#1B2338;
  --kept:#4FC7B4; --dropped:#E0A84A;
  --map-old:#2E5559; --map-link:#33405F; --map-dim:#232C44;
}
.gradio-container { width:100% !important; max-width:1440px !important; margin:0 auto !important; }
.gradio-container a { color:var(--ink); text-decoration-color:var(--rule); text-underline-offset:3px; }
.gradio-container a:hover { text-decoration-color:var(--kept); }
.gradio-container a:focus-visible, .gradio-container button:focus-visible {
  outline:2px solid var(--kept); outline-offset:2px; border-radius:2px; }

/* masthead */
.nv-masthead { padding:20px 0 2px; display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 18px; }
.nv-masthead h1 { font-family:var(--serif); font-weight:500; font-size:27px; line-height:1.15;
  letter-spacing:-0.01em; color:var(--ink); margin:0; }
.nv-masthead p { font-size:15px; color:var(--graphite); margin:0; max-width:68ch; }

/* the Ask desk: question + refine on the left, results in the center, the AI's steps on the right */
.nv-desk { gap:32px !important; align-items:flex-start !important; }
.nv-center { min-width:0; }
.nv-pane-h { font-size:13px; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  color:var(--graphite); margin:4px 0 10px; }
.nv-side .nv-ask textarea { font-size:17px !important; padding:12px 14px !important; }
.nv-side .nv-refine { margin-top:8px; padding-top:14px; border-top:1px solid var(--rule); }
.nv-more, .nv-more button { border:1px solid var(--rule) !important; }
.nv-flush, .nv-flush .html-container { padding:0 !important; }
/* when the three panes wrap (tablets, phones), show the short steps panel before the long results */
@media (max-width: 1100px) { .nv-steps { order:2; } .nv-center { order:3; } }

/* question box is the hero */
.nv-ask textarea { font-family:var(--serif) !important; font-size:21px !important; line-height:1.4 !important;
  padding:14px 16px !important; }
.nv-ask .submit-button { background:var(--kept) !important; color:#fff !important; font-weight:600;
  border-radius:6px !important; padding:0 20px !important; margin-left:10px; min-height:44px; align-self:center; }
.dark .nv-ask .submit-button { color:#0E1A2B !important; }
.nv-ask .submit-button:hover { filter:brightness(.92); }
.nv-filters { gap:12px !important; }
.nv-hint { color:var(--graphite); font-size:15px; max-width:68ch; }

/* the sieve */
.nv-sieve { margin:0 0 4px; }
.nv-step { display:grid; grid-template-columns:1fr 4rem; align-items:center; gap:4px 10px; padding:6px 0;
  font-size:15px; color:var(--ink); }
.nv-step .nv-rail { grid-column:1 / span 2; grid-row:2; }
.nv-step .label { color:var(--graphite); }
.nv-step .count { text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }
.nv-rail { position:relative; height:14px; }
.nv-rail .was { position:absolute; inset:0 auto 0 0; border:1.5px solid var(--dropped); border-radius:2px;
  opacity:.75; }
.nv-rail .kept { position:absolute; inset:0 auto 0 0; background:var(--kept); border-radius:2px; }
.nv-step.note .label { grid-column:1 / span 2; font-style:italic; font-family:var(--serif); font-size:17px; }
.nv-step.new .kept { animation:grow .6s cubic-bezier(.2,.7,.2,1) both; transform-origin:left; }
@keyframes grow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .nv-step.new .kept { animation:none; } }
.nv-sieve details { margin-top:8px; font-size:13px; color:var(--graphite); }
.nv-sieve details summary { cursor:pointer; font-size:15px; }
.nv-sieve details ol { padding-left:20px; overflow-wrap:anywhere; }
.nv-steps-empty { font-size:15px; color:var(--graphite); }
.nv-steps-empty ol { padding-left:20px; margin:8px 0 0; }
.nv-steps-empty li { margin:4px 0; }
.nv-legend { font-size:13px; color:var(--graphite); margin-top:4px; }
.nv-legend i { display:inline-block; width:18px; height:8px; vertical-align:middle; margin:0 4px 0 0;
  border-radius:2px; }
.nv-legend .k { background:var(--kept); }
.nv-legend .d { border:1.5px solid var(--dropped); margin-left:12px; }

/* answer prose */
.nv-answer { font-family:var(--serif); font-size:17px; line-height:1.55; color:var(--ink); }
.nv-answer .nv-prose { max-width:68ch; }
.nv-answer .nv-prose p, .nv-answer .nv-prose li { font-family:var(--serif) !important; font-size:17px !important;
  line-height:1.55 !important; }
.nv-answer > .nv-hint, .nv-answer .nv-hint { font-family:var(--serif); font-size:17px !important; }
button.nv-reset, .nv-reset button { border:1px solid var(--rule) !important; padding:6px 14px !important;
  font-size:13px; }
.nv-answer h3 { font-family:var(--serif); font-weight:500; font-size:27px; line-height:1.2; margin:0 0 4px; }
.nv-answer .nv-prose p.nv-note { font-family:var(--body-font, inherit) !important; font-size:13px !important;
  color:var(--graphite); margin:0 0 12px; }
.nv-answer h2, .nv-answer h4 { font-family:var(--serif); font-weight:600; font-size:21px; }
.nv-answer details { margin-top:24px; font-size:15px; color:var(--graphite); }
.nv-answer details summary { cursor:pointer; font-family:var(--body-font, inherit); }

/* references under the summary */
.nv-refs { margin-top:28px; }
.nv-refs h2 { font-family:var(--serif); font-weight:500; font-size:21px; margin:0; color:var(--ink); }
.nv-refs .nv-count { font-size:13px; color:var(--graphite); margin:2px 0 10px; }
.nv-refs ol { list-style:none; padding:0; margin:0; }
.nv-refs li { display:grid; grid-template-columns:2.6rem 1fr; padding:14px 0; border-top:1px solid var(--rule); }
.nv-refs li .n { font-family:var(--serif); font-size:17px; color:var(--kept); line-height:1.35;
  font-variant-numeric:tabular-nums; }
.nv-refs .t { font-family:var(--serif); font-size:19px; line-height:1.35; font-weight:500; }
.nv-item a.t, .nv-refs a.t { display:block; padding:0 !important; text-decoration:none; color:var(--ink);
  text-align:left; }
.nv-item, .nv-refs li, .nv-step { text-align:left; }
.nv-item a.t:hover, .nv-refs a.t:hover { text-decoration:underline; text-decoration-color:var(--kept); }
.nv-item .m a.pdf { margin-left:12px; padding:0 !important; }
.nv-refs .m, .nv-item .m { font-size:13px; color:var(--graphite); margin:4px 0 0; }
.nv-refs .d { font-size:15px; line-height:1.55; margin:6px 0 0; color:var(--ink); max-width:75ch; }
.nv-refs .d .lbl { font-size:12px; font-weight:600; letter-spacing:.04em; text-transform:uppercase;
  margin-right:6px; color:var(--kept); }
.nv-refs .d.ab .lbl { color:var(--graphite); }
.nv-refs .empty, .nv-rows .empty { color:var(--graphite); font-size:15px; }
.nv-score { display:inline-block; margin-left:10px; padding:0 6px; border-radius:3px; font-weight:600;
  font-variant-numeric:tabular-nums; color:var(--kept); background:color-mix(in srgb, var(--kept) 12%, transparent); }
.nv-score.low { color:var(--ink); font-weight:500; background:transparent; border:1.5px solid var(--dropped); }
.nv-warn { margin:0 0 10px; padding:8px 12px; font-size:15px; line-height:1.5; color:var(--ink);
  border-left:3px solid var(--dropped); background:color-mix(in srgb, var(--dropped) 9%, transparent); }

/* result rows (search, latest) */
.nv-rows { max-width:860px; }
.nv-rows .nv-count { font-size:15px; color:var(--graphite); margin:4px 0 8px; }
.nv-item { padding:14px 0; border-top:1px solid var(--rule); }
.nv-item .t { font-family:var(--serif); font-size:21px; line-height:1.3; font-weight:500; }
.nv-item .s { font-size:15px; line-height:1.55; color:var(--ink); margin:6px 0 0; max-width:75ch; }
.nv-item mark { background:color-mix(in srgb, var(--kept) 14%, transparent); color:inherit;
  padding:0; border-radius:2px; }
.nv-kind { display:inline-block; width:.9em; text-align:center; margin-right:4px; color:var(--graphite); }

/* latest, by day */
.nv-day { margin-top:28px; }
.nv-day:first-of-type { margin-top:8px; }
.nv-day h2 { font-family:var(--serif); font-weight:500; font-size:21px; margin:0; padding-top:12px;
  color:var(--ink); border-top:2px solid var(--ink); }
.nv-day .nv-count { font-size:13px; margin:2px 0 4px; }
.nv-day .more { font-size:13px; color:var(--graphite); margin:10px 0 0; }

/* sources table */
.nv-sources table { border-collapse:collapse; font-size:15px; max-width:760px; width:100%; }
.nv-sources th { text-align:left; font-weight:600; color:var(--graphite); padding:6px 12px 6px 0; }
.nv-sources td { padding:7px 12px 7px 0; border-top:1px solid var(--rule); vertical-align:top; }
.nv-sources td.n { text-align:right; font-variant-numeric:tabular-nums; }
.nv-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
.nv-dot.ok { background:var(--kept); } .nv-dot.bad { background:var(--dropped); }

/* about: technical summary */
.nv-about { font-size:15px; line-height:1.55; color:var(--ink); }
.nv-about .md { max-width:none !important; }
.nv-about p, .nv-about ol, .nv-about ul, .nv-about h2, .nv-about h3 { max-width:76ch; }
.nv-about h2 { font-family:var(--serif); font-weight:500; font-size:34px; line-height:1.15; margin:8px 0 12px; }
.nv-about h3 { font-family:var(--serif); font-weight:500; font-size:27px; line-height:1.2; margin:40px 0 10px; }
.nv-about p, .nv-about li { font-size:15px !important; }
.nv-about table { display:block; overflow-x:auto; width:100%; border-collapse:collapse !important;
  border:none !important; margin:8px 0 4px; font-size:14px; }
.nv-about th, .nv-about td { border:none !important; border-bottom:1px solid var(--rule) !important;
  background:transparent !important; text-align:left; vertical-align:top; }
.nv-about th { font-weight:600; color:var(--graphite); padding:8px 18px 8px 0 !important; white-space:nowrap; }
.nv-about td { padding:12px 18px 12px 0 !important; }
.nv-about td:nth-child(1) { width:16%; min-width:150px; }
.nv-about td:nth-child(2) { width:22%; min-width:180px; color:var(--graphite); }
.nv-about td:nth-child(3) { width:42%; min-width:300px; }
.nv-about td:nth-child(4) { width:20%; min-width:170px; font-size:13px; color:var(--graphite); }
.nv-about tr, .nv-about thead, .nv-about tbody { border:none !important; background:transparent !important; }
.nv-about code { font-size:.9em; background:color-mix(in srgb, var(--rule) 45%, transparent); padding:0 4px;
  border-radius:3px; white-space:normal; word-break:break-word; }

/* landing: the research map, collection stats, and new papers per day in the right-hand pane */
.nv-landing { gap:16px !important; }
.nv-landing.nv-off { display:none !important; }
.nv-tiles { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); border-radius:4px; overflow:hidden; margin:4px 0 0; }
.nv-tiles.two { grid-template-columns:repeat(2, minmax(0, 1fr)); margin:0 0 10px; }
.nv-trend { padding-bottom:18px; margin-bottom:14px; border-bottom:1px solid var(--rule); }
.nv-trend .nv-tile .v { font-size:21px; }
.nv-trend .nv-bar .tip { white-space:normal; width:max-content; max-width:200px; }
.nv-tile { background:var(--paper); padding:12px 14px; }
.nv-tile .k { font-size:13px; color:var(--graphite); margin:0; }
.nv-tile .v { font-family:var(--serif); font-size:27px; line-height:1.2; font-weight:500; color:var(--ink);
  margin:2px 0 0; font-variant-numeric:tabular-nums; }
.nv-tile .sub { font-size:13px; color:var(--graphite); margin:2px 0 0; }
.nv-land-h { font-family:var(--serif); font-weight:500; font-size:21px; color:var(--ink); margin:0; }
.nv-land-note { font-size:13px; color:var(--graphite); margin:2px 0 12px; max-width:68ch; }
.nv-chart { position:relative; padding:0 0 0 2.6rem; }
.nv-chart .grid { position:absolute; left:2.6rem; right:0; border-top:1px dashed var(--rule); }
.nv-chart .ylab { position:absolute; left:0; width:2.2rem; text-align:right; font-size:12px; color:var(--graphite);
  transform:translateY(-50%); font-variant-numeric:tabular-nums; }
.nv-bars { position:relative; display:flex; align-items:flex-end; gap:2px; height:150px;
  border-bottom:1px solid var(--graphite); }
.nv-bar { position:relative; flex:1 1 0; height:100%; display:flex; align-items:flex-end; outline:none; }
.nv-bar i { display:block; width:100%; background:var(--kept); border-radius:4px 4px 0 0; min-height:0; }
.nv-bar.today i { background:color-mix(in srgb, var(--kept) 45%, transparent); }
.nv-bar:hover i, .nv-bar:focus-visible i { filter:brightness(1.15); }
.nv-bar .tip { display:none; position:absolute; bottom:calc(100% + 6px); left:50%; transform:translateX(-50%);
  z-index:3; white-space:nowrap; padding:6px 9px; border-radius:4px; font-size:13px; line-height:1.4;
  background:var(--ink); color:var(--paper); pointer-events:none; }
.nv-bar:first-child .tip, .nv-bar:nth-child(2) .tip, .nv-bar:nth-child(3) .tip { left:0; transform:none; }
.nv-bar:nth-last-child(-n+3) .tip { left:auto; right:0; transform:none; }
.nv-bar:hover .tip, .nv-bar:focus-visible .tip { display:block; }
.nv-xlabs { position:relative; height:18px; margin-top:4px; font-size:12px; color:var(--graphite); }
.nv-xlabs span { position:absolute; transform:translateX(-50%); white-space:nowrap; }
.nv-xlabs span:last-child { transform:translateX(-100%); }
.nv-chart-table { margin-top:8px; font-size:13px; color:var(--graphite); }
.nv-chart-table summary { cursor:pointer; }
.nv-chart-table table { border-collapse:collapse; margin-top:6px; font-variant-numeric:tabular-nums; }
.nv-chart-table td, .nv-chart-table th { padding:2px 14px 2px 0; text-align:left; }
.nv-topics { display:flex; flex-wrap:wrap; gap:8px; margin:0; padding:0; list-style:none; }
.nv-topics button { font-size:13px; padding:5px 10px; border:1px solid var(--rule); border-radius:999px;
  background:var(--field); color:var(--ink); cursor:pointer; }
.nv-topics button:hover { border-color:var(--kept); }
.nv-topics button span { color:var(--graphite); margin-left:6px; font-variant-numeric:tabular-nums; }

.nv-map { position:relative; height:560px; border:1px solid var(--rule); border-radius:4px; overflow:hidden;
  background:var(--paper); }
.nv-map-canvas { position:absolute; inset:0; }
.nv-map-labels { position:absolute; inset:0; pointer-events:none; overflow:hidden; }
.nv-map-label { position:absolute; left:0; top:0; pointer-events:auto; cursor:pointer; white-space:nowrap;
  font-size:12px; font-weight:600; letter-spacing:.01em; color:var(--ink); padding:2px 7px; border-radius:3px;
  border:none; background:color-mix(in srgb, var(--paper) 78%, transparent); }
.nv-map-label:hover { color:var(--kept); }
.nv-map-bar { position:absolute; top:10px; left:10px; right:10px; display:flex; gap:8px; z-index:2;
  pointer-events:none; }
.nv-map-bar > * { pointer-events:auto; }
@media (min-width: 641px) {  /* the open card sits on the right: keep the bar beside it */
  .nv-map:has(.nv-map-card:not([hidden])) .nv-map-bar { right:calc(min(340px, 100% - 20px) + 20px); } }
.nv-map-find { flex:0 1 320px; min-width:0; font-size:14px; padding:6px 10px; border:1px solid var(--rule);
  border-radius:4px; background:var(--field); color:var(--ink); }
.nv-map-reset { font-size:13px; padding:6px 12px; border:1px solid var(--rule); border-radius:4px;
  background:var(--field); color:var(--ink); cursor:pointer; white-space:nowrap; }
.nv-map-hint { position:absolute; left:10px; bottom:8px; margin:0; font-size:12px; color:var(--graphite);
  pointer-events:none; z-index:2; }
.nv-map-msg { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; margin:0;
  font-family:var(--serif); font-size:17px; color:var(--graphite); text-align:center; padding:24px; }
.nv-map-msg[hidden] { display:none; }
.nv-map-card { position:absolute; top:10px; right:10px; bottom:10px; width:min(340px, calc(100% - 20px));
  overflow-y:auto; z-index:3; padding:14px 16px; border:1px solid var(--rule); border-radius:4px; text-align:left;
  background:color-mix(in srgb, var(--paper) 94%, transparent); }
.nv-map-card[hidden] { display:none; }
.nv-map-x { position:absolute; top:6px; right:8px; border:none; background:none; font-size:20px; line-height:1;
  color:var(--graphite); cursor:pointer; padding:4px; }
.nv-map-kicker { display:block; font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase;
  color:var(--kept); margin:0 24px 4px 0; padding:0; border:none; background:none; text-align:left; cursor:pointer; }
p.nv-map-kicker { cursor:default; }
.nv-map-title { display:block; font-family:var(--serif); font-size:19px; line-height:1.3; font-weight:500;
  color:var(--ink); margin:0 20px 0 0; text-decoration:none; text-align:left; padding:0 !important; }
a.nv-map-title:hover { text-decoration:underline; text-decoration-color:var(--kept); }
.nv-map-meta { font-size:13px; color:var(--graphite); margin:4px 0 0; }
.nv-map-sub { font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--graphite);
  margin:14px 0 4px; }
.nv-map-card ol { list-style:none; padding:0; margin:6px 0 0; }
.nv-map-card li { padding:7px 0; border-top:1px solid var(--rule); }
.nv-map-card li button { display:block; text-align:left; border:none; background:none; padding:0; cursor:pointer;
  font-family:var(--serif); font-size:15px; line-height:1.35; color:var(--ink); }
.nv-map-card li button:hover { color:var(--kept); }
.nv-map-card li span { display:block; font-size:12px; color:var(--graphite); margin-top:2px; }
.nv-map .scene-tooltip { font-family:inherit; font-size:13px; line-height:1.4; max-width:320px; white-space:normal;
  padding:6px 9px; border-radius:4px; background:var(--ink) !important; color:var(--paper) !important; }

@media (max-width: 640px) {
  .nv-map { height:440px; }
  .nv-map-hint { display:none; }
  .nv-map-card { top:auto; height:55%; }
  .nv-tiles { grid-template-columns:repeat(2, minmax(0, 1fr)); }
  .nv-tile .v { font-size:21px; }
  .nv-about table, .nv-about tbody, .nv-about tr, .nv-about td { display:block; width:auto !important; }
  .nv-about thead { display:none; }
  .nv-about tr { padding:12px 0; border-bottom:1px solid var(--rule) !important; }
  .nv-about td { min-width:0 !important; border:none !important; padding:2px 0 !important; }
  .nv-about td:nth-child(1) { font-size:17px; }
  .nv-about td:nth-child(3) { padding-top:6px !important; }
  .nv-masthead h1 { font-size:24px; }
  .nv-ask textarea { font-size:17px !important; }
  .nv-item .t { font-size:17px; }
}
"""


def theme() -> gr.themes.Base:
    t = gr.themes.Base(
        font=[gr.themes.GoogleFont("Hanken Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_sm,
    )
    return t.set(
        body_background_fill="#F5F7F9", body_background_fill_dark="#141B2D",
        body_text_color="#1B2440", body_text_color_dark="#E6EAF2",
        body_text_color_subdued="#5B6478", body_text_color_subdued_dark="#9AA4B8",
        background_fill_primary="#F5F7F9", background_fill_primary_dark="#141B2D",
        background_fill_secondary="#EDF0F4", background_fill_secondary_dark="#1B2338",
        block_background_fill="#F5F7F9", block_background_fill_dark="#141B2D",
        block_border_width="0px", block_shadow="none", block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        panel_background_fill="#F5F7F9", panel_background_fill_dark="#141B2D",
        border_color_primary="#D7DCE5", border_color_primary_dark="#2A3350",
        input_background_fill="#FFFFFF", input_background_fill_dark="#1B2338",
        input_border_color="#C9D0DC", input_border_color_dark="#334064",
        input_border_color_focus="#0F7B6C", input_border_color_focus_dark="#4FC7B4",
        color_accent="#0F7B6C", color_accent_soft="#DDEFEC", color_accent_soft_dark="#1E3A3A",
        link_text_color="#1B2440", link_text_color_dark="#E6EAF2",
        button_primary_background_fill="#0F7B6C", button_primary_background_fill_dark="#4FC7B4",
        button_primary_background_fill_hover="#0B6457", button_primary_background_fill_hover_dark="#3DB3A0",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#0E1A2B",
        button_secondary_background_fill="transparent", button_secondary_background_fill_dark="transparent",
        button_secondary_border_color="#C9D0DC", button_secondary_border_color_dark="#334064",
        checkbox_label_background_fill_selected="#DDEFEC", checkbox_label_background_fill_selected_dark="#1E3A3A",
        shadow_drop="none", layout_gap="18px",
    )


# ---------------------------------------------------------------- small formatting helpers
def human_date(iso: str) -> str:
    try:
        return parse_iso(iso).strftime("%-d %b %Y")
    except Exception:
        return iso[:10]


def ago(iso: str | None) -> str:
    if not iso:
        return "not yet"
    mins = int((datetime.now(timezone.utc) - parse_iso(iso)).total_seconds() // 60)
    if mins < 2:
        return "just now"
    if mins < 90:
        return f"{mins} minutes ago"
    hours = round(mins / 60)
    return f"{hours} hours ago" if hours < 36 else f"{round(hours / 24)} days ago"


def _meta(source: str, source_type: str, published: str, upvotes: int = 0, extra: str = "") -> str:
    kind = ('<span class="nv-kind" title="Paper" aria-label="Paper">●</span>' if source_type == "paper"
            else '<span class="nv-kind" title="Blog post or news" aria-label="Blog post or news">◆</span>')
    parts = [html.escape(source), human_date(published)]
    if upvotes:
        parts.append(f"{upvotes} upvote{'s' if upvotes != 1 else ''}")
    return f'<p class="m">{kind}{", ".join(parts)}{extra}</p>'


def masthead_html(store: CorpusStore) -> str:
    s = store.stats()
    if not s["total"]:
        line = ("The collection is being gathered. This takes a few minutes after the app starts; "
                "Keyword search and the assistant will work once it is ready.")
    else:
        line = (f"{s['total']:,} papers and posts from arXiv, Hugging Face Daily Papers and AI lab blogs, "
                f"updated {ago(s['last_refresh'])}.")
    return f'<header class="nv-masthead"><h1>AI Research Navigator</h1><p>{line}</p></header>'


# ---------------------------------------------------------------- landing: stats, papers per day, topics
def _tile(label: str, value: str, sub: str = "") -> str:
    sub = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<div class="nv-tile"><p class="k">{label}</p><p class="v">{value}</p>{sub}</div>'


def _week_change(week: int, prev: int) -> str:
    if not prev:
        return "the first week collected" if week else ""
    diff = week - prev
    if diff == 0:
        return f"same as the week before ({prev:,})"
    pct = round(abs(diff) / prev * 100)
    return f"{'up' if diff > 0 else 'down'} {pct}% on the week before ({prev:,})"


def papers_chart(days: list[tuple[date, int, int]]) -> str:
    """Papers per publication day as HTML bars (they keep real text sizes at any width), with a table view."""
    if not days:
        return ""
    peak = max(p for _, p, _ in days)
    top = max(10, math.ceil(peak / 10) * 10) if peak <= 100 else math.ceil(peak / 50) * 50
    today = days[-1][0]
    bars = []
    for d, papers, posts in days:
        posts_line = f", {_plural(posts, 'blog post')}" if posts else ""
        tip = f"{d.strftime('%a %-d %b')}: {_plural(papers, 'paper')}{posts_line}" + \
              (" (so far today)" if d == today else "")
        bars.append(f'<span class="nv-bar{" today" if d == today else ""}" tabindex="0" aria-label="{tip}">'
                    f'<i style="height:{100 * papers / top:.1f}%"></i><span class="tip">{tip}</span></span>')
    n = len(days)
    ticks = [i for i in range(n - 1, -1, -7)][::-1]
    xlabs = "".join(f'<span style="left:{(i + 0.5) / n * 100:.2f}%">{days[i][0].strftime("%-d %b")}</span>'
                    for i in ticks)
    grid = "".join(f'<span class="grid" style="top:{100 - f * 100:.0f}%"></span>'
                   f'<span class="ylab" style="top:{(1 - f) * 150:.0f}px">{round(top * f):,}</span>'
                   for f in (1, 0.5))
    total = sum(p for _, p, _ in days)
    best = max(days, key=lambda x: x[1])
    rows_ = "".join(f"<tr><td>{d.strftime('%a %-d %b')}</td><td>{p:,}</td><td>{b:,}</td></tr>"
                    for d, p, b in reversed(days))
    return (f'<div class="nv-chart" role="group" aria-label="{total:,} papers over the last {n} days, '
            f'busiest on {best[0].strftime("%-d %b")} with {best[1]:,}">{grid}'
            f'<div class="nv-bars">{"".join(bars)}</div><div class="nv-xlabs">{xlabs}</div></div>'
            f'<details class="nv-chart-table"><summary>Show as a table</summary><table><thead><tr><th>Day</th>'
            f'<th>Papers</th><th>Blog posts</th></tr></thead><tbody>{rows_}</tbody></table></details>')


def map_intro_html(topics: list[dict]) -> str:
    """The caption under the research map, with the largest topics as fly-to chips."""
    biggest = sorted(enumerate(topics), key=lambda gt: -gt[1]["size"])[:8]
    chips = "".join(f'<li><button type="button" data-nv-topic="{g}">{html.escape(t["label"])}'
                    f'<span>{t["size"]:,}</span></button></li>' for g, t in biggest)
    pick = f' Pick a topic to fly there:</p><ul class="nv-topics">{chips}</ul>' if chips else "</p>"
    return (f'<section><h2 class="nv-land-h">Map of the papers</h2><p class="nv-land-note">Every paper is a point, '
            f'placed by what it is about. Papers on the same subject cluster around a topic, named by its most '
            f'typical title words, and each paper is linked to the papers closest to it in meaning. Newer papers '
            f'are a stronger teal.{pick}</section>')


def landing_html(stats: dict) -> str:
    """The collection's numbers, under the research map on the Ask tab before a question."""
    if not stats["total"]:
        return ('<p class="nv-hint">The collection is being gathered. Its numbers and the map of its papers '
                'appear here once it is ready.</p>')
    tiles = "".join([
        _tile("In the collection", f"{stats['total']:,}"),
        _tile("Papers", f"{stats['papers']:,}"),
        _tile("Blog posts and news", f"{stats['posts']:,}"),
        _tile("Sources", f"{stats['sources']:,}"),
    ])
    return f'<div class="nv-tiles">{tiles}</div>'


def trend_html(stats: dict) -> str:
    """New papers this week against last week, and per day, for the top of the right-hand pane."""
    if not stats["total"]:
        return ""
    days = stats["days"]
    tiles = (_tile("Last 7 days", f"{stats['week']:,}", _week_change(stats["week"], stats["prev_week"]))
             + _tile("The 7 days before", f"{stats['prev_week']:,}"))
    return (f'<section class="nv-trend"><p class="nv-pane-h">New papers, last {len(days)} days</p>'
            f'<div class="nv-tiles two">{tiles}</div><p class="nv-land-note">Per day, by publication date. arXiv '
            f'announces nothing at weekends, so those bars stay low.</p>{papers_chart(days)}</section>')


# ---------------------------------------------------------------- result rows
def row(it: NewsItem, query: str = "", chars: int = 380) -> str:
    summary = it.summary[:chars] + ("…" if len(it.summary) > chars else "")
    authors = ", ".join(it.authors[:3]) + (" and others" if len(it.authors) > 3 else "")
    extra = f", {html.escape(authors)}" if authors else ""
    if it.pdf_url:
        extra += f'<a class="pdf" href="{html.escape(it.pdf_url)}" target="_blank" rel="noopener">PDF</a>'

    return (f'<article class="nv-item"><a class="t" href="{html.escape(it.url)}" target="_blank" rel="noopener">'
            f'{highlight(it.title, query)}</a>{_meta(it.source, it.source_type, it.published, it.upvotes, extra)}'
            f'<p class="s">{highlight(summary, query)}</p></article>')


def rows(items: list[NewsItem], query: str = "", count_line: str = "") -> str:
    if not items:
        return ('<div class="nv-rows"><p class="empty">Nothing matches. Try fewer words, a longer date range, '
                'or clear the source filter.</p></div>')
    head = f'<p class="nv-count">{count_line}</p>' if count_line else ""
    return f'<div class="nv-rows">{head}{"".join(row(it, query) for it in items)}</div>'


# ---------------------------------------------------------------- latest, by day
LATEST_DAYS = 10
DAY_PAGE = 50
ALL_DAYS = "all"


def day_groups(items: list[NewsItem], today: date | None = None) -> list[tuple[date, list[NewsItem]]]:
    """Items from the last LATEST_DAYS calendar days (UTC, today included), newest day first. Within a day the
    most upvoted come first, so Daily Papers picks and lab posts sit above the 0-upvote arXiv flood."""
    today = today or datetime.now(timezone.utc).date()
    first = today - timedelta(days=LATEST_DAYS - 1)
    days: dict[date, list[NewsItem]] = {}
    for it in items:
        try:
            d = parse_iso(it.published).astimezone(timezone.utc).date()
        except Exception:
            continue
        if first <= d <= today:
            days.setdefault(d, []).append(it)
    for day in days.values():
        day.sort(key=lambda it: (it.upvotes, it.published), reverse=True)
    return sorted(days.items(), reverse=True)


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'s' if n != 1 else ''}"


def day_heading(d: date, items: list[NewsItem]) -> str:
    papers = sum(it.source_type == "paper" for it in items)
    parts = [_plural(papers, "paper") if papers else "", _plural(len(items) - papers, "blog post")
             if len(items) > papers else ""]
    return (f'<h2>{d.strftime("%A %-d %b %Y")}</h2>'
            f'<p class="nv-count">{", ".join(p for p in parts if p)}</p>')


def latest_html(groups: list[tuple[date, list[NewsItem]]], per_day: int) -> str:
    if not groups:
        return f'<div class="nv-rows"><p class="empty">Nothing new in the last {LATEST_DAYS} days.</p></div>'
    total = sum(len(g) for _, g in groups)
    out = [f'<p class="nv-count">{total:,} papers and posts from the last {LATEST_DAYS} days, grouped by the day '
           f'they were published. Most upvoted first.</p>']
    for d, items in groups:
        more = (f'<p class="more">Showing the top {per_day} of {len(items):,}. Pick this day under Day to see '
                f'them all.</p>' if len(items) > per_day else "")
        out.append(f'<section class="nv-day">{day_heading(d, items)}'
                   f'{"".join(row(it) for it in items[:per_day])}{more}</section>')
    return f'<div class="nv-rows">{"".join(out)}</div>'


def latest_day_html(d: date, items: list[NewsItem], shown: int) -> str:
    if not items:
        return '<div class="nv-rows"><p class="empty">Nothing from this day matches the filters.</p></div>'
    note = f'<p class="more">Showing {min(shown, len(items)):,} of {len(items):,}.</p>'
    return (f'<div class="nv-rows"><section class="nv-day">{day_heading(d, items)}'
            f'{"".join(row(it) for it in items[:shown])}{note}</section></div>')


# ---------------------------------------------------------------- assistant outputs
REF_PAGE = 25
SEARCH_PAGE = 50
REF_SORTS = [("Relevance", "relevance"), ("Newest", "date"), ("Oldest", "oldest"), ("Most upvoted", "upvotes")]


def view_refs(refs: list[Candidate], sort: str = "relevance", source_type: str = "any",
              window: str = "Any time", sources: list[str] | None = None) -> list[tuple[int, Candidate]]:
    """(number, reference) pairs after the view filters and sort. Numbers come from relevance order, so the
    summary's [n] citations still point at the right item after re-sorting."""
    days = DATE_WINDOWS.get(window)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days) if days else None
    out = [(n, c) for n, c in enumerate(refs, 1)
           if (not sources or c["source"] in sources)
           and (source_type in (None, "any", "all") or c.get("source_type") == source_type)
           and (cutoff is None or parse_iso(c["published"]) >= cutoff)]
    if sort in ("date", "oldest"):
        out.sort(key=lambda p: p[1]["published"], reverse=sort == "date")
    elif sort == "upvotes":
        out.sort(key=lambda p: (p[1].get("upvotes", 0), -p[0]), reverse=True)
    return out


def _pct(rel: float) -> str:
    return "<1%" if rel < 0.005 else f"{round(rel * 100)}%"


def _score_badge(c: Candidate) -> str:
    """Match % = the cross-encoder's estimate that the item is relevant to the question."""
    rel = c.get("relevance")
    if c.get("weak"):
        label = f"Low match {_pct(rel)}" if rel is not None else "Low match"
        if c.get("rejected"):
            label += " · the model judged it off-topic"
        return f'<span class="nv-score low">{label}</span>'
    return f'<span class="nv-score">Match {_pct(rel)}</span>' if rel is not None else ""


def _ref_item(n: int, c: Candidate) -> str:
    if c.get("digest"):
        body = f'<p class="d"><span class="lbl">Model summary</span>{html.escape(c["digest"])}</p>'
    else:
        text = c.get("summary", "")
        text = text[:320].rsplit(" ", 1)[0] + "…" if len(text) > 320 else text
        body = f'<p class="d ab"><span class="lbl">From the abstract</span>{html.escape(text)}</p>' if text else ""
    return (f'<li id="nv-ref-{n}"><span class="n">[{n}]</span><div>'
            f'<a class="t" href="{html.escape(c["url"])}" target="_blank" rel="noopener">{html.escape(c["title"])}</a>'
            f'{_meta(c["source"], c.get("source_type", "paper"), c["published"], c.get("upvotes", 0), _score_badge(c))}'
            f'{body}</div></li>')


def references_html(refs: list[Candidate], sort: str = "relevance", source_type: str = "any",
                    window: str = "Any time", sources: list[str] | None = None, shown: int = REF_PAGE,
                    summarized: int = 0) -> str:
    if not refs:
        return ""
    view = view_refs(refs, sort, source_type, window, sources)
    order = dict(REF_SORTS)
    sort_name = next((k for k, v in order.items() if v == sort), "relevance").lower()
    head = f"{len(refs):,} reference{'s' if len(refs) != 1 else ''} · sorted by {sort_name}"
    if len(view) < len(refs):
        head += f" · {len(view):,} match your filters"
        hidden = summarized - sum(1 for n, _ in view if n <= summarized)
        if hidden > 0:
            head += f" ({hidden} cited in the summary {'is' if hidden == 1 else 'are'} hidden)"
    if len(view) > shown:
        head += f" · showing {shown:,}"
    if any(c.get("relevance") is not None for c in refs):
        head += " · match % is how relevant the reranker judges each one to your question"
    items = "".join(_ref_item(n, c) for n, c in view[:shown])
    body = f"<ol>{items}</ol>" if items else '<p class="empty">No reference matches these filters.</p>'
    return (f'<section class="nv-refs"><h2>References</h2><p class="nv-count">{head}</p>{weak_warning(refs)}'
            f'{body}</section>')


def weak_warning(refs: list[Candidate]) -> str:
    weak = sum(1 for c in refs if c.get("weak"))
    if not weak:
        return ""
    strong = len(refs) - weak
    if strong:
        text = (f"Only {strong} reference{' matched' if strong == 1 else 's matched'} your question well. "
                f"{weak} more with low match scores {'is' if weak == 1 else 'are'} shown so you have at least "
                f"{len(refs)}; treat {'it' if weak == 1 else 'them'} as leads, not answers.")
    else:
        text = (f"No reference matched your question well. These {weak} are the closest, with low match scores; "
                "treat them as leads, not answers. Try other words or a broader question.")
    return f'<p class="nv-warn" role="note">{text}</p>'


def has_more(refs: list[Candidate], shown: int, sort: str = "relevance", source_type: str = "any",
             window: str = "Any time", sources: list[str] | None = None) -> bool:
    return len(view_refs(refs, sort, source_type, window, sources)) > shown


# kept for callers that want markdown (e.g. MCP clients reading the API output)
def format_references(refs: list[Candidate]) -> str:
    return "\n\n".join(f"[{n}] [{c['title']}]({c['url']}) ({c['source']}, {human_date(c['published'])})"
                       + (f", match {_pct(c['relevance'])}" if c.get("relevance") is not None else "")
                       + (" (low match)" if c.get("weak") else "")
                       + (f"\n{c['digest']}" if c.get("digest") else "") for n, c in enumerate(refs, 1))


STEP_LABEL = {
    "retrieve": "Pulled from the collection",
    "rerank": "Relevant after reranking",
    "grade_documents": "Reviewed and kept by the model",
    "live_search": "Found by a live arXiv search",
    "generate": "In the results",
}
STEP_NOTE = {
    "analyze_query": "Reading your question",
    "rewrite_query": "Too few were relevant, so the question was reworded and searched again",
}
STEPS_EMPTY = ("<div class='nv-steps-empty'>As the assistant works, this panel shows how the collection is "
               "narrowed, with the real count at each step:<ol>"
               "<li>Pull candidates by keyword (BM25) and meaning (embeddings).</li>"
               "<li>Rerank them with a cross-encoder; keep every one that is relevant.</li>"
               "<li>The model reviews the best ones and writes a short summary of each.</li>"
               "<li>If too few are relevant: reword the question, then search arXiv live.</li></ol></div>")

def _bar(count: int, was: int | None, total: int) -> str:
    scale = math.log10(max(total, 10) + 1)

    def pct(n: int) -> float:
        return min(100.0, round(100 * math.log10(n + 1) / scale, 1)) if n > 0 else 0.0

    was_el = f'<span class="was" style="width:{pct(was)}%"></span>' if was is not None and was > count else ""
    return f'<span class="nv-rail" aria-hidden="true">{was_el}<span class="kept" style="width:{pct(count)}%"></span></span>'


def sieve_html(total: int, steps: list[dict], trace: list[str] | None = None, working: bool = False) -> str:
    """The narrowing from the whole collection to the items used in the answer, one row per step."""
    if not steps and not working:
        return STEPS_EMPTY
    out = [f'<div class="nv-step"><span class="label">In the collection</span>{_bar(total, None, total)}'
           f'<span class="count">{total:,}</span></div>']
    pool = total
    for i, st in enumerate(steps):
        new = " new" if i == len(steps) - 1 else ""
        node, count = st["node"], st.get("count")
        if node in STEP_NOTE:
            if node == "analyze_query" and i != len(steps) - 1:
                continue  # only show "reading your question" while it is the current step
            out.append(f'<div class="nv-step note{new}"><span class="label">{STEP_NOTE[node]}</span></div>')
            continue
        if node == "live_search":
            out.append(f'<div class="nv-step{new}"><span class="label">{STEP_LABEL[node]}</span>'
                       f'{_bar(count, None, total)}<span class="count">+{count}</span></div>')
            pool = count
            continue
        # outline = what this step filtered out; only retrieve (from the collection) and rerank (from the pool) narrow
        was = {"retrieve": total, "rerank": pool}.get(node)
        label = STEP_LABEL.get(node, node)
        if node == "generate" and st.get("summarized"):
            label += f" (summary from the best {st['summarized']})"
        out.append(f'<div class="nv-step{new}"><span class="label">{label}</span>'
                   f'{_bar(count or 0, was, total)}<span class="count">{count or 0:,}</span></div>')
        if node == "retrieve":
            pool = count or 0
    legend = ('<p class="nv-legend"><i class="k"></i>kept<i class="d"></i>filtered out at this step</p>'
              if len(steps) > 1 else "")
    detail = ""
    if trace:
        lines = "".join(f"<li>{re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'<strong>\1</strong>', html.escape(t))}</li>"
                        for t in trace)
        detail = f"<details><summary>Technical details</summary><ol>{lines}</ol></details>"
    return f'<section class="nv-sieve" aria-label="How the collection was narrowed">{"".join(out)}{legend}{detail}</section>'


def answer_md(question: str, body: str, earlier: list[tuple[str, str]], note: str = "") -> str:
    note = f"<p class='nv-note'>{html.escape(note)}</p>\n\n" if note else ""
    md = f"<div class='nv-prose'>\n\n### {html.escape(question)}\n\n{note}{body}\n\n</div>"
    if earlier:
        items = "\n\n".join(f"**{html.escape(q)}**\n\n{a}" for q, a in reversed(earlier))
        md += f"\n\n<details><summary>Earlier in this session ({len(earlier)})</summary>\n\n{items}\n\n</details>"
    return md


# ---------------------------------------------------------------- layout
def build_ui(store: CorpusStore, source_names: list[str], assistant_fn: Callable,
             refresh_fn: Callable[[], str], pull_mode: bool = False) -> gr.Blocks:

    def search(query: str, sources: list[str], source_type: str, window: str, sort: str) -> str:
        """Keyword (BM25) search over collected AI papers and posts. Quote a phrase to require it verbatim.

        Args:
            query: keywords, e.g. 'mixture of experts "load balancing"'
            sources: restrict to these sources (empty = all)
            source_type: "any", "paper" or "blog"
            window: one of "Any time", "Last 24 hours", "Last 3 days", "Last week", "Last month", "Last 3 months"
            sort: "relevance", "date" or "upvotes"
        """
        return search_page(query, sources, source_type, window, sort, SEARCH_PAGE)[0]

    def search_page(query: str, sources: list[str], source_type: str, window: str, sort: str,
                    shown: int) -> tuple[str, int]:
        hits = keyword_search(store, query, sources or None, source_type, DATE_WINDOWS.get(window), sort)
        n = len(hits)
        line = f"{n:,} result{'s' if n != 1 else ''}" + (f", showing the first {shown:,}" if n > shown else "")
        return rows([h[0] for h in hits[:shown]], query, line), n

    def search_more(query, sources, source_type, window, sort, shown):
        shown = int(shown or 0) + SEARCH_PAGE
        out, n = search_page(query, sources, source_type, window, sort, shown)
        return out, shown, gr.update(visible=n > shown)

    def search_reset(query, sources, source_type, window, sort):
        n = len(keyword_search(store, query, sources or None, source_type, DATE_WINDOWS.get(window), sort))
        return SEARCH_PAGE, gr.update(visible=n > SEARCH_PAGE)

    def refine(results: dict, sort: str, source_type: str, window: str, sources: list[str], shown: int = REF_PAGE):
        refs = (results or {}).get("refs", [])
        return (references_html(refs, sort, source_type, window, sources, shown, (results or {}).get("summarized", 0)),
                shown, gr.update(visible=has_more(refs, shown, sort, source_type, window, sources)))

    def refine_more(results, sort, source_type, window, sources, shown):
        return refine(results, sort, source_type, window, sources, int(shown or 0) + REF_PAGE)

    def latest_groups(sources, source_type):
        # one extra day: keyword_search's window is rolling, day_groups cuts at calendar days
        hits = keyword_search(store, "", sources or None, source_type, LATEST_DAYS + 1, "date")
        return day_groups([h[0] for h in hits])

    def latest(sources: list[str], source_type: str, day: str = ALL_DAYS, per_day: float = 10) -> str:
        """Papers and posts from the last 10 days, grouped by publication date (UTC), most upvoted first.

        Args:
            sources: source names to keep; empty means all
            source_type: "any", "paper" or "blog"
            day: "all" for every day (top `per_day` each), or one date as YYYY-MM-DD for that whole day
            per_day: how many items to show under each date when day is "all"
        """
        return latest_page(sources, source_type, day, per_day, DAY_PAGE)[0]

    def latest_page(sources, source_type, day, per_day, shown) -> tuple[str, int]:
        groups = latest_groups(sources, source_type)
        if not day or day == ALL_DAYS:
            return latest_html(groups, int(per_day)), 0
        items = next((g for d, g in groups if d.isoformat() == day), [])
        return latest_day_html(date.fromisoformat(day), items, int(shown)), len(items)

    def latest_more(sources, source_type, day, per_day, shown):
        shown = int(shown or 0) + DAY_PAGE
        out, n = latest_page(sources, source_type, day, per_day, shown)
        return out, shown, gr.update(visible=n > shown)

    def latest_reset(sources, source_type, day, per_day):
        return DAY_PAGE, gr.update(visible=latest_page(sources, source_type, day, per_day, DAY_PAGE)[1] > DAY_PAGE)

    def latest_days():
        days = [(f'{d.strftime("%a %-d %b")} ({len(g):,})', d.isoformat()) for d, g in latest_groups(None, "any")]
        return gr.update(choices=[(f"All {LATEST_DAYS} days", ALL_DAYS)] + days, value=ALL_DAYS)

    def status() -> str:
        """Collection size, last update time and the health of each source."""
        s = store.stats()
        names = sorted(set(s["by_source"]) | set(s["source_status"]), key=lambda k: -s["by_source"].get(k, 0))
        trs = []
        for name in names:
            st = s["source_status"].get(name, "")
            ok = st.startswith("ok")
            note = "working" if ok else (html.escape(st.removeprefix("error: "))[:120] or "not fetched yet")
            trs.append(f'<tr><td><span class="nv-dot {"ok" if ok else "bad"}"></span>{html.escape(name)}</td>'
                       f'<td class="n">{s["by_source"].get(name, 0):,}</td><td>{note}</td></tr>')
        where = ("New items are collected every 12 hours by a GitHub Action and saved to a Hugging Face "
                 "dataset; this app downloads the latest copy every hour." if pull_mode else
                 "This app collects new items itself every few hours.")
        return (f'<div class="nv-sources"><p class="nv-hint">{s["total"]:,} items, updated {ago(s["last_refresh"])}. '
                f'{where}</p><table><thead><tr><th>Source</th><th style="text-align:right">Items</th>'
                f'<th>Last update</th></tr></thead><tbody>{"".join(trs)}</tbody></table></div>')

    def refresh_now() -> tuple[str, str, str]:
        """Update the collection now: fetch all sources (fetch mode, takes minutes) or download the latest
        copy from the Hugging Face dataset (pull mode)."""
        msg = refresh_fn()
        return f'<p class="nv-hint">{html.escape(msg)}</p>', status(), masthead_html(store)

    def map_data(*_) -> dict:  # Gradio passes the JS call's arguments; this takes none
        """The research map (topics, papers, links); fetched by the map's own script, not part of the API."""
        return research_map(store)

    def landing() -> tuple[str, str, str]:
        s = collection_stats(store)
        return map_intro_html(research_map(store)["topics"]), landing_html(s), trend_html(s)

    with gr.Blocks(title="AI Research Navigator") as demo:
        masthead = gr.HTML(masthead_html(store))
        with gr.Tabs():
            with gr.Tab("Ask"):
                with gr.Row(elem_classes="nv-desk", equal_height=False):
                    with gr.Column(scale=1, min_width=260, elem_classes="nv-side"):
                        msg = gr.Textbox(placeholder=ASK_PLACEHOLDER, show_label=False, lines=3, max_lines=8,
                                         elem_classes="nv-ask", autofocus=True)
                        ask = gr.Button("Ask", variant="primary")
                        gr.Examples(EXAMPLES, inputs=msg, label="Or try one of these")
                        with gr.Column(elem_classes="nv-refine"):
                            gr.HTML('<p class="nv-pane-h">Refine the list</p>', elem_classes="nv-flush")
                            a_sort = gr.Dropdown(REF_SORTS, value="relevance", label="Sort by")
                            a_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                            a_window = gr.Dropdown(list(DATE_WINDOWS), value="Any time", label="Published")
                            a_sources = gr.Dropdown(source_names, multiselect=True, label="Sources",
                                                    info="Empty means all")
                        start_over = gr.Button("Start a new conversation", size="sm", variant="secondary",
                                               elem_classes="nv-reset")
                    with gr.Column(scale=3, min_width=380, elem_classes="nv-center"):
                        answer = gr.Markdown(f"<p class='nv-hint'>{ASK_EMPTY}</p>", elem_classes="nv-answer")
                        refs = gr.HTML(elem_classes="nv-flush")
                        more = gr.Button("Show more references", size="sm", variant="secondary", visible=False,
                                         elem_classes="nv-more")
                        # shown until a question is asked; hidden by a CSS class so the 3D map stays mounted
                        with gr.Column(elem_classes="nv-landing"):
                            gr.HTML(MAP_HTML, head=MAP_HEAD, js_on_load=MAP_JS, server_functions=[map_data],
                                    elem_classes="nv-flush")
                            map_intro = gr.HTML(elem_classes="nv-flush")
                            stats = gr.HTML(elem_classes="nv-flush")
                    with gr.Column(scale=1, min_width=260, elem_classes="nv-steps"):
                        trend = gr.HTML(elem_classes="nv-flush")  # stays while an answer is shown
                        gr.HTML('<p class="nv-pane-h">What the assistant did</p>', elem_classes="nv-flush")
                        sieve = gr.HTML(STEPS_EMPTY, elem_classes="nv-flush")
                conv = gr.State([])
                results = gr.State({})
                shown = gr.State(REF_PAGE)
                # History and results live in gr.State: hidden from the API/MCP schema, so callers only pass
                # (message, sources, source_type, window, sort) and can't send a malformed chat history.
                ask_inputs = [msg, conv, a_sources, a_type, a_window, a_sort]
                ask_outputs = [answer, refs, sieve, msg, conv, results, shown]
                view = [results, a_sort, a_type, a_window, a_sources]
                # hide the landing in the browser, registered first: the page runs a trigger's listeners in
                # order and would otherwise hold this one until the answer is done
                for trigger in (msg.submit, ask.click):
                    trigger(None, js=LANDING_JS.format(op="add"), api_visibility="private")
                msg.submit(assistant_fn, ask_inputs, ask_outputs).then(
                    refine, view, [refs, shown, more], api_visibility="private")
                ask.click(assistant_fn, ask_inputs, ask_outputs, api_visibility="private").then(
                    refine, view, [refs, shown, more], api_visibility="private")
                for comp in (a_sort, a_type, a_window, a_sources):
                    comp.change(refine, view, [refs, shown, more], api_visibility="private")
                more.click(refine_more, view + [shown], [refs, shown, more], api_visibility="private")
                start_over.click(None, js=LANDING_JS.format(op="remove"), api_visibility="private")
                start_over.click(lambda: (f"<p class='nv-hint'>{ASK_EMPTY}</p>", "", STEPS_EMPTY, "", [], {},
                                          REF_PAGE, gr.update(visible=False)), None,
                                 ask_outputs + [more], api_visibility="private")

            with gr.Tab("Search"):
                q = gr.Textbox(placeholder='Keywords, e.g. diffusion "video editing"', show_label=False,
                               submit_btn="Search", elem_classes="nv-ask")
                with gr.Row(elem_classes="nv-filters"):
                    s_sources = gr.Dropdown(source_names, multiselect=True, label="Sources", info="Empty means all")
                    s_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                    s_window = gr.Dropdown(list(DATE_WINDOWS), value="Any time", label="Published")
                    s_sort = gr.Radio([("Best match", "relevance"), ("Newest", "date"), ("Most upvoted", "upvotes")],
                                      value="relevance", label="Sort by")
                s_results = gr.HTML('<p class="nv-hint">Type keywords and press Enter. Put a phrase in quotes '
                                    'to match it exactly.</p>')
                s_more = gr.Button("Show more results", size="sm", variant="secondary", visible=False,
                                   elem_classes="nv-more")
                s_shown = gr.State(SEARCH_PAGE)
                inputs = [q, s_sources, s_type, s_window, s_sort]
                q.submit(search, inputs, s_results).then(search_reset, inputs, [s_shown, s_more],
                                                          api_visibility="private")
                for comp in (s_sources, s_type, s_window, s_sort):
                    comp.change(search, inputs, s_results, api_visibility="private").then(
                        search_reset, inputs, [s_shown, s_more], api_visibility="private")
                s_more.click(search_more, inputs + [s_shown], [s_results, s_shown, s_more],
                             api_visibility="private")

            with gr.Tab("Latest"):
                with gr.Row(elem_classes="nv-filters"):
                    l_sources = gr.Dropdown(source_names, multiselect=True, label="Sources", info="Empty means all")
                    l_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                    l_day = gr.Dropdown([(f"All {LATEST_DAYS} days", ALL_DAYS)], value=ALL_DAYS, label="Day",
                                        info="Pick a day to see all of it")
                    l_per = gr.Slider(5, 50, value=10, step=5, label="Per day", info=f"With All {LATEST_DAYS} days")
                feed = gr.HTML()
                l_more = gr.Button("Show more from this day", size="sm", variant="secondary", visible=False,
                                   elem_classes="nv-more")
                l_shown = gr.State(DAY_PAGE)
                l_inputs = [l_sources, l_type, l_day, l_per]
                for comp in l_inputs:
                    comp.change(latest, l_inputs, feed, api_visibility="private").then(
                        latest_reset, l_inputs, [l_shown, l_more], api_visibility="private")
                l_more.click(latest_more, l_inputs + [l_shown], [feed, l_shown, l_more], api_visibility="private")

            with gr.Tab("Sources"):
                stat = gr.HTML()
                btn = gr.Button("Download the latest copy" if pull_mode else "Collect new items now",
                                variant="secondary", size="sm")
                out = gr.HTML()
                btn.click(refresh_now, None, [out, stat, masthead]).then(
                    latest_days, None, l_day, api_visibility="private").then(
                    landing, None, [map_intro, stats, trend], api_visibility="private")

            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD, elem_classes="nv-about")

        demo.load(latest_days, None, l_day, api_visibility="private")
        demo.load(latest, l_inputs, feed, api_name="latest_feed")
        demo.load(status, None, stat, api_name="status")
        demo.load(lambda: masthead_html(store), None, masthead, api_visibility="private")
        demo.load(landing, None, [map_intro, stats, trend], api_visibility="private")
    return demo
