/* Inference Atlas — app engine
   Data contracts:
     MODULES   : [{ id, num, title, tagline, focus, outcome, units:[Unit] }]
     Unit      : { id, title, tier, summary, why, core[], math[], tables[], code[],
                   pitfalls[], build[], checks[], res[] }
     RESOURCES : [{ id, kind, title, url, src, year, note, essential, topics[] }]
     DRILLS    : [{ id, kind, level, module, q, a }]
     REFERENCE : { hardware, models, formulas, glossary, pitfalls, checklist }
*/
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  /* inline markup: `code`, **bold**, [text](url) */
  const rich = (s) => esc(s)
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, (_, c) => `<strong>${c}</strong>`)
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      (_, t, u) => `<a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`);

  const MODULES = window.MODULES || [];
  const RESOURCES = (function () {
    const raw = window.RESOURCES;
    if (Array.isArray(raw)) return raw;
    if (raw && typeof raw === 'object') return Object.keys(raw).map((k) => raw[k]);
    return [];
  })();
  const DRILLS = window.DRILLS || [];
  const REF = window.REFERENCE || {};

  const UNITS = [];
  MODULES.forEach((m) => m.units.forEach((u, i) => {
    u.module = m; u.indexInModule = i; u.globalIndex = UNITS.length; UNITS.push(u);
  }));
  const UNIT_BY_ID = new Map(UNITS.map((u) => [u.id, u]));
  const RES_BY_ID = new Map(RESOURCES.map((r) => [r.id, r]));

  const TIERS = {
    1: { label: 'Foundations', cls: 't1' },
    2: { label: 'Core', cls: 't2' },
    3: { label: 'Advanced', cls: 't3' },
    4: { label: 'Frontier', cls: 't4' }
  };
  const KIND_LABEL = {
    paper: 'paper', blog: 'essay', docs: 'docs', video: 'video',
    code: 'code', book: 'book', tool: 'tool', course: 'course'
  };

  const store = {
    read(k, fallback) {
      try { const v = localStorage.getItem(k); return v === null ? fallback : JSON.parse(v); }
      catch (e) { return fallback; }
    },
    write(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode */ } }
  };

  const state = {
    view: 'track',
    moduleId: null,
    unitId: null,
    libFilter: 'all',
    libQuery: '',
    drillFilter: 'all',
    refTab: 'formulas',
    done: new Set(store.read('llmi-done', [])),
    starred: new Set(store.read('llmi-starred', [])),
    openDrills: new Set(),
    searchIndex: [],
    searchSel: 0
  };

  const persist = () => {
    store.write('llmi-done', [...state.done]);
    store.write('llmi-starred', [...state.starred]);
  };

  /* ---------------------------------------------------------------- utils */
  const pct = (a, b) => (b ? Math.round((a / b) * 100) : 0);
  const moduleDone = (m) => m.units.filter((u) => state.done.has(u.id)).length;

  const fmt = (n, digits) => {
    if (!isFinite(n)) return '—';
    if (n === 0) return '0';
    const d = digits === undefined ? 2 : digits;
    const abs = Math.abs(n);
    if (abs >= 1e12) return (n / 1e12).toFixed(d) + 'T';
    if (abs >= 1e9) return (n / 1e9).toFixed(d) + 'B';
    if (abs >= 1e6) return (n / 1e6).toFixed(d) + 'M';
    if (abs >= 1e3) return (n / 1e3).toFixed(d) + 'K';
    if (abs >= 1) return n.toFixed(d);
    if (abs >= 0.001) return n.toFixed(4);
    return n.toExponential(2);
  };
  const gb = (bytes) => (bytes / 1024 ** 3);

  /* ------------------------------------------------------------- fragments */
  const tierTag = (t) => `<span class="tag ${TIERS[t].cls}">${TIERS[t].label}</span>`;

  const listBlock = (title, items, cls) => {
    if (!items || !items.length) return '';
    return `<div class="prose-section"><h2>${esc(title)}</h2>
      <ul class="clean-list ${cls || ''}">${items.map((i) => `<li>${rich(i)}</li>`).join('')}</ul></div>`;
  };

  const formulaBlock = (f) => `
    <div class="formula">
      ${f.title ? `<div class="formula-title">${esc(f.title)}</div>` : ''}
      <pre><code>${esc(f.body)}</code></pre>
      ${f.note ? `<div class="formula-note">${rich(f.note)}</div>` : ''}
    </div>`;

  const tableBlock = (t) => `
    <div class="prose-section">
      ${t.title ? `<h2>${esc(t.title)}</h2>` : ''}
      <div class="table-wrap"><table class="data">
        <thead><tr>${t.cols.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead>
        <tbody>${t.rows.map((r) => `<tr>${r.map((c) => `<td>${rich(c)}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div>
      ${t.note ? `<p class="res-src" style="margin-top:8px">${rich(t.note)}</p>` : ''}
    </div>`;

  const codeBlock = (c) => `
    <div class="formula">
      ${c.title ? `<div class="formula-title">${esc(c.title)}</div>` : ''}
      <pre><code>${esc(c.body)}</code></pre>
      ${c.note ? `<div class="formula-note">${rich(c.note)}</div>` : ''}
    </div>`;

  const resItem = (r) => {
    if (!r) return '';
    const kind = KIND_LABEL[r.kind] || r.kind;
    return `<div class="res-item">
      <span class="res-kind ${esc(r.kind)}">${esc(kind)}</span>
      <div class="res-body">
        <a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>${r.essential ? ' <span class="star" title="Essential">*</span>' : ''}
        <div class="res-src">${esc([r.src, r.year].filter(Boolean).join(' · '))}</div>
        ${r.note ? `<p>${rich(r.note)}</p>` : ''}
      </div>
    </div>`;
  };

  const resolveRes = (ids) => (ids || []).map((id) => RES_BY_ID.get(id)).filter(Boolean);

  /* ------------------------------------------------------------- rendering */
  function render() {
    const v = $('view');
    const map = {
      track: renderTrack, module: renderModule, unit: renderUnit,
      library: renderLibrary, reference: renderReference,
      calculators: renderCalculators, drills: renderDrills, progress: renderProgress
    };
    v.innerHTML = (map[state.view] || renderTrack)();
    if (state.view === 'calculators') initCalculators();
    bindViewEvents();
    updateChrome();
    v.scrollTop = 0;
    window.scrollTo({ top: 0, behavior: 'auto' });
  }

  function renderTrack() {
    const total = UNITS.length;
    const done = state.done.size;
    const papers = RESOURCES.filter((r) => r.kind === 'paper').length;
    return `
      <section class="page-intro">
        <p class="eyebrow">LLM inference &amp; serving</p>
        <h1>Everything between a prompt and a token, in order.</h1>
        <p class="lede">A complete, ordered curriculum for engineers who want to run and optimize
        LLM inference for real. It starts at the roofline and arithmetic of a forward pass, moves
        through attention kernels, KV cache management, serving schedulers, quantization,
        speculative decoding, and distributed serving, and ends at the unit economics that
        decide whether any of it is worth doing. Every unit carries the formulas, the source
        papers, the code to read, and the questions you should be able to answer.</p>
      </section>

      <div class="stats-grid">
        <div class="stat"><span class="stat-value">${MODULES.length}</span><span class="stat-label">Modules</span></div>
        <div class="stat"><span class="stat-value">${total}</span><span class="stat-label">Units</span></div>
        <div class="stat"><span class="stat-value">${papers}</span><span class="stat-label">Papers</span></div>
        <div class="stat"><span class="stat-value">${DRILLS.length}</span><span class="stat-label">Drills</span></div>
      </div>

      <div class="callout info">
        <strong>How to use this</strong>
        It is ordered, not dated. Modules 0&ndash;3 are the load-bearing ones: skipping them makes
        everything later a bag of tricks. Read the unit, do the build task, then answer the checks
        without looking. Tick a unit only when the checks are answerable cold.
      </div>

      <div class="section-heading">
        <h2>The track</h2>
        <p>${done} of ${total} units complete</p>
      </div>
      <div class="card-grid">
        ${MODULES.map((m) => {
          const d = moduleDone(m);
          return `<button class="card module-card" data-module="${esc(m.id)}">
            <span class="module-index">MODULE ${esc(m.num)}</span>
            <h3>${esc(m.title)}</h3>
            <p>${esc(m.tagline)}</p>
            <div class="module-foot">
              <span>${m.units.length} units</span>
              <span class="mini-track"><span class="mini-fill" style="width:${pct(d, m.units.length)}%"></span></span>
              <span>${d}/${m.units.length}</span>
            </div>
          </button>`;
        }).join('')}
      </div>

      <div class="section-heading"><h2>What this is not</h2></div>
      <ul class="clean-list">
        <li>Not a daily calendar. Move at whatever pace the material allows; the order is the product.</li>
        <li>Not a prompt-engineering or fine-tuning course. This is the serving side of the stack.</li>
        <li>Not vendor-neutral by accident: vLLM, SGLang and TensorRT-LLM are studied as source code,
        because the design decisions live there, not in the marketing pages.</li>
      </ul>`;
  }

  function renderModule() {
    const m = MODULES.find((x) => x.id === state.moduleId) || MODULES[0];
    const d = moduleDone(m);
    const idx = MODULES.indexOf(m);
    return `
      <div class="breadcrumb"><button data-goto="track">Curriculum</button><span>/</span><span>Module ${esc(m.num)}</span></div>
      <header class="detail-header">
        <div>
          <p class="eyebrow" style="margin-top:14px">Module ${esc(m.num)}</p>
          <h1>${esc(m.title)}</h1>
          <p class="lede">${rich(m.focus)}</p>
        </div>
        <div class="detail-actions">
          <div class="info-panel" style="min-width:180px">
            <h3>Progress</h3>
            <div class="progress-copy"><span>${d} / ${m.units.length}</span><span>${pct(d, m.units.length)}%</span></div>
            <div class="progress-track"><div class="progress-fill" style="width:${pct(d, m.units.length)}%"></div></div>
          </div>
        </div>
      </header>

      <div class="callout"><strong>Deliverable</strong>${rich(m.outcome)}</div>

      <div class="section-heading"><h2>Units</h2><p>Work them in order</p></div>
      <div class="unit-list">
        ${m.units.map((u, i) => unitRow(u, i + 1)).join('')}
      </div>

      <div class="prev-next">
        <button class="pn" data-module="${esc((MODULES[idx - 1] || m).id)}" ${idx === 0 ? 'disabled' : ''}>
          <small>Previous module</small><strong>${esc(idx === 0 ? '—' : MODULES[idx - 1].title)}</strong></button>
        <button class="pn next" data-module="${esc((MODULES[idx + 1] || m).id)}" ${idx === MODULES.length - 1 ? 'disabled' : ''}>
          <small>Next module</small><strong>${esc(idx === MODULES.length - 1 ? '—' : MODULES[idx + 1].title)}</strong></button>
      </div>`;
  }

  const unitRow = (u, n) => `
    <button class="unit-row ${state.done.has(u.id) ? 'done' : ''}" data-unit="${esc(u.id)}">
      <span class="unit-num">${state.done.has(u.id) ? '[x]' : String(n).padStart(2, '0')}</span>
      <span class="unit-body"><strong>${esc(u.title)}</strong><span>${esc(u.summary)}</span></span>
      <span class="unit-meta">${tierTag(u.tier)}<span class="count">${(u.res || []).length} refs</span></span>
    </button>`;

  function renderUnit() {
    const u = UNIT_BY_ID.get(state.unitId);
    if (!u) return renderTrack();
    const m = u.module;
    const prev = UNITS[u.globalIndex - 1];
    const next = UNITS[u.globalIndex + 1];
    const done = state.done.has(u.id);
    const res = resolveRes(u.res);
    const essential = res.filter((r) => r.essential);

    return `
      <div class="breadcrumb">
        <button data-goto="track">Curriculum</button><span>/</span>
        <button data-module="${esc(m.id)}">Module ${esc(m.num)}</button><span>/</span>
        <span>Unit ${String(u.indexInModule + 1).padStart(2, '0')}</span>
      </div>
      <header class="detail-header">
        <div>
          <div style="margin-top:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            ${tierTag(u.tier)}<span class="res-src">${esc(m.title)}</span>
          </div>
          <h1>${esc(u.title)}</h1>
          <p class="lede">${rich(u.summary)}</p>
        </div>
        <div class="detail-actions">
          <button class="button ${done ? '' : 'primary'}" data-toggle-done="${esc(u.id)}">${done ? 'Completed &#10003;' : 'Mark complete'}</button>
        </div>
      </header>

      <div class="detail-layout">
        <div>
          <div class="prose-section">
            <h2>Why this unit exists</h2>
            ${u.why.split('\n\n').map((p) => `<p>${rich(p)}</p>`).join('')}
          </div>

          ${listBlock('Core ideas', u.core)}

          ${(u.math || []).length ? `<div class="prose-section"><h2>The arithmetic</h2>
            ${u.math.map(formulaBlock).join('')}</div>` : ''}

          ${(u.tables || []).map(tableBlock).join('')}

          ${(u.code || []).length ? `<div class="prose-section"><h2>Code &amp; commands</h2>
            ${u.code.map(codeBlock).join('')}</div>` : ''}

          ${listBlock('Traps that catch people', u.pitfalls)}

          ${(u.build || []).length ? `<div class="prose-section"><h2>Build task</h2>
            <div class="callout"><strong>Produce something, not notes</strong>
            <ul class="clean-list" style="margin-top:8px">${u.build.map((b) => `<li>${rich(b)}</li>`).join('')}</ul></div></div>` : ''}

          ${(u.checks || []).length ? `<div class="prose-section"><h2>Answer these cold</h2>
            <ol class="clean-list">${u.checks.map((c) => `<li>${rich(c)}</li>`).join('')}</ol></div>` : ''}

          ${res.length ? `<div class="prose-section"><h2>Sources</h2>
            <div>${res.map(resItem).join('')}</div></div>` : ''}
        </div>

        <aside class="side-panel">
          <div class="info-panel">
            <h3>Unit at a glance</h3>
            <div class="out-line"><span>Tier</span><span>${TIERS[u.tier].label}</span></div>
            <div class="out-line"><span>Sources</span><span>${res.length}</span></div>
            <div class="out-line"><span>Must-read</span><span>${essential.length}</span></div>
            <div class="out-line"><span>Checks</span><span>${(u.checks || []).length}</span></div>
          </div>
          ${essential.length ? `<div class="info-panel"><h3>Start here</h3>
            <ul class="plain-list">${essential.slice(0, 5).map((r) =>
              `<li><a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>
               <div class="res-src">${esc(r.src || '')}</div></li>`).join('')}</ul></div>` : ''}
          <div class="info-panel">
            <h3>Module units</h3>
            <ul class="plain-list">${m.units.map((x, i) =>
              `<li><button class="button ghost" style="width:100%;text-align:left;font-weight:${x.id === u.id ? 700 : 400};min-height:34px;padding:6px 8px;font-size:13px;border-color:${x.id === u.id ? 'var(--cp-accent)' : 'var(--cp-border)'}" data-unit="${esc(x.id)}">${String(i + 1).padStart(2, '0')} &nbsp;${esc(x.title)}</button></li>`).join('')}</ul>
          </div>
        </aside>
      </div>

      <div class="prev-next">
        <button class="pn" ${prev ? `data-unit="${esc(prev.id)}"` : 'disabled'}>
          <small>Previous</small><strong>${esc(prev ? prev.title : '—')}</strong></button>
        <button class="pn next" ${next ? `data-unit="${esc(next.id)}"` : 'disabled'}>
          <small>Next</small><strong>${esc(next ? next.title : '—')}</strong></button>
      </div>`;
  }

  /* -------------------------------------------------------------- library */
  function renderLibrary() {
    const kinds = ['all', 'essential', 'paper', 'blog', 'docs', 'video', 'code', 'course', 'tool'];
    const q = state.libQuery.trim().toLowerCase();
    let list = RESOURCES.slice();
    if (state.libFilter === 'essential') list = list.filter((r) => r.essential);
    else if (state.libFilter !== 'all') list = list.filter((r) => r.kind === state.libFilter);
    if (q) list = list.filter((r) => (r.title + ' ' + (r.src || '') + ' ' + (r.note || '') + ' ' + (r.topics || []).join(' ')).toLowerCase().includes(q));

    const groups = {};
    list.forEach((r) => { (groups[r.group || 'Other'] = groups[r.group || 'Other'] || []).push(r); });
    const order = window.LIBRARY_GROUPS || Object.keys(groups);

    return `
      <section class="page-intro">
        <p class="eyebrow">Library</p>
        <h1>Papers, essays, docs, and code worth your time.</h1>
        <p class="lede">${RESOURCES.length} sources, deduplicated and grouped. Items marked
        <span class="star">*</span> are the ones you cannot skip &mdash; read those in full, skim the rest
        with a specific question in mind.</p>
      </section>

      <div class="field" style="margin-bottom:14px">
        <input id="lib-search" type="search" placeholder="Filter by title, author, venue, or topic" value="${esc(state.libQuery)}" autocomplete="off">
      </div>
      <div class="filter-row">
        ${kinds.map((k) => `<button class="chip ${state.libFilter === k ? 'active' : ''}" data-lib-filter="${k}">${k === 'all' ? 'All' : k === 'essential' ? 'Must read' : (KIND_LABEL[k] || k)}</button>`).join('')}
      </div>
      <p class="res-src" style="margin:-8px 0 20px">${list.length} shown</p>

      ${order.filter((g) => groups[g]).map((g) => `
        <div class="section-heading"><h2>${esc(g)}</h2><p>${groups[g].length}</p></div>
        <div class="card" style="padding:4px 16px">${groups[g].map(resItem).join('')}</div>
      `).join('') || '<div class="empty-state"><h2>Nothing matches</h2><p>Try a broader filter.</p></div>'}`;
  }

  /* ------------------------------------------------------------ reference */
  function renderReference() {
    const tabs = [
      ['formulas', 'Formula sheet'],
      ['hardware', 'Hardware'],
      ['models', 'Model configs'],
      ['pitfalls', 'Benchmarking traps'],
      ['glossary', 'Glossary']
    ];
    let body = '';
    if (state.refTab === 'formulas') {
      body = (REF.formulas || []).map((sec) => `
        <div class="section-heading"><h2>${esc(sec.title)}</h2></div>
        ${sec.blurb ? `<p class="lede" style="font-size:15px;margin-bottom:6px">${rich(sec.blurb)}</p>` : ''}
        ${sec.items.map(formulaBlock).join('')}`).join('');
    } else if (state.refTab === 'hardware') {
      const h = REF.hardware;
      body = `${tableBlock(h)}
        <div class="callout warn"><strong>Read the ratio column first</strong>
        The ops:byte ratio is the machine balance point: the arithmetic intensity your kernel must
        exceed to be compute bound. Decode at batch 1 sits at roughly 1&ndash;2 FLOP/byte, which is why
        it is memory bound on every accelerator in this table by two orders of magnitude.</div>`;
    } else if (state.refTab === 'models') {
      body = tableBlock(REF.models);
    } else if (state.refTab === 'pitfalls') {
      body = `<div class="unit-list">${(REF.pitfalls || []).map((p, i) => `
        <div class="card" style="padding:16px">
          <div style="display:flex;gap:10px;align-items:baseline">
            <span class="unit-num">${String(i + 1).padStart(2, '0')}</span>
            <div><strong>${rich(p.title)}</strong><p style="margin:6px 0 0;color:var(--cp-text-soft);font-size:14px">${rich(p.body)}</p></div>
          </div>
        </div>`).join('')}</div>`;
    } else {
      body = `<div class="table-wrap"><table class="data">
        <thead><tr><th>Term</th><th>Definition</th></tr></thead>
        <tbody>${(REF.glossary || []).map((g) => `<tr><td><strong>${esc(g.term)}</strong></td><td>${rich(g.def)}</td></tr>`).join('')}</tbody>
      </table></div>`;
    }

    return `
      <section class="page-intro">
        <p class="eyebrow">Reference</p>
        <h1>The numbers, in one place.</h1>
        <p class="lede">Formulas you should be able to reproduce on a whiteboard, the hardware
        constants they plug into, and the mistakes that make published benchmarks meaningless.</p>
      </section>
      <div class="filter-row">
        ${tabs.map(([k, l]) => `<button class="chip ${state.refTab === k ? 'active' : ''}" data-ref-tab="${k}">${l}</button>`).join('')}
      </div>
      ${body}`;
  }

  /* --------------------------------------------------------------- drills */
  function renderDrills() {
    const kinds = ['all', 'quant', 'concept', 'design', 'debug'];
    const labels = { all: 'All', quant: 'Quantitative', concept: 'Conceptual', design: 'System design', debug: 'Debugging' };
    const list = state.drillFilter === 'all' ? DRILLS : DRILLS.filter((d) => d.kind === state.drillFilter);
    return `
      <section class="page-intro">
        <p class="eyebrow">Drills</p>
        <h1>If you cannot do these, you have not learned it.</h1>
        <p class="lede">${DRILLS.length} problems: quantitative sizing, conceptual traps, system
        design, and production debugging. Work them with a calculator and no notes, then open the
        answer. Numbers use the hardware table in the reference tab.</p>
      </section>
      <div class="filter-row">
        ${kinds.map((k) => `<button class="chip ${state.drillFilter === k ? 'active' : ''}" data-drill-filter="${k}">${labels[k]}</button>`).join('')}
      </div>
      <p class="res-src" style="margin:-8px 0 18px">${list.length} shown ·
        <button class="button ghost" style="min-height:28px;padding:2px 10px;font-size:12px" data-drill-all="1">Toggle all</button></p>
      ${list.map((d, i) => {
        const open = state.openDrills.has(d.id);
        return `<div class="drill">
          <button class="drill-head" data-drill="${esc(d.id)}" aria-expanded="${open}">
            <span class="drill-idx">${String(i + 1).padStart(2, '0')}</span>
            <span><span class="drill-q">${rich(d.q)}</span>
              <span style="display:block;margin-top:7px"><span class="tag">${esc(labels[d.kind] || d.kind)}</span>
              ${d.level ? `<span class="tag ${d.level === 'hard' ? 't4' : d.level === 'medium' ? 't3' : 't2'}">${esc(d.level)}</span>` : ''}
              ${d.module ? `<span class="tag">${esc(d.module)}</span>` : ''}</span></span>
            <span class="drill-toggle">${open ? '&minus;' : '+'}</span>
          </button>
          ${open ? `<div class="drill-body">${d.a.split('\n\n').map((p) =>
            p.startsWith('```') ? `<pre><code>${esc(p.replace(/```/g, '').trim())}</code></pre>` : `<p>${rich(p)}</p>`).join('')}</div>` : ''}
        </div>`;
      }).join('')}`;
  }

  /* ------------------------------------------------------------- progress */
  function renderProgress() {
    const total = UNITS.length;
    const done = state.done.size;
    const byTier = {};
    UNITS.forEach((u) => {
      byTier[u.tier] = byTier[u.tier] || { total: 0, done: 0 };
      byTier[u.tier].total++;
      if (state.done.has(u.id)) byTier[u.tier].done++;
    });
    return `
      <section class="page-intro">
        <p class="eyebrow">Progress</p>
        <h1>${done} of ${total} units.</h1>
        <p class="lede">Stored in this browser's local storage only. No account, no server.
        Export if you want it on another device.</p>
      </section>
      <div class="stats-grid">
        ${Object.keys(TIERS).map((t) => {
          const b = byTier[t] || { total: 0, done: 0 };
          return `<div class="stat"><span class="stat-value">${b.done}/${b.total}</span><span class="stat-label">${TIERS[t].label}</span></div>`;
        }).join('')}
      </div>
      <div class="section-heading"><h2>By module</h2></div>
      <div class="unit-list">
        ${MODULES.map((m) => {
          const d = moduleDone(m);
          return `<button class="unit-row" data-module="${esc(m.id)}">
            <span class="unit-num">${esc(m.num)}</span>
            <span class="unit-body"><strong>${esc(m.title)}</strong>
              <span class="mini-track" style="max-width:none;margin-top:8px;display:block"><span class="mini-fill" style="width:${pct(d, m.units.length)}%"></span></span></span>
            <span class="unit-meta"><span class="count">${d}/${m.units.length}</span></span>
          </button>`;
        }).join('')}
      </div>
      <div class="section-heading"><h2>Data</h2></div>
      <div class="detail-actions">
        <button class="button" data-export="1">Export progress</button>
        <button class="button" data-import="1">Import progress</button>
        <button class="button" data-reset="1">Reset</button>
      </div>`;
  }

  /* ---------------------------------------------------------- calculators */
  const HW = () => (REF.hardwarePresets || []);
  const MD = () => (REF.modelPresets || []);

  function renderCalculators() {
    const hwOpts = HW().map((h, i) => `<option value="${i}">${esc(h.name)}</option>`).join('');
    const mdOpts = MD().map((m, i) => `<option value="${i}">${esc(m.name)}</option>`).join('');
    return `
      <section class="page-intro">
        <p class="eyebrow">Calculators</p>
        <h1>Do the sizing before you rent the GPU.</h1>
        <p class="lede">Every number below comes from a formula in the reference sheet. They are
        first-order models: they will get you within a factor of ~1.3 of a well-tuned engine, which
        is close enough to decide what to buy, what to quantize, and what to stop trying.</p>
      </section>

      <div class="calc-grid">
        <!-- 1. memory & KV -->
        <div class="calc">
          <div class="calc-head"><h3>Memory &amp; KV cache planner</h3>
            <p>Weights, KV bytes per token, and how many concurrent requests actually fit.</p></div>
          <div class="calc-body">
            <div class="field"><label>Model preset</label><select id="mem-preset"><option value="-1">Custom</option>${mdOpts}</select></div>
            <div class="field-row">
              <div class="field"><label>Layers</label><input id="mem-layers" type="number" value="32" min="1"></div>
              <div class="field"><label>Hidden size d_model</label><input id="mem-dmodel" type="number" value="4096" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>KV heads (GQA)</label><input id="mem-kvheads" type="number" value="8" min="1"></div>
              <div class="field"><label>Head dim</label><input id="mem-headdim" type="number" value="128" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Params (billions)</label><input id="mem-params" type="number" value="8" step="0.1" min="0.1"></div>
              <div class="field"><label>Weight dtype (bytes)</label><select id="mem-wbytes">
                <option value="2">FP16 / BF16 (2)</option><option value="1">FP8 / INT8 (1)</option>
                <option value="0.5">INT4 / FP4 (0.5)</option><option value="4">FP32 (4)</option></select></div>
            </div>
            <div class="field-row">
              <div class="field"><label>KV dtype (bytes)</label><select id="mem-kvbytes">
                <option value="2">FP16 / BF16 (2)</option><option value="1">FP8 (1)</option><option value="0.5">INT4 (0.5)</option></select></div>
              <div class="field"><label>Context per request</label><input id="mem-ctx" type="number" value="8192" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>GPU preset</label><select id="mem-gpu">${hwOpts}</select></div>
              <div class="field"><label>GPU count (TP)</label><input id="mem-gpus" type="number" value="1" min="1"></div>
            </div>
            <div class="field"><label>Framework / activation overhead (%)</label><input id="mem-over" type="number" value="10" min="0" max="60"></div>
          </div>
          <div class="calc-out" id="mem-out"></div>
          <p class="calc-note">KV bytes/token = 2 &times; layers &times; kv_heads &times; head_dim &times; dtype. MLA caches one latent rather than separate K and V, so set kv_heads &times; head_dim to <em>half</em> the latent width.</p>
        </div>

        <!-- 2. roofline / decode -->
        <div class="calc">
          <div class="calc-head"><h3>Roofline &amp; decode throughput</h3>
            <p>Where you sit on the roofline, and the token rate that follows from it.</p></div>
          <div class="calc-body">
            <div class="field"><label>GPU preset</label><select id="rf-gpu">${hwOpts}</select></div>
            <div class="field-row">
              <div class="field"><label>GPU count</label><input id="rf-gpus" type="number" value="1" min="1"></div>
              <div class="field"><label>Params (billions)</label><input id="rf-params" type="number" value="8" step="0.1" min="0.1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Weight bytes/param</label><select id="rf-wbytes">
                <option value="2">FP16 / BF16 (2)</option><option value="1">FP8 / INT8 (1)</option>
                <option value="0.5">INT4 (0.5)</option></select></div>
              <div class="field"><label>Batch size</label><input id="rf-batch" type="number" value="1" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>KV bytes/token/req</label><input id="rf-kvpt" type="number" value="131072" min="0"></div>
              <div class="field"><label>Avg context</label><input id="rf-ctx" type="number" value="4096" min="1"></div>
            </div>
            <div class="field"><label>Achieved bandwidth efficiency (%)</label><input id="rf-eff" type="number" value="80" min="10" max="100"></div>
          </div>
          <div class="calc-out" id="rf-out"></div>
          <p class="calc-note">Decode reads all weights plus the live KV every step. Time per step &asymp; bytes_moved / (BW &times; efficiency) when memory bound.</p>
        </div>

        <!-- 3. prefill / TTFT -->
        <div class="calc">
          <div class="calc-head"><h3>Prefill &amp; TTFT</h3>
            <p>Prefill is compute bound. This is the FLOP budget and the time it buys.</p></div>
          <div class="calc-body">
            <div class="field"><label>GPU preset</label><select id="pf-gpu">${hwOpts}</select></div>
            <div class="field-row">
              <div class="field"><label>GPU count</label><input id="pf-gpus" type="number" value="1" min="1"></div>
              <div class="field"><label>Params (billions)</label><input id="pf-params" type="number" value="8" step="0.1" min="0.1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Prompt tokens</label><input id="pf-tokens" type="number" value="4096" min="1"></div>
              <div class="field"><label>MFU (%)</label><input id="pf-mfu" type="number" value="45" min="1" max="100"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Layers</label><input id="pf-layers" type="number" value="32" min="1"></div>
              <div class="field"><label>d_model</label><input id="pf-dmodel" type="number" value="4096" min="1"></div>
            </div>
            <div class="field"><label>Precision peak</label><select id="pf-prec">
              <option value="bf16">BF16 dense</option><option value="fp8">FP8 dense</option><option value="fp4">FP4 dense</option></select></div>
          </div>
          <div class="calc-out" id="pf-out"></div>
          <p class="calc-note">Prefill FLOPs &asymp; 2 &times; N_params &times; T + 4 &times; L &times; d &times; T&sup2; (attention scores). The quadratic term is what makes long prompts hurt.</p>
        </div>

        <!-- 4. speculative decoding -->
        <div class="calc">
          <div class="calc-head"><h3>Speculative decoding payoff</h3>
            <p>Leviathan et al.'s expectation, adjusted for draft cost. Includes the loss regime.</p></div>
          <div class="calc-body">
            <div class="field-row">
              <div class="field"><label>Acceptance rate &alpha;</label><input id="sd-alpha" type="number" value="0.7" step="0.01" min="0.01" max="0.99"></div>
              <div class="field"><label>Draft length &gamma;</label><input id="sd-gamma" type="number" value="4" min="1" max="16"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Draft cost ratio c</label><input id="sd-c" type="number" value="0.15" step="0.01" min="0.001" max="1"></div>
              <div class="field"><label>Verify overhead (%)</label><input id="sd-vo" type="number" value="10" min="0" max="200"></div>
            </div>
          </div>
          <div class="calc-out" id="sd-out"></div>
          <p class="calc-note">E[tokens/iter] = (1 &minus; &alpha;<sup>&gamma;+1</sup>) / (1 &minus; &alpha;). Speedup = E / (1 + c&gamma; + overhead). The optimal &gamma; is shown; beyond it, drafting cost eats the win.</p>
        </div>

        <!-- 5. cost -->
        <div class="calc">
          <div class="calc-head"><h3>Cost per million tokens</h3>
            <p>The only number that decides whether an optimization ships.</p></div>
          <div class="calc-body">
            <div class="field-row">
              <div class="field"><label>GPU $/hour</label><input id="co-rate" type="number" value="2.99" step="0.01" min="0"></div>
              <div class="field"><label>GPU count</label><input id="co-gpus" type="number" value="1" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Output tokens/sec (peak)</label><input id="co-tps" type="number" value="2400" min="1"></div>
              <div class="field"><label>Utilization (%)</label><input id="co-util" type="number" value="45" min="1" max="100"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Prefill tokens/sec</label><input id="co-ptps" type="number" value="20000" min="1"></div>
              <div class="field"><label>Input:output ratio</label><input id="co-ratio" type="number" value="10" min="0.1" step="0.1"></div>
            </div>
            <div class="field"><label>Compare against API price ($/1M output)</label><input id="co-api" type="number" value="0.60" step="0.01" min="0"></div>
          </div>
          <div class="calc-out" id="co-out"></div>
          <p class="calc-note">Cost/1M = ($/hr &times; GPUs) / (tok/s &times; 3600 &times; utilization) &times; 10&#8310;. Utilization, not sticker price, dominates.</p>
        </div>

        <!-- 6. capacity / Little's law -->
        <div class="calc">
          <div class="calc-head"><h3>Capacity &amp; SLO (Little's Law)</h3>
            <p>Concurrency, arrival rate, and whether your latency budget survives.</p></div>
          <div class="calc-body">
            <div class="field-row">
              <div class="field"><label>Arrival rate (req/s)</label><input id="ca-qps" type="number" value="20" step="0.1" min="0.1"></div>
              <div class="field"><label>Output tokens/request</label><input id="ca-otok" type="number" value="500" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Per-stream tokens/s</label><input id="ca-itl" type="number" value="45" min="1"></div>
              <div class="field"><label>TTFT target (ms)</label><input id="ca-ttft" type="number" value="500" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Measured TTFT (ms)</label><input id="ca-ttftm" type="number" value="300" min="1"></div>
              <div class="field"><label>Max KV slots (tokens)</label><input id="ca-slots" type="number" value="400000" min="1"></div>
            </div>
            <div class="field"><label>Avg context per request (tokens)</label><input id="ca-ctx" type="number" value="3000" min="1"></div>
          </div>
          <div class="calc-out" id="ca-out"></div>
          <p class="calc-note">Little's Law: concurrency = arrival rate &times; residence time. If required concurrency exceeds KV capacity, the scheduler preempts and p99 detonates.</p>
        </div>

        <!-- 7. parallelism -->
        <div class="calc">
          <div class="calc-head"><h3>Parallelism &amp; communication</h3>
            <p>Tensor-parallel all-reduce volume against the link you actually have.</p></div>
          <div class="calc-body">
            <div class="field-row">
              <div class="field"><label>TP degree</label><input id="pa-tp" type="number" value="8" min="1"></div>
              <div class="field"><label>Layers</label><input id="pa-layers" type="number" value="80" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>d_model</label><input id="pa-dmodel" type="number" value="8192" min="1"></div>
              <div class="field"><label>Tokens in step (batch)</label><input id="pa-tokens" type="number" value="64" min="1"></div>
            </div>
            <div class="field-row">
              <div class="field"><label>Interconnect GB/s (per GPU)</label><input id="pa-bw" type="number" value="450" min="1"></div>
              <div class="field"><label>Activation bytes</label><select id="pa-abytes"><option value="2">BF16 (2)</option><option value="1">FP8 (1)</option></select></div>
            </div>
            <div class="field"><label>Step time budget (ms)</label><input id="pa-step" type="number" value="20" step="0.5" min="0.5"></div>
          </div>
          <div class="calc-out" id="pa-out"></div>
          <p class="calc-note">Ring all-reduce moves 2(P&minus;1)/P &times; message bytes per GPU. Two all-reduces per layer (after attention out-proj and after MLP down-proj).</p>
        </div>
      </div>`;
  }

  function initCalculators() {
    const num = (id) => parseFloat(($(id) || {}).value) || 0;
    const hw = (id) => HW()[parseInt(($(id) || {}).value, 10)] || HW()[0] || {};

    /* --- memory --- */
    const memCalc = () => {
      const L = num('mem-layers'), kvh = num('mem-kvheads'), hd = num('mem-headdim');
      const P = num('mem-params') * 1e9, wb = parseFloat($('mem-wbytes').value);
      const kvb = parseFloat($('mem-kvbytes').value), ctx = num('mem-ctx');
      const g = hw('mem-gpu'), n = Math.max(1, num('mem-gpus')), over = num('mem-over') / 100;
      const weights = P * wb;
      const kvPerTok = 2 * L * kvh * hd * kvb;
      const totalMem = (g.mem || 80) * 1024 ** 3 * n;
      const free = totalMem * (1 - over) - weights;
      const tokens = free > 0 ? Math.floor(free / kvPerTok) : 0;
      const reqs = Math.floor(tokens / Math.max(1, ctx));
      $('mem-out').innerHTML = `
        <div class="out-line"><span>Weights</span><span>${fmt(gb(weights))} GiB</span></div>
        <div class="out-line"><span>Total GPU memory</span><span>${fmt(gb(totalMem))} GiB</span></div>
        <div class="out-line"><span>KV bytes / token</span><span>${fmt(kvPerTok / 1024)} KiB</span></div>
        <div class="out-line"><span>KV budget left</span><span style="color:${free <= 0 ? 'var(--cp-danger)' : 'inherit'}">${fmt(gb(Math.max(0, free)))} GiB</span></div>
        <div class="out-line hero"><span>Cacheable tokens</span><span>${free <= 0 ? 'OOM' : fmt(tokens, 1)}</span></div>
        <div class="out-line"><span>Concurrent reqs @ ${fmt(ctx, 0)} ctx</span><span>${free <= 0 ? '0' : reqs}</span></div>
        <div class="out-line"><span>KV per request</span><span>${fmt(kvPerTok * ctx / 1024 ** 3)} GiB</span></div>`;
    };
    ['mem-layers', 'mem-dmodel', 'mem-kvheads', 'mem-headdim', 'mem-params', 'mem-wbytes',
      'mem-kvbytes', 'mem-ctx', 'mem-gpu', 'mem-gpus', 'mem-over'].forEach((id) =>
        $(id) && $(id).addEventListener('input', memCalc));
    $('mem-preset') && $('mem-preset').addEventListener('change', (e) => {
      const m = MD()[parseInt(e.target.value, 10)];
      if (!m) return;
      $('mem-layers').value = m.layers; $('mem-dmodel').value = m.dmodel;
      $('mem-kvheads').value = m.kvHeads; $('mem-headdim').value = m.headDim;
      $('mem-params').value = m.params;
      memCalc();
    });

    /* --- roofline --- */
    const rfCalc = () => {
      const g = hw('rf-gpu'), n = Math.max(1, num('rf-gpus'));
      const P = num('rf-params') * 1e9, wb = parseFloat($('rf-wbytes').value);
      const B = Math.max(1, num('rf-batch')), kvpt = num('rf-kvpt'), ctx = num('rf-ctx');
      const eff = num('rf-eff') / 100;
      const bw = (g.bw || 3350) * 1e9 * n * eff;
      const peak = (g.bf16 || 989) * 1e12 * n * (wb <= 1 ? (g.fp8 ? g.fp8 / g.bf16 : 2) : 1);
      const bytes = P * wb + B * ctx * kvpt;
      const flops = 2 * P * B;
      const ai = flops / bytes;
      const opsByte = ((g.bf16 || 989) * 1e12) / ((g.bw || 3350) * 1e9);
      const tStep = Math.max(bytes / bw, flops / peak);
      const bound = (bytes / bw) > (flops / peak) ? 'memory' : 'compute';
      const critB = Math.round(opsByte / 2 * (wb / 2));
      $('rf-out').innerHTML = `
        <div class="out-line"><span>Machine ops:byte</span><span>${fmt(opsByte, 0)} FLOP/byte</span></div>
        <div class="out-line"><span>Arithmetic intensity</span><span>${fmt(ai, 2)} FLOP/byte</span></div>
        <div class="out-line"><span>Bound by</span><span style="color:${bound === 'memory' ? 'var(--cp-warning)' : 'var(--cp-success)'}">${bound}</span></div>
        <div class="out-line"><span>Bytes moved / step</span><span>${fmt(gb(bytes))} GiB</span></div>
        <div class="out-line"><span>Time / decode step</span><span>${fmt(tStep * 1000, 2)} ms</span></div>
        <div class="out-line"><span>Per-stream rate</span><span>${fmt(1 / tStep, 1)} tok/s</span></div>
        <div class="out-line hero"><span>Aggregate throughput</span><span>${fmt(B / tStep, 0)} tok/s</span></div>
        <div class="out-line"><span>Batch to reach compute bound</span><span>&asymp; ${fmt(critB, 0)}</span></div>`;
    };
    ['rf-gpu', 'rf-gpus', 'rf-params', 'rf-wbytes', 'rf-batch', 'rf-kvpt', 'rf-ctx', 'rf-eff']
      .forEach((id) => $(id) && $(id).addEventListener('input', rfCalc));

    /* --- prefill --- */
    const pfCalc = () => {
      const g = hw('pf-gpu'), n = Math.max(1, num('pf-gpus'));
      const P = num('pf-params') * 1e9, T = num('pf-tokens'), mfu = num('pf-mfu') / 100;
      const L = num('pf-layers'), d = num('pf-dmodel');
      const prec = $('pf-prec').value;
      const peakMap = { bf16: g.bf16, fp8: g.fp8 || g.bf16 * 2, fp4: g.fp4 || (g.fp8 || g.bf16 * 2) * 2 };
      const peak = (peakMap[prec] || g.bf16) * 1e12 * n;
      const dense = 2 * P * T;
      const attn = 4 * L * d * T * T;
      const total = dense + attn;
      const t = total / (peak * mfu);
      $('pf-out').innerHTML = `
        <div class="out-line"><span>Dense GEMM FLOPs</span><span>${fmt(dense)}</span></div>
        <div class="out-line"><span>Attention score FLOPs</span><span>${fmt(attn)}</span></div>
        <div class="out-line"><span>Attention share</span><span>${fmt(100 * attn / total, 1)}%</span></div>
        <div class="out-line"><span>Effective compute</span><span>${fmt(peak * mfu / 1e12, 1)} TFLOP/s</span></div>
        <div class="out-line hero"><span>Prefill time (TTFT floor)</span><span>${fmt(t * 1000, 1)} ms</span></div>
        <div class="out-line"><span>Prefill throughput</span><span>${fmt(T / t, 0)} tok/s</span></div>`;
    };
    ['pf-gpu', 'pf-gpus', 'pf-params', 'pf-tokens', 'pf-mfu', 'pf-layers', 'pf-dmodel', 'pf-prec']
      .forEach((id) => $(id) && $(id).addEventListener('input', pfCalc));

    /* --- speculative --- */
    const sdCalc = () => {
      const a = Math.min(0.999, Math.max(0.001, num('sd-alpha')));
      const gm = Math.max(1, Math.round(num('sd-gamma')));
      const c = num('sd-c'), vo = num('sd-vo') / 100;
      const E = (g_) => (1 - Math.pow(a, g_ + 1)) / (1 - a);
      const speedup = (g_) => E(g_) / (1 + c * g_ + vo);
      let best = 1, bestS = 0;
      for (let i = 1; i <= 16; i++) { const s = speedup(i); if (s > bestS) { bestS = s; best = i; } }
      const s = speedup(gm);
      $('sd-out').innerHTML = `
        <div class="out-line"><span>E[accepted tokens / iter]</span><span>${fmt(E(gm), 2)}</span></div>
        <div class="out-line"><span>Cost per iteration</span><span>${fmt(1 + c * gm + vo, 2)}&times; target step</span></div>
        <div class="out-line hero"><span>Speedup at &gamma;=${gm}</span><span>${fmt(s, 2)}&times;</span></div>
        <div class="out-line"><span>Verdict</span><span style="color:${s > 1 ? 'var(--cp-success)' : 'var(--cp-danger)'}">${s > 1 ? 'net win' : 'net loss'}</span></div>
        <div class="out-line"><span>Optimal &gamma;</span><span>${best} (${fmt(bestS, 2)}&times;)</span></div>
        <div class="out-line"><span>Break-even &alpha; at &gamma;=${gm}</span><span>${fmt(breakEvenAlpha(gm, c, vo), 3)}</span></div>`;
    };
    const breakEvenAlpha = (g_, c, vo) => {
      let lo = 0.001, hi = 0.999;
      for (let i = 0; i < 60; i++) {
        const mid = (lo + hi) / 2;
        const E = (1 - Math.pow(mid, g_ + 1)) / (1 - mid);
        (E / (1 + c * g_ + vo) < 1) ? lo = mid : hi = mid;
      }
      return hi;
    };
    ['sd-alpha', 'sd-gamma', 'sd-c', 'sd-vo'].forEach((id) => $(id) && $(id).addEventListener('input', sdCalc));

    /* --- cost --- */
    const coCalc = () => {
      const rate = num('co-rate') * Math.max(1, num('co-gpus'));
      const tps = Math.max(1, num('co-tps')), util = num('co-util') / 100;
      const ptps = Math.max(1, num('co-ptps')), ratio = num('co-ratio');
      const api = num('co-api');
      const outCost = rate / (tps * 3600 * util) * 1e6;
      const inCost = rate / (ptps * 3600 * util) * 1e6;
      const blended = (inCost * ratio + outCost) / (ratio + 1);
      $('co-out').innerHTML = `
        <div class="out-line"><span>Output cost</span><span>$${fmt(outCost, 3)} / 1M</span></div>
        <div class="out-line"><span>Input (prefill) cost</span><span>$${fmt(inCost, 3)} / 1M</span></div>
        <div class="out-line hero"><span>Blended @ ${fmt(ratio, 1)}:1</span><span>$${fmt(blended, 3)} / 1M</span></div>
        <div class="out-line"><span>Cost per hour</span><span>$${fmt(rate, 2)}</span></div>
        <div class="out-line"><span>Break-even utilization vs API</span><span>${api > 0 ? fmt(100 * (rate / (tps * 3600 * api / 1e6)) / 1, 1) + '%' : '—'}</span></div>
        <div class="out-line"><span>vs API price</span><span style="color:${outCost < api ? 'var(--cp-success)' : 'var(--cp-danger)'}">${api > 0 ? (outCost <= api ? fmt(api / outCost, 2) + '&times; cheaper self-hosted' : fmt(outCost / api, 2) + '&times; more expensive self-hosted') : '&mdash;'}</span></div>`;
    };
    ['co-rate', 'co-gpus', 'co-tps', 'co-util', 'co-ptps', 'co-ratio', 'co-api']
      .forEach((id) => $(id) && $(id).addEventListener('input', coCalc));

    /* --- capacity --- */
    const caCalc = () => {
      const qps = num('ca-qps'), outTok = num('ca-otok'), rate = Math.max(1, num('ca-itl'));
      const ttftT = num('ca-ttft'), ttftM = num('ca-ttftm');
      const slots = num('ca-slots'), ctx = Math.max(1, num('ca-ctx'));
      const residence = ttftM / 1000 + outTok / rate;
      const conc = qps * residence;
      const kvCap = Math.floor(slots / ctx);
      const maxQps = kvCap / residence;
      $('ca-out').innerHTML = `
        <div class="out-line"><span>Residence time / request</span><span>${fmt(residence, 2)} s</span></div>
        <div class="out-line hero"><span>Required concurrency</span><span>${fmt(conc, 1)}</span></div>
        <div class="out-line"><span>KV-limited concurrency</span><span>${kvCap}</span></div>
        <div class="out-line"><span>Headroom</span><span style="color:${conc <= kvCap ? 'var(--cp-success)' : 'var(--cp-danger)'}">${conc <= kvCap ? fmt(kvCap / conc, 2) + '&times;' : 'over by ' + fmt(conc - kvCap, 1)}</span></div>
        <div class="out-line"><span>Max sustainable QPS</span><span>${fmt(maxQps, 1)} req/s</span></div>
        <div class="out-line"><span>Aggregate token rate</span><span>${fmt(conc * rate, 0)} tok/s</span></div>
        <div class="out-line"><span>TTFT SLO</span><span style="color:${ttftM <= ttftT ? 'var(--cp-success)' : 'var(--cp-danger)'}">${ttftM <= ttftT ? 'met' : 'violated'}</span></div>`;
    };
    ['ca-qps', 'ca-otok', 'ca-itl', 'ca-ttft', 'ca-ttftm', 'ca-slots', 'ca-ctx']
      .forEach((id) => $(id) && $(id).addEventListener('input', caCalc));

    /* --- parallelism --- */
    const paCalc = () => {
      const tp = Math.max(1, num('pa-tp')), L = num('pa-layers'), d = num('pa-dmodel');
      const T = num('pa-tokens'), bw = num('pa-bw') * 1e9, ab = parseFloat($('pa-abytes').value);
      const step = num('pa-step') / 1000;
      const msg = T * d * ab;
      const perGpu = 2 * (tp - 1) / tp * msg;
      const perLayer = 2 * perGpu;
      const total = L * perLayer;
      const t = tp > 1 ? total / bw : 0;
      $('pa-out').innerHTML = `
        <div class="out-line"><span>Activation message</span><span>${fmt(msg / 1024, 1)} KiB</span></div>
        <div class="out-line"><span>All-reduce bytes / GPU / layer</span><span>${fmt(perLayer / 1024, 1)} KiB</span></div>
        <div class="out-line"><span>Total comm / step / GPU</span><span>${fmt(total / 1024 ** 2, 2)} MiB</span></div>
        <div class="out-line hero"><span>Comm time / step</span><span>${fmt(t * 1000, 2)} ms</span></div>
        <div class="out-line"><span>Share of step budget</span><span style="color:${t / step > 0.25 ? 'var(--cp-danger)' : 'var(--cp-success)'}">${fmt(100 * t / step, 1)}%</span></div>
        <div class="out-line"><span>Sync points / step</span><span>${2 * L}</span></div>
        <div class="out-line"><span>Weights per GPU factor</span><span>1/${tp}</span></div>`;
    };
    ['pa-tp', 'pa-layers', 'pa-dmodel', 'pa-tokens', 'pa-bw', 'pa-abytes', 'pa-step']
      .forEach((id) => $(id) && $(id).addEventListener('input', paCalc));

    memCalc(); rfCalc(); pfCalc(); sdCalc(); coCalc(); caCalc(); paCalc();
  }

  /* ------------------------------------------------------------ navigation */
  function setView(view, opts) {
    opts = opts || {};
    state.view = view;
    if (opts.moduleId) state.moduleId = opts.moduleId;
    if (opts.unitId) state.unitId = opts.unitId;
    closeSidebar();
    render();
    const hash = view === 'unit' ? `#/unit/${state.unitId}`
      : view === 'module' ? `#/module/${state.moduleId}` : `#/${view}`;
    if (location.hash !== hash) history.pushState(null, '', hash);
  }

  function applyHash() {
    const h = (location.hash || '').replace(/^#\/?/, '');
    const [a, b] = h.split('/');
    if (a === 'unit' && UNIT_BY_ID.has(b)) { state.view = 'unit'; state.unitId = b; state.moduleId = UNIT_BY_ID.get(b).module.id; }
    else if (a === 'module' && MODULES.some((m) => m.id === b)) { state.view = 'module'; state.moduleId = b; }
    else if (['track', 'library', 'reference', 'calculators', 'drills', 'progress'].includes(a)) state.view = a;
    else state.view = 'track';
    render();
  }

  function updateChrome() {
    const total = UNITS.length, done = state.done.size;
    $('progress-label').textContent = `${done} / ${total}`;
    $('progress-fill').style.width = pct(done, total) + '%';
    $('nav-units').textContent = total;
    $('nav-res').textContent = RESOURCES.length;
    $('nav-drills').textContent = DRILLS.length;
    $('nav-done').textContent = pct(done, total) + '%';
    const active = ['module', 'unit'].includes(state.view) ? 'track' : state.view;
    document.querySelectorAll('.nav-button').forEach((b) => b.classList.toggle('active', b.dataset.view === active));
    document.querySelectorAll('.mobile-nav button').forEach((b) => b.classList.toggle('active', b.dataset.mobileView === active));
    document.querySelectorAll('.module-button').forEach((b) =>
      b.classList.toggle('active', state.view !== 'track' && b.dataset.module === state.moduleId));
  }

  function renderModuleNav() {
    $('module-list').innerHTML = MODULES.map((m) =>
      `<button class="module-button" data-module="${esc(m.id)}"><span>${esc(m.num)} &nbsp;${esc(m.title)}</span><span class="count">${moduleDone(m)}/${m.units.length}</span></button>`).join('');
  }

  /* ---------------------------------------------------------------- events */
  function bindViewEvents() {
    const v = $('view');
    v.querySelectorAll('[data-module]').forEach((el) => el.addEventListener('click', () => setView('module', { moduleId: el.dataset.module })));
    v.querySelectorAll('[data-unit]').forEach((el) => el.addEventListener('click', () => setView('unit', { unitId: el.dataset.unit })));
    v.querySelectorAll('[data-goto]').forEach((el) => el.addEventListener('click', () => setView(el.dataset.goto)));
    v.querySelectorAll('[data-toggle-done]').forEach((el) => el.addEventListener('click', () => {
      const id = el.dataset.toggleDone;
      if (state.done.has(id)) { state.done.delete(id); toast('Marked incomplete'); }
      else { state.done.add(id); toast('Unit complete'); }
      persist(); renderModuleNav(); render();
    }));
    v.querySelectorAll('[data-lib-filter]').forEach((el) => el.addEventListener('click', () => { state.libFilter = el.dataset.libFilter; render(); }));
    v.querySelectorAll('[data-ref-tab]').forEach((el) => el.addEventListener('click', () => { state.refTab = el.dataset.refTab; render(); }));
    v.querySelectorAll('[data-drill-filter]').forEach((el) => el.addEventListener('click', () => { state.drillFilter = el.dataset.drillFilter; render(); }));
    v.querySelectorAll('[data-drill]').forEach((el) => el.addEventListener('click', () => {
      const id = el.dataset.drill;
      state.openDrills.has(id) ? state.openDrills.delete(id) : state.openDrills.add(id);
      render();
    }));
    const all = v.querySelector('[data-drill-all]');
    if (all) all.addEventListener('click', () => {
      const list = state.drillFilter === 'all' ? DRILLS : DRILLS.filter((d) => d.kind === state.drillFilter);
      const anyClosed = list.some((d) => !state.openDrills.has(d.id));
      list.forEach((d) => anyClosed ? state.openDrills.add(d.id) : state.openDrills.delete(d.id));
      render();
    });
    const libSearch = v.querySelector('#lib-search');
    if (libSearch) {
      libSearch.addEventListener('input', debounce((e) => {
        state.libQuery = e.target.value;
        const pos = e.target.selectionStart;
        render();
        const el = $('lib-search');
        if (el) { el.focus(); el.setSelectionRange(pos, pos); }
      }, 220));
    }
    const ex = v.querySelector('[data-export]');
    if (ex) ex.addEventListener('click', exportProgress);
    const im = v.querySelector('[data-import]');
    if (im) im.addEventListener('click', importProgress);
    const rs = v.querySelector('[data-reset]');
    if (rs) rs.addEventListener('click', () => {
      if (confirm('Clear all progress on this device?')) {
        state.done.clear(); state.starred.clear(); persist(); renderModuleNav(); render(); toast('Progress cleared');
      }
    });
  }

  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

  function exportProgress() {
    const blob = new Blob([JSON.stringify({ v: 1, done: [...state.done] }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'inference-atlas-progress.json';
    a.click();
    URL.revokeObjectURL(a.href);
    toast('Exported');
  }

  function importProgress() {
    const input = document.createElement('input');
    input.type = 'file'; input.accept = 'application/json';
    input.onchange = () => {
      const f = input.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const data = JSON.parse(reader.result);
          (data.done || []).forEach((id) => UNIT_BY_ID.has(id) && state.done.add(id));
          persist(); renderModuleNav(); render(); toast('Imported');
        } catch (e) { toast('Could not read that file'); }
      };
      reader.readAsText(f);
    };
    input.click();
  }

  /* ---------------------------------------------------------------- search */
  function buildSearchIndex() {
    const idx = [];
    UNITS.forEach((u) => idx.push({
      kind: 'unit', id: u.id, title: u.title, sub: `${u.module.title} · ${TIERS[u.tier].label}`,
      text: (u.title + ' ' + u.summary + ' ' + u.why + ' ' + (u.core || []).join(' ') + ' ' + (u.checks || []).join(' ')).toLowerCase()
    }));
    RESOURCES.forEach((r) => idx.push({
      kind: r.kind, id: r.id, title: r.title, sub: [r.src, r.year].filter(Boolean).join(' · '), url: r.url,
      text: (r.title + ' ' + (r.src || '') + ' ' + (r.note || '') + ' ' + (r.topics || []).join(' ')).toLowerCase()
    }));
    DRILLS.forEach((d) => idx.push({
      kind: 'drill', id: d.id, title: d.q.length > 90 ? d.q.slice(0, 90) + '…' : d.q, sub: `Drill · ${d.kind}`,
      text: (d.q + ' ' + d.a).toLowerCase()
    }));
    MODULES.forEach((m) => idx.push({
      kind: 'module', id: m.id, title: m.title, sub: `Module ${m.num}`,
      text: (m.title + ' ' + m.tagline + ' ' + m.focus).toLowerCase()
    }));
    state.searchIndex = idx;
  }

  function openSearch() {
    $('search-backdrop').classList.add('open');
    const i = $('search-input');
    i.value = ''; i.focus();
    renderSearch('');
  }
  const closeSearch = () => $('search-backdrop').classList.remove('open');

  function renderSearch(q) {
    q = q.trim().toLowerCase();
    const terms = q.split(/\s+/).filter(Boolean);
    let hits = state.searchIndex;
    if (terms.length) hits = hits.filter((h) => terms.every((t) => h.text.includes(t)));
    hits = hits.slice(0, 40);
    state.searchSel = 0;
    $('search-results').innerHTML = hits.length ? hits.map((h, i) =>
      `<button class="search-result ${i === 0 ? 'selected' : ''}" data-si="${i}">
        <span><strong>${esc(h.title)}</strong><small>${esc(h.sub)}</small></span>
        <span class="result-kind">${esc(h.kind === 'unit' ? 'unit' : KIND_LABEL[h.kind] || h.kind)}</span>
      </button>`).join('')
      : `<p class="search-help" style="border:0">${q ? 'No matches.' : 'Type to search units, papers, drills, and modules.'}</p>`;
    state.searchHits = hits;
    $('search-results').querySelectorAll('[data-si]').forEach((el) =>
      el.addEventListener('click', () => openResult(parseInt(el.dataset.si, 10))));
  }

  function openResult(i) {
    const h = (state.searchHits || [])[i];
    if (!h) return;
    closeSearch();
    if (h.kind === 'unit') setView('unit', { unitId: h.id });
    else if (h.kind === 'module') setView('module', { moduleId: h.id });
    else if (h.kind === 'drill') { state.openDrills.add(h.id); state.drillFilter = 'all'; setView('drills'); }
    else if (h.url) window.open(h.url, '_blank', 'noopener');
  }

  function moveSel(delta) {
    const els = $('search-results').querySelectorAll('.search-result');
    if (!els.length) return;
    els[state.searchSel] && els[state.searchSel].classList.remove('selected');
    state.searchSel = (state.searchSel + delta + els.length) % els.length;
    els[state.searchSel].classList.add('selected');
    els[state.searchSel].scrollIntoView({ block: 'nearest' });
  }

  /* ----------------------------------------------------------------- misc */
  let toastTimer;
  function toast(msg) {
    const t = $('toast');
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 1800);
  }
  const closeSidebar = () => { $('sidebar').classList.remove('open'); $('scrim').classList.remove('open'); };

  /* ------------------------------------------------------------------ init */
  function init() {
    renderModuleNav();
    buildSearchIndex();

    document.querySelectorAll('.nav-button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
    document.querySelectorAll('.mobile-nav button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.mobileView)));
    $('module-list').addEventListener('click', (e) => {
      const b = e.target.closest('[data-module]');
      if (b) setView('module', { moduleId: b.dataset.module });
    });
    $('menu-button').addEventListener('click', () => {
      $('sidebar').classList.toggle('open');
      $('scrim').classList.toggle('open', $('sidebar').classList.contains('open'));
    });
    $('scrim').addEventListener('click', closeSidebar);
    $('theme-button').addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      store.write('llmi-theme', next);
    });
    $('search-button').addEventListener('click', openSearch);
    $('search-backdrop').addEventListener('click', (e) => { if (e.target === $('search-backdrop')) closeSearch(); });
    $('search-input').addEventListener('input', (e) => renderSearch(e.target.value));
    $('search-input').addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') { e.preventDefault(); moveSel(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); moveSel(-1); }
      else if (e.key === 'Enter') { e.preventDefault(); openResult(state.searchSel); }
    });
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearch(); }
      else if (e.key === 'Escape') { closeSearch(); closeSidebar(); }
      else if (e.key === '/' && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) { e.preventDefault(); openSearch(); }
    });
    window.addEventListener('popstate', applyHash);
    window.addEventListener('hashchange', applyHash);
    applyHash();
  }

  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', init) : init();
})();
