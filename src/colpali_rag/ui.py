"""Showcase UI for the ColPali RAG web app.

Hero header with live stats, quick-start question chips, drag-&-drop PDF
upload, chat with citations + page thumbnails, and a collapsible retrieval
inspector that visualizes the fusion of the three retrieval legs per query
(data comes from GET /v1/query -> response.trace).
"""

UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ColPali RAG</title>
<style>
  :root {
    --bg:#0a0e14; --panel:#121a26; --line:#1f2b3d; --txt:#dbe6f2; --mut:#8399b2;
    --acc:#3ddc84; --c:#46d38d; --d:#57a7ff; --s:#ffb45c; --f:#b48cff;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,"Segoe UI",sans-serif; background:var(--bg); color:var(--txt); height:100vh; display:flex; flex-direction:column; }
  header { background:linear-gradient(90deg,#0e1a2a 0%,#0c2418 60%,#0a0e14 100%); border-bottom:1px solid var(--line); padding:14px 22px 10px; }
  header .row { display:flex; align-items:baseline; gap:14px; }
  header h1 { margin:0; font-size:19px; letter-spacing:.3px;
    background:linear-gradient(90deg,#fff,#7fe0b0); -webkit-background-clip:text; background-clip:text; color:transparent; }
  header .tag { color:var(--mut); font-size:12px; }
  #stats { margin-left:auto; color:var(--mut); font-size:12px; white-space:nowrap; }
  #stats b { color:var(--txt); }
  #chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .chip-q { background:var(--panel); border:1px solid var(--line); color:var(--mut); font-size:12px;
    padding:5px 11px; border-radius:999px; cursor:pointer; transition:border-color .15s,color .15s; max-width:340px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .chip-q:hover { border-color:var(--acc); color:var(--txt); }
  #upload { padding:10px 22px; border-bottom:1px solid var(--line); display:flex; flex-direction:column; gap:8px; }
  #drop { border:1px dashed var(--line); border-radius:10px; padding:12px; text-align:center; color:var(--mut); font-size:13px; cursor:pointer; transition:border-color .15s; }
  #drop.over { border-color:var(--acc); color:var(--txt); }
  #uploadbar { display:flex; gap:8px; align-items:center; }
  #file { display:none; }
  #pick { background:var(--panel); border:1px solid var(--line); color:var(--txt); padding:7px 14px; border-radius:8px; cursor:pointer; font-size:13px; }
  #upbtn { background:var(--acc); border:0; color:#04110a; font-weight:700; padding:7px 16px; border-radius:8px; cursor:pointer; }
  #upbtn:disabled { opacity:.5; cursor:wait; }
  #upstatus { color:var(--mut); font-size:12px; flex:1; }
  #sources { display:flex; flex-wrap:wrap; gap:6px; font-size:12px; }
  .chip { background:var(--panel); border:1px solid var(--line); color:var(--mut); padding:3px 9px; border-radius:999px; }
  #chat { flex:1; overflow-y:auto; padding:16px 22px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:860px; display:flex; flex-direction:column; }
  .msg.user { align-self:flex-end; align-items:flex-end; }
  .msg.bot { align-self:flex-start; align-items:flex-start; }
  .bubble { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; white-space:pre-wrap; line-height:1.55; font-size:14px; }
  .msg.user .bubble { background:#14261c; border-color:#244736; }
  .meta { margin-top:8px; font-size:11px; color:var(--mut); }
  .mut { color:var(--mut); font-size:12px; margin-top:8px; }
  .empty { color:var(--mut); font-size:13px; }
  .thumbs { display:flex; gap:10px; overflow-x:auto; margin-top:10px; padding-bottom:4px; }
  .thumbs .thumb { flex:0 0 auto; }
  .thumbs img { height:105px; border-radius:6px; border:1px solid var(--line); background:#fff; cursor:pointer; display:block; }
  .thumbs .cap { font-size:10px; color:var(--mut); margin-top:3px; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .badges { margin-left:4px; font-size:9px; letter-spacing:.5px; }
  .bdg { display:inline-block; width:14px; height:14px; line-height:14px; text-align:center; border-radius:3px; font-weight:700; color:#06130c; margin-left:2px; }
  .bdg.c { background:var(--c); } .bdg.d { background:var(--d); } .bdg.s { background:var(--s); } .bdg.f { background:var(--f); }
  /* retrieval inspector */
  .insp { margin-top:10px; border:1px solid var(--line); border-radius:10px; overflow:hidden; width:100%; }
  .inspbar { width:100%; background:var(--panel); border:0; color:var(--mut); font-size:11.5px; padding:8px 12px; cursor:pointer; text-align:left; display:flex; align-items:center; gap:8px; }
  .inspbar:hover { color:var(--txt); }
  .insp-dot { width:8px; height:8px; border-radius:50%; background:var(--f); box-shadow:0 0 6px var(--f); }
  .inspbar b { color:var(--txt); }
  .inspbody { background:#0d1420; border-top:1px solid var(--line); padding:10px 12px; }
  .tchips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
  .tchip { background:var(--panel); border:1px solid var(--line); border-radius:6px; color:var(--mut); font-size:10.5px; padding:3px 8px; }
  .tchip b { color:var(--txt); }
  .legs { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:10px; }
  .leg { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px; }
  .leg h5 { margin:0 0 6px; font-size:10.5px; letter-spacing:.4px; text-transform:uppercase; }
  .leg h5 .bdg { vertical-align:middle; }
  .legrow { font-size:10.5px; color:var(--mut); display:flex; align-items:center; gap:6px; margin:3px 0; }
  .legrow .r { color:var(--txt); font-weight:700; width:12px; }
  .legrow .nm { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:52%; }
  .bar { flex:1; height:5px; border-radius:3px; background:#1a2536; overflow:hidden; }
  .bar i { display:block; height:100%; background:var(--c); }
  .leg.d .bar i { background:var(--d); } .leg.s .bar i { background:var(--s); }
  .fused { border-top:1px solid var(--line); padding-top:8px; }
  .fused h6 { margin:0 0 6px; font-size:10.5px; letter-spacing:.4px; text-transform:uppercase; color:var(--f); }
  .fusedrow { font-size:11.5px; color:var(--txt); display:flex; align-items:center; gap:8px; padding:3px 0; border-bottom:1px dashed #16202f; }
  .fusedrow:last-child { border-bottom:0; }
  .fusedrow .r { color:var(--mut); width:14px; }
  .fusedrow .nm { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .fusedrow .sc { color:var(--mut); font-size:10.5px; }
  #composer { border-top:1px solid var(--line); padding:12px 22px; display:flex; gap:8px; align-items:flex-end; }
  #q { flex:1; background:var(--panel); border:1px solid var(--line); color:var(--txt); padding:10px 13px; border-radius:8px; font-size:14px; resize:none; font-family:inherit; }
  #q:focus { outline:none; border-color:var(--acc); }
  #go { background:var(--acc); border:0; color:#04110a; font-weight:700; padding:10px 18px; border-radius:8px; cursor:pointer; }
  #go:disabled { opacity:.5; cursor:wait; }
  @media (max-width:720px){ .legs { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div class="row">
    <h1>ColPali RAG</h1><span class="tag">visual + hybrid document retrieval</span>
    <span id="stats">booting…</span>
  </div>
  <div id="chips"></div>
</header>
<section id="upload">
  <div id="drop">Drop PDFs here or click to choose — they are indexed and you can ask about them right away</div>
  <div id="uploadbar">
    <input type="file" id="file" accept=".pdf" multiple>
    <button id="pick" onclick="document.getElementById('file').click()">choose files</button>
    <button id="upbtn" onclick="uploadFiles()">upload</button>
    <span id="upstatus"></span>
  </div>
  <div id="sources"></div>
</section>
<main id="chat"></main>
<section id="composer">
  <textarea id="q" rows="2" placeholder="Ask about your documents… e.g. what does the revenue chart show?"></textarea>
  <button id="go" onclick="ask()">Ask</button>
</section>
<script>
const $ = id => document.getElementById(id);
const esc = s => (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const sleep = ms => new Promise(r => setTimeout(r, ms));

const QUICK = [
  'What does the revenue chart show for each product line?',
  'How does solid-state battery energy density compare with LFP?',
  'What does the attention gate figure show about cross-attention during decoding?',
  'What code activates the Global Widgets API?'
];

function addMsg(role, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = html;
  $('chat').appendChild(d);
  $('chat').scrollTop = $('chat').scrollHeight;
  return d;
}

function legRow(t, i, max) {
  const nm = (t.src || '').slice(0, 16);
  const w = max > 0 ? Math.max(4, Math.round(Math.abs(t.score) / max * 100)) : 0;
  return '<div class="legrow"><span class="r">' + (i + 1) + '</span>' +
    '<span class="nm" title="' + esc(t.src) + '">' + esc(nm) + '·p' + t.page + '</span>' +
    '<div class="bar"><i style="width:' + w + '%"></i></div>' + t.score.toFixed(2) + '</div>';
}

function inspectorHTML(t) {
  if (!t) return '';
  const peaks = { colpali: 0, dense: 0, sparse: 0 };
  ['colpali','dense','sparse'].forEach(k => {
    (t.legs[k] || []).forEach(x => { peaks[k] = Math.max(peaks[k], Math.abs(x.score)); });
  });
  const times = [
    ['encode', t.encode_ms], ['colpali', t.colpali_ms], ['dense', t.dense_ms],
    ['sparse', t.sparse_ms], ['fusion', t.fusion_ms]
  ];
  const tchips = times.map(x => '<span class="tchip">' + x[0] + ' <b>' + (x[1] || 0).toFixed(1) + '</b>ms</span>').join('') +
    '<span class="tchip">total <b>' + (t.total_ms || 0).toFixed(1) + '</b>ms</span>';
  const col = (k, label, cls) => {
    const rows = (t.legs[k] || []).slice(0, 5);
    return '<div class="leg ' + cls + '"><h5><span class="bdg ' + cls + '">' + label[0] + '</span> ' + label + '</h5>' +
      rows.map((x, i) => legRow(x, i, peaks[k])).join('') + '</div>';
  };
  const fusedRows = (t.fused || []).slice(0, 6).map((x, i) =>
    '<div class="fusedrow"><span class="r">' + (i + 1) + '</span>' +
    '<span class="nm" title="' + esc(x.src) + '">' + esc((x.src||'').slice(0, 30)) + ' · p' + x.page + '</span>' +
    '<span class="badges">' + (x.legs || []).map(l => '<span class="bdg ' + l[0] + '">' + l[0].toUpperCase()[0] + '</span>').join('') + '</span>' +
    '<span class="sc">' + x.score.toFixed(4) + '</span></div>').join('');
  return '<div class="insp">' +
    '<button class="inspbar" onclick="const b=this.nextElementSibling;b.hidden=!b.hidden">' +
    '<span class="insp-dot"></span> retrieval internals · <b>' + (t.fused||[]).length + '</b> fused pages · <b>' + (t.total_ms||0).toFixed(1) + ' ms</b> · click to collapse/expand</button>' +
    '<div class="inspbody"><div class="tchips">' + tchips + '</div>' +
    '<div class="legs">' + col('colpali','ColPali','c') + col('dense','Dense','d') + col('sparse','Sparse','s') + '</div>' +
    '<div class="fused"><h6>RRF fusion ranking</h6>' + fusedRows + '</div></div></div>';
}

async function refreshInfo() {
  try {
    const s = await (await fetch('/v1/sources')).json();
    const pts = await (await fetch('/v1/collection')).json();
    const docs = s.length, points = pts.points || 0;
    $('stats').innerHTML = '<b>' + docs + '</b> doc' + (docs === 1 ? '' : 's') + ' · <b>' + points + '</b> pages';
    $('sources').innerHTML = s.length ? s.map(x =>
      '<span class="chip">' + esc(x.src) + ' · ' + x.pages + 'p</span>').join('')
      : '<span class="chip">no documents indexed yet</span>';
    if (!docs && !$('chat').children.length) {
      addMsg('bot', '<div class="bubble"><span class="empty">Welcome. Drop a PDF above (or click upload) — then ask me anything about it, or try one of the quick-start questions.</span></div>');
    }
  } catch (e) { $('stats').textContent = 'server starting…'; }
}

function askChip(i) {
  $('q').value = QUICK[i];
  ask();
}

function renderChips() {
  $('chips').innerHTML = QUICK.map((q, i) => '<span class="chip-q" title="' + esc(q) + '" onclick="askChip(' + i + ')">' + esc(q) + '</span>').join('');
}

async function ask() {
  const q = $('q').value.trim();
  if (!q) return;
  const go = $('go');
  addMsg('user', '<div class="bubble">' + esc(q) + '</div>');
  $('q').value = '';
  go.disabled = true;
  const waitEl = addMsg('bot', '<div class="bubble"><span class="empty">retrieving…</span></div>');
  try {
    const r = await fetch('/v1/query', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ query: q, top_k: 6, generate: true }) });
    if (!r.ok) {
      const errBody = await r.json().catch(() => ({}));
      throw new Error('HTTP ' + r.status + (errBody.detail ? ' — ' + errBody.detail : '') + (r.status === 401 ? ' (server requires an API key the web UI cannot supply)' : ''));
    }
    const d = await r.json();
    const srcLabel = d.generation_backend ? esc(d.generation_backend + (d.generation_model ? ' · ' + d.generation_model : '')) : '';
    const sources = (d.citations && d.citations.length) ? '<div class="mut">sources: ' + esc(d.citations.join(' · ')) + '</div>' : '';
    const note = d.generation_note ? '<div class="mut">' + esc(d.generation_note) + '</div>' : '';
    const meta = d.answer ? '<div class="meta">answered by ' + (srcLabel || 'retrieval only') + (d.latency_ms ? ' · ' + d.latency_ms + ' ms' : '') + '</div>' : '';
    const thumbs = (d.pages && d.pages.length) ? '<div class="thumbs">' + d.pages.map(p =>
        '<div class="thumb">' +
        (p.image ? '<img src="/assets/' + esc(p.image) + '" loading="lazy" title="' + esc((p.text || '').slice(0, 180)) + '" onclick="window.open(this.src)">' : '') +
        '<div class="cap">' + esc(p.src) + ' · p' + p.page + ' · ' + p.score.toFixed(3) + '<span class="badges">' + (p.legs || []).map(l => '<span class="bdg ' + l[0] + '">' + l[0].toUpperCase()[0] + '</span>').join('') + '</span></div></div>').join('')
      : '';
    const ans = (d.answer && d.answer !== '""') ? esc(d.answer)
      : '<span class="empty">' + esc(d.generation_note || 'No answer (no pages retrieved — upload a PDF first).') + '</span>';
    const insp = inspectorHTML(d.trace);
    waitEl.innerHTML = '<div class="bubble">' + ans + meta + sources + note + '</div>' + thumbs + insp;
    if (d.latency_ms) {
      $('stats').innerHTML = $('stats').innerHTML.replace('· last query', '') + ' · last query <b>' + d.latency_ms + ' ms</b>';
    }
  } catch (e) {
    waitEl.innerHTML = '<div class="bubble"><span class="empty">error: ' + esc(e.message) + '</span></div>';
  } finally {
    go.disabled = false;
    $('q').focus();
  }
}

async function uploadFiles(picked) {
  const files = Array.from(picked || $('file').files || []);
  if (!files.length) return;
  const btn = $('upbtn'), st = $('upstatus');
  btn.disabled = true;
  for (const f of files) {
    if (!f.name.toLowerCase().endsWith('.pdf')) { st.textContent = f.name + ' skipped (only PDF files are supported)'; continue; }
    st.textContent = 'uploading ' + f.name + '…';
    try {
      const fd = new FormData(); fd.append('file', f);
      const r = await fetch('/v1/ingest', { method:'POST', body: fd });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) { st.textContent = f.name + ' failed: ' + (body.detail || ('HTTP ' + r.status)); continue; }
      await pollJob(body.job_id, f.name);
    } catch (e) { st.textContent = f.name + ' failed: ' + e.message; }
  }
  $('file').value = '';
  btn.disabled = false;
  refreshInfo();
}

async function pollJob(id, name) {
  for (;;) {
    await sleep(1200);
    let j;
    try { j = await (await fetch('/v1/ingest/' + id)).json(); }
    catch (e) { continue; }
    if (j.status === 'done' || j.status === 'partial') {
      const rep = j.report || {};
      $('upstatus').textContent = name + ' indexed · ' + rep.pages + ' pages in ' + rep.duration_s + 's' +
        ((j.status === 'partial' && rep.errors && rep.errors.length) ? ' (' + esc(rep.errors.join('; ')) + ')' : '') +
        ' — you can ask about it now.';
      return;
    }
    if (j.status === 'error') {
      const rep = j.report || {};
      $('upstatus').textContent = name + ' failed: ' + ((rep.errors || ['unknown error']).join('; '));
      return;
    }
  }
}

const drop = $('drop');
drop.addEventListener('click', () => $('file').click());
['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => {
  const files = Array.from(e.dataTransfer.files || []);
  if (files.length) uploadFiles(files);
});
$('file').addEventListener('change', () => uploadFiles());
$('q').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } });
renderChips();
refreshInfo();
</script>
</body>
</html>
"""