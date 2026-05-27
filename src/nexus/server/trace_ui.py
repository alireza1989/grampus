"""Inline HTML/JS UI for the execution trace viewer."""

from __future__ import annotations

TRACE_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Nexus &mdash; Trace Viewer</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0f1117;
      --surface: #1a1d27;
      --text: #e2e4f0;
      --muted: #8b90b0;
      --accent: #7c3aed;
      --accent-hover: #6d28d9;
      --border: #2a2d3d;
      --error: #ef4444;
      --success: #22c55e;

      --c-started: #7c3aed;
      --c-completed: #7c3aed;
      --c-llm: #60a5fa;
      --c-tool-called: #f97316;
      --c-tool-result: #22c55e;
      --c-human: #fbbf24;
      --c-failed: #ef4444;
      --c-safety: #ef4444;
      --c-budget: #ef4444;
      --c-memory: #14b8a6;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 14px;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    header {
      padding: 12px 24px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      flex-shrink: 0;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    header h1 { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }

    .header-agent { color: var(--muted); font-size: 12px; }

    .spacer { flex: 1; }

    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 9px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .badge-idle       { background: #1f2937; color: #6b7280; }
    .badge-connecting { background: #1f2937; color: #9ca3af; }
    .badge-live       { background: #14532d; color: #22c55e; display: inline-flex; align-items: center; gap: 5px; }
    .badge-done       { background: #14532d; color: #22c55e; }
    .badge-error      { background: #450a0a; color: #ef4444; }
    .badge-history    { background: #1e1b4b; color: #a78bfa; }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.3; }
    }

    .dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: #22c55e;
      animation: pulse 1.5s ease-in-out infinite;
    }

    .main { display: flex; flex: 1; overflow: hidden; }

    /* ── Left panel ── */
    .left-panel {
      width: 420px;
      flex-shrink: 0;
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .session-bar {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      display: flex;
      gap: 8px;
      flex-shrink: 0;
    }

    .session-input {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      font-size: 13px;
      padding: 7px 10px;
      font-family: monospace;
    }

    .session-input:focus { outline: none; border-color: var(--accent); }
    .session-input::placeholder { color: var(--muted); }

    .btn-watch {
      padding: 7px 16px;
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      transition: background 0.15s;
      white-space: nowrap;
    }

    .btn-watch:hover { background: var(--accent-hover); }

    .timeline { flex: 1; overflow-y: auto; padding: 4px 0; }

    .empty-state {
      color: var(--muted);
      font-style: italic;
      padding: 24px 16px;
      text-align: center;
      font-size: 13px;
    }

    @keyframes fadein {
      from { opacity: 0; transform: translateY(3px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .event-row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 7px 14px;
      cursor: pointer;
      border-left: 3px solid transparent;
      animation: fadein 0.2s ease;
      transition: background 0.1s;
    }

    .event-row:hover    { background: #1e2135; }
    .event-row.selected { background: #1e1530; border-left-color: var(--accent); }

    .ev-icon    { font-size: 13px; width: 18px; text-align: center; flex-shrink: 0; }
    .ev-time    { font-family: monospace; font-size: 11px; color: var(--muted); flex-shrink: 0; width: 58px; }
    .ev-label   { font-size: 11px; font-weight: 700; flex-shrink: 0; width: 110px; text-transform: uppercase; letter-spacing: 0.03em; }
    .ev-summary { font-size: 12px; color: var(--muted); overflow: hidden; white-space: nowrap; text-overflow: ellipsis; flex: 1; }

    /* ── Right panel ── */
    .right-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

    .detail-wrap { flex: 1; overflow-y: auto; padding: 20px; }

    .detail-empty { color: var(--muted); font-style: italic; text-align: center; padding-top: 80px; font-size: 13px; }

    .detail-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 14px;
    }

    .detail-title { font-size: 15px; font-weight: 600; }
    .detail-meta  { font-size: 12px; color: var(--muted); margin-top: 3px; font-family: monospace; }

    .btn-close {
      background: none;
      border: none;
      color: var(--muted);
      cursor: pointer;
      font-size: 18px;
      padding: 2px 6px;
      border-radius: 4px;
      line-height: 1;
    }

    .btn-close:hover { color: var(--text); background: var(--border); }

    .detail-payload {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      font-family: monospace;
      font-size: 12px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
      overflow-x: auto;
    }

    /* ── Footer ── */
    footer {
      padding: 7px 20px;
      border-top: 1px solid var(--border);
      background: var(--surface);
      display: flex;
      align-items: center;
      gap: 20px;
      flex-shrink: 0;
      font-size: 12px;
    }

    .stat { display: flex; align-items: center; gap: 6px; }
    .stat-label { color: var(--muted); }
    .stat-value { font-weight: 600; font-family: monospace; }

    .cost-badge {
      background: #1a2e1a;
      color: #4ade80;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 600;
      font-family: monospace;
      font-size: 11px;
    }

    .ev-count { color: var(--muted); font-family: monospace; font-size: 12px; }
  </style>
</head>
<body>

<header>
  <h1>Nexus &mdash; Trace Viewer</h1>
  <span class="header-agent" id="agent-name"></span>
  <span class="spacer"></span>
  <span class="badge badge-idle" id="status-badge">Idle</span>
</header>

<div class="main">

  <div class="left-panel">
    <div class="session-bar">
      <input type="text" class="session-input" id="session-input"
             placeholder="session-id" autocomplete="off" spellcheck="false" />
      <button class="btn-watch" id="btn-watch">Watch</button>
    </div>
    <div class="timeline" id="timeline">
      <p class="empty-state" id="timeline-empty">Enter a session ID and click Watch.</p>
    </div>
  </div>

  <div class="right-panel">
    <div class="detail-wrap" id="detail-wrap">
      <p class="detail-empty">Click an event to inspect its payload.</p>
    </div>
  </div>

</div>

<footer>
  <div class="stat"><span class="stat-label">Steps</span><span class="stat-value" id="stat-steps">0</span></div>
  <div class="stat"><span class="stat-label">Tokens</span><span class="stat-value" id="stat-tokens">0</span></div>
  <div class="stat"><span class="stat-label">Cost</span><span class="cost-badge" id="stat-cost">$0.0000</span></div>
  <span class="spacer"></span>
  <span class="ev-count" id="ev-count">0 events</span>
</footer>

<script>
  'use strict';

  const TYPE_META = {
    'agent.started':              { icon: '▶', color: 'var(--c-started)',    label: 'Started' },
    'agent.completed':            { icon: '✓', color: 'var(--c-completed)',  label: 'Completed' },
    'agent.failed':               { icon: '✗', color: 'var(--c-failed)',     label: 'Failed' },
    'agent.llm_called':           { icon: '⚡', color: 'var(--c-llm)',        label: 'LLM Called' },
    'agent.tool_called':          { icon: '🔧', color: 'var(--c-tool-called)',  label: 'Tool Called' },
    'agent.tool_result':          { icon: '↩', color: 'var(--c-tool-result)', label: 'Tool Result' },
    'agent.memory_read':          { icon: '🧠', color: 'var(--c-memory)',   label: 'Mem Read' },
    'agent.memory_written':       { icon: '🧠', color: 'var(--c-memory)',   label: 'Mem Write' },
    'agent.safety_violation':     { icon: '🛡', color: 'var(--c-safety)',   label: 'Safety' },
    'agent.human_input_requested':{ icon: '⏸', color: 'var(--c-human)',     label: 'Human Input' },
    'agent.budget_exceeded':      { icon: '⚠', color: 'var(--c-budget)',    label: 'Budget' },
  };

  let eventSource = null;
  let selectedRow = null;
  let userScrolled = false;
  let stats = { steps: 0, tokens: 0, cost: 0, count: 0 };

  function _esc(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function _fmtTime(iso) {
    try { return new Date(iso).toTimeString().slice(0, 8); } catch (_) { return ''; }
  }

  function _brief(ev) {
    const p = ev.payload || {};
    switch (ev.event_type) {
      case 'agent.started':    return 'model=' + (p.model || '');
      case 'agent.completed':  return 'steps=' + (p.steps || 0) + '  cost=$' + (+(p.cost_usd || 0)).toFixed(4);
      case 'agent.llm_called': return 'step=' + (p.step || 0) + '  in=' + (p.input_tokens || 0) + '  out=' + (p.output_tokens || 0);
      case 'agent.tool_called':return String(p.tool || '');
      case 'agent.tool_result':return (p.ok ? '✓ ' : '✗ ') + (p.tool || '');
      case 'agent.human_input_requested': return String(p.question || '').slice(0, 60);
      default: return Object.keys(p).slice(0, 2).map(function(k) { return k + '=' + String(p[k]).slice(0, 20); }).join('  ');
    }
  }

  function setStatus(s) {
    var el = document.getElementById('status-badge');
    el.className = 'badge';
    if (s === 'CONNECTING') { el.classList.add('badge-connecting'); el.textContent = 'Connecting'; }
    else if (s === 'LIVE')  { el.classList.add('badge-live'); el.innerHTML = '<span class="dot"></span>Live'; }
    else if (s === 'DONE')  { el.classList.add('badge-done'); el.textContent = 'Done'; }
    else if (s === 'ERROR') { el.classList.add('badge-error'); el.textContent = 'Error'; }
    else                    { el.classList.add('badge-idle'); el.textContent = 'Idle'; }
  }

  function resetTimeline() {
    var tl = document.getElementById('timeline');
    tl.innerHTML = '<p class="empty-state" id="timeline-empty">Loading…</p>';
    selectedRow = null;
    document.getElementById('detail-wrap').innerHTML = '<p class="detail-empty">Click an event to inspect its payload.</p>';
    userScrolled = false;
    stats = { steps: 0, tokens: 0, cost: 0, count: 0 };
    renderStats();
  }

  function renderStats() {
    document.getElementById('stat-steps').textContent = stats.steps;
    document.getElementById('stat-tokens').textContent = stats.tokens.toLocaleString();
    document.getElementById('stat-cost').textContent = '$' + stats.cost.toFixed(4);
    document.getElementById('ev-count').textContent = stats.count + ' event' + (stats.count === 1 ? '' : 's');
  }

  function updateStats(ev) {
    if (ev.event_type === 'agent.llm_called') {
      stats.steps++;
      var p = ev.payload || {};
      stats.tokens += (p.input_tokens || 0) + (p.output_tokens || 0);
    }
    if (ev.event_type === 'agent.completed') {
      stats.cost += (ev.payload || {}).cost_usd || 0;
    }
    stats.count++;
    renderStats();
  }

  function showDetail(row, ev) {
    if (selectedRow) selectedRow.classList.remove('selected');
    selectedRow = row;
    row.classList.add('selected');

    var meta = TYPE_META[ev.event_type] || { icon: '•', color: 'var(--muted)', label: ev.event_type };
    var payload = JSON.stringify(ev.payload || {}, null, 2);

    document.getElementById('detail-wrap').innerHTML =
      '<div class="detail-header">' +
        '<div>' +
          '<div class="detail-title" style="color:' + meta.color + '">' + meta.icon + ' ' + _esc(meta.label) + '</div>' +
          '<div class="detail-meta">seq=' + ev.sequence_number + ' &middot; ' + _esc(ev.timestamp) + '</div>' +
          '<div class="detail-meta">agent=' + _esc(ev.agent_id) + ' &middot; session=' + _esc(ev.session_id) + '</div>' +
        '</div>' +
        '<button class="btn-close" id="btn-close-detail" title="Close">&#x2715;</button>' +
      '</div>' +
      '<pre class="detail-payload">' + _esc(payload) + '</pre>';

    document.getElementById('btn-close-detail').addEventListener('click', function() {
      if (selectedRow) selectedRow.classList.remove('selected');
      selectedRow = null;
      document.getElementById('detail-wrap').innerHTML = '<p class="detail-empty">Click an event to inspect its payload.</p>';
    });
  }

  function appendEvent(ev) {
    updateStats(ev);
    var tl = document.getElementById('timeline');
    var empty = tl.querySelector('.empty-state');
    if (empty) empty.remove();

    var meta = TYPE_META[ev.event_type] || { icon: '•', color: 'var(--muted)', label: ev.event_type };
    var row = document.createElement('div');
    row.className = 'event-row';
    row.innerHTML =
      '<span class="ev-icon">' + meta.icon + '</span>' +
      '<span class="ev-time">' + _esc(_fmtTime(ev.timestamp)) + '</span>' +
      '<span class="ev-label" style="color:' + meta.color + '">' + _esc(meta.label) + '</span>' +
      '<span class="ev-summary">' + _esc(_brief(ev)) + '</span>';

    row.addEventListener('click', function() { showDetail(row, ev); });

    var atBottom = tl.scrollTop + tl.clientHeight >= tl.scrollHeight - 20;
    tl.appendChild(row);
    if (!userScrolled || atBottom) { tl.scrollTop = tl.scrollHeight; }
  }

  async function watchSession(sessionId) {
    if (!sessionId) return;
    if (eventSource) { eventSource.close(); eventSource = null; }
    resetTimeline();
    setStatus('CONNECTING');

    var url = new URL(window.location.href);
    url.searchParams.set('session', sessionId);
    window.history.replaceState({}, '', url);

    var lastType = null;
    try {
      var r = await fetch('/trace/' + encodeURIComponent(sessionId) + '/history');
      if (r.ok) {
        var data = await r.json();
        if (data.events && data.events.length > 0) {
          data.events.forEach(appendEvent);
          lastType = data.events[data.events.length - 1].event_type;
        }
      }
    } catch (_) {}

    if (lastType === 'agent.completed' || lastType === 'agent.failed') {
      setStatus('DONE');
      return;
    }

    eventSource = new EventSource('/trace/' + encodeURIComponent(sessionId) + '/stream');
    setStatus('LIVE');

    eventSource.onmessage = function(e) {
      var d;
      try { d = JSON.parse(e.data); } catch (_) { return; }
      if (d.heartbeat) return;
      if (d.done) {
        setStatus('DONE');
        eventSource.close();
        eventSource = null;
        return;
      }
      appendEvent(d);
    };

    eventSource.onerror = function() {
      setStatus('ERROR');
      if (eventSource) { eventSource.close(); eventSource = null; }
    };
  }

  document.getElementById('timeline').addEventListener('scroll', function() {
    var tl = document.getElementById('timeline');
    userScrolled = tl.scrollTop + tl.clientHeight < tl.scrollHeight - 40;
  });

  document.getElementById('btn-watch').addEventListener('click', function() {
    var sid = document.getElementById('session-input').value.trim();
    if (sid) watchSession(sid);
  });

  document.getElementById('session-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      var sid = e.target.value.trim();
      if (sid) watchSession(sid);
    }
  });

  (function() {
    fetch('/health').then(function(r) { return r.ok ? r.json() : null; }).then(function(d) {
      if (d && d.agent_name) document.getElementById('agent-name').textContent = d.agent_name;
    }).catch(function() {});

    var params = new URLSearchParams(window.location.search);
    var sid = params.get('session');
    if (sid) {
      document.getElementById('session-input').value = sid;
      watchSession(sid);
    }
  })();
</script>
</body>
</html>"""
