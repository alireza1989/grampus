"""Inline HTML/JS UI for the human-in-the-loop interface."""

from __future__ import annotations

UI_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Grampus — Human-in-the-loop</title>
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
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 14px;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      flex-shrink: 0;
    }

    header h1 {
      font-size: 18px;
      font-weight: 600;
      letter-spacing: -0.01em;
    }

    header p {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }

    .main {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    /* ── Left panel ── */
    .sidebar {
      width: 320px;
      flex-shrink: 0;
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .sidebar-header {
      padding: 16px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }

    .session-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
    }

    .empty-state {
      color: var(--muted);
      font-style: italic;
      padding: 24px 8px;
      text-align: center;
    }

    .session-card {
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      cursor: pointer;
      margin-bottom: 8px;
      background: var(--surface);
      transition: border-color 0.15s, background 0.15s;
    }

    .session-card:hover {
      border-color: var(--accent);
      background: #1e2135;
    }

    .session-card.active {
      border-color: var(--accent);
      background: #1e1530;
    }

    .session-card .sid {
      font-family: monospace;
      font-size: 12px;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .session-card .preview {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .session-card .since {
      color: var(--muted);
      font-size: 11px;
      margin-top: 6px;
    }

    /* ── Right panel ── */
    .convo-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .convo-panel.hidden { display: none; }

    .convo-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      flex-shrink: 0;
    }

    .convo-header .session-title {
      font-family: monospace;
      font-size: 13px;
      color: var(--muted);
    }

    .btn-close {
      background: none;
      border: none;
      color: var(--muted);
      cursor: pointer;
      font-size: 18px;
      line-height: 1;
      padding: 4px 8px;
      border-radius: 4px;
    }

    .btn-close:hover { color: var(--text); background: var(--border); }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .msg {
      max-width: 75%;
      padding: 10px 14px;
      border-radius: 10px;
      line-height: 1.5;
      font-size: 13px;
    }

    .msg-system {
      align-self: center;
      max-width: 90%;
      background: none;
      color: var(--muted);
      font-style: italic;
      font-size: 12px;
      text-align: center;
      padding: 4px 0;
    }

    .msg-user {
      align-self: flex-end;
      background: var(--accent);
      color: #fff;
      border-bottom-right-radius: 3px;
    }

    .msg-assistant {
      align-self: flex-start;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      border-bottom-left-radius: 3px;
    }

    .msg-tool {
      align-self: flex-start;
      background: none;
      color: var(--muted);
      font-size: 12px;
      font-family: monospace;
      border: 1px dashed var(--border);
      border-radius: 6px;
    }

    .msg .ts {
      font-size: 10px;
      opacity: 0.55;
      margin-top: 4px;
      display: block;
    }

    .input-area {
      padding: 16px;
      border-top: 1px solid var(--border);
      background: var(--surface);
      flex-shrink: 0;
    }

    textarea {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-size: 13px;
      padding: 10px 12px;
      resize: none;
      font-family: inherit;
      line-height: 1.5;
    }

    textarea:focus {
      outline: none;
      border-color: var(--accent);
    }

    textarea::placeholder { color: var(--muted); }

    .input-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
    }

    .btn-send {
      padding: 8px 20px;
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      white-space: nowrap;
      transition: background 0.15s;
    }

    .btn-send:hover:not(:disabled) { background: var(--accent-hover); }
    .btn-send:disabled { opacity: 0.5; cursor: not-allowed; }

    .status-msg {
      font-size: 12px;
      margin-left: 4px;
    }

    .status-msg.success { color: var(--success); }
    .status-msg.error   { color: var(--error); }

    .placeholder-panel {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-style: italic;
    }
  </style>
</head>
<body>

<header>
  <h1 id="agent-name">Nexus</h1>
  <p>Human-in-the-loop</p>
</header>

<div class="main">

  <!-- Left: pending sessions -->
  <aside class="sidebar">
    <div class="sidebar-header">Pending Sessions</div>
    <div class="session-list" id="session-list">
      <p class="empty-state">No agents waiting for input.</p>
    </div>
  </aside>

  <!-- Right: placeholder shown when nothing is selected -->
  <div class="placeholder-panel" id="placeholder-panel">
    Select a session to view the conversation.
  </div>

  <!-- Right: conversation panel -->
  <div class="convo-panel hidden" id="convo-panel">
    <div class="convo-header">
      <span class="session-title" id="convo-session-title"></span>
      <button class="btn-close" id="btn-close" title="Close">&#x2715;</button>
    </div>
    <div class="messages" id="messages"></div>
    <div class="input-area">
      <textarea id="response-input" rows="3" placeholder="Type your response…"></textarea>
      <div class="input-row">
        <button class="btn-send" id="btn-send">Send Response</button>
        <span class="status-msg" id="status-msg"></span>
      </div>
    </div>
  </div>

</div>

<script>
  'use strict';

  let _activeSessionId = null;
  let _renderedSids = [];

  // ── Health ──────────────────────────────────────────────────────────────
  async function fetchHealth() {
    try {
      const r = await fetch('/health');
      if (!r.ok) return;
      const d = await r.json();
      document.getElementById('agent-name').textContent =
        (d.agent_name || 'Nexus') + ' — Human-in-the-loop';
    } catch (_) {}
  }

  // ── Pending sessions ────────────────────────────────────────────────────
  function renderPending(sessions) {
    const list = document.getElementById('session-list');
    const incomingSids = sessions.map(s => s.session_id);

    // Remove cards no longer present
    _renderedSids
      .filter(sid => !incomingSids.includes(sid))
      .forEach(sid => {
        const el = document.getElementById('card-' + sid);
        if (el) el.remove();
        if (_activeSessionId === sid) closePanel();
      });

    if (incomingSids.length === 0) {
      list.innerHTML = '<p class="empty-state">No agents waiting for input.</p>';
      _renderedSids = [];
      return;
    }

    // Remove stale empty-state if present
    const empty = list.querySelector('.empty-state');
    if (empty) empty.remove();

    // Add new cards (preserve existing ones)
    incomingSids.forEach(sid => {
      if (!document.getElementById('card-' + sid)) {
        const s = sessions.find(x => x.session_id === sid);
        const card = document.createElement('div');
        card.className = 'session-card' + (sid === _activeSessionId ? ' active' : '');
        card.id = 'card-' + sid;
        card.innerHTML =
          '<div class="sid">' + _esc(sid) + '</div>' +
          '<div class="preview">' + _esc(s.last_message || '') + '</div>' +
          '<div class="since">Waiting since ' + _esc(s.waiting_since || '') + '</div>';
        card.addEventListener('click', () => loadState(sid));
        list.appendChild(card);
      }
    });

    _renderedSids = incomingSids;
  }

  // ── State / conversation ────────────────────────────────────────────────
  async function loadState(sessionId) {
    _activeSessionId = sessionId;

    // Highlight active card
    document.querySelectorAll('.session-card').forEach(c => c.classList.remove('active'));
    const card = document.getElementById('card-' + sessionId);
    if (card) card.classList.add('active');

    try {
      const r = await fetch('/agents/' + encodeURIComponent(sessionId) + '/state');
      if (!r.ok) {
        showStatus('Failed to load state: ' + r.status, 'error');
        return;
      }
      const state = await r.json();
      renderConversation(sessionId, state.messages);
    } catch (e) {
      showStatus('Error: ' + e.message, 'error');
    }
  }

  function renderConversation(sessionId, messages) {
    document.getElementById('placeholder-panel').style.display = 'none';
    const panel = document.getElementById('convo-panel');
    panel.classList.remove('hidden');
    document.getElementById('convo-session-title').textContent = sessionId;

    const box = document.getElementById('messages');
    box.innerHTML = '';
    messages.forEach(m => {
      const div = document.createElement('div');
      const role = (m.role || '').toLowerCase();
      div.className = 'msg msg-' + role;
      div.innerHTML =
        '<span>' + _esc(m.content || '') + '</span>' +
        '<span class="ts">' + _esc(m.timestamp || '') + '</span>';
      box.appendChild(div);
    });
    box.scrollTop = box.scrollHeight;

    document.getElementById('response-input').value = '';
    clearStatus();
  }

  function closePanel() {
    _activeSessionId = null;
    document.getElementById('convo-panel').classList.add('hidden');
    document.getElementById('placeholder-panel').style.display = '';
    document.querySelectorAll('.session-card').forEach(c => c.classList.remove('active'));
  }

  // ── Send response ───────────────────────────────────────────────────────
  async function sendResponse(sessionId) {
    const textarea = document.getElementById('response-input');
    const text = textarea.value.trim();
    if (!text) return;

    const btn = document.getElementById('btn-send');
    btn.disabled = true;
    btn.textContent = 'Sending…';
    clearStatus();

    try {
      const r = await fetch('/agents/' + encodeURIComponent(sessionId) + '/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: text }),
      });

      const data = await r.json();

      if (!r.ok) {
        showStatus('Error: ' + (data.error || r.status), 'error');
        return;
      }

      showStatus('✓ Agent resumed', 'success');
      textarea.value = '';

      if (data.still_waiting) {
        // Agent paused again — reload state
        await loadState(sessionId);
      } else {
        closePanel();
      }
    } catch (e) {
      showStatus('Error: ' + e.message, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Send Response';
    }
  }

  // ── Status helpers ──────────────────────────────────────────────────────
  function showStatus(msg, type) {
    const el = document.getElementById('status-msg');
    el.textContent = msg;
    el.className = 'status-msg ' + (type || '');
  }

  function clearStatus() {
    const el = document.getElementById('status-msg');
    el.textContent = '';
    el.className = 'status-msg';
  }

  function _esc(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── SSE listener ────────────────────────────────────────────────────────
  const es = new EventSource('/ui/events');
  es.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      renderPending(d.sessions || []);
    } catch (_) {}
  };
  es.onerror = () => {};

  // ── Wire up static handlers ─────────────────────────────────────────────
  document.getElementById('btn-close').addEventListener('click', closePanel);
  document.getElementById('btn-send').addEventListener('click', () => {
    if (_activeSessionId) sendResponse(_activeSessionId);
  });

  fetchHealth();
</script>
</body>
</html>"""
