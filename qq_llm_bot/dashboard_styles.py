from __future__ import annotations

DASHBOARD_CSS = r"""    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172033;
      --muted: #647084;
      --line: #dfe4ec;
      --accent: #1769e0;
      --accent-soft: #e8f1ff;
      --danger: #b42318;
      --warn: #9a6700;
      --ok: #137a3a;
      --shadow: 0 8px 24px rgba(23, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { font-size: 20px; margin: 0; }
    h2 { font-size: 18px; margin: 0 0 14px; }
    h3 { font-size: 15px; margin: 0 0 10px; }
    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      min-width: 0;
    }
    .status input { width: 220px; }
    main {
      display: grid;
      grid-template-columns: 220px 1fr;
      min-height: calc(100vh - 62px);
    }
    nav {
      padding: 18px 14px;
      border-right: 1px solid var(--line);
      background: #fbfcfe;
    }
    .tab {
      width: 100%;
      border: 0;
      background: transparent;
      color: var(--text);
      display: block;
      text-align: left;
      padding: 10px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      margin-bottom: 4px;
    }
    .tab.active {
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 650;
    }
    .content { padding: 22px 24px 42px; min-width: 0; }
    .section { display: none; }
    .section.active { display: block; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
      margin: 0 0 16px;
      padding: 14px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    input, select {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--text);
      min-width: 150px;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      min-width: 0;
      padding: 0;
      accent-color: var(--accent);
    }
    button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 12px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.danger {
      background: var(--danger);
      border-color: var(--danger);
      color: #fff;
    }
    button.warn {
      background: #fff8e1;
      border-color: #f0c36a;
      color: var(--warn);
    }
    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }
    .panel, .item {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel { padding: 16px; margin-bottom: 14px; }
    .item { padding: 14px; margin-bottom: 10px; }
    .kv {
      display: grid;
      grid-template-columns: minmax(90px, 160px) 1fr;
      gap: 8px 12px;
      font-size: 14px;
    }
    .key { color: var(--muted); }
    .muted { color: var(--muted); }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 2px 8px;
      background: #eef1f5;
      color: var(--muted);
      font-size: 12px;
      margin: 0 4px 4px 0;
    }
    .pill.warn { background: #fff3cd; color: var(--warn); }
    .pill.danger { background: #fdecea; color: var(--danger); }
    .pill.ok { background: #e7f6ec; color: var(--ok); }
    .memory {
      border-top: 1px solid var(--line);
      padding-top: 9px;
      margin-top: 9px;
      font-size: 13px;
    }
    .message-text {
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.5;
      margin-top: 8px;
    }
    code {
      display: inline-block;
      max-width: 100%;
      overflow-wrap: anywhere;
      padding: 4px 6px;
      border-radius: 5px;
      background: #f1f3f6;
      color: #24324a;
    }
    .commands {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      align-items: center;
    }
    .pending-title {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .selection-count {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
    }
    .sticker-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
    }
    .sticker-card {
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
    }
    .metric-label {
      color: var(--muted);
      font-size: 12px;
    }
    .metric-value {
      font-size: 22px;
      font-weight: 700;
      margin-top: 4px;
    }
    .usage-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    .usage-table th,
    .usage-table td {
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }
    .usage-table th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .sticker-media {
      width: 100%;
      aspect-ratio: 1;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      object-fit: contain;
    }
    .empty {
      color: var(--muted);
      padding: 18px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .error {
      color: var(--danger);
      background: #fdecea;
      border: 1px solid #f5c2c0;
      padding: 10px 12px;
      border-radius: 8px;
      margin-bottom: 12px;
      display: none;
    }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      nav {
        display: flex;
        overflow-x: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .tab { white-space: nowrap; width: auto; }
      .content { padding: 16px; }
      .kv { grid-template-columns: 1fr; }
    }
"""
