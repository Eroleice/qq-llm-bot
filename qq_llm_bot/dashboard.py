from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import AppConfig


def register_dashboard_routes(driver: Any, storage: BotStorage, config: AppConfig) -> None:
    app = getattr(driver, "server_app", None)
    if app is None:
        return
    if getattr(app.state, "qq_llm_bot_dashboard_registered", False):
        return
    app.state.qq_llm_bot_dashboard_registered = True

    route_prefix = config.dashboard.route_prefix
    api_prefix = config.dashboard.api_prefix

    @app.get(route_prefix, response_class=HTMLResponse)
    async def dashboard_page(request: Request) -> HTMLResponse:
        _ensure_authorized(request, config)
        return HTMLResponse(_DASHBOARD_HTML.replace("__API_PREFIX__", api_prefix))

    @app.get(f"{api_prefix}/groups")
    async def dashboard_groups(request: Request) -> dict[str, object]:
        _ensure_authorized(request, config)
        return {"groups": storage.list_dashboard_groups()}

    @app.get(f"{api_prefix}/persona")
    async def dashboard_persona(request: Request) -> dict[str, object]:
        _ensure_authorized(request, config)
        return storage.get_dashboard_persona()

    @app.get(f"{api_prefix}/users")
    async def dashboard_users(
        request: Request,
        group_id: str = "",
        user_id: str = "",
        limit: int = 100,
    ) -> dict[str, object]:
        _ensure_authorized(request, config)
        return {
            "items": storage.list_dashboard_user_cognition(
                group_id=group_id.strip(),
                user_id=user_id.strip(),
                limit=_clamp_limit(limit, 10, 300),
            )
        }

    @app.get(f"{api_prefix}/messages")
    async def dashboard_messages(
        request: Request,
        group_id: str = "",
        user_id: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 200,
    ) -> dict[str, object]:
        _ensure_authorized(request, config)
        start_time = _date_to_timestamp(date_from.strip(), end=False)
        end_time = _date_to_timestamp(date_to.strip(), end=True)
        return {
            "items": storage.list_dashboard_messages(
                group_id=group_id.strip(),
                user_id=user_id.strip(),
                start_time=start_time,
                end_time=end_time,
                limit=_clamp_limit(limit, 10, 1000),
            )
        }

    @app.get(f"{api_prefix}/pending")
    async def dashboard_pending(request: Request, limit: int = 100) -> dict[str, object]:
        _ensure_authorized(request, config)
        return {"items": storage.list_dashboard_pending(limit=_clamp_limit(limit, 10, 300))}

    @app.get(f"{api_prefix}/stickers")
    async def dashboard_stickers(
        request: Request,
        group_id: str = "",
        limit: int = 200,
    ) -> dict[str, object]:
        _ensure_authorized(request, config)
        items = storage.list_dashboard_stickers(
            group_id=group_id.strip(),
            limit=_clamp_limit(limit, 10, 500),
        )
        return {
            "items": [
                item
                for item in items
                if _resolve_sticker_file(config, str(item.get("local_path", ""))) is not None
            ]
        }

    @app.get(f"{api_prefix}/stickers/{{sticker_id}}/image")
    async def dashboard_sticker_image(request: Request, sticker_id: int) -> FileResponse:
        _ensure_authorized(request, config)
        asset = storage.get_sticker_asset(sticker_id)
        if asset is None or not asset.enabled:
            raise HTTPException(status_code=404, detail="sticker not found")
        image_path = _resolve_sticker_file(config, asset.local_path)
        if image_path is None:
            raise HTTPException(status_code=404, detail="sticker image not found")
        return FileResponse(image_path)


def _ensure_authorized(request: Request, config: AppConfig) -> None:
    expected = _dashboard_token(config)
    if not expected:
        return
    supplied = request.query_params.get("token") or request.headers.get("x-dashboard-token", "")
    if supplied != expected:
        raise HTTPException(status_code=401, detail="dashboard token is required")


def _dashboard_token(config: AppConfig) -> str:
    return config.dashboard.access_token or os.getenv(config.dashboard.access_token_env, "")


def _date_to_timestamp(value: str, end: bool) -> int | None:
    if not value:
        return None
    try:
        day = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid date: {value}") from exc
    if end:
        day = day + timedelta(days=1)
    return int(day.timestamp())


def _clamp_limit(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _resolve_sticker_file(config: AppConfig, local_path: str) -> Path | None:
    raw_path = str(local_path).strip()
    if not raw_path:
        return None
    try:
        path = Path(raw_path).resolve()
        root = config.resolve_path(config.stickers.storage_dir).resolve()
    except OSError:
        return None
    if not path.is_relative_to(root):
        return None
    if not path.exists() or not path.is_file():
        return None
    return path


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QQ LLM Bot 看板</title>
  <style>
    :root {
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
  </style>
</head>
<body>
  <header>
    <h1>QQ LLM Bot 看板</h1>
    <div class="status">
      <span id="statusText">准备读取数据</span>
      <input id="tokenInput" type="password" placeholder="dashboard token" />
      <button id="saveTokenBtn">保存</button>
    </div>
  </header>
  <main>
    <nav>
      <button class="tab active" data-tab="persona">自我设定</button>
      <button class="tab" data-tab="users">成员认知</button>
      <button class="tab" data-tab="messages">群聊记录</button>
      <button class="tab" data-tab="stickers">表情包</button>
      <button class="tab" data-tab="pending">Pending</button>
    </nav>
    <section class="content">
      <div id="errorBox" class="error"></div>

      <div id="persona" class="section active">
        <h2>机器人自我设定</h2>
        <div class="grid">
          <div class="panel">
            <h3>稳定/当前人设</h3>
            <div id="personaState" class="kv"></div>
          </div>
          <div class="panel">
            <h3>自我记忆</h3>
            <div id="selfMemories"></div>
          </div>
        </div>
      </div>

      <div id="users" class="section">
        <h2>机器人对群员的认知</h2>
        <div class="toolbar">
          <label>群号<select id="usersGroup"></select></label>
          <label>QQ ID<input id="usersUser" placeholder="可留空" /></label>
          <label>数量<input id="usersLimit" type="number" value="100" min="10" max="300" /></label>
          <button class="primary" id="loadUsersBtn">查询</button>
        </div>
        <div id="usersList"></div>
      </div>

      <div id="messages" class="section">
        <h2>已入库群聊记录</h2>
        <div class="toolbar">
          <label>群号<select id="messagesGroup"></select></label>
          <label>发言人<input id="messagesUser" placeholder="QQ ID，可留空" /></label>
          <label>开始日期<input id="dateFrom" type="date" /></label>
          <label>结束日期<input id="dateTo" type="date" /></label>
          <label>数量<input id="messagesLimit" type="number" value="200" min="10" max="1000" /></label>
          <button class="primary" id="loadMessagesBtn">查询</button>
        </div>
        <div id="messagesList"></div>
      </div>

      <div id="pending" class="section">
        <h2>待确认与冲突记忆</h2>
        <div class="toolbar">
          <label>数量<input id="pendingLimit" type="number" value="100" min="10" max="300" /></label>
          <button class="primary" id="loadPendingBtn">刷新</button>
        </div>
        <div id="pendingList"></div>
      </div>

      <div id="stickers" class="section">
        <h2>可使用表情包</h2>
        <div class="toolbar">
          <label>群号<select id="stickersGroup"></select></label>
          <label>数量<input id="stickersLimit" type="number" value="200" min="10" max="500" /></label>
          <button class="primary" id="loadStickersBtn">刷新</button>
        </div>
        <div id="stickersList"></div>
      </div>
    </section>
  </main>
  <script>
    const API_PREFIX = "__API_PREFIX__";
    const state = { groups: [] };
    const qs = new URLSearchParams(location.search);
    const tokenFromUrl = qs.get("token") || "";
    if (tokenFromUrl) localStorage.setItem("qqBotDashboardToken", tokenFromUrl);
    document.getElementById("tokenInput").value = localStorage.getItem("qqBotDashboardToken") || "";

    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".section").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
      });
    });

    document.getElementById("saveTokenBtn").addEventListener("click", () => {
      localStorage.setItem("qqBotDashboardToken", document.getElementById("tokenInput").value.trim());
      loadAll();
    });
    document.getElementById("loadUsersBtn").addEventListener("click", loadUsers);
    document.getElementById("loadMessagesBtn").addEventListener("click", loadMessages);
    document.getElementById("loadStickersBtn").addEventListener("click", loadStickers);
    document.getElementById("loadPendingBtn").addEventListener("click", loadPending);

    async function api(path, params = {}) {
      const url = new URL(API_PREFIX + path, location.origin);
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && String(value).trim() !== "") {
          url.searchParams.set(key, value);
        }
      });
      const token = localStorage.getItem("qqBotDashboardToken") || "";
      if (token) url.searchParams.set("token", token);
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${await response.text()}`);
      }
      return response.json();
    }

    function showError(error) {
      const box = document.getElementById("errorBox");
      box.style.display = "block";
      box.textContent = error ? String(error.message || error) : "";
    }
    function clearError() {
      const box = document.getElementById("errorBox");
      box.style.display = "none";
      box.textContent = "";
    }
    function setStatus(text) {
      document.getElementById("statusText").textContent = text;
    }
    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }
    function formatTime(seconds) {
      if (!seconds) return "";
      return new Date(seconds * 1000).toLocaleString();
    }
    function tokenParam() {
      const token = localStorage.getItem("qqBotDashboardToken") || "";
      return token ? `?token=${encodeURIComponent(token)}` : "";
    }
    function stickerImageSrc(item) {
      return `${API_PREFIX}/stickers/${encodeURIComponent(item.id)}/image${tokenParam()}`;
    }
    function statusPill(status) {
      const cls = status === "conflict" ? "danger" : status === "pending_confirmation" ? "warn" : "ok";
      return `<span class="pill ${cls}">${escapeHtml(status)}</span>`;
    }
    function memoryHtml(memory) {
      return `
        <div class="memory">
          <div>
            <span class="pill">#${memory.id}</span>
            <span class="pill">${escapeHtml(memory.kind)}</span>
            ${statusPill(memory.status)}
            <span class="pill">${escapeHtml(memory.claim_scope)}</span>
          </div>
          <div class="message-text">${escapeHtml(memory.content)}</div>
          <div class="muted">
            conf=${Number(memory.confidence).toFixed(2)}
            imp=${Number(memory.importance).toFixed(2)}
            · ${formatTime(memory.updated_at)}
          </div>
          </div>`;
    }
    function factHtml(fact) {
      return `
        <div class="memory">
          <div>
            <span class="pill">#${fact.id}</span>
            <span class="pill">${escapeHtml(fact.fact_type)}</span>
            ${statusPill(fact.status)}
            <span class="pill">${escapeHtml(fact.claim_scope)}</span>
          </div>
          <div class="message-text">${escapeHtml(fact.claim_text)}</div>
          <div class="muted">
            topic=${escapeHtml(fact.topic)}
            stance=${escapeHtml(fact.stance || "-")}
            conf=${Number(fact.confidence).toFixed(2)}
            · ${formatTime(fact.updated_at)}
          </div>
        </div>`;
    }
    function traitsHtml(traits) {
      if (!traits || !Object.keys(traits).length) return "";
      return Object.entries(traits).map(([key, value]) => {
        const rendered = Array.isArray(value) ? value.join("、") : String(value ?? "");
        return `<div class="key">${escapeHtml(key)}</div><div>${escapeHtml(rendered || "(empty)")}</div>`;
      }).join("");
    }
    function attachmentsHtml(attachments) {
      if (!attachments || !attachments.length) return "";
      return attachments.map((item) => {
        const image = item.url
          ? `<div style="margin-top:8px">
              <img src="${escapeHtml(item.url)}" alt="image"
                   style="max-width:220px;max-height:160px;border:1px solid var(--line);border-radius:6px" />
             </div>`
          : "";
        const link = item.url
          ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">打开图片</a>`
          : escapeHtml(item.file || "image");
        return `
          <div class="memory">
            <span class="pill">${escapeHtml(item.attachment_type)}</span>
            ${link}
            ${image}
            ${item.summary ? `<div class="message-text">摘要：${escapeHtml(item.summary)}</div>` : ""}
          </div>`;
      }).join("");
    }
    function mentionsHtml(mentions) {
      if (!mentions || !mentions.length) return "";
      return `<div style="margin-top:8px">${
        mentions.map((item) => {
          const name = item.display_name && item.display_name !== item.user_id
            ? `${item.display_name} / QQ ${item.user_id}`
            : `QQ ${item.user_id}`;
          return `<span class="pill">@${escapeHtml(name)}${item.is_bot ? " bot" : ""}</span>`;
        }).join(" ")
      }</div>`;
    }
    function renderEmpty(target, text) {
      document.getElementById(target).innerHTML = `<div class="empty">${escapeHtml(text)}</div>`;
    }
    function fillGroupSelect(id) {
      const select = document.getElementById(id);
      select.innerHTML = `<option value="">全部</option>` + state.groups.map((group) => (
        `<option value="${escapeHtml(group)}">${escapeHtml(group)}</option>`
      )).join("");
    }

    async function loadGroups() {
      const data = await api("/groups");
      state.groups = data.groups || [];
      fillGroupSelect("usersGroup");
      fillGroupSelect("messagesGroup");
      fillGroupSelect("stickersGroup");
    }
    async function loadPersona() {
      const data = await api("/persona");
      const persona = data.persona_state || [];
      document.getElementById("personaState").innerHTML = persona.length ? persona.map((item) => (
        `<div class="key">${escapeHtml(item.key)}</div><div>${escapeHtml(item.value)}</div>`
      )).join("") : `<div class="empty">暂无人设状态。</div>`;
      const memories = data.self_memories || [];
      document.getElementById("selfMemories").innerHTML = memories.length
        ? memories.map(memoryHtml).join("")
        : `<div class="empty">暂无自我记忆。</div>`;
    }
    async function loadUsers() {
      clearError();
      setStatus("读取成员认知");
      try {
        const data = await api("/users", {
          group_id: document.getElementById("usersGroup").value,
          user_id: document.getElementById("usersUser").value,
          limit: document.getElementById("usersLimit").value,
        });
        const items = data.items || [];
        if (!items.length) return renderEmpty("usersList", "暂无成员画像、FACT 或关系记录。");
        document.getElementById("usersList").innerHTML = items.map((item) => {
          const relation = item.relationship || {};
          const profile = item.profile || null;
          const facts = item.facts || [];
          const nickname = item.nickname || item.display_name || "";
          const memberLabel = nickname ? `${nickname} (${item.user_id})` : `QQ ${item.user_id}`;
          return `
            <div class="item">
              <div>
                <span class="pill">${escapeHtml(memberLabel)}</span>
              </div>
              <div class="kv" style="margin-top:10px">
                <div class="key">亲近</div><div>${relation.closeness ?? 0}</div>
                <div class="key">信任</div><div>${relation.trust ?? 0}</div>
                <div class="key">熟悉</div><div>${relation.familiarity ?? 0}</div>
                <div class="key">紧张</div><div>${relation.tension ?? 0}</div>
                <div class="key">关系洞察</div><div>${escapeHtml(relation.summary || "(empty)")}</div>
              </div>
              ${profile ? `
                <div class="memory">
                  <div>
                    <span class="pill">profile v${escapeHtml(profile.version)}</span>
                    <span class="pill">facts ${escapeHtml(profile.fact_count)}</span>
                  </div>
                  <div class="message-text">${escapeHtml(profile.summary)}</div>
                  ${traitsHtml(profile.traits) ? `<div class="kv" style="margin-top:10px">${traitsHtml(profile.traits)}</div>` : ""}
                </div>` : `<div class="memory muted">暂无全局画像。</div>`}
              ${facts.length ? facts.map(factHtml).join("") : `<div class="memory muted">暂无 accepted FACT。</div>`}
            </div>`;
        }).join("");
        setStatus(`成员认知 ${items.length} 条`);
      } catch (error) {
        showError(error);
      }
    }
    async function loadMessages() {
      clearError();
      setStatus("读取群聊记录");
      try {
        const data = await api("/messages", {
          group_id: document.getElementById("messagesGroup").value,
          user_id: document.getElementById("messagesUser").value,
          date_from: document.getElementById("dateFrom").value,
          date_to: document.getElementById("dateTo").value,
          limit: document.getElementById("messagesLimit").value,
        });
        const items = data.items || [];
        if (!items.length) return renderEmpty("messagesList", "暂无符合条件的群聊记录。");
        document.getElementById("messagesList").innerHTML = items.map((item) => `
          <div class="item">
            <div>
              <span class="pill">#${item.id}</span>
              <span class="pill">group ${escapeHtml(item.group_id)}</span>
              <span class="pill">QQ ${escapeHtml(item.user_id)}</span>
              <span class="pill">${escapeHtml(item.sender_name || item.sender_role || "sender")}</span>
            </div>
            <div class="muted" style="margin-top:6px">
              ${formatTime(item.time)} · message=${escapeHtml(item.message_id)}
            </div>
            <div class="message-text">${escapeHtml(item.plain_text || item.raw_message)}</div>
            ${mentionsHtml(item.mentions)}
            ${attachmentsHtml(item.attachments)}
          </div>
        `).join("");
        setStatus(`群聊记录 ${items.length} 条`);
      } catch (error) {
        showError(error);
      }
    }
    async function copyText(text) {
      await navigator.clipboard.writeText(text);
      setStatus("已复制命令");
    }
    async function loadStickers() {
      clearError();
      setStatus("读取表情包");
      try {
        const data = await api("/stickers", {
          group_id: document.getElementById("stickersGroup").value,
          limit: document.getElementById("stickersLimit").value,
        });
        const items = data.items || [];
        if (!items.length) return renderEmpty("stickersList", "暂无可使用表情包。");
        document.getElementById("stickersList").innerHTML = `
          <div class="sticker-grid">
            ${items.map((item) => {
              const tags = item.tags || [];
              const command = item.delete_command || `#bot stickers delete ${item.id}`;
              return `
                <div class="item sticker-card">
                  <img class="sticker-media" src="${escapeHtml(stickerImageSrc(item))}" alt="sticker #${escapeHtml(item.id)}" />
                  <div>
                    <span class="pill">#${escapeHtml(item.id)}</span>
                    <span class="pill">group ${escapeHtml(item.group_id)}</span>
                    ${item.mood ? `<span class="pill">${escapeHtml(item.mood)}</span>` : ""}
                  </div>
                  <div class="message-text">${escapeHtml(item.trigger || item.usage || item.description)}</div>
                  ${tags.length ? `<div>${tags.map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}</div>` : ""}
                  ${item.description ? `<div class="muted">${escapeHtml(item.description)}</div>` : ""}
                  <div class="muted">
                    hits=${Number(item.hit_count || 0)}
                    conf=${Number(item.confidence || 0).toFixed(2)}
                    · ${formatTime(item.updated_at)}
                  </div>
                  <div class="commands">
                    <code>${escapeHtml(command)}</code>
                    <button onclick="copyText('${escapeHtml(command)}')">复制删除</button>
                  </div>
                </div>`;
            }).join("")}
          </div>`;
        setStatus(`表情包 ${items.length} 个`);
      } catch (error) {
        showError(error);
      }
    }
    async function loadPending() {
      clearError();
      setStatus("读取 pending");
      try {
        const data = await api("/pending", {
          limit: document.getElementById("pendingLimit").value,
        });
        const items = data.items || [];
        if (!items.length) return renderEmpty("pendingList", "暂无待确认 FACT 或冲突记忆。");
        document.getElementById("pendingList").innerHTML = items.map((item) => `
          <div class="item">
            <div>
              <span class="pill">#${item.id}</span>
              <span class="pill">${escapeHtml(item.item_type || "memory")}</span>
              <span class="pill">${
                item.item_type === "fact"
                  ? `user:${escapeHtml(item.subject_user_id)}`
                  : `${escapeHtml(item.owner_type)}:${escapeHtml(item.owner_id)}`
              }</span>
              <span class="pill">${escapeHtml(item.fact_type || item.kind)}</span>
              ${statusPill(item.status)}
            </div>
            <div class="message-text">${escapeHtml(item.claim_text || item.content)}</div>
            <div class="muted">
              source=${escapeHtml(item.source_user_id)}
              subject=${escapeHtml(item.subject_user_id)}
              · ${formatTime(item.updated_at)}
            </div>
            <div class="commands">
              <code>${escapeHtml(item.approve_command)}</code>
              <button onclick="copyText('${escapeHtml(item.approve_command)}')">复制批准</button>
              <code>${escapeHtml(item.reject_command)}</code>
              <button onclick="copyText('${escapeHtml(item.reject_command)}')">复制拒绝</button>
            </div>
          </div>
        `).join("");
        setStatus(`Pending ${items.length} 条`);
      } catch (error) {
        showError(error);
      }
    }
    async function loadAll() {
      clearError();
      setStatus("读取数据");
      try {
        await loadGroups();
        await Promise.all([loadPersona(), loadUsers(), loadMessages(), loadStickers(), loadPending()]);
        setStatus("数据已更新");
      } catch (error) {
        showError(error);
        setStatus("读取失败");
      }
    }
    loadAll();
  </script>
</body>
</html>
"""
