from __future__ import annotations

DASHBOARD_JS = r"""    const API_PREFIX = "__API_PREFIX__";
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
    document.getElementById("loadLlmUsageBtn").addEventListener("click", loadLlmUsage);
    document.getElementById("loadQaBlocksBtn").addEventListener("click", loadQaBlocks);
    document.getElementById("loadPendingBtn").addEventListener("click", loadPending);
    document.getElementById("selectAllPendingBtn").addEventListener("click", selectAllPending);
    document.getElementById("clearPendingSelectionBtn").addEventListener("click", clearPendingSelection);
    document.getElementById("bulkApprovePendingBtn").addEventListener("click", () => bulkManagePending("approve"));
    document.getElementById("bulkRejectPendingBtn").addEventListener("click", () => bulkManagePending("reject"));

    async function api(path, params = {}, options = {}) {
      const url = new URL(API_PREFIX + path, location.origin);
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && String(value).trim() !== "") {
          url.searchParams.set(key, value);
        }
      });
      const token = localStorage.getItem("qqBotDashboardToken") || "";
      if (token) url.searchParams.set("token", token);
      const fetchOptions = { method: options.method || "GET" };
      if (options.body !== undefined) {
        fetchOptions.headers = { "content-type": "application/json" };
        fetchOptions.body = JSON.stringify(options.body);
      }
      const response = await fetch(url, fetchOptions);
      const text = await response.text();
      let data = {};
      try {
        data = text ? JSON.parse(text) : {};
      } catch {
        data = {};
      }
      if (!response.ok) {
        const detail = data.detail || text || response.statusText;
        throw new Error(`${response.status} ${detail}`);
      }
      return data;
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
    function formatInteger(value) {
      return Number(value || 0).toLocaleString();
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
          <div class="commands">
            <button class="danger" onclick="manageMemory(${Number(memory.id)}, 'forget')">删除</button>
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
          <div class="commands">
            <button class="danger" onclick="manageFact(${Number(fact.id)}, 'forget')">删除</button>
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
    function contextLinesHtml(title, lines) {
      if (!lines || !lines.length) return "";
      return `
        <div class="memory">
          <div class="key">${escapeHtml(title)}</div>
          <div class="message-text">${escapeHtml(lines.join("\n"))}</div>
        </div>`;
    }
    function qaBlockHtml(item) {
      const categories = item.qa_categories || [];
      return `
        <div class="item">
          <div>
            <span class="pill">#${escapeHtml(item.id)}</span>
            <span class="pill">group ${escapeHtml(item.group_id)}</span>
            <span class="pill">QQ ${escapeHtml(item.user_id)}</span>
            <span class="pill">${escapeHtml(item.sender_name || item.sender_role || "sender")}</span>
            <span class="pill danger">QA block</span>
            ${categories.map((category) => `<span class="pill warn">${escapeHtml(category)}</span>`).join("")}
          </div>
          <div class="muted" style="margin-top:6px">
            ${formatTime(item.created_at)} · message=${escapeHtml(item.message_id)}
            · conf=${Number(item.qa_confidence || 0).toFixed(2)}
          </div>
          <div class="memory">
            <div class="key">触发消息</div>
            <div class="message-text">${escapeHtml(item.trigger_text || item.raw_message)}</div>
          </div>
          <div class="memory">
            <div class="key">被拦回复</div>
            <div class="message-text">${escapeHtml(item.candidate_reply)}</div>
          </div>
          <div class="memory">
            <div class="key">QA 理由</div>
            <div class="message-text">${escapeHtml(item.qa_reason || "(empty)")}</div>
          </div>
          <details class="memory">
            <summary>上下文与决策</summary>
            <div class="kv" style="margin-top:10px">
              <div class="key">模式/动作</div><div>${escapeHtml(item.mode)} / ${escapeHtml(item.action)}</div>
              <div class="key">价值类型</div><div>${escapeHtml(item.value_type || "-")} ${Number(item.value_score || 0).toFixed(2)}</div>
              <div class="key">聊天密度</div><div>${escapeHtml(item.traffic_level || "-")}</div>
              <div class="key">决策理由</div><div>${escapeHtml(item.decision_reason || "(empty)")}</div>
            </div>
            ${contextLinesHtml("当前发言人近期主线", item.speaker_recent_messages)}
            ${contextLinesHtml("其他发言近期话题参考", item.other_recent_messages)}
            ${contextLinesHtml("最近群聊", item.recent_messages)}
            ${contextLinesHtml("最近图片", item.recent_image_descriptions)}
          </details>
        </div>`;
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
      fillGroupSelect("qaBlocksGroup");
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
    async function loadQaBlocks() {
      clearError();
      setStatus("读取 QA 拦截归档");
      try {
        const data = await api("/qa-blocks", {
          group_id: document.getElementById("qaBlocksGroup").value,
          user_id: document.getElementById("qaBlocksUser").value,
          date_from: document.getElementById("qaBlocksDateFrom").value,
          date_to: document.getElementById("qaBlocksDateTo").value,
          limit: document.getElementById("qaBlocksLimit").value,
        });
        const items = data.items || [];
        if (!items.length) return renderEmpty("qaBlocksList", "暂无符合条件的 QA 拦截记录。");
        document.getElementById("qaBlocksList").innerHTML = items.map(qaBlockHtml).join("");
        setStatus(`QA 拦截 ${items.length} 条`);
      } catch (error) {
        showError(error);
      }
    }
    async function copyText(text) {
      await navigator.clipboard.writeText(text);
      setStatus("已复制命令");
    }
    function activeTab() {
      const active = document.querySelector(".tab.active");
      return active ? active.dataset.tab : "persona";
    }
    function setBusy(isBusy) {
      document.querySelectorAll("button").forEach((button) => {
        button.disabled = Boolean(isBusy);
      });
    }
    async function refreshActiveTab() {
      const tab = activeTab();
      if (tab === "persona") return loadPersona();
      if (tab === "users") return loadUsers();
      if (tab === "messages") return loadMessages();
      if (tab === "stickers") return loadStickers();
      if (tab === "llmUsage") return loadLlmUsage();
      if (tab === "qaBlocks") return loadQaBlocks();
      if (tab === "pending") return loadPending();
    }
    async function runAction(path, options = {}, doneText = "操作已完成") {
      clearError();
      setStatus("提交操作");
      setBusy(true);
      try {
        await api(path, {}, options);
        await refreshActiveTab();
        setStatus(doneText);
      } catch (error) {
        showError(error);
        setStatus("操作失败");
      } finally {
        setBusy(false);
      }
    }
    async function manageFact(factId, action) {
      const labels = { approve: "批准", reject: "拒绝", forget: "删除" };
      if (action === "forget" && !confirm(`确认删除 FACT #${factId}？`)) return;
      await runAction(
        `/facts/${encodeURIComponent(factId)}/${encodeURIComponent(action)}`,
        { method: "POST" },
        `FACT #${factId} 已${labels[action] || "更新"}`
      );
    }
    async function manageMemory(memoryId, action) {
      const labels = { approve: "批准", reject: "拒绝", forget: "删除" };
      if (action === "forget" && !confirm(`确认删除记忆 #${memoryId}？`)) return;
      await runAction(
        `/memories/${encodeURIComponent(memoryId)}/${encodeURIComponent(action)}`,
        { method: "POST" },
        `记忆 #${memoryId} 已${labels[action] || "更新"}`
      );
    }
    async function managePending(itemType, itemId, action) {
      if (itemType === "fact") return manageFact(itemId, action);
      return manageMemory(itemId, action);
    }
    function pendingCheckboxes() {
      return Array.from(document.querySelectorAll(".pending-check"));
    }
    function selectedPendingItems() {
      return pendingCheckboxes()
        .filter((checkbox) => checkbox.checked)
        .map((checkbox) => ({
          id: Number(checkbox.dataset.itemId),
          item_type: checkbox.dataset.itemType || "memory",
        }))
        .filter((item) => Number.isFinite(item.id) && item.id > 0);
    }
    function updatePendingSelection() {
      const count = selectedPendingItems().length;
      document.getElementById("pendingSelectionText").textContent = `已选 ${count} 条`;
    }
    function selectAllPending() {
      pendingCheckboxes().forEach((checkbox) => {
        checkbox.checked = true;
      });
      updatePendingSelection();
    }
    function clearPendingSelection() {
      pendingCheckboxes().forEach((checkbox) => {
        checkbox.checked = false;
      });
      updatePendingSelection();
    }
    async function bulkManagePending(action) {
      const items = selectedPendingItems();
      if (!items.length) {
        setStatus("请先选择 pending 项");
        return;
      }
      const label = action === "approve" ? "批准" : "驳回";
      if (!confirm(`确认${label}选中的 ${items.length} 条 pending？`)) return;
      await runAction(
        "/pending/bulk",
        { method: "POST", body: { action, items } },
        `已${label} ${items.length} 条 pending`
      );
    }
    async function setStickerEnabled(stickerId, enabled) {
      await runAction(
        `/stickers/${encodeURIComponent(stickerId)}/${enabled ? "enable" : "disable"}`,
        { method: "POST" },
        `表情包 #${stickerId} 已${enabled ? "启用" : "停用"}`
      );
    }
    async function deleteSticker(stickerId) {
      if (!confirm(`确认删除表情包 #${stickerId}？本地图片也会尝试删除。`)) return;
      await runAction(
        `/stickers/${encodeURIComponent(stickerId)}`,
        { method: "DELETE" },
        `表情包 #${stickerId} 已删除`
      );
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
                    sent=${Number(item.send_count || 0)}
                    conf=${Number(item.confidence || 0).toFixed(2)}
                    · ${formatTime(item.updated_at)}
                  </div>
                  <div class="commands">
                    <button class="warn" onclick="setStickerEnabled(${Number(item.id)}, false)">停用</button>
                    <button class="danger" onclick="deleteSticker(${Number(item.id)})">删除</button>
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
    async function loadLlmUsage() {
      clearError();
      setStatus("读取 LLM 用量");
      try {
        const data = await api("/llm-usage", {
          hours: document.getElementById("llmUsageHours").value,
          limit: document.getElementById("llmUsageLimit").value,
        });
        const summary = data.summary || {};
        const calls = Number(summary.calls || 0);
        const avgTokens = calls ? Math.round(Number(summary.total_tokens || 0) / calls) : 0;
        document.getElementById("llmUsageSummary").innerHTML = `
          <div class="metric-grid">
            <div class="metric">
              <div class="metric-label">调用次数</div>
              <div class="metric-value">${formatInteger(calls)}</div>
            </div>
            <div class="metric">
              <div class="metric-label">总 token</div>
              <div class="metric-value">${formatInteger(summary.total_tokens)}</div>
            </div>
            <div class="metric">
              <div class="metric-label">Prompt token</div>
              <div class="metric-value">${formatInteger(summary.prompt_tokens)}</div>
            </div>
            <div class="metric">
              <div class="metric-label">Completion token</div>
              <div class="metric-value">${formatInteger(summary.completion_tokens)}</div>
            </div>
            <div class="metric">
              <div class="metric-label">平均 token / 次</div>
              <div class="metric-value">${formatInteger(avgTokens)}</div>
            </div>
            <div class="metric">
              <div class="metric-label">字符数</div>
              <div class="metric-value">${formatInteger(Number(summary.prompt_chars || 0) + Number(summary.completion_chars || 0))}</div>
            </div>
          </div>
          <div class="muted">
            范围：${formatTime(summary.first_at) || "无记录"} - ${formatTime(summary.last_at) || "无记录"}。
            provider 不返回 token 时，token 会显示为 0，可参考字符数。
          </div>
        `;

        const byPurpose = data.by_purpose || [];
        document.getElementById("llmUsageByPurpose").innerHTML = byPurpose.length ? `
          <div class="panel">
            <h3>按 purpose / model 汇总</h3>
            <table class="usage-table">
              <thead>
                <tr>
                  <th>purpose</th>
                  <th>model</th>
                  <th>calls</th>
                  <th>prompt</th>
                  <th>completion</th>
                  <th>total</th>
                  <th>chars</th>
                  <th>last</th>
                </tr>
              </thead>
              <tbody>
                ${byPurpose.map((item) => `
                  <tr>
                    <td><span class="pill">${escapeHtml(item.purpose || "(empty)")}</span></td>
                    <td>${escapeHtml(item.model || "-")}</td>
                    <td>${formatInteger(item.calls)}</td>
                    <td>${formatInteger(item.prompt_tokens)}</td>
                    <td>${formatInteger(item.completion_tokens)}</td>
                    <td>${formatInteger(item.total_tokens)}</td>
                    <td>${formatInteger(Number(item.prompt_chars || 0) + Number(item.completion_chars || 0))}</td>
                    <td>${formatTime(item.last_at)}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        ` : `<div class="empty">暂无 LLM 用量记录。</div>`;

        const recent = data.recent || [];
        document.getElementById("llmUsageRecent").innerHTML = recent.length ? `
          <div class="panel">
            <h3>最近调用</h3>
            ${recent.map((item) => `
              <div class="item">
                <div>
                  <span class="pill">#${escapeHtml(item.id)}</span>
                  <span class="pill">${escapeHtml(item.purpose || "(empty)")}</span>
                  <span class="pill">${escapeHtml(item.model || "-")}</span>
                </div>
                <div class="muted" style="margin-top:6px">${formatTime(item.created_at)}</div>
                <div class="kv" style="margin-top:10px">
                  <div class="key">prompt tokens</div><div>${formatInteger(item.prompt_tokens)}</div>
                  <div class="key">completion tokens</div><div>${formatInteger(item.completion_tokens)}</div>
                  <div class="key">total tokens</div><div>${formatInteger(item.total_tokens)}</div>
                  <div class="key">prompt chars</div><div>${formatInteger(item.prompt_chars)}</div>
                  <div class="key">completion chars</div><div>${formatInteger(item.completion_chars)}</div>
                </div>
              </div>
            `).join("")}
          </div>
        ` : "";
        setStatus(`LLM 用量 ${calls} 次`);
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
        if (!items.length) {
          renderEmpty("pendingList", "暂无待确认 FACT 或冲突记忆。");
          updatePendingSelection();
          return;
        }
        document.getElementById("pendingList").innerHTML = items.map((item) => {
          const itemType = item.item_type || "memory";
          return `
          <div class="item">
            <div class="pending-title">
              <input
                class="pending-check"
                type="checkbox"
                data-item-id="${Number(item.id)}"
                data-item-type="${escapeHtml(itemType)}"
                onchange="updatePendingSelection()"
                aria-label="选择 pending #${escapeHtml(item.id)}"
              />
              <span class="pill">#${item.id}</span>
              <span class="pill">${escapeHtml(itemType)}</span>
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
              <button class="primary" onclick="managePending('${escapeHtml(itemType)}', ${Number(item.id)}, 'approve')">批准</button>
              <button class="danger" onclick="managePending('${escapeHtml(itemType)}', ${Number(item.id)}, 'reject')">拒绝</button>
              <code>${escapeHtml(item.approve_command)}</code>
              <button onclick="copyText('${escapeHtml(item.approve_command)}')">复制批准</button>
              <code>${escapeHtml(item.reject_command)}</code>
              <button onclick="copyText('${escapeHtml(item.reject_command)}')">复制拒绝</button>
            </div>
          </div>
        `;
        }).join("");
        updatePendingSelection();
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
        await Promise.all([
          loadPersona(),
          loadUsers(),
          loadMessages(),
          loadStickers(),
          loadLlmUsage(),
          loadQaBlocks(),
          loadPending(),
        ]);
        setStatus("数据已更新");
      } catch (error) {
        showError(error);
        setStatus("读取失败");
      }
    }
    loadAll();
"""
