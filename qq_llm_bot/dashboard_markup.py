from __future__ import annotations

DASHBOARD_HTML_PREFIX = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QQ LLM Bot 看板</title>
  <style>
"""

DASHBOARD_HTML_AFTER_STYLE = r"""  </style>
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
      <button class="tab" data-tab="llmUsage">LLM 用量</button>
      <button class="tab" data-tab="qaBlocks">QA 拦截</button>
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
          <button id="selectAllPendingBtn">全选</button>
          <button id="clearPendingSelectionBtn">清空选择</button>
          <button class="primary" id="bulkApprovePendingBtn">批量批准</button>
          <button class="danger" id="bulkRejectPendingBtn">批量驳回</button>
          <span id="pendingSelectionText" class="muted selection-count">已选 0 条</span>
        </div>
        <div id="pendingList"></div>
      </div>

      <div id="qaBlocks" class="section">
        <h2>Final QA 拦截归档</h2>
        <div class="toolbar">
          <label>群号<select id="qaBlocksGroup"></select></label>
          <label>发言人<input id="qaBlocksUser" placeholder="QQ ID，可留空" /></label>
          <label>开始日期<input id="qaBlocksDateFrom" type="date" /></label>
          <label>结束日期<input id="qaBlocksDateTo" type="date" /></label>
          <label>数量<input id="qaBlocksLimit" type="number" value="100" min="10" max="500" /></label>
          <button class="primary" id="loadQaBlocksBtn">查询</button>
        </div>
        <div id="qaBlocksList"></div>
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

      <div id="llmUsage" class="section">
        <h2>LLM token 用量</h2>
        <div class="toolbar">
          <label>时间范围
            <select id="llmUsageHours">
              <option value="1">最近 1 小时</option>
              <option value="6">最近 6 小时</option>
              <option value="24" selected>最近 24 小时</option>
              <option value="168">最近 7 天</option>
              <option value="720">最近 30 天</option>
            </select>
          </label>
          <label>明细数量<input id="llmUsageLimit" type="number" value="100" min="10" max="500" /></label>
          <button class="primary" id="loadLlmUsageBtn">刷新</button>
        </div>
        <div id="llmUsageSummary"></div>
        <div id="llmUsageByPurpose"></div>
        <div id="llmUsageRecent"></div>
      </div>
    </section>
  </main>
  <script>
"""

DASHBOARD_HTML_SUFFIX = r"""  </script>
</body>
</html>
"""
