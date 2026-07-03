# QQ LLM Bot

这是一个 `NoneBot2 + OneBot V11` 的本地 QQ 群机器人骨架，用于连接已经部署好的 NapCat WebSocket，并在本地处理群消息、白名单、管理员指令、三态参与策略和后续 LLM agents。

## 快速开始

1. 修改 [config.toml](config.toml)：
   - `napcat.ws_url`：NapCat OneBot V11 正向 WebSocket 地址。
   - `napcat.access_token`：NapCat 端配置的 token。
   - `bot.admin_ids`：管理员 QQ ID。
   - `bot.enabled_groups`：允许生效的群号。

2. 安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

3. 启动：

```powershell
python bot.py
```

4. 打开本地看板：

```text
http://127.0.0.1:8080/dashboard
```

## 管理指令

默认支持 `#bot ...` 和 `/bot ...`：

```text
#bot status
#bot mode silent
#bot mode passive
#bot mode active
#bot whitelist list
#bot whitelist add <group_id>
#bot whitelist remove <group_id>
#bot admin list
#bot admin add <qq_id>
#bot admin remove <qq_id>
#bot memory lexicon [term]
#bot memory pending
#bot memory conflicts
#bot memory approve <memory_id>
#bot memory reject <memory_id>
#bot facts user <qq_id>
#bot facts pending
#bot facts approve <fact_id>
#bot facts reject <fact_id>
#bot profile <qq_id>
#bot stickers list [数量]
#bot stickers enable <sticker_id>
#bot stickers disable <sticker_id>
#bot stickers delete <sticker_id>
#bot relation <qq_id>
#bot persona show
#bot persona self
#bot persona self pending
#bot persona self conflicts
#bot why
#bot forget <memory_id>
#bot llm status
#bot llm test
```

三种群状态：

- `silent`：只记录与观察，普通消息不回复。
- `passive`：仅在被 @、被叫昵称、或被明确提及时回复。
- `active`：允许主动参与，但受冷却、最近发言频率、话题相关性和 LLM 决策共同约束。

## 当前落地范围

- 已接入 NoneBot2 + OneBot V11 正向 WebSocket 配置。
- 已实现管理员与群白名单配置。
- 已实现群状态机：`silent | passive | active`。
- 已实现群消息 SQLite 记录、上下文检索、成员 FACT、全局成员画像、自我记忆、群复盘和关系记录。
- 已实现结构化 LLM agents pipeline：感知、FACT 抽取、关系变化、参与决策、回复生成、回复后自我记忆账本。
- 已实现保守记忆写入：低置信候选拒绝，冲突候选标记为 `conflict`，不会覆盖旧记忆。
- 已实现成员认知防投毒策略：本人 FACT 达阈值直接采信，第三方转述按信任度进入 `accepted` 或 `pending_confirmation`，管理员可批准或拒绝。
- 已实现受控词条学习：发现疑似网络用语、玩梗或圈层黑话时，可按配置联网搜索并形成 `group/lexicon` 记忆。
- 已实现自我叙事治理：机器人可生成轻量虚构的自我偏好、习惯和小经历，但必须先通过一致性检查并写入 `self` 记忆后再引用。
- 已实现本地 Web 看板：按 tab 查看机器人自我设定、成员 FACT/画像与关系、群聊记录、待确认/冲突记忆。
- 已实现多模态识图：可解析 OneBot 图片消息，调用支持 `image_url` 的 OpenAI-compatible 模型生成图片摘要、OCR 和保守群记忆。
- 已实现表情包学习与发送：可识别聊天图片中的表情包/梗图，分析适合使用的场景，下载保存到本地，并在回复时保守选择合适表情发送。

## 本地看板

启动机器人后访问：

```text
http://127.0.0.1:8080/dashboard
```

看板包含：

- `自我设定`：展示稳定/当前人设和机器人自己的 self memory。
- `成员认知`：按群号和 QQ ID 查询成员全局画像、accepted FACT、亲近、信任、熟悉、紧张和关系摘要。
- `群聊记录`：按群号、发言人、开始日期、结束日期筛选已入库消息。
- `表情包`：查看现存可使用表情包、触发条件和复制删除命令。
- `Pending`：展示 `pending_confirmation` 和 `conflict` 记忆，并生成可直接复制到 QQ 群里的确认命令。

网页不会直接批准或拒绝 pending，仍需要管理员在 QQ 群里发送命令，例如：

```text
#bot memory approve <memory_id>
#bot memory reject <memory_id>
#bot facts approve <fact_id>
#bot facts reject <fact_id>
#bot persona self approve <memory_id>
#bot persona self reject <memory_id>
```

配置项：

```toml
[dashboard]
enabled = true
route_prefix = "/dashboard"
api_prefix = "/api/dashboard"
access_token = ""
access_token_env = "QQ_LLM_BOT_DASHBOARD_TOKEN"
```

如果设置了 `access_token` 或 `.env` 中的 `QQ_LLM_BOT_DASHBOARD_TOKEN`，访问时使用：

```text
http://127.0.0.1:8080/dashboard?token=你的token
```

成员 FACT 与画像阈值：

```toml
[facts]
fact_confidence_threshold = 0.75
third_party_trust_threshold = 70
third_party_confidence_threshold = 0.85
profile_fact_threshold = 5
```

## 接入 OpenAI-compatible LLM

1. 在 [config.toml](config.toml) 中修改：

```toml
[llm]
provider = "openai-compatible"
model = "你的模型名"
base_url = "https://你的 API 地址/v1"
api_key_env = "OPENAI_API_KEY"
temperature = 0.8
max_tokens = 256
timeout_seconds = 30.0
```

2. 新建 `.env`，写入：

```text
OPENAI_API_KEY=你的 API key
```

3. 启动后在管理员所在群测试：

```text
#bot llm status
#bot llm test 用一句话打个招呼
```

LLM 当前用于回复生成、感知分析、成员 FACT 抽取、全局画像聚合、关系变化、主动参与决策和周期性群复盘。所有这些 agents 都要求结构化 JSON；解析失败时会安全降级为启发式观察或保守回复。

## 图片理解

如果 `[llm]` 中配置的模型支持 OpenAI-compatible 多模态 `image_url` 输入，可以启用：

```toml
[vision]
enabled = true
model = "" # 留空则复用 [llm].model
max_images_per_message = 3
detail = "low"
timeout_seconds = 45.0
remember_threshold = 0.78
```

图片处理流程：

- 从 OneBot V11 `image` segment 提取 `url/file/summary` 并落库。
- 单条消息图片超过 `max_images_per_message` 时只抽样理解，抽样包含第一张和最后一张。
- 调用视觉模型输出图片描述、OCR 文本、话题、图片类型和可选群记忆。
- 非表情包图片会形成成员 FACT：文字内容图记录“用户对此内容感兴趣”，纯图片记录“用户对这类图片感兴趣”。
- 相同图片 URL 的识图结果会进入本地缓存，重复表情包只调用一次视觉模型。
- 图片摘要会进入回复上下文，机器人能回答“这图是什么”之类的问题。
- 图片附件和视觉摘要会显示在看板的群聊记录里。

隐私策略默认保守：不做人脸身份识别，不猜测真实人物身份，不把身份证、手机号、住址、密码等敏感信息写入长期记忆。

## 表情包学习

表情包功能依赖图片理解。启用后，机器人会从群聊图片中识别明显的表情包、梗图或反应图，分析它适合表达的情绪和使用场景，并下载到本地：

```toml
[vision]
enabled = true

[stickers]
enabled = true
storage_dir = "data/stickers"
min_confidence = 0.72
selection_threshold = 0.68
max_context_stickers = 24
download_timeout_seconds = 20.0
max_download_bytes = 8388608
send_cooldown_seconds = 120
```

机器人只会在已经决定要回复时考虑附带表情；如果语境严肃、匹配度不够或处于冷却期，就只发文字。管理员可管理本群表情库：

```text
#bot stickers list [数量]
#bot stickers disable <sticker_id>
#bot stickers enable <sticker_id>
#bot stickers delete <sticker_id>
```

`delete` 会移除表情库记录，并删除本地保存的表情图片。

## 自我叙事治理

机器人可以逐渐形成自己的轻量人设，但遵循“先入账，再引用”：

- 可自动生成：`self_preference`、`self_hobby`、`self_habit`、`self_past_event`、`self_background`。
- 不自动生成：真实住址、学校、公司、亲属关系、恋爱关系、线下见面或现实行动能力。
- 稳定偏好和边界发生冲突时，新记忆会进入 `conflict`，不会覆盖旧记忆。
- 普通小经历可多条共存，避免所有经历都互相冲突。

查看和管理：

```text
#bot persona self
#bot persona self pending
#bot persona self conflicts
#bot persona self approve <memory_id>
#bot persona self reject <memory_id>
#bot persona self forget <memory_id>
```

## 词条联网学习

默认关闭真实联网搜索。需要启用时修改 [config.toml](config.toml)：

```toml
[lexicon]
enabled = true
provider = "duckduckgo" # disabled | duckduckgo | serper | searxng
min_interval_seconds = 300
max_terms_per_message = 1
max_results = 5
confidence_threshold = 0.78
timeout_seconds = 10.0
```

如果使用 `serper`，把 key 放到 `.env`：

```text
WEB_SEARCH_API_KEY=你的搜索 API key
```

词条会写入本群的 `lexicon` 记忆，可用 `#bot memory lexicon` 或 `#bot memory lexicon <term>` 查看。`silent` 模式下仍可学习，但不会因此发言。
