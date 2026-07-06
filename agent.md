# Agent Notes

本文件给后来接手本仓库的 Codex/agent 使用。先读这里，再跑测试或清理临时文件。

## 项目与常用命令

- 这是 Python 3.10+ 项目，依赖和测试工具在 `pyproject.toml` 里。
- 默认在仓库根目录 `E:\vibe\qq-llm-bot` 运行命令。
- 跑测试、ruff 和项目 Python 命令必须使用项目内虚拟环境 `.\.venv\Scripts\python.exe`，不要直接用系统 `python`：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

- 如果虚拟环境缺依赖，安装开发依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- 如果 `.venv` 不存在，不要直接用系统 Python 跑测试；先创建项目内虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- PowerShell 里 extras 必须加引号：`".[dev]"` 或 `".[dev]"` 等价形式。

## 沙箱/权限注意

- 当前 Codex 桌面环境通常不需要、也不能申请提权；运行 `exec_command` 时不要传 `sandbox_permissions`。
- 如果遇到权限错误，不要第一反应申请更高权限。先检查是不是 Windows 文件锁、测试进程未退出、SQLite 临时文件未释放，或命令误删/误写了受保护目录。
- 手工改文件时用 `apply_patch`。不要用 shell 重定向、`cat > file`、PowerShell here-string 等方式直接写项目文件。

## 测试临时目录

测试大量使用 `tempfile.TemporaryDirectory()` 和 SQLite。不要让临时文件落到 Windows 系统临时目录；测试进程会通过 `tests/conftest.py` 自动把 Python 临时目录指到仓库内已忽略的 `.tmp`。

跑 pytest 之外的脚本前，也默认先设置这些环境变量：

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TMP = (Resolve-Path .tmp).Path
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMPDIR = (Resolve-Path .tmp).Path
.\.venv\Scripts\python.exe -m pytest
```

同一个 PowerShell 会话里后续的 `python`/`pytest` 会继承这些环境变量。开新终端后要重新设置一次。新增测试或脚本时，如果需要显式创建临时目录，优先使用项目内 `.tmp`：

```python
tmp_root = Path.cwd() / ".tmp"
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
    ...
```

现有测试里的临时目录统一通过 `tests/test_cognitive_loop.py` 的 `_project_temp_directory()` 创建。`test_config()` 也会把相对 SQLite 路径重定向到 `.tmp`，避免 `Path("unused.sqlite3")` 之类的占位路径意外落到仓库根目录。

注意：历史测试中曾经有 `TemporaryDirectory(dir=Path.cwd())`，失败或中断时可能在仓库根目录留下旧的 `tmp*` 目录。它们通常是测试残留，不是源码；新测试不要再这样写。

安全清理流程：

```powershell
$root = (Get-Location).Path
Get-ChildItem -Directory -Filter "tmp*" | ForEach-Object {
    $full = $_.FullName
    if ($full.StartsWith($root + [IO.Path]::DirectorySeparatorChar)) {
        Remove-Item -LiteralPath $full -Recurse -Force
    }
}
```

清理前先确认这些目录确实是测试残留。不要删除 `data/`、`.venv/`、`.git/`、`.codex/` 或用户明确保留的目录。

## SQLite/数据库文件

- 正常测试应使用临时库：`test_config(Path(tmp) / "bot.sqlite3")` 或内存库 `:memory:`。
- 不要让测试写真实运行库 `data/bot.sqlite3`，也不要依赖本地 `config.toml`。
- 新增存储相关测试时，模式是：

```python
with tempfile.TemporaryDirectory() as tmp:
    config = test_config(Path(tmp) / "bot.sqlite3")
    storage = BotStorage.from_config(config)
    storage.setup()
```

- `Path("unused.sqlite3")` 只适合不会实际打开数据库的纯逻辑测试。如果测试会调用 `BotStorage.from_config(...).setup()`，必须换成临时目录里的真实路径。
- 如果出现 `PermissionError`、`database is locked`、`WinError 32`，先确认没有仍在运行的 pytest/bot/server 进程持有文件。等进程退出后，只清理 `.tmp/` 或 `tmp*/` 下的测试 SQLite 文件及其 `-wal`、`-shm`，不要动 `data/`。

## 推荐验证节奏

- 改动较小时先跑目标测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cognitive_loop.py -q -x
```

- 需要缩小范围时用 `-k`：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cognitive_loop.py -q -k "dashboard or stickers"
```

- 收尾再跑：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

## 编码与输出

仓库里有大量中文文本。PowerShell 默认编码有时会把中文显示成乱码，但这不一定表示文件损坏。

读取中文文件时优先显式 UTF-8：

```powershell
Get-Content -Raw -Encoding UTF8 README.md
```

如果测试输出中文乱码，可在同一个 PowerShell 会话里先设置：

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 不需要真实外部服务

- 单元测试使用 `FakeLLM`、`FakeSearch` 等桩对象，不应要求真实 OpenAI key、NapCat 连接或联网。
- 不要为了跑测试修改 `.env`、`config.toml` 或真实机器人配置。
- FastAPI/Pillow 相关测试在依赖缺失时可能跳过；若目标改动涉及这些路径，应先安装 `".[dev]"` 和项目依赖再验证。

## Git/忽略规则

`.gitignore` 已忽略 `.tmp/`、`.pytest_cache/`、`.ruff_cache/`、`.venv/`、`data/`、SQLite 文件、`.env` 和本地配置。提交前检查：

```powershell
git status --short
```

只提交源码、测试和文档。不要提交本地数据库、缓存、临时目录、`.env`、`config.toml`。
