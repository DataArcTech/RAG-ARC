# Repository Guidelines

## Non-Negotiables（强约束）

**任何不符合软件工程规范的代码都不允许合入。**

- 禁止硬编码：配置/常量/开关/模型参数/阈值/路径等必须统一管理（`config/`、环境变量、配置文件），禁止“散落式”常量。
- 禁止不受控的全局状态：避免 `global` 变量/隐式单例/不可追踪的全局缓存；必须可配置、可测试、可复现。
- Prompt 必须可管理：禁止把 prompt 字符串散落在业务代码里；应集中在 `core/`/`config/` 的可维护位置并具备版本/复用策略。
- 禁止补丁/猴子补丁：不得为了“某次运行成功”临时打补丁；尤其**禁止滥用正则**等脆弱手段掩盖根因。
- 禁止“掩耳盗铃式”fallback：fallback 不能跳过主逻辑或吞掉错误；必须对失败原因可观测、可定位。
- 控制复杂度：避免产生 >1000 行且缺乏可维护拆分的文件/模块；遵循分层与职责单一。
- 文档同步：更新代码后必须同步更新相应文档。
- 不允许降级实现：除非用户明确同意，否则不得以“功能降级/绕过核心逻辑”作为解决方案。
- 修复 Bug 的流程：先在沙箱环境写最小复现脚本/测试用例定位问题并稳定复现，再开始修复与回归验证。
- 在每次写代码前，先看下是否有任何skills可用。

## Project Structure

- `api/`: FastAPI HTTP API, routers, MCP servers (`api/mcp/`).
- `application/`: business workflows (RAG inference, knowledge, accounts).
- `core/`: core RAG building blocks (parsing/chunking, retrieval, rerank, prompts, utils).
- `encapsulation/` + `framework/`: adapters, shared abstractions, config plumbing.
- `config/`: JSON + Python configuration (`config/json_configs/` is the main entry point).
- `test/`: pytest suite (`test_*.py`).
- Runtime/artifacts: `data/`, `local/`, and `models/` (often large; don’t commit generated contents).

## Build, Test, and Development Commands

- Install deps (creates venv): `uv sync`
- Install dev deps (pytest, httpx): `uv sync --extra dev`
- Run API locally: `uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- Run CLI (Typer entrypoint): `uv run rag-arc --help`
- Run tests: `uv run pytest`
- Docker (recommended): `./build.sh` (build images) then `./start.sh` (start services) and `./stop.sh` (stop).

## Coding Style & Naming

- Python 3.11+ (see `pyproject.toml`); use 4-space indentation.
- Keep modules layered: `core/` should stay framework-agnostic; put service wiring in `application/`/`api/`.
- Prefer explicit types for request/response models (Pydantic v2) and stable, descriptive names.

## Testing Guidelines

- Framework: `pytest` + `pytest-asyncio` (async tests supported).
- Naming: `test_*.py` and `test_*` functions.
- Integration tests are opt-in and may require Postgres/Redis/Neo4j and model access; enable via env flags like `RUN_RAGARC_INTEGRATION_TESTS=1` (see `README.md`).

## Commits & Pull Requests

- Follow existing history prefixes: `feat: ...`, `fix: ...`, `optimize: ...` (imperative, concise subject).
- PRs should include: summary, how to run/verify (`uv run pytest` or specific test), and any config changes (`.env` / `config/json_configs/*`).

## Security & Configuration

- Never commit secrets from `.env`; update `.env.example` when adding new config knobs.
- Any new config knob must have a single source of truth (env/config file) and a documented default/behavior.
- Changing model/provider settings for Docker deployments typically requires rebuilding the image (`./build.sh`).
