## 意图路由（Semantic Router）

本目录实现 RAG-ARC 的「意图识别 / 意图路由」，底层使用第三方包
[`semantic-router`](https://github.com/aurelio-labs/semantic-router)（语义向量路由）。

意图路由位于对话链路最前面，用来决定：
- **intent**（稳定的意图标签），以及
- **action**：`rag` | `web_only` | `no_retrieval`

核心设计目标：
- **配置驱动**：意图/样例/阈值全部集中在 `config/core/intent_routing/intent_router.toml`，便于后期逐步完善。
- **嵌入解耦**：意图分类可使用与 RAG 不同的 embedding 模型/服务（本地小模型 vs 远程服务）。
- **支持多轮**：使用轻量 topic stack 识别 `topic_switch` / `return_to_topic`，并可做话题范围内的 history 裁剪。
- **Prompt 可管理**：与意图相关的系统提示词集中在 `core/prompts/intent_routing_prompts.py`，避免散落在业务代码中。

### 在链路中的位置

RAG 推理在最开始调用 `application.intent_routing.IntentRoutingService.route(...)`：
- `action == no_retrieval`：不检索，也不会调用 query rewrite 模型（query 原样使用），直接快速回答（对话模式）。
- `action == rag`：走内部检索/重排（并会 query rewrite）。
- `action == web_only`：走联网搜索（并会 query rewrite）；联网能力由请求参数控制。

### 当前意图分类（7 类）

由 `config/core/intent_routing/intent_router.toml` 定义：

1) `RAG_REQUIRED`（`rag`）
- “请根据我上传的资料回答这个问题，并给出处。”
- “对比 A 和 B 的区别是什么？请引用依据。”

2) `FOLLOWUP_RAG_REQUIRED`（`rag`，requires history）
- “回到刚才的主题，请再从资料里找一下证据。”
- “继续查资料，给我更准确的条款/数字并给出处。”

3) `WEB_ONLY`（`web_only`）
- “帮我网上查一下最新消息并给链接。”
- “今天天气咋样？”

4) `FOLLOWUP_NO_RAG`（`no_retrieval`，requires history）
- “详细展开说说。”
- “继续刚才那个话题，解释得更清楚一些。”

5) `NO_RETRIEVAL`（`no_retrieval`）
- “谢谢。”
- “不用查资料，直接说结论。”

6) `TASK_EXECUTION`（`no_retrieval`）
- “帮我生成一个 SQL 建表语句。”
- “把下面这段文本改写得更清晰：...”

7) `CLARIFY_REQUIRED`（`no_retrieval`）
- “这个是什么意思？”（缺上下文/关键信息）
- “帮我看看这个。”（缺关键条件）

8) `ANSWER_DISSATISFIED`（`no_retrieval`，requires history）
- “不对，你没回答我的问题。”
- “太泛了，给更具体的步骤。”

补充说明：
- `requires_history=true` 的意图（follow-up 类）在首轮会通过 semantic-router 官方 `route_filter` 自动排除。
- `enable_web_search` 由前端控制：当本次请求 `enable_web_search=false` 时，会通过 `route_filter` 排除 `WEB_ONLY`
  （能力 gating）；如果极端情况下仍命中 `WEB_ONLY`，业务上按普通 `rag` 处理。
- `ANSWER_DISSATISFIED` 会强制 topic 判定为 `same_topic`，避免反馈句触发无意义的 topic_switch。

### 多轮对话主题管理（仅 session 内）

topic stack 属于「派生状态」：
- 以 session_id 为 key 写入 Redis（TTL）。
- 缓存 miss 时，用最近 `max_user_messages` 条用户消息重建（仅使用 user 消息，不使用 assistant）。
- 通过 topic centroid 的余弦相似度判断：
  - `same_topic`
  - `topic_switch`
  - `return_to_topic`

相关参数在 `config/core/intent_routing/intent_router.toml` 的 `[intent_router.topic_stack]`。

### 新增/迭代意图的推荐流程

1) 在 `config/core/intent_routing/intent_router.toml` 增加新的 `[[intent_router.intents]]`：
- `name`：稳定、全大写（后续尽量不改名）
- `action`：`rag` / `web_only` / `no_retrieval`
- `utterances`：建议 10–30 条（尽量通用领域，不做场景特化）
- `threshold`：先用 `0.50` 起步，后续再拟合
- 若是“只在多轮出现”的意图，设 `requires_history=true`

2) 在 `config/core/intent_routing/intent_router_eval.toml` 增加标注样例（用于评估/拟合阈值）。

3) 用 semantic-router 官方阈值拟合/评估：
- `uv run python scripts/intent_routing/fit_intent_router_thresholds.py`

4) 如果这个意图需要强约束回答风格/策略，把 prompt 统一加到：
- `core/prompts/intent_routing_prompts.py`

5) 补充测试（建议至少覆盖）：
- 单 session：`bash test/e2e/intent_routing/e2e_intent_router_qwen_local.sh`
- 多用户并发：`INTENT_CONCURRENCY_USERS=16 INTENT_CONCURRENCY_WORKERS=8 bash test/e2e/intent_routing/e2e_intent_router_concurrency_qwen_local.sh`
- 全链路冒烟：`bash test/e2e/rag/e2e_rag_pipeline_test_pdf.sh`

### 本地意图 Embedding（Qwen）环境变量

意图 embedding 通过 TOML 占位符 + 环境变量配置（示例见 `.env.example`）：
- `INTENT_QWEN_EMBEDDING_MODEL_NAME`（例如 `Qwen/Qwen3-Embedding-0.6B`）
- `INTENT_EMBEDDING_DEVICE`（`cpu` / `cuda`）
- `INTENT_EMBEDDING_CACHE_FOLDER`（例如 `./models/Qwen`）
