# KG 增强 Phase 1 评审报告（commit `961c859` 及后续修复回归）

## 0. 范围与结论

本报告基于 `docs-proj/kg增强计划/` 文档约束与目标，对 `961c859 enhance kg phase 1` 的实现做工程评审，并在“真实服务 + 真实文档”环境下完成端到端验证与并发压测。整体结论：

- Phase 1 的核心交付（schema 治理、chunk‑triples 契约、fact provenance、定型的 Cypher 工具族、PPR 有向模式与缓存、导出语义增强、kg_ingest_stats 并发语义修复）与文档描述基本一致，且大多以配置/抽象层分隔落地，工程方向正确。
- 真实服务测试中发现 2 个会影响线上可用性/安全性的缺陷：DeepSearch 报告生成在 outline 阶段会 500、`/user/me` 会泄露 `hashed_password`。已补齐复现脚本并做了针对性修复，且 `uv run pytest` 全量通过。

---

## 1. 文档对齐检查（与 Phase 1 交付点逐项对照）

对照 `docs-proj/kg增强计划/README.md` 的 Phase 1 清单：

- Schema 治理
  - 位置：`core/knowledge_graph/schema.py`、`kg_schema.yml`
  - 评审：predicate 归一化 + allowlist + direction_sensitive 集合的设计合理，且 domain 化为 Phase 2 的离线流水线预留了接口。
- chunk‑triples 契约（precision‑first）
  - 位置：`encapsulation/database/graph_db/pruned_hipporag_neo4j_indexing.py`
  - 评审：端点强约束 + 歧义丢弃能有效降低“共现幻觉/实体混淆”风险，符合生产指南的“显式边验证”方向。
- Fact provenance（证据聚合 + 上限）
  - 位置：`encapsulation/database/utils/fact_provenance.py`
  - 评审：在 `RELATES_TO` 边上附加来源 chunk 证据聚合可追溯性显著提升；上限配置避免边属性无限膨胀是必要的工程约束。
- 方向敏感 PPR（+ 缓存）
  - 位置：`core/retrieval/graph_retrieveal/pruned_hipporag_neo4j_ppr.py`、`encapsulation/database/graph_db/pruned_hipporag_neo4j_cache.py`
  - 评审：`ppr_directed_mode=auto` 的 gating 逻辑与 schema direction_sensitive 的联动符合“方向性反转”类故障模式的治理思路。
- DeepSearch 确定性图算子（Neo4j Cypher）
  - 位置：`core/graph_adapter/cypher.py`、`core/deepsearch/tools/fast/graph_ops_*.py`
  - 评审：只读 Cypher guard + 工具族拆分清晰，便于后续在 production 指南 01–12 中逐项补齐确定性能力。
- 图导出语义增强
  - 位置：`encapsulation/database/utils/graph_export_utils_neo4j.py`
- kg_ingest_stats 并发语义修复（持久化）
  - 位置：`encapsulation/database/graph_db/pruned_hipporag_neo4j.py`、`encapsulation/database/graph_db/pruned_hipporag_neo4j_indexing.py`

---

## 2. 真实服务测试与覆盖范围

### 2.1 启动方式

- 依赖服务：已由用户通过 `./start.sh --services-only` 启动
- 主服务：`uv run uvicorn main:app --host 0.0.0.0 --port 8000`

### 2.2 测试数据

- 文档：`docs-proj/英文港险产品/` 下 4 个 PDF（均上传并完成索引）

### 2.3 覆盖与结果

- HippoRAG 问答（`/rag_inference/chat`）
  - 能基于上传的 PDF 正确回答“可选保单币种”等问题，并能返回 evidence/chunks。
- DeepSearch（`/deepsearch/run`）
  - 修复后可稳定完成计划 → 推理 → 报告输出，且 `tool_runs` 中包含工具调用记录。
- 用户相关功能（`/auth/register`、`/auth/token`、`/user/me`）
  - 修复后 `/user/me` 返回已脱敏字段，不再包含 `hashed_password`。
- 并发测试
  - 使用 `local/tmp/concurrency_stress_api.py` 进行并发压测：DeepSearch 异步任务与 HippoRAG chat 并行发起，均完成且无 5xx/超时。

---

## 3. 发现的问题、复现脚本与修复

### 3.1 DeepSearch 报告生成偶发 500（outline 阶段 schema invalid）

- 现象：`/deepsearch/run` 可能返回 500，错误形如 `Report outline generation failed ... outline_schema_invalid raw=[]`
- 根因（工程层面）：outline prompt 依赖 evidence index 的 id 字段，但证据索引的字段名与 prompt/契约存在不一致，且 outline gate 过严导致线上可用性受 LLM 输出波动影响。
- 修复：
  - `core/deepsearch/memory/evidence_bank.py`：evidence index 同时输出 `chunk_id` 与兼容字段 `evidence_id`
  - `core/deepsearch/report/llm_writer.py`：补齐 outline 阶段的“缺失 evidence_ids 自动填充”与“JSON schema 可解析但不合规时的可用性兜底 outline”
- 复现/验证脚本：`local/tmp/repro_deepsearch_report_outline_bug.py`

### 3.2 `/user/me` 泄露 `hashed_password`

- 现象：`/user/me` 返回 ORM 对象的完整字段，包含 `hashed_password`
- 风险：属于高风险信息泄露（即使是 hash，也不应对客户端暴露）
- 修复：`api/routers/user.py` 增加 `UserMeResponse`，只返回必要字段
- 复现/验证脚本：`local/tmp/repro_user_me_leaks_credentials.py`

---

## 4. 软件工程规范评审（优点与风险点）

### 4.1 做得好的地方（建议保留）

- 分层意识较强：`core/` 侧保持框架无关，Neo4j/FAISS 等实现放在 `encapsulation/`，符合仓库规范。
- Schema/工具/缓存均配置化：路径与开关多数有 config/env 驱动，为 Phase 2 的门禁与回归留出接口。
- 可测试性显著增强：Phase 1 引入了大量针对 schema、导出语义、PPR directed mode、deterministic tools 的测试用例（本次修改后 `uv run pytest` 全量通过）。

### 4.2 需要改进的工程风险点（建议纳入 Phase 2）

- **避免“全局副作用/猴子补丁”**：例如 `main.py` 对 `logging.Logger.addHandler` 的全局替换属于高风险操作，建议改为显式 logger/handler 装配。
- **启动期环境变量缺失的降级策略**：启动日志中仍可见多个 placeholder 未设置告警（如 `GRAPH_STORAGE_PATH` 等）。建议统一默认值策略（配置层提供默认，不在运行期散落 fallback）。
- **文本归一化真源统一**：已把 `core/utils/text_processing.py` 设为真源是正确方向，建议进一步排查旧路径（`encapsulation/...`）是否仍存在重复实现/语义漂移风险。

---

## 5. 改进方向（与文档 Phase 2 对齐的可执行建议）

- Schema 离线流水线与门禁（对齐 `docs-proj/kg增强计划/Schema构建指南-AutoSchemaKG-SHIELD-MFI.md`）
  - schema version/fingerprint 与缓存/索引重建联动
  - predicate coverage/rejection ratio、alias fragmentation、direction consistency、provenance cost 指标化
- DeepSearch 工具链可用性与能力声明
  - tool 与 adapter capability 对齐（避免“配置启用但 adapter 不支持”的隐性断裂）
  - 为每个 deterministic tool 建立最小可复现样例与回归集（覆盖生产指南 01–12）
- 生产回归集建设
  - 基于 `docs-proj/kg增强计划/知识图谱替代RAG生产指南-严谨版.md` 的 12 类故障模式，每类沉淀 5–20 条小样本 + ground truth + 自动验证脚本

---

## 6. 本次评审新增/修改文件清单（便于 code review）

- 修复：`core/deepsearch/memory/evidence_bank.py`
- 修复：`core/deepsearch/report/llm_writer.py`
- 修复：`api/routers/user.py`
- 复现脚本：`local/tmp/repro_deepsearch_report_outline_bug.py`
- 复现脚本：`local/tmp/repro_user_me_leaks_credentials.py`
- 并发脚本：`local/tmp/concurrency_stress_api.py`

---

## 7. 如何复现实测（快速命令）

- DeepSearch outline 修复验证：`uv run python local/tmp/repro_deepsearch_report_outline_bug.py`
- `/user/me` 脱敏验证：`uv run python local/tmp/repro_user_me_leaks_credentials.py`
- 并发压测：`RAGARC_TOKEN=... DS_CONCURRENCY=3 CHAT_CONCURRENCY=4 uv run python local/tmp/concurrency_stress_api.py`

