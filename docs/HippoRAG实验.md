# HippoRAG 实验报告（图检索 + 全链路生成）

本文汇总 HippoRAG（`rag-arc-hipporag` / Neo4j pruned HippoRAG）在本次回归集上的实验参数与结果，并给出默认配置结论。

范围与原则
- 覆盖：`query -> rewrite -> (dense/bm25/hipporag) -> RRF -> rerank -> LLM` 以及“图路单独评测”。
- 不覆盖：DeepSearch 链路（DeepSearch 的有向图 deterministic tools 见 `docs/kg.md`）。
- 原则：不做领域/产品 hardcode；不做“对比/区别/vs”正则特化；所有门控/阈值必须可配置、可观测、可回滚。

回归集
- Q1–Q5：来自 `local/1.md`
- MH1–MH4：来自 `local/scripts/multihop_recall_eval.py`（用于验证图在“多文件/多约束”上的价值）

本机实验上下文（用于复现）
- owner_id：`2a16b821-0e49-44c7-a5bb-96fd141f7772`
- KG schema：`KG_SCHEMA_PATH=./fin_kg_schema.yml`
- 模型：以 `.env` 中 OpenAI-compatible endpoint 为准（本轮未使用本地模型）

---

## 1. 对比对象与实验变量

图检索对比对象（命名约定）
- `upstream(full_graph_ref)`：上游 HippoRAG “全图 + 无向 PPR” 的参考实现（仅用于离线对比，**不作为线上依赖**）
- `rag-arc(pruned_base)`：本仓库 Neo4j pruned HippoRAG 的旧默认行为（未做候选补全、reset 聚合=overwrite、PPR 终止阈值不做 degree-normalize、damping=0.5、directed=auto）
- `rag-arc(pruned_optimized)`：本仓库 Neo4j pruned HippoRAG 的新默认行为（见第 3 节）

核心实验变量（图路）
- 子图候选是否“补全”（dense file closure）
- reset 权重聚合：`overwrite` vs `sum_avg`
- PPR 近似算法：push-PPR 的阈值策略（hub-bias 抑制）
- damping_factor（topic-sensitivity）
- 有向 PPR：`ppr_directed_mode=off/auto/on`

---

## 2. 图检索（graph-only）实验结论

2.1 关键发现：有向 PPR 对检索是负收益
- 在本回归集上，`ppr_directed_mode=auto/on` 会显著增加漂移/漏召回，且带来额外开销。
- 因此：**通用检索默认使用无向 PPR（`ppr_directed_mode=off`）**；方向性能力保留给 DeepSearch / fast graph tools 的确定性图操作（见 `docs/kg.md`），而不是默认注入到 PPR 随机游走里。

2.2 pruned 的核心短板：候选集不全
- pruned 方案的 `_expand_subgraph()` 是成本友好的，但会把“同一目标文件里的其他关键 chunks”裁剪掉。
- 结果是：即使 PPR 参数再好，也无法把“不在子图里的 chunk”排进 topK。

2.3 通用补救：dense file closure（候选补全）
- 当 dense topK 高度集中在某一个文件上时，把该文件的 chunks 注入 PPR 子图（仍然由 PPR 排序，非直接返回 dense）。
- 该策略只依赖“文件集中度”，不依赖领域词表/正则。

2.4 经典 PPR 优化：抑制 hub 漂移 + 提高 topic-sensitivity
- 使用 degree-normalized 的 push 阈值（降低高 degree 节点吞噬概率质量）
- 更小的 damping_factor（更强的个性化重启）

2.5 聚合结果（来自本机图路 sweep 汇总）
- 在 `dense_file_prior=off` 的隔离条件下，`pruned+closure+sumavg+push_deg+damp0.3` 的聚合指标最好：`avg_hit30=17.33`
- 对照：
  - `upstream(full_graph_ref)`：`avg_hit30=11.22`
  - `rag-arc(pruned_base)`：`avg_hit30=9.00`

注：详细 per-question/per-variant 数据来源于本轮生成的（旧）多份报告与 json trace，已在第 5 节统一归档复现方式；本文件只保留结论与关键对比点。

---

## 3. 已合入的默认算法改动（以及如何关闭）

本节对应 `docs/HippoRAG图检索实验结论.md` 的 “## 3. 现象解释” 中可泛化改动，已作为**默认行为**合入 Neo4j pruned HippoRAG 检索（可在 config 里关闭/回滚）。

3.1 默认关闭有向 PPR（检索阶段）
- `ppr_directed_mode` 默认改为 `off`
- 原因：本回归集上 directed PPR 是负收益；方向敏感的需求应由 DeepSearch / deterministic graph ops 处理（见 `docs/kg.md`）

3.2 默认启用 dense file closure（候选补全）
- `dense_file_closure_enabled=True`
- 门控参数：`dense_file_closure_top_k / min_ratio / min_margin / max_chunks`

3.3 reset 权重聚合更接近 upstream
- `entity_reset_weight_aggregation=sum_avg`

3.4 push-PPR 默认开启 hub-bias 抑制
- `ppr_backend=push`
- `ppr_push_threshold_mode=residual_over_degree`
- `ppr_push_target_degree_penalty_gamma=0.5`

3.5 damping_factor 默认调低
- `damping_factor=0.3`（更 topic-sensitive，减少扩散漂移）

配置位置（JSON）：
- HippoRAG Q&A：`config/json_configs/rag_inference.json`
- DeepSearch graph adapter：`config/json_configs/deepsearch_service.json`

配置位置（Config 类默认值）：
- `config/core/retrieval/pruned_hipporag_neo4j_config.py`

---

## 4. 全链路（含 LLM 生成）补充验证

说明：此前多数实验只评估“召回/覆盖”；本节补齐“包含 LLM 生成”的对比，以确认图路默认改动不会把错误证据带进答案。

4.1 实验脚本（可复用、可配置）
- `scripts/hipporag/experiments/rag_inference_eval.py`

4.2 本机运行记录（示例）
- multipath（dense+bm25+graph）对比：`local/runtime/rag_inference_eval_all_densebm25_20260118_165944`
  - variants：`all+graph_base` / `all+graph_optimized` / `dense_bm25`
- graph-only（优化配置）：`local/runtime/rag_inference_eval_graph_only_opt_20260118_170503`

4.3 观察到的现象（摘要）
- 对单文件强指向问题（例如“保费折扣/保单贷款规则”这类条款定位），`dense_bm25` 与 `all+graph_*` 都能稳定把目标文件占满 top chunks，答案稳定引用正确文件。
- 对多文件/对比倾向问题（如 Q1、MH1、MH4），`all+graph_optimized` 通常能把两份目标文件更均衡地带入候选（file_distribution 更接近 1:1）。
- 但“多文件归因 + LLM 生成”仍存在风险：当候选池混入同类型但非目标文件的 chunks 时，LLM 可能会“挑错文件作对比对象”。这属于融合/证据约束问题，不应通过领域 hardcode 解决；后续建议见第 6 节。

补充：从本机输出抽样 5 条问题的“top 文件分布”与生成现象（仅作定性参考）
- Q1（对比类）：`all+graph_optimized` 的 top_files 更均衡（约 5/5），`graph-only` 容易塌缩到单文件并导致“无法比较”的回答。
- Q4/Q5（单文件条款）：`dense_bm25`、`all+graph_*`、`graph-only` 基本都能把目标文件占满 top chunks，生成稳定。
- MH3（单文件多约束）：`all+graph_*`、`dense_bm25` 可稳定召回并生成；`graph-only` 在本轮出现误判（回答“没有折扣/贷款规则”）。
- MH4（多文件归因）：`dense_bm25` 往往倾向“只回答单文件并声明缺失”，而 `all+graph_*`/`graph-only` 更容易引入“错误的第二文件”（需要后续的通用证据治理来约束）。

---

## 5. 复现方式（命令清单）

图路召回对比（graph-only）
- `uv run python local/scripts/hipporag_graph_recall_compare.py --owner-id <OWNER_ID> --k 30`

多跳倾向召回评测（graph value probe）
- `uv run python local/scripts/multihop_recall_eval.py --owner-id <OWNER_ID> --k 30`

全链路（rewrite+retrieve+rerank+LLM）评测
- `uv run python scripts/hipporag/experiments/rag_inference_eval.py --owner-id <OWNER_ID> --questions-path <PATH>`

---

## 6. 结论与后续工作

6.1 结论（本轮可落地）
- 默认无向 PPR 是正确选择：更稳、对回归集更一致。
- pruned 方案仍有必要：相比 full-graph，成本更可控；通过“候选补全 + 去 hub + 更 topic-sensitive”可在不切全图的前提下显著提升图路覆盖。

6.2 后续（不在本次合入范围，但建议立项）
- “图路是否参与融合”的通用门控：应基于可观测信号（例如候选文件的 query-title 词面覆盖、provenance-groundability、一致性评分），而不是句式/正则硬编码。
- 对多文件问题的证据约束：LLM 侧应被要求“对比对象必须来自不同 source_file_id 且都在 top evidence 中”，否则降级为单文件回答并解释缺失原因（这是可泛化的证据治理，不是领域 hardcode）。

---

## 7. 提交拆分建议（本次工作区）

建议将本次工作拆成 7 次提交（按影响面递增，便于 review/回滚）：
1) `fix: keep rag_inference.chat backward-compatible`
2) `fix: make tantivy bm25 index loading resilient (meta.json guard)`
3) `fix: normalize graph chunk ids and clean dangling RELATES_TO provenance`
4) `feat: query variants defaults + docs + tests (zh-Hans/en/zh-Hant)`
5) `feat: stabilize pruned hipporag retrieval defaults (undirected PPR + closure + sum_avg + hub-bias controls)`
6) `docs: consolidate hipporag docs (diagnosis + experiments) and document directed-PPR rationale`
7) `chore: promote useful hipporag scripts into scripts/ (experiments vs maintenance)`
