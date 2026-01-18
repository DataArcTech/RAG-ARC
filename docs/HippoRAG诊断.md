# HippoRAG 诊断与修复记录（rag-arc-hipporag / Neo4j 图检索）

本文档按 “问题发现 -> 问题复现 -> 问题解决” 的方式，记录本次 `rag-arc-hipporag`（Neo4j 图检索）召回不稳的定位过程与修复点。

范围与约束
- 仅覆盖：query -> rewrite -> 三路检索（faiss/bm25/hipporag）-> RRF -> rerank -> RAGQA
- 不覆盖：deepsearch 链路
- 原则：通用领域通用优化；不为 “对比/区别/vs” 句式做正则特化；所有开关/阈值必须配置化且可观测
 - 说明：检索阶段的 HippoRAG PPR 默认使用无向模式（`ppr_directed_mode=off`），方向敏感能力保留给 deepsearch/fast graph tools（见 `docs/kg.md`）

关键上下文（本次问题）
- owner_id：`2a16b821-0e49-44c7-a5bb-96fd141f7772`
- 回归问题：`local/1.md`（Q1~Q5）
- 代表性目标文件（示例）：`保誠信守明天多元貨幣計劃-产品小册子.pdf`
- 代表性 source_file_id（示例）：`2217c878-cfdc-41b7-8460-f80201736308`

--------------------------------------------------------------------------------
1. 问题发现（Symptoms）

用户反馈的典型现象（抽象后、通用领域）
- 单文件强指向问题：dense/bm25 能命中目标文件，但图检索（HippoRAG）命中不稳定，甚至 0-hit。
- 图路命中错文件时，会把错误 chunk 混入融合候选池，导致 rerank/生成阶段出现“引用存在但来源不对/证据不全”。

--------------------------------------------------------------------------------
2. 问题复现（Reproducible Experiments）

2.1 全链路复现：逐路打印三路检索 + 融合 + rerank
用于判断问题发生在：query rewrite / 某一路检索 / 融合 / rerank。

推荐脚本：
`uv run python local/scripts/debug.py --owner-id <OWNER_ID> --query '<QUERY>' --per-retriever --show-candidates --print-top 20`

重点观察项（必看）
- `rewritten_query`：是否发生意图漂移
- per-retriever 文件分布：dense/bm25/hipporag 各自 topK 是否合理
- fused candidates：RRF 后 top30 文件分布是否被噪声“挤掉主信号”
- rerank topK：最终 top10 是否稳定包含目标文件（单文件条款类应接近满命中）

2.2 图路单独复现：只跑 HippoRAG（避免被 dense/bm25 掩盖）
用于验证 “图路是否真的在拖累/增益”。

推荐脚本（仅图路）：
`uv run python local/scripts/hipporag_graph_recall_compare.py --owner-id <OWNER_ID> --k 30`

2.3 图索引一致性：索引路径/环境变量
先保证 indexing 与 retrieval 读取同一套目录，否则调参无意义。

检查 `.env` 或运行时环境变量（示例）
- `GRAPH_STORAGE_PATH=./local/data/graph_index_neo4j`
- `FAISS_INDEX_PATH=./local/data/unified_faiss_index`
- `BM25_INDEX_PATH=./local/data/unified_bm25_index`

2.4 图索引完整性自检：chunk embedding 覆盖率（本次的主因）
问题形态
- Neo4j 里 `(:Chunk)` 数据齐全，但本地 `index_chunk_embeddings.pkl` 覆盖不全；
- 图检索内部的 dense scoring 依赖该 embedding cache；一旦缺失，缺失 chunk 的向量会退化为全 0，导致：
  - dense topK “看不到”目标文件的 chunks
  - `dense_file_prior` 可能被错误的 top_file_id 诱导，进一步放大 PPR 漂移

自检（只看缺失，不写入）
`uv run python local/scripts/neo4j_backfill_chunk_embeddings.py --owner-id <OWNER_ID> --dry-run`

修复（先修数据再看算法）
`uv run python local/scripts/neo4j_backfill_chunk_embeddings.py --owner-id <OWNER_ID>`

--------------------------------------------------------------------------------
3. 问题定位（Root Causes）

3.1 图路 chunk embeddings 大量缺失（导致 dense 退化为 0 向量）
证据（owner 维度，修复前）
- owner 下 Neo4j chunks：8717
- `index_chunk_embeddings.pkl` 命中：4564
- 缺失：4153（缺失 chunk 在图检索时会被补成全 0 向量）

进一步证据（单文件，修复前）
- 目标文件 `2217c878-...` 在 Neo4j 中 chunks=161
- 缓存命中=0（即这份文件在图检索 dense scoring 中“不可见”）

根因（工程问题，不是算法问题）
- 图索引增量更新时 `chunk.id` 的类型/序列化不一致（UUID 对象 vs str），导致：
  - `batch_generate_embeddings(chunk_ids=...)` 传入的 chunk_id 列表与 Neo4j 中 `Chunk.chunk_id`（字符串）无法匹配
  - embedding 没生成、没落盘，检索侧加载旧的 `index_chunk_embeddings.pkl` 后出现大规模缺失

修复（代码）
- `encapsulation/database/graph_db/pruned_hipporag_neo4j_indexing_ingest.py`：统一把 `chunk_id` 写入 Neo4j 前强制转为非空字符串（并复用到 mention/fact provenance）。
- `encapsulation/database/graph_db/pruned_hipporag_neo4j_indexing_ops.py`：`update_index()` 里把 `new_chunk_ids` 归一化为字符串。
- `encapsulation/database/graph_db/pruned_hipporag_neo4j_embeddings.py`：`batch_generate_embeddings()` 对传入的 chunk_ids/entity_ids 做字符串归一化。
- 回归测试：`test/test_pruned_hipporag_update_index_normalizes_chunk_ids.py`

3.2 `dense_file_prior` 的门控缺陷会放大漂移（尤其在 embedding 缺失/错误时）
现象
- 仅靠 dense topK 的 `top_ratio` 门控会出现 “错误但很集中” 的情况；
- 一旦 prior 选错文件，会直接拉偏 PPR reset，导致图路系统性漂移。

修复（通用、无领域 hardcode）
- 增加基于文件名标题（`title=`）的词法一致性门控（CJK bigram + ASCII token）。
- 增加 `dense_file_prior_lexical_min_top_ratio`：只在 top_ratio 已较高时才启用词法门控，避免误伤多文件问题。

相关代码/配置
- `core/retrieval/graph_retrieveal/dense_file_prior.py`
- `core/retrieval/graph_retrieveal/pruned_hipporag_neo4j_graph.py`
- `config/core/retrieval/pruned_hipporag_neo4j_config.py`

3.3 Chunk 删除/重建时 provenance 悬挂（`RELATES_TO.source_chunk_ids`）
问题
- `RELATES_TO.source_chunk_ids` 是 append-only 的证据列表；若 chunk 被删/重建而不清理，会累积 dangling chunk_id，影响图重建/分析，并可能影响检索侧 groundability/置信度。

修复（代码 + 单测）
- `encapsulation/database/graph_db/pruned_hipporag_neo4j_indexing_ops.py`：`delete_chunks()` 删除 chunk 后清理 `RELATES_TO.source_chunk_ids`
- `test/test_pruned_hipporag_delete_chunks.py`

--------------------------------------------------------------------------------
4. 实验结果（2026-01-18）

4.1 修复前（embedding 缺失时）现象摘要
- dense/bm25：能命中目标文件（例如 Q4：Dense 16/30，BM25 30/30）
- graph：目标文件 0/30（因为该文件所有 chunks 的 embedding 缺失，dense scoring 退化）

4.2 修复后（owner 全量 backfill 后）图路稳定性恢复
运行（图路单独回归）：
`uv run python local/scripts/hipporag_graph_recall_compare.py --owner-id 2a16b821-0e49-44c7-a5bb-96fd141f7772 --k 30`

结果（摘要）
- Q1/Q2/Q3/Q4/Q5：图路均可稳定把目标文件打入 topK（可参考 `dense_top_file_ratio` 与 `top_files` 分布）。

本机输出留档（用于复查）
- `local/runtime/debug_runs_20260118_112929/graph_recall_post_backfill.txt`（先回填单文件）
- `local/runtime/debug_runs_20260118_113610/graph_recall_dense_on_lex_off.txt`（owner 全量回填后，dense_file_prior on）
- `local/runtime/debug_runs_20260118_114045/multihop_dense_prior_on.txt`（多跳倾向评测）

--------------------------------------------------------------------------------
5. 回归验证清单（Regression Checklist）

5.1 检索层（不看最终答案）
- 对 `local/1.md` 每条问句跑：
  - `local/scripts/debug.py --per-retriever --show-candidates`
  - 重点看：per-retriever 文件分布 / fused candidates / rerank topK

5.2 图索引健康检查（建议作为日常运维）
- owner 维度 chunk embedding 覆盖率必须接近 100%（否则图路 dense scoring 会系统性失真）
- 建议每次大规模重建/迁移后跑一次：
  - `local/scripts/neo4j_backfill_chunk_embeddings.py --owner-id <OWNER_ID> --dry-run`
