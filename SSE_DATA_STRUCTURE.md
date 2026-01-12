# SSE 返回数据结构文档

## 概述

当开启 DeepSearch (`enable_deepsearch=true`) 和联网搜索 (`enable_web_search=true`) 时，SSE 流式返回的数据结构如下。

## 通用 SSE 包装格式

所有 SSE 事件都使用 `sse_json_wrapped` 包装，格式为：

```json
{
  "code": 200,
  "message": "success",
  "data": { /* 实际数据 */ },
  "request_id": "uuid-string"
}
```

## 事件类型序列

### 1. 初始消息（Initial Message）

**格式：** OpenAI-compatible chat completion chunk

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "chatcmpl-xxx",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
      {
        "index": 0,
        "delta": {
          "role": "assistant",
          "content": "",
          "refusal": null
        },
        "finish_reason": null,
        "logprobs": null
      }
    ],
    "service_tier": null,
    "system_fingerprint": null,
    "usage": null
  },
  "request_id": "uuid-string"
}
```

---

### 2. DeepSearch 进度事件（DeepSearch Progress Events）

**事件类型：** `tool_calls` with `rag_arc_progress`

#### 2.1 created（初始化）

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "chatcmpl-xxx",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
      {
        "index": 0,
        "delta": {
          "tool_calls": [
            {
              "index": 0,
              "id": "call_deepsearch_progress_xxx",
              "type": "function",
              "function": {
                "name": "rag_arc_progress",
                "arguments": "{\"stage\":\"deepsearch\",\"deepsearch_stage\":\"created\",\"status\":\"running\",\"message\":\"DeepSearch 初始化...\",\"run_id\":\"xxx\",\"config_fingerprint\":\"xxx\",\"v\":1,\"type\":\"progress\",\"ts_ms\":1234567890,\"request_id\":\"xxx\",\"seq\":1}"
              }
            }
          ]
        },
        "finish_reason": null,
        "logprobs": null
      }
    ]
  },
  "request_id": "uuid-string"
}
```

**arguments 内容：**
```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "created",
  "status": "running",
  "message": "DeepSearch 初始化...",
  "run_id": "xxx",
  "config_fingerprint": "xxx",
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 1
}
```

#### 2.2 planned（计划生成）

```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "planned",
  "status": "running",
  "message": "正在生成搜索计划...",
  "plan_steps_count": 4,
  "plan_steps": [
    {
      "step_id": "step-1",
      "description": "步骤描述",
      "channel": "graph",
      "metadata": {}
    }
  ],
  "plan_metadata": {
    "plan_id": "xxx",
    "mode": "xxx",
    "artifact_path": "local/deepsearch_runs/xxx_plan.json",
    "created_at": "2026-01-12T14:25:07Z"
  },
  "plan_id": "xxx",
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 2
}
```

#### 2.3 reasoned（图谱推理）

```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "reasoned",
  "status": "running",
  "message": "正在进行图谱推理...",
  "reasoning_trace": {
    "reasoning_steps": [...],
    "tool_results": [...],
    "evidences": [...]
  },
  "reasoning_steps": [...],
  "reasoning_steps_count": 5,
  "tool_results": [...],
  "tool_calls_count": 3,
  "last_tool": "graph_query",
  "evidences": [...],
  "evidence_count": 10,
  "completed_steps": 5,
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 3
}
```

#### 2.4 gap_evaluated（缺口检测）

```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "gap_evaluated",
  "status": "running",
  "message": "正在检测知识缺口...",
  "gap_result": {
    "should_trigger_external": true,
    "reason": "缺少实时信息"
  },
  "should_trigger_external": true,
  "gap_reason": "缺少实时信息",
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 4
}
```

#### 2.5 external_invoked（外部搜索）

```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "external_invoked",
  "status": "running",
  "message": "正在进行外部搜索...",
  "external_calls": [
    {
      "provider": "tavily",
      "query": "搜索查询",
      "results": [...]
    }
  ],
  "external_calls_count": 2,
  "total_calls": 2,
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 5
}
```

#### 2.6 reported（报告生成）

```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "reported",
  "status": "running",
  "message": "正在生成报告...",
  "report_payload": {
    "answer": "完整的报告答案...",
    "structured_report": {...},
    "references": [...]
  },
  "answer": "完整的报告答案...",
  "answer_length": 5000,
  "structured_report": {...},
  "references": [...],
  "references_count": 5,
  "evidence_count": 10,
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 6
}
```

#### 2.7 quality_gated（质量检查）

```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "quality_gated",
  "status": "running",
  "message": "正在进行质量检查...",
  "quality_gates": [...],
  "quality_gates_count": 2,
  "quality_passed": true,
  "should_iterate": false,
  "round": 1,
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 7
}
```

#### 2.8 done（完成）

```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "done",
  "status": "completed",
  "message": "DeepSearch 完成",
  "run_id": "xxx",
  "cost_telemetry": {
    "plan": 3000,
    "graph_reasoning_r1": 25000,
    "report_r1": 40000
  },
  "stage_history": [...],
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 8
}
```

#### 2.9 failed（失败）

```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "failed",
  "status": "failed",
  "message": "DeepSearch 执行失败",
  "errors": [
    {
      "error": "错误信息",
      "timestamp": "2026-01-12T14:25:07Z"
    }
  ],
  "run_id": "xxx",
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 9
}
```

---

### 3. RAG 处理进度事件（RAG Progress Events）

**事件类型：** `tool_calls` with `rag_arc_progress`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "chatcmpl-xxx",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
      {
        "index": 0,
        "delta": {
          "tool_calls": [
            {
              "index": 0,
              "id": "call_progress_xxx",
              "type": "function",
              "function": {
                "name": "rag_arc_progress",
                "arguments": "{\"stage\":\"prepare\",\"status\":\"start\",\"v\":1,\"type\":\"progress\",\"ts_ms\":1234567890,\"request_id\":\"xxx\",\"seq\":10}"
              }
            }
          ]
        },
        "finish_reason": null,
        "logprobs": null
      }
    ]
  },
  "request_id": "uuid-string"
}
```

**常见的 RAG 进度事件 stages：**
- `prepare` - 准备阶段
- `generate` - 生成阶段
- `web_search` - 联网搜索（如果启用）

**web_search 进度事件示例：**
```json
{
  "stage": "web_search",
  "status": "running",
  "message": "正在搜索网络...",
  "query": "搜索查询",
  "results_count": 5,
  "v": 1,
  "type": "progress",
  "ts_ms": 1234567890,
  "request_id": "xxx",
  "seq": 11
}
```

---

### 4. 内容流式输出（Content Streaming）

**事件类型：** OpenAI-compatible chat completion chunk with content delta

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "chatcmpl-xxx",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
      {
        "index": 0,
        "delta": {
          "content": "这是"
        },
        "finish_reason": null,
        "logprobs": null
      }
    ],
    "service_tier": null,
    "system_fingerprint": null,
    "usage": null
  },
  "request_id": "uuid-string"
}
```

**注意：** 内容会分多次发送，每次发送一小段文本。

---

### 5. 标题事件（Title Event）

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "type": "title",
    "title": "生成的对话标题"
  },
  "request_id": "uuid-string"
}
```

---

### 6. 来源事件（Sources Event）

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "type": "sources",
    "sources": [
      {
        "id": "chunk-id",
        "title": "文档标题",
        "description": "文档描述",
        "url": "文档URL（如果有）",
        "score": 0.95
      }
    ],
    "id": "session-id"
  },
  "request_id": "uuid-string"
}
```

---

### 7. 最终负载事件（Final Payload Event）

**事件类型：** `tool_calls` with `rag_arc_payload`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "chatcmpl-xxx",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
      {
        "index": 0,
        "delta": {
          "tool_calls": [
            {
              "index": 0,
              "id": "call_message-id",
              "type": "function",
              "function": {
                "name": "rag_arc_payload",
                "arguments": "{\"message\":{\"id\":\"xxx\",\"content\":{\"role\":\"assistant\",\"content\":\"完整回答\"},\"created_at\":\"2026-01-12T14:25:07Z\"},\"chunks\":[...],\"subgraph\":{...},\"evidence\":{...}}"
              }
            }
          ]
        },
        "finish_reason": null,
        "logprobs": null
      }
    ]
  },
  "request_id": "uuid-string"
}
```

**arguments 内容：**
```json
{
  "message": {
    "id": "message-id",
    "content": {
      "role": "assistant",
      "content": "完整的回答内容"
    },
    "created_at": "2026-01-12T14:25:07Z"
  },
  "chunks": [
    {
      "id": "chunk-id",
      "content": "chunk内容",
      "score": 0.95,
      "metadata": {}
    }
  ],
  "subgraph": {
    "nodes": [...],
    "edges": [...],
    "chunks": [...]
  },
  "evidence": {
    "chunks": [...],
    "entities": [...],
    "facts": [...]
  }
}
```

---

### 8. 结束事件（Final Chunk）

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "chatcmpl-xxx",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
      {
        "index": 0,
        "delta": {},
        "finish_reason": "stop",
        "logprobs": null
      }
    ],
    "service_tier": null,
    "system_fingerprint": null,
    "usage": null
  },
  "request_id": "uuid-string"
}
```

---

### 9. SSE 结束标记

```
data: [DONE]

```

---

## 完整事件序列示例

1. **初始消息** - 设置 assistant role
2. **DeepSearch created** - DeepSearch 初始化
3. **DeepSearch planned** - 计划生成
4. **DeepSearch reasoned** - 图谱推理（可能多次）
5. **DeepSearch gap_evaluated** - 缺口检测
6. **DeepSearch external_invoked** - 外部搜索（如果触发）
7. **DeepSearch reported** - 报告生成
8. **DeepSearch quality_gated** - 质量检查
9. **DeepSearch done** - DeepSearch 完成
10. **RAG prepare** - RAG 准备阶段
11. **RAG web_search** - 联网搜索（如果启用）
12. **RAG generate** - 生成阶段
13. **Content chunks** - 内容流式输出（多次）
14. **Title event** - 标题生成
15. **Sources event** - 来源信息
16. **Final payload** - 最终负载
17. **Final chunk** - 结束标记
18. **[DONE]** - SSE 结束

---

## 注意事项

1. **所有进度事件**都通过 `rag_arc_progress` tool call 发送
2. **DeepSearch 进度事件**的 `arguments` 是 JSON 字符串，需要解析
3. **内容流式输出**会分多次发送，每次一小段
4. **所有事件**都包含 `request_id` 用于追踪
5. **序列号 `seq`** 在进度事件中递增，用于排序

