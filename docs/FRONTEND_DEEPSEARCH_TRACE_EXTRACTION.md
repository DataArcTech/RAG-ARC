# 前端提取 DeepSearch Trace Events 指南

## 概述

当启用 DeepSearch 时，后端会通过 SSE (Server-Sent Events) 流式返回 DeepSearch 的思考过程（trace events）。这些事件以 OpenAI 兼容的格式发送，前端需要解析并提取这些内容。

**参考文件**: 实际 SSE 输出示例请参考 `/test/sse.txt`

## SSE 事件格式

### 1. 标准响应包装

所有 SSE 事件都使用以下格式（注意：实际传输时以 `data: ` 前缀开头）：

```
data: {"code":200,"message":"success","data":{...},"request_id":"uuid-string"}
```

解析后的 JSON 结构：

```json
{
  "code": 200,
  "message": "success",
  "data": { /* 实际数据 */ },
  "request_id": "uuid-string"
}
```

### 2. DeepSearch Trace Event 格式（实际示例）

DeepSearch trace events 在 `data` 字段中包含 OpenAI chat completion chunk。以下是来自 `sse.txt` 的实际示例：

**实际 SSE 行**:
```
data: {"code":200,"message":"success","data":{"id":"chatcmpl-51549d350c2248ec8a6fb94f507cd69c","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_deepsearch_trace_4d71d913845c41e1952ba6a78e74a564","type":"function","function":{"name":"rag_arc_trace","arguments":"{\"tag\":\"think\",\"content\":\"<think>\\nPlanning the research workflow.\\nmode=react\\nmax_steps=8\\nexternal_channel_allowed=True\\n</think>\",\"meta\":{\"stage\":\"plan\",\"mode\":\"react\",\"max_steps\":8,\"external_channel_allowed\":true}}"}}]},"finish_reason":null,"index":0,"logprobs":null}],"created":1768275172,"model":"google/gemini-2.5-flash-preview-09-2025","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null},"request_id":"5de23e02-8562-4233-933d-ea05ac93232d"}
```

**解析后的结构**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "chatcmpl-51549d350c2248ec8a6fb94f507cd69c",
    "object": "chat.completion.chunk",
    "created": 1768275172,
    "model": "google/gemini-2.5-flash-preview-09-2025",
    "choices": [{
      "index": 0,
      "delta": {
        "tool_calls": [{
          "index": 0,
          "id": "call_deepsearch_trace_4d71d913845c41e1952ba6a78e74a564",
          "type": "function",
          "function": {
            "name": "rag_arc_trace",
            "arguments": "{\"tag\":\"think\",\"content\":\"<think>\\nPlanning the research workflow.\\nmode=react\\nmax_steps=8\\nexternal_channel_allowed=True\\n</think>\",\"meta\":{\"stage\":\"plan\",\"mode\":\"react\",\"max_steps\":8,\"external_channel_allowed\":true}}"
          }
        }]
      },
      "finish_reason": null
    }]
  },
  "request_id": "5de23e02-8562-4233-933d-ea05ac93232d"
}
```

**解析 `function.arguments` 后的内容**:
```json
{
  "tag": "think",
  "content": "<think>\nPlanning the research workflow.\nmode=react\nmax_steps=8\nexternal_channel_allowed=True\n</think>",
  "message": "正在生成搜索计划...",
  "meta": {
    "stage": "plan",
    "mode": "react",
    "max_steps": 8,
    "external_channel_allowed": true
  }
}
```

**注意**: `rag_arc_trace` 事件现在包含 `message` 字段，提供人类可读的过程描述，方便前端直接显示，无需解析 weaver 格式的 `content`。

### 3. DeepSearch Progress Event 格式（实际示例）

DeepSearch 进度事件使用类似的格式，但 function name 为 `rag_arc_progress`。以下是来自 `sse.txt` 的实际示例：

**实际 SSE 行**:
```
data: {"code":200,"message":"success","data":{"id":"chatcmpl-51549d350c2248ec8a6fb94f507cd69c","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_deepsearch_progress_3b9ae1c646c945739bac89632c97574a","type":"function","function":{"name":"rag_arc_progress","arguments":"{\"stage\":\"deepsearch\",\"deepsearch_stage\":\"reasoned\",\"status\":\"running\",\"message\":\"正在进行图谱推理...\",\"reasoning_trace\":{...},\"v\":1,\"type\":\"progress\",\"ts_ms\":1768275195462,\"request_id\":\"5de23e02-8562-4233-933d-ea05ac93232d\",\"seq\":3}"}}]},"finish_reason":null,"index":0,"logprobs":null}],"created":1768275172,"model":"google/gemini-2.5-flash-preview-09-2025","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null},"request_id":"5de23e02-8562-4233-933d-ea05ac93232d"}
```

**解析 `function.arguments` 后的内容**:
```json
{
  "stage": "deepsearch",
  "deepsearch_stage": "reasoned",
  "status": "running",
  "message": "正在进行图谱推理...",
  "reasoning_trace": { /* 详细的推理追踪信息 */ },
  "v": 1,
  "type": "progress",
  "ts_ms": 1768275195462,
  "request_id": "5de23e02-8562-4233-933d-ea05ac93232d",
  "seq": 3
}
```

## 前端提取步骤

### 步骤 1: 解析 SSE 事件

**重要**: SSE 事件以 `data: ` 前缀开头，需要先去除该前缀再解析 JSON。

```javascript
// 示例：使用 fetch 接收 SSE（EventSource 不支持 POST，推荐使用 fetch）
async function streamChat(sessionId, query, enableDeepSearch = true) {
  const response = await fetch(`/rag_inference/stream_chat/${sessionId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      enable_deepsearch: enableDeepSearch
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.slice(6); // 移除 "data: " 前缀
        try {
          const data = JSON.parse(jsonStr);
          handleSSEEvent(data);
        } catch (e) {
          console.error('Failed to parse SSE event:', e, jsonStr);
        }
      }
    }
  }
}

function handleSSEEvent(data) {
  // 检查响应格式
  if (data.code !== 200) {
    console.error('Error:', data.message);
    return;
  }
  
  // 提取实际数据
  const chunk = data.data;
  
  // 处理 trace 或 progress 事件
  if (chunk.choices && chunk.choices[0].delta.tool_calls) {
    chunk.choices[0].delta.tool_calls.forEach(toolCall => {
      if (toolCall.function.name === 'rag_arc_trace') {
        handleTraceEvent(toolCall);
      } else if (toolCall.function.name === 'rag_arc_progress') {
        handleProgressEvent(toolCall);
      }
    });
  }
}
```

### 步骤 2: 提取 Trace Event

```javascript
function handleTraceEvent(toolCall) {
  // 解析 arguments (JSON 字符串)
  const args = JSON.parse(toolCall.function.arguments);
  
  // 提取关键信息
  const traceInfo = {
    tag: args.tag,           // trace 类型: "think", "write_outline", "tool_call", "tool_response"
    content: args.content,   // weaver 格式的内容: "<tag>...</tag>"
    message: args.message,   // 人类可读的过程描述（新增字段）
    meta: args.meta,         // 元数据对象
    callId: toolCall.id      // 调用 ID（格式: "call_deepsearch_trace_xxx"）
  };
  
  // 可以直接使用 message 字段显示简要信息
  displayTraceMessage(traceInfo.message);
  
  // 如果需要详细信息，可以解析 weaver 格式的内容
  const weaverParsed = parseWeaverContent(args.content);
  
  // 根据 tag 类型处理
  switch (args.tag) {
    case 'think':
      displayThinking(weaverParsed, args.meta);
      break;
    case 'write_outline':
      displayPlanOutline(weaverParsed, args.meta);
      break;
    case 'tool_call':
      displayToolCall(weaverParsed, args.meta);
      break;
    case 'tool_response':
      displayToolResponse(weaverParsed, args.meta);
      break;
    default:
      console.log('Unknown trace tag:', args.tag, weaverParsed);
  }
}
```

### 步骤 3: 解析 Weaver 格式内容

Weaver 格式使用 XML-like 标签包装内容。注意：内容中的换行符是 `\n`（在 JSON 字符串中表示为 `\\n`）。

**实际示例 1 - think tag**:
```xml
<think>
Planning the research workflow.
mode=react
max_steps=8
external_channel_allowed=True
</think>
```

**实际示例 2 - tool_call tag**:
```xml
<tool_call>
tool=graph.think call_id=2967e218e5094c01ae5b2d526317d8e7 plan_step=think_init
tool_meta={"profile":"H","determinism":"llm_heavy"}
about=Structured pause that digests current context before the next hop. Evidence: derived think notes (NOT citeable).
routing={"can_route_remote":false,"prefer_remote":false,"has_local":true,"default_mcp_server":null}
coverage={"evidence_count":0,"coverage_ratio":0.0,"coverage_score":0.0,"completed_steps":0,"total_steps":4}
</tool_call>
```

**实际示例 3 - tool_response tag**:
```xml
<tool_response>
tool=graph_adapter.query call_id=c2e8a936c7604d98a06b58493cae03a9
evidence_count=5 sample=3
- 05db9e12-e147-4842-ab7e-bba0e63690c6 source=hipporag score=0.1823
  With an expenses like mortgages,  children's  education,  aging  parents' care, and planning for  your own future, savings alone  aren't enough...
</tool_response>
```

**实际示例 4 - write_outline tag**:
```xml
<write_outline>
Plan ID: e134e42ce9e345dc9069f0cce8884c79
Question: 你给我说一下原神是啥
Mode: react
External allowed: True
Note: coarse macro plan; tool selection happens during execution.
Steps:
1. plan_01 [graph] enabled=True requires_external=False
   首先，在知识图谱中查找"原神"这一实体的基本信息...
2. plan_02 [text] enabled=True requires_external=False
   基于第一步检索到的信息，提取并总结原神的主要特点...
</write_outline>
```

解析函数：

```javascript
function parseWeaverContent(weaverContent) {
  // 注意：weaverContent 中的 \n 在 JSON 中是 \\n，需要先处理
  // 如果是从 JSON.parse 得到的字符串，\n 已经是真正的换行符
  // 如果是从原始 JSON 字符串中提取，可能需要先替换 \\n 为 \n
  
  // 提取标签名和内容（使用非贪婪匹配，支持多行）
  const tagMatch = weaverContent.match(/<(\w+)>(.*?)<\/\1>/s);
  if (!tagMatch) {
    return { tag: 'unknown', content: weaverContent, rawContent: weaverContent };
  }
  
  const tag = tagMatch[1];
  const content = tagMatch[2].trim();
  
  // 解析键值对格式的内容
  const lines = content.split('\n');
  const parsed = {};
  const rawLines = [];
  
  lines.forEach(line => {
    const trimmedLine = line.trim();
    if (!trimmedLine) return;
    
    rawLines.push(trimmedLine);
    
    // 匹配 key=value 格式
    const match = trimmedLine.match(/^(\w+)=(.*)$/);
    if (match) {
      const key = match[1];
      let value = match[2].trim();
      
      // 尝试解析 JSON 值（如果值看起来像 JSON）
      if (value.startsWith('{') || value.startsWith('[')) {
        try {
          value = JSON.parse(value);
        } catch (e) {
          // 解析失败，保持字符串格式
        }
      }
      
      parsed[key] = value;
    }
  });
  
  return {
    tag: tag,
    rawContent: content,
    rawLines: rawLines,
    parsed: parsed
  };
}
```

### 步骤 4: 提取 Progress Event

```javascript
function handleProgressEvent(toolCall) {
  // 解析 arguments
  const progress = JSON.parse(toolCall.function.arguments);
  
  // 提取进度信息（基于 sse.txt 中的实际格式）
  const progressInfo = {
    stage: progress.stage,                    // "deepsearch"
    deepsearchStage: progress.deepsearch_stage, // "reasoned", "gap_evaluated", "reported", "quality_gated" 等
    status: progress.status,                   // "running", "completed", "failed"
    message: progress.message,                 // 中文消息，如 "正在进行图谱推理..."
    requestId: progress.request_id,            // 请求 ID
    seq: progress.seq,                        // 序列号
    tsMs: progress.ts_ms,                     // 时间戳（毫秒）
    
    // 根据 deepsearch_stage 的不同，可能包含以下字段：
    reasoningTrace: progress.reasoning_trace,  // 当 stage 为 "reasoned" 时
    gapResult: progress.gap_result,            // 当 stage 为 "gap_evaluated" 时
    reportPayload: progress.report_payload,    // 当 stage 为 "reported" 时
    qualityGates: progress.quality_gates,      // 当 stage 为 "quality_gated" 时
    
    // 其他可能的字段
    planStepsCount: progress.plan_steps_count,
    reasoningStepsCount: progress.reasoning_steps_count,
    toolCallsCount: progress.tool_calls_count,
    externalCallsCount: progress.external_calls_count,
    errors: progress.errors
  };
  
  // 更新 UI 显示进度
  updateProgressUI(progressInfo);
}
```

**常见的 `deepsearch_stage` 值**:
- `"reasoned"`: 正在进行图谱推理
- `"gap_evaluated"`: 正在检测知识缺口
- `"reported"`: 正在生成报告
- `"quality_gated"`: 正在进行质量检查

## Trace Event 类型说明

根据 `sse.txt` 中的实际数据，DeepSearch trace events 包含以下 tag 类型：

### 1. `think` - 思考过程

**示例**:
```xml
<think>
Planning the research workflow.
mode=react
max_steps=8
external_channel_allowed=True
</think>
```

**另一个示例**:
```xml
<think>
Question classification (computable routing).
{"source": "llm", "model": "gpt-4o-mini", "is_computable": false, "reasons": ["The question asks for a definition or explanation."], "suggested_tools": []}
</think>
```

**meta 字段可能包含**:
- `stage`: "plan", "question_classification", "think_init", "reflection" 等
- `mode`: "react"
- `max_steps`: 数字
- `external_channel_allowed`: 布尔值
- `classification`: 对象（当 stage 为 "question_classification" 时）

### 2. `write_outline` - 写入计划大纲

**示例**:
```xml
<write_outline>
Plan ID: e134e42ce9e345dc9069f0cce8884c79
Question: 你给我说一下原神是啥
Mode: react
External allowed: True
Note: coarse macro plan; tool selection happens during execution.
Steps:
1. plan_01 [graph] enabled=True requires_external=False
   首先，在知识图谱中查找"原神"这一实体的基本信息...
2. plan_02 [text] enabled=True requires_external=False
   基于第一步检索到的信息，提取并总结原神的主要特点...
</write_outline>
```

### 3. `tool_call` - 工具调用

**示例**:
```xml
<tool_call>
tool=graph.think call_id=2967e218e5094c01ae5b2d526317d8e7 plan_step=think_init
tool_meta={"profile":"H","determinism":"llm_heavy"}
about=Structured pause that digests current context before the next hop...
routing={"can_route_remote":false,"prefer_remote":false,"has_local":true}
coverage={"evidence_count":0,"coverage_ratio":0.0,"completed_steps":0,"total_steps":4}
</tool_call>
```

**另一个示例**:
```xml
<tool_call>
tool=graph_adapter.query call_id=c2e8a936c7604d98a06b58493cae03a9 plan_step=plan_01
</tool_call>
```

**meta 字段可能包含**:
- `call_id`: 工具调用 ID
- `tool_name`: 工具名称（如 "graph.think", "graph_adapter.query"）
- `plan_step`: 计划步骤 ID

### 4. `tool_response` - 工具响应

**示例**:
```xml
<tool_response>
tool=graph.think call_id=2967e218e5094c01ae5b2d526317d8e7
</tool_response>
```

**另一个示例（包含证据）**:
```xml
<tool_response>
tool=graph_adapter.query call_id=c2e8a936c7604d98a06b58493cae03a9
evidence_count=5 sample=3
- 05db9e12-e147-4842-ab7e-bba0e63690c6 source=hipporag score=0.1823
  With an expenses like mortgages,  children's  education...
- a85fb406-bbe8-4dd8-9e18-17edd2b9aaa4 source=hipporag score=0.1690
  is equivalent to a trust product...
</tool_response>
```

**meta 字段可能包含**:
- `call_id`: 工具调用 ID
- `tool_name`: 工具名称
- `plan_step`: 计划步骤 ID
- `ok`: 布尔值，表示调用是否成功
- `route`: "local" 或 "remote"

## 完整示例

```javascript
class DeepSearchTraceHandler {
  constructor() {
    this.traces = [];
    this.progress = null;
  }
  
  // 处理从 SSE 流中读取的原始数据行
  handleSSELine(line) {
    if (!line.startsWith('data: ')) {
      return;
    }
    
    const jsonStr = line.slice(6); // 移除 "data: " 前缀
    try {
      const wrapped = JSON.parse(jsonStr);
      this.handleSSEEvent(wrapped);
    } catch (error) {
      console.error('Error parsing SSE line:', error, jsonStr);
    }
  }
  
  handleSSEEvent(wrapped) {
    if (wrapped.code !== 200) {
      console.error('SSE Error:', wrapped.message);
      return;
    }
    
    const chunk = wrapped.data;
    
    if (!chunk.choices || !chunk.choices[0].delta.tool_calls) {
      return;
    }
    
    chunk.choices[0].delta.tool_calls.forEach(toolCall => {
      const functionName = toolCall.function.name;
      const argumentsStr = toolCall.function.arguments;
      
      if (functionName === 'rag_arc_trace') {
        this.handleTrace(JSON.parse(argumentsStr), toolCall.id);
      } else if (functionName === 'rag_arc_progress') {
        this.handleProgress(JSON.parse(argumentsStr));
      }
    });
  }
  
  handleTrace(args, callId) {
    const trace = {
      tag: args.tag,
      content: args.content,
      message: args.message,  // 人类可读的过程描述
      meta: args.meta,
      callId: callId,
      timestamp: Date.now()
    };
    
    this.traces.push(trace);
    
    // 解析 weaver 内容（如果需要详细信息）
    const parsed = this.parseWeaver(args.content);
    
    // 触发 UI 更新
    this.onTraceReceived(trace, parsed);
  }
  
  handleProgress(progress) {
    this.progress = progress;
    this.onProgressUpdated(progress);
  }
  
  parseWeaver(weaverContent) {
    const match = weaverContent.match(/<(\w+)>(.*?)<\/\1>/s);
    if (!match) {
      return { tag: 'unknown', content: weaverContent, rawContent: weaverContent };
    }
    
    const tag = match[1];
    const content = match[2].trim();
    const lines = content.split('\n');
    const parsed = {};
    
    lines.forEach(line => {
      const trimmedLine = line.trim();
      if (!trimmedLine) return;
      
      const keyValueMatch = trimmedLine.match(/^(\w+)=(.*)$/);
      if (keyValueMatch) {
        const key = keyValueMatch[1];
        let value = keyValueMatch[2].trim();
        
        // 尝试解析 JSON 值
        if (value.startsWith('{') || value.startsWith('[')) {
          try {
            value = JSON.parse(value);
          } catch (e) {
            // 保持字符串格式
          }
        }
        
        parsed[key] = value;
      }
    });
    
    return {
      tag: tag,
      rawContent: content,
      parsed: parsed
    };
  }
  
  onTraceReceived(trace, parsed) {
    // 实现 UI 更新逻辑
    console.log('Trace received:', trace.tag, trace.message, parsed);
    
    // 可以直接使用 message 字段显示简要信息
    this.updateTraceMessageUI(trace.message);
    
    // 示例：根据不同的 tag 类型更新 UI（显示详细信息）
    switch (trace.tag) {
      case 'think':
        this.updateThinkingUI(parsed, trace.meta);
        break;
      case 'write_outline':
        this.updatePlanOutlineUI(parsed);
        break;
      case 'tool_call':
        this.updateToolCallUI(parsed, trace.meta);
        break;
      case 'tool_response':
        this.updateToolResponseUI(parsed, trace.meta);
        break;
    }
  }
  
  onProgressUpdated(progress) {
    // 实现进度更新逻辑
    console.log('Progress updated:', progress.deepsearch_stage, progress.message);
    this.updateProgressUI(progress);
  }
  
  // UI 更新方法（需要根据实际 UI 框架实现）
  updateTraceMessageUI(message) {
    // 显示简要的过程描述（使用 message 字段）
    // 例如：在进度条或状态栏中显示
  }
  
  updateThinkingUI(parsed, meta) {
    // 显示思考过程（详细信息）
  }
  
  updatePlanOutlineUI(parsed) {
    // 显示计划大纲
  }
  
  updateToolCallUI(parsed, meta) {
    // 显示工具调用
  }
  
  updateToolResponseUI(parsed, meta) {
    // 显示工具响应
  }
  
  updateProgressUI(progress) {
    // 显示进度信息
  }
}

// 使用示例：使用 fetch API 接收 SSE
async function streamChatWithDeepSearch(sessionId, query) {
  const handler = new DeepSearchTraceHandler();
  
  const response = await fetch(`/rag_inference/stream_chat/${sessionId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      enable_deepsearch: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    
    // 保留最后一个不完整的行
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.trim()) {
        handler.handleSSELine(line);
      }
    }
  }
  
  // 处理剩余的 buffer
  if (buffer.trim()) {
    handler.handleSSELine(buffer);
  }
}

// 调用示例
streamChatWithDeepSearch('session-123', '你给我说一下原神是啥');
```

## 注意事项

1. **SSE 格式**: SSE 事件以 `data: ` 前缀开头，需要先去除该前缀再解析 JSON
2. **JSON 解析**: `function.arguments` 字段是 JSON 字符串，需要先使用 `JSON.parse()` 解析
3. **Message 字段**: 
   - `rag_arc_trace` 事件现在包含 `message` 字段，提供人类可读的过程描述
   - `rag_arc_progress` 事件也包含 `message` 字段，显示当前阶段的进度
   - 可以直接使用 `message` 字段显示简要信息，无需解析 weaver 格式
4. **换行符处理**: Weaver 格式内容中的换行符在 JSON 字符串中是 `\n`（转义后为 `\\n`），解析后会自动转换为真正的换行符
5. **Weaver 格式**: `content` 字段包含 XML-like 标签（如 `<think>...</think>`），需要解析提取实际内容（如果需要详细信息）
6. **事件顺序**: Trace events 和 progress events 可能交错到达，需要按接收顺序处理
7. **错误处理**: 
   - 某些 trace events 可能包含错误信息，需要适当处理
   - 如果 `status` 为 `"failed"`，表示 DeepSearch 执行失败，可能需要回退到 RAG
8. **性能考虑**: 
   - 大量 trace events 可能影响性能，考虑节流或虚拟滚动
   - 建议使用增量更新而非全量重渲染
   - 如果只需要显示简要信息，可以直接使用 `message` 字段，避免解析 weaver 格式
9. **实际数据格式**: 本文档中的所有示例都基于 `/test/sse.txt` 中的实际输出，请参考该文件了解最新的数据格式

## 实际数据格式参考

本文档中的所有格式说明和示例都基于 `/test/sse.txt` 文件中的实际 SSE 输出。该文件包含了完整的 DeepSearch 执行过程的 SSE 事件流，包括：

- 多个 `rag_arc_trace` 事件，展示了不同的 tag 类型（`think`, `write_outline`, `tool_call`, `tool_response`）
- 多个 `rag_arc_progress` 事件，展示了不同的 `deepsearch_stage`（`reasoned`, `gap_evaluated`, `reported`, `quality_gated`）
- 完整的 weaver 格式内容示例
- 实际的 meta 字段结构

**建议**: 在实现前端解析逻辑时，请直接参考 `sse.txt` 文件中的实际数据格式，以确保解析逻辑的准确性。

## 相关文件

- **实际 SSE 输出示例**: `/test/sse.txt` - 包含完整的 DeepSearch SSE 事件流
- **后端实现**: `api/routers/rag_inference_modules/stream_chat/deepsearch_handler.py` - DeepSearch 处理和 SSE 流式输出
- **Weaver 渲染**: `api/routers/deepsearch_weaver_render.py` - Trace 事件格式化为 weaver 格式
- **SSE 工具**: `api/sse.py` - SSE 事件构建工具函数
- **Trace 系统**: `core/deepsearch/trace/context.py` - Trace 事件发射机制
