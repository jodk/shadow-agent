# Learning ReAct Agent

这是一个适合阅读和修改的 Python ReAct Agent。它不依赖 LangGraph，
直接用一个循环展示模型如何请求工具、程序如何执行工具，以及工具结果如何返回模型。

支持：

- OpenAI 兼容的模型接口
- Python 函数工具
- YAML 声明的 REST API 工具
- MCP stdio 工具
- `SKILL.md + skill.yaml + tools.py` 形式的 Skill，并按需加载 Skill 正文
- SQLite 原文记忆、FTS5/BM25 关键词检索、Chroma 中文向量检索和 RRF 融合
- 后台异步向量索引，不阻塞当前问答
- OpenAI 兼容模型流、Agent 事件流和浏览器 SSE 实时输出
- SQLite 会话列表、会话删除和跨会话长期记忆
- 每轮输入/输出 Token 与会话累计 Token 统计
- 命令行聊天、FastAPI 接口和原生 HTML/CSS/JavaScript 聊天页面
- 控制台与滚动文件日志

## 推荐学习顺序

如果你对 Python 和 Agent 都不熟悉，不建议一开始按目录顺序逐文件阅读。
可以按照下面的路径学习：

1. 阅读 `examples/skills/calculator/tools.py`，理解普通函数如何通过 `@tool`
   变成工具。
2. 阅读 `src/react_agent/tools/base.py`，理解所有工具共有的 `Tool` 数据结构。
3. 阅读 `src/react_agent/agent.py`，重点观察模型调用、工具执行、工具结果回填
   这三个步骤。
4. 阅读 `tests/test_agent.py`，通过假的 ScriptedModel 看清楚两轮模型调用之间
   消息发生了什么变化。
5. 阅读 `src/react_agent/model.py`，理解 Python 对象如何映射到 OpenAI
   `/chat/completions` JSON。
6. 阅读 `src/react_agent/tools/rest.py` 和 `tools/mcp.py`，理解不同来源的能力
   如何适配成统一 Tool。
7. 最后阅读 `factory.py`、`api.py` 和 `cli.py`，了解完整程序如何启动。

## Python 语法速查

项目中频繁使用以下 Python 语法：

```python
def add(left: int, right: int) -> int:
    return left + right
```

`left: int` 是参数类型标注，`-> int` 是返回值类型标注。Python 默认不会仅凭
类型标注阻止错误类型，本项目使用 Pydantic 在工具执行前进行实际校验。

```python
async def complete(...):
    response = await client.post(...)
```

`async def` 定义异步函数，调用后得到协程。`await` 等待网络请求完成，同时允许
事件循环处理其他任务。最外层通常通过 `asyncio.run(...)` 启动。

```python
async for event in agent.stream(messages):
    ...
```

`stream()` 是异步生成器。它通过 `yield` 分批返回事件，因此网页可以先显示
模型生成片段或“正在调用工具”，不用等最终回答全部生成。

```python
@tool
def multiply(left: float, right: float) -> float:
    return left * right
```

`@tool` 是装饰器，相当于：

```python
def multiply(left: float, right: float) -> float:
    return left * right

multiply = tool(multiply)
```

装饰后，`multiply` 已经是一个 `Tool` 对象，而不再只是原始函数。

```python
built = await build_agent(config, base_dir)
agent = built.agent
```

`build_agent` 返回一个 `BuiltAgent` 对象，其中包含 Agent、MCP Provider、
Skill 元数据和能力路由器。

```python
tools = [adapt(item) for item in response.tools]
```

这是列表推导式，是下面写法的简写：

```python
tools = []
for item in response.tools:
    tools.append(adapt(item))
```

类型标注中常见的含义：

- `List[Tool]`：由多个 `Tool` 组成的列表。
- `Dict[str, Any]`：键是字符串，值可以是任意类型的字典。
- `Optional[str]`：值可以是字符串，也可以是 `None`。
- `Tuple[Agent, List[Provider]]`：返回两个元素组成的元组。
- `Protocol`：只要求对象具有指定方法，不要求继承某个具体父类。

## ReAct 循环

核心代码位于 `src/react_agent/agent.py`：

```text
messages + tool schemas
        |
        v
      model
       |  \
       |   \ ordinary answer -> finish
       v
   tool_calls
       |
       v
 execute tools
       |
       v
 append tool messages -> call model again
```

模型的“推理”留在模型内部，应用只处理公开的 assistant 消息和 tool calls，
避免要求模型输出隐藏思维过程。

## 上下文记忆与 Token 控制

项目不会为了节省 token 而删除历史原文。每条消息会完整写入 SQLite：

```text
data/memory.db
├── messages       # 完整 payload，包含 tool_calls、tool_call_id 等字段
├── messages_fts   # SQLite FTS5 虚拟表，负责 BM25
└── ../chroma/     # 实际路径为 memory.db 同目录下的 Chroma 向量目录
```

每次请求给模型的上下文由四个阶段组成：

1. 最近消息：保证当前对话连续性；
2. BM25 命中消息：找回精确的名称、数字、技术词和用户原话；
3. 向量命中消息：找回没有共享相同关键词但语义相近的历史消息；
4. RRF 融合结果：决定哪些旧消息最终进入模型上下文。

SQLite 中保存的是完整历史，只有“发送给模型的窗口”被裁剪，因此不会因为
token 控制而丢失细节。对应配置：

```yaml
agent:
  memory_db: "./data/memory.db"
  recent_messages: 8
  keyword_memory_limit: 5
  vector_memory_limit: 5
  fused_memory_limit: 8
  rrf_k: 60
  cross_session_memory: true
```

### SQLite + Chroma 混合检索

当前重构后的混合检索由两个本地存储协同完成：

```text
data/
├── memory.db        # 完整原文、会话数据、SQLite FTS5/BM25 索引
├── chroma/          # Chroma 持久化向量索引
└── models/
    └── shibing624--text2vec-base-chinese/  # 中文 embedding 模型缓存
```

应用层分别执行 SQLite FTS5/BM25 和 Chroma 向量检索，再使用 RRF 合并两份排名列表。这样精确的
姓名、数字、文件名和错误码可以由 BM25 找回，表达方式不同但语义相近的内容
可以由向量检索补充。RRF 不直接比较两种检索器的原始分数，只比较排名。

默认使用 `shibing624/text2vec-base-chinese`，也可以改成 BGE 中文模型。
模型加载规则如下：

1. 如果 `data/models/<模型名>` 已存在且包含模型文件，直接离线加载；
2. 如果目录不存在，首次后台索引时联网下载；
3. 下载完成后保存到 `memory.db` 同级的 `models/` 目录；
4. 下载或 embedding 失败不会影响 SQLite 原文保存和 BM25 问答。

默认配置使用 `embedding_device: "cpu"`。macOS Apple Silicon 虽然通常能使用
PyTorch MPS，但 SentenceTransformer 在 asyncio 后台线程中加载或编码时，部分
PyTorch、Metal 和 macOS 版本组合可能直接触发 `segmentation fault`。这是原生库
崩溃，Python 的 `try/except` 无法捕获，所以本学习项目优先使用更稳定的 CPU。
确认当前环境稳定后可以显式改为 `"mps"`；NVIDIA CUDA 环境可以改为 `"cuda"`。

加载日志会打印配置设备和实际设备，例如：

```text
embedding_device_selected model=... configured=cpu actual=cpu
embedding_model_loaded model=... source=local device=cpu
embedding_documents_started model=... document_count=1 device=cpu
embedding_documents_finished model=... vector_dimension=768 device=cpu
```

模型加载和 `encode()` 共用同一把异步锁。即使后台索引和在线向量检索同时
发生，也不会让多个线程并发调用同一个 SentenceTransformer/PyTorch 模型实例。

安装向量依赖：

```bash
pip install -e ".[vector]"
```

### 异步索引时序

保存消息时先同步写入 SQLite，保证原文不丢失，然后把向量索引任务放入后台
队列。当前问答不会等待模型下载、embedding 生成或 Chroma 写入：

```text
写入消息
   ├── SQLite messages + FTS5 trigger       -> 立即可用
   └── 后台 worker -> 中文模型 -> embedding -> Chroma
                                      -> 后续请求可使用
```

因此首次启动或模型下载期间，BM25 仍然可以正常工作；向量结果会在后台索引
完成后逐步参与检索。应用关闭时会等待已排队的索引任务完成。

### BM25 和向量检索使用 RRF 融合

向量检索适合“意思相近但用词不同”的内容，但对精确数字、文件名、函数名、
错误码和专有名词不一定可靠。BM25 对这些内容更稳定。因此当前策略不是二选一，
而是分别排序后使用 Reciprocal Rank Fusion（RRF）融合：

```text
BM25 排名列表  ─┐
                 ├─ RRF(rank) ─> 融合排名 ─> 最终召回
向量排名列表  ──┘
```

RRF 使用下面的公式：

```text
RRF(d) = Σ 1 / (k + rank(d))
```

默认 `k=60`。RRF 只比较排名，不直接比较 BM25 分数和向量相似度，
因此不需要把两个检索器的分数强行归一化。一个消息同时出现在 BM25 和向量
结果中时，会累加两路排名贡献，通常优先进入最终上下文。

`fused_memory_limit` 控制 RRF 后最终召回的旧消息数量。
召回消息会被放在单独的 system 消息中，并明确标识为“较早的原始消息”，
避免模型把历史内容误认为当前新指令。

## 安装

项目要求 Python 3.9 或更高版本，当前开发环境使用 Python 3.12。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,vector]"
```

需要 MCP 时额外安装：

```bash
pip install -e ".[mcp]"
```

复制示例配置：

```bash
cp config.example.yaml config.yaml
```

然后在 `config.yaml` 中直接填写 `base_url`、`api_key` 和 `model`。
这是按项目要求采用的纯 YAML 配置，API Key 是明文；`config.yaml` 已加入
`.gitignore`，不要上传或分享该文件。

任何支持 OpenAI `/chat/completions` 和原生 tool calling 的兼容服务都可以接入：

```yaml
model:
  base_url: "https://api.openai.com/v1"
  api_key: "your-key"
  model: "gpt-4.1-mini"
```

## 运行

命令行交互：

```bash
python -m react_agent --config config.yaml chat
```

发送单条问题：

```bash
python -m react_agent --config config.yaml chat "计算 12 乘以 8"
```

查看已经加载的工具：

```bash
python -m react_agent --config config.yaml tools
```

启动网页：

```bash
python -m react_agent --config config.yaml serve --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。页面使用原生 HTML、CSS 和 JavaScript，
没有前端构建步骤。页面当前提供：

- 左侧会话列表，按最近活动时间排序；
- 新建、切换、清空和删除会话；
- 用户消息、流式模型回答和工具调用过程；
- 本轮输入 Token、本轮输出 Token、本轮总 Token 和会话累计 Token；
- 停止当前流式请求。

修改前端文件后不需要执行 npm 构建。重新启动服务并刷新页面即可；`app.js`
使用版本化 URL，避免浏览器继续使用旧的 SSE 解析代码。

## 配置说明

项目只读取 YAML 配置。完整示例见 `config.example.yaml`，主要配置如下：

```yaml
model:
  base_url: "https://api.openai.com/v1"
  api_key: "replace-with-your-api-key"
  model: "gpt-4.1-mini"
  temperature: 0.2
  timeout_seconds: 60

logging:
  level: INFO
  console: true
  file: "./data/agent.log"
  max_bytes: 10485760
  backup_count: 5

agent:
  system_prompt: |
    You are a helpful assistant.
  max_steps: 10
  memory_db: "./data/memory.db"
  recent_messages: 8
  keyword_memory_limit: 5
  vector_memory_limit: 5
  vector_enabled: true
  embedding_model: "shibing624/text2vec-base-chinese"
  embedding_device: "cpu"
  vector_workers: 1
  fused_memory_limit: 8
  rrf_k: 60
  cross_session_memory: true
```

相对路径都以 `config.yaml` 所在目录为基准。`cross_session_memory: true`
表示检索长期记忆时会排除当前会话并从其他会话召回相关原文；当前会话连续性
仍由最近消息保证。

## 日志

默认同时输出到终端和 `data/agent.log`。日志文件达到 `max_bytes` 后滚动，
最多保留 `backup_count` 个历史文件。日志覆盖以下关键阶段：

- 应用启动、关闭和加载的工具数量；
- 模型请求、流式完成、耗时和 Token usage；
- SQLite 消息写入、BM25 检索、向量检索和 RRF 融合；
- embedding 模型本地发现、下载、加载设备和编码；
- Chroma 写入、查询和会话向量删除；
- 工具调用的开始、结束和错误。

日志不会记录 API Key、完整系统提示词、完整用户消息或完整工具返回正文。

## Python 工具

`@tool` 会读取函数签名并生成 OpenAI 工具 Schema：

```python
from react_agent.tools import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get weather for a city."""
    return {"city": city, "temperature": 24, "unit": unit}
```

Skill 模块通过 `TOOLS` 暴露工具：

```python
TOOLS = [get_weather]
```

同步和异步函数都支持。调用前会根据类型标注校验参数；异常会被包装成
`{"ok": false, "error": "..."}` 并作为工具消息交给模型。

## REST API 工具

REST 工具使用 YAML 声明，示例见 `examples/rest_tools.yaml`：

```yaml
tools:
  - name: find_user
    description: Find a user by ID.
    method: GET
    url: https://api.example.com/users/{user_id}
    parameters:
      type: object
      properties:
        user_id: {type: integer}
        verbose: {type: boolean}
      required: [user_id]
      additionalProperties: false
    path_params: [user_id]
    query_params: [verbose]
    response_path: data
```

可用字段：

- `path_params`：替换 URL 中的 `{name}`。
- `query_params`：添加 URL 查询参数。
- `body_params`：构造 JSON 请求体。
- `headers`：添加固定请求头，可用于 API 认证。
- `response_path`：从 JSON 响应提取点分隔路径，例如 `data.items`。
- `timeout_seconds`：覆盖默认 30 秒超时。

模型不能自行指定主机，实际 URL 固定在 REST 工具配置中。

## MCP 工具

`config.yaml` 中声明 stdio MCP Server：

```yaml
tools:
  rest:
    files: []
  mcp_servers:
    - name: filesystem
      transport: stdio
      command: npx
      args:
        - "-y"
        - "@modelcontextprotocol/server-filesystem"
        - "./workspace"
```

启动时，`MCPToolProvider` 完成初始化并调用 `tools/list`。每个 MCP 工具会被
转换成项目内部的 `Tool`，Agent 调用时再转发至 `tools/call`。
首版只实现 stdio，是为了让 MCP 生命周期保持清晰；HTTP transport 可以在同一
Provider 接口上继续扩展。

## Skill

示例目录：

```text
examples/skills/calculator/
├── SKILL.md
├── skill.yaml
└── tools.py
```

`SKILL.md` 不会在启动时追加到全局系统提示词。启动时只读取 Skill 的名称、
描述、关键词、工具名称和文件路径；每轮请求先用 BM25 + Chroma 向量检索并通过
RRF 排名，只有命中的 Skill 才会把完整 `SKILL.md` 放入本轮提示词。这样 Skill
数量很大时不会每轮重复消耗全部文档 Token。

```yaml
enabled: true
name: calculator
description: 执行精确的数学运算
keywords: [计算, 数学, 加法, 乘法]
python_module: tools.py
```

`tools.py` 可以提供 `TOOLS` 列表。Skill 是模型使用工具的说明和相关资源集合；
真正可执行的动作仍然通过统一 Tool 接口完成。

### 完整 Skill 样例

`examples/skills/text-analysis` 是一个包含详细中文注释的完整样例：

```text
examples/skills/text-analysis/
├── SKILL.md       # 告诉模型适用场景、使用规则和工具选择方式
├── skill.yaml     # 控制是否启用，并指定 Python 工具模块
└── tools.py       # 三个可以实际执行的 Python 工具
```

它提供以下工具：

- `count_text`：统计字符数、非空白字符数、中文字符数、英文单词数和行数。
- `count_keyword`：统计关键词次数，并支持是否区分大小写。
- `summarize_text_info`：返回文本长度、首行和内容预览。

示例配置已经包含整个 `examples/skills` 目录，因此无需单独修改配置。可以先检查
Skill 中的工具是否成功加载：

```bash
python -m react_agent --config config.yaml tools
```

输出中应该包含：

```text
count_text
count_keyword
summarize_text_info
```

然后进入命令行聊天：

```bash
python -m react_agent --config config.yaml chat
```

可以尝试以下问题：

```text
请精确统计下面文本的字符数、中文字符数和英文单词数：
你好，ReAct Agent! Hello world.
```

或者：

```text
忽略大小写，统计下面文本中 agent 出现了多少次：
Agent calls tools. The AGENT gets results.
```

命中 Skill 后，所属工具和直接命中的工具 Schema 才会暴露给模型；配置中的
`always_tools` 或工具的 `always_expose: true` 会常驻暴露。工具 Schema 和
Skill 正文都有独立 token 预算，工具返回结构化结果，模型再把结果组织成
自然语言回答。对应自动测试位于 `tests/test_text_analysis_skill.py`。

### 渐进式能力暴露

每轮问题会经过应用侧预路由：

```text
用户问题
   ├── SQLite FTS5/BM25
   └── Chroma capability collection
             ↓
           RRF
             ↓
命中的 Skill 正文 + 相关工具 Schema + always_tools
```

对应配置：

```yaml
agent:
  always_tools: []
  capability_keyword_limit: 10
  capability_vector_limit: 10
  capability_skill_limit: 3
  capability_tool_limit: 8
  capability_rrf_k: 60
  capability_skill_token_budget: 2000
  capability_tool_token_budget: 3000
```

能力目录使用同一个 `memory.db` 的独立 `capabilities` 表和
`capabilities_fts` FTS5 表；向量索引使用同一个 Chroma 持久化目录下独立的
`capabilities` collection。能力向量在后台同步，不阻塞当前问答。若 embedding
加载或向量检索失败，本轮仍可使用 BM25；若整个能力路由失败，生产流程应降级到
核心工具和无 Skill 说明继续回答。

## HTTP API

- `POST /api/chat/stream`：发送消息，返回标准 `text/event-stream` SSE 事件流。
  如果模型服务支持 OpenAI 兼容的 `stream=true`，回答文本会通过
  `assistant_delta` 事件逐片段发送，前端会实时更新回答；工具调用仍会等参数
  聚合完成后执行。因为请求需要通过 POST 发送 JSON，前端使用 `fetch` 读取响应
  流并按照 SSE 的 `event:`、`data:` 和空行边界解析，而不是使用只支持 GET 的
  原生 `EventSource`。
- `GET /api/sessions`：返回会话列表、标题、预览和累计 Token。
- `GET /api/sessions/{session_id}`：读取 SQLite 中的完整会话消息和累计 Token。
- `GET /api/sessions/{session_id}/usage`：单独读取会话累计 Token。
- `DELETE /api/sessions/{session_id}`：删除消息、Token 统计和对应 Chroma 向量。
- `GET /api/tools`：查看注册的工具定义。每轮真正发送给模型的工具会由能力路由
  按问题动态筛选。

请求示例：

```json
{
  "session_id": "demo",
  "message": "12 乘以 8 是多少？"
}
```

事件类型包括 `connected`、`capability_route`、`model_started`、`model_usage`、`assistant_delta`、
`tool_started`、`tool_finished`、`assistant_completed`、`error` 和 `done`。

一段实际 SSE 响应类似：

```text
event: connected
data: {"session_id":"demo"}

event: assistant_delta
data: {"content":"计算结果","step":1}

event: assistant_delta
data: {"content":"是 96。","step":1}

event: done
data: {"content":"计算结果是 96。","usage":{"prompt_tokens":20,"completion_tokens":8,"total_tokens":28,"estimated":false},"session_usage":{"prompt_tokens":20,"completion_tokens":8,"total_tokens":28,"estimated":false}}
```

前端使用增量 SSE 状态机处理网络分块，支持 `LF`、`CRLF` 和单独 `CR`，
也能处理 JSON 内容或换行符恰好被拆到两个网络 chunk 的情况。

### Token 统计

模型服务返回 `usage` 时，页面显示模型提供的准确值。模型服务未返回 usage 时，
项目会分别估算输入和输出 Token，并把 `estimated` 标记为 `true`。一次 ReAct
运行可能因为工具调用而多次请求模型，本轮统计会累加所有模型步骤；会话统计
持久化在 SQLite 的 `session_usage` 表中。

## 项目结构

```text
src/react_agent/
├── agent.py          # ReAct 循环
├── model.py          # OpenAI 兼容客户端
├── config.py         # YAML 配置
├── factory.py        # 装配模型、工具和 Skill
├── api.py            # FastAPI、会话 API 和浏览器 SSE
├── cli.py            # 命令行入口
├── events.py         # Agent 流式事件结构
├── logging_config.py # 控制台与滚动文件日志
├── memory.py         # SQLite、FTS5/BM25、RRF 和后台索引
├── skills.py         # Skill 加载
├── vector_store.py   # SentenceTransformer 和 Chroma
├── tools/
│   ├── base.py       # Tool 与注册表
│   ├── python.py     # Python 装饰器
│   ├── rest.py       # REST 工具
│   └── mcp.py        # MCP 适配
└── static/
    ├── index.html    # 原生聊天页面结构
    ├── styles.css    # 页面布局和样式
    └── app.js        # 会话、Token 统计和 SSE 状态机
```

## 测试

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
node --check src/react_agent/static/app.js
```

测试中的模型和 HTTP 服务都使用本地替身，不需要真实 API Key，也不会下载
真实 embedding 模型。
