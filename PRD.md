# PRD — 电商智能客服 Agent

> **状态:** 生效中  
> **最后更新:** 2026-10-06  
> **用途:** 本文件是项目唯一需求源。任何实现决策与本文件冲突时，以本文件为准。

---

## 1. 项目定位

面向电商场景的智能客服系统 Agent，支持三类核心场景：

| 场景 | 说明 | 示例 |
|------|------|------|
| 商品咨询 | 基于知识库回答产品相关问题 | "这款耳机支持蓝牙5.0吗？" |
| 订单查询 | 通过工具调用查询订单/物流状态 | "帮我查一下ORD-1001的物流" |
| 退换货政策问答 | 基于知识库回答退换货、退款等政策 | "7天无理由退货怎么操作？" |

**不做的事情：**
- 不做实际支付、扣款、转账操作
- 不做用户身份认证（OAuth/SSO）—— 仅用 API Key 鉴权
- 不做多租户 / 多品牌隔离
- 不做前端 UI（仅提供 API）
- 不做模型微调或训练
- 不接入真实第三方物流/支付 API —— 使用 SQLite 种子数据模拟

---

## 2. 技术栈（锁定）

| 层 | 技术 | 备注 |
|----|------|------|
| 语言 | Python 3.12 | 唯一语言 |
| Agent 框架 | LangGraph | 多节点状态机工作流 |
| Web 框架 | FastAPI | 含 SSE 流式输出 |
| 向量数据库 | ChromaDB | 本地持久化，cosine 距离 |
| Embedding | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) | 多语言模型，不替换 |
| LLM | OpenAI API（`gpt-4.1-mini` 默认） | 通过 `openai` SDK 调用 |
| 数据库 | SQLite | 状态持久化（用户/会话/工单/订单/trace） |
| 容器 | Docker + docker-compose | 部署基础 |
| CI | GitHub Actions | lint + test |
| 测试 | pytest | 单元 + 集成 |

**禁止引入的技术：**
- 不使用 LangChain（仅用 LangGraph）
- 不使用 Redis、PostgreSQL 或其他外部数据库
- 不使用 Celery、RabbitMQ 或消息队列
- 不使用 Kubernetes
- 不使用前端框架（React/Vue 等）
- 不引入 rank-bm25、ElasticSearch 或其他检索引擎 —— 检索仅用 ChromaDB 向量检索
- 不引入 Prometheus、Grafana 或外部监控系统

---

## 3. 系统架构

### 3.1 LangGraph 工作流节点

```
START → guardrails → route → [tool | retrieve] → assess → generate → END
```

| 节点 | 职责 |
|------|------|
| `guardrails` | 输入安全检查：拦截敏感/恶意内容 |
| `route` | 意图路由：判断走工具调用还是知识检索 |
| `tool` | 工具执行：check_order / check_shipment / create_ticket |
| `retrieve` | 知识检索：ChromaDB 向量检索 |
| `assess` | 置信度评估：低置信度触发转人工 |
| `generate` | 答案生成：OpenAI API 调用或 fallback |

### 3.2 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/answer` | 同步问答 |
| POST | `/answer/stream` | SSE 流式输出 |
| POST | `/tickets` | 创建工单 |
| GET | `/tickets` | 工单列表 |
| GET | `/tickets/{id}` | 工单详情 |
| PATCH | `/tickets/{id}` | 更新工单状态 |
| GET | `/sessions` | 会话列表 |
| GET | `/sessions/{id}` | 会话详情（含消息历史） |
| GET | `/traces` | trace 列表 |
| GET | `/traces/{id}` | trace 详情（含事件） |
| GET | `/users/{id}` | 用户信息 |
| PUT | `/users/{id}` | 更新用户信息 |
| GET | `/health` | 健康检查 |
| GET | `/ready` | 就绪检查 |

---

## 4. 核心功能需求

### 4.1 知识库 RAG 检索

- **知识库内容：** 电商 FAQ，覆盖商品咨询、订单流程、退换货政策、支付方式、物流配送、售后服务
- **知识库格式：** Markdown 文件，Q/A 对格式
- **知识库规模目标：** ≥100 条 QA 对（当前仅 5 条，需扩充）
- **分块策略：** 优先按 Q/A 块分割；长段落按 chunk_size=220 / overlap=40 分割
- **检索方式：** ChromaDB 向量检索，cosine 距离，top_k=3
- **多语言：** 中英文混合检索（依赖 multilingual embedding 模型）

**检索质量目标：**
- 300 条标注测试集上，检索召回率 ≥ 91%
- 同义词和口语化表达可正确匹配

### 4.2 工具调用

现有工具（3 个）：

| 工具名 | 触发条件 | 功能 |
|--------|----------|------|
| `check_order` | 包含订单号 + 订单关键词 | 查询订单状态 |
| `check_shipment` | 包含订单号 + 物流关键词 | 查询物流信息 |
| `create_ticket` | 包含人工/投诉/客服关键词 | 创建人工工单 |

**工具检测：** 基于关键词匹配（中英文），非 LLM 分类
**工具优先级：** shipment > order > ticket（有订单号时优先走物流/订单）

### 4.3 转人工机制

**自动转人工（低置信度）：**
- 检索置信度 < min_score（默认 0.12）时触发
- 返回提示消息 + escalated=true
- 目标：误转人工率 ≤ 8%（基于测试集调整阈值）

**强制转人工（高风险意图）：**
- 关键词命中：投诉、退款、complaint、refund 等 → 直接 create_ticket
- guardrails 拦截：hack、exploit、injection 等敏感内容 → block + escalated

**指标：**
- 误转人工率：由 30% 降至 ≤ 8%（通过阈值调优实现）

### 4.4 SSE 流式输出

- 端点：`POST /answer/stream`
- 协议：Server-Sent Events（`text/event-stream`）
- 实现：异步生成器 + `ThreadPoolExecutor` 衔接同步模型调用
- 结束标记：`data: [DONE]\n\n`

### 4.5 限流策略

- **粒度：** 每用户（user_id）或每 API Key
- **限制：** 每分钟 10 次（可通过环境变量配置，当前默认 20 次/60 秒，需调整到 10 次）
- **响应：** 429 状态码 + Retry-After 头

### 4.6 可观测性

- **trace：** 每次请求生成 trace_id，记录完整处理链路
- **事件：** guardrails_checked → route_selected → tool_executed/retrieval_completed → retrieval_assessed → answer_generated
- **存储：** SQLite traces + trace_events 表
- **查询：** 通过 `/traces` API 查询

### 4.7 认证

- **方式：** API Key（`X-API-Key` 请求头）
- **配置：** 环境变量 `SUPPORT_AGENT_API_KEYS`（逗号分隔）
- **可关闭：** `SUPPORT_AGENT_REQUIRE_API_KEY=false`

---

## 5. 评测需求

### 5.1 模拟对话评测

- **规模：** ≥ 500 轮模拟对话
- **评测维度：**
  - 答案正确性（与标注答案对比）
  - 意图路由准确性（工具调用 vs 知识检索 vs 转人工）
  - 安全拦截准确性（敏感内容是否被正确拦截）
- **通过率目标：** 人工评测答案通过率 ≥ 87%

### 5.2 检索评测

- **测试集：** 300 条标注查询（含期望匹配的知识块）
- **指标：** 召回率 ≥ 91%
- **对比基线：** TF-IDF 方案召回率 84%

### 5.3 性能评测

- **平均响应时间：** ≤ 2.5 秒（含 OpenAI API 调用）
- **无 OpenAI 时（fallback）：** ≤ 500ms

### 5.4 评测工具

- `support_agent/eval_runner.py` —— 离线评测脚本
- `support_agent/data/eval_dataset.json` —— 评测数据集
- 评测结果输出到 stdout，包含通过率、召回率、平均延迟

---

## 6. 测试需求

### 6.1 覆盖率

- **目标：** ≥ 78%（当前已有测试基础）
- **工具：** pytest + coverage

### 6.2 测试类型

| 类型 | 范围 | 示例 |
|------|------|------|
| 单元测试 | 各模块独立功能 | guardrails 拦截、工具检测、分块逻辑 |
| 集成测试 | 端到端链路 | agent.answer_with_metadata 完整流程 |
| API 测试 | HTTP 端点 | FastAPI TestClient 测试各端点 |

### 6.3 CI/CD

- **平台：** GitHub Actions
- **流水线：**
  1. `ruff check` — 代码风格检查
  2. `pytest --cov` — 运行测试 + 覆盖率报告
  3. 覆盖率 < 78% 时 CI 失败

---

## 7. 部署

- **容器化：** Dockerfile + docker-compose.yml
- **基础镜像：** `python:3.12-slim`
- **端口：** 8000
- **环境变量：** 见 `.env.example`
- **数据持久化：** SQLite 文件 + ChromaDB 目录挂载

---

## 8. 数据边界

### 8.1 种子数据（模拟数据，非真实）

| 数据 | 内容 |
|------|------|
| 用户 | user-1（VIP）、user-2（standard） |
| 订单 | ORD-1001（paid, HKD 299）、ORD-1002（shipped, HKD 88） |
| 物流 | ORD-1002 → SF Express, SF123456789HK, in_transit |

### 8.2 知识库数据

- 存放于 `support_agent/data/` 目录
- 格式：Markdown（Q/A 对）
- 语言：中文为主，支持英文

### 8.3 数据安全

- 不存储真实用户个人信息
- 不存储真实支付信息
- 所有数据均为模拟/演示用途

---

## 9. 当前状态 vs 目标差距

| 维度 | 当前状态 | 目标 | 差距 |
|------|----------|------|------|
| 知识库规模 | 5 条 QA | ≥100 条 QA | 需大量扩充 |
| 评测数据集 | 存在但不完整 | 500+ 轮对话 + 300 条检索 | 需补充 |
| 检索召回率 | 未测量 | ≥91% | 需基准测试 |
| 答案通过率 | 未测量 | ≥87% | 需评测 |
| 响应时间 | 未测量 | ≤2.5s | 需基准测试 |
| 误转人工率 | 未测量 | ≤8% | 需阈值调优 |
| 限流配置 | 20次/60s | 10次/60s | 需调整 |
| 测试覆盖率 | ~78%（估计） | ≥78% | 需确认 |
| CI/CD | 未配置 | GitHub Actions | 需新建 |
| eval_dataset.json | README提到但缺失 | 完整数据集 | 需创建 |

---

## 10. 禁止事项

1. **不引入本文件「技术栈」之外的框架或数据库**
2. **不实现多租户、RBAC、A/B测试等超出范围的功能**
3. **不使用 LangChain（仅 LangGraph）**
4. **不接入真实外部 API（物流、支付、CRM）**
5. **不做前端 UI**
6. **不做模型微调/训练**
7. **不添加 Redis、PostgreSQL、Prometheus 等外部依赖**
8. **不在知识库中编造不存在的产品或政策** —— 所有 FAQ 内容需合理且一致
9. **不修改已通过测试的核心工作流节点顺序**（guardrails → route → tool/retrieve → assess → generate）
10. **评测数据中不使用 LLM 生成的答案作为标注** —— 标注答案须人工编写或人工审核

---

## 11. 决策参考

当实现过程中遇到不确定时，按以下优先级决策：

1. **本 PRD 文件** —— 需求和边界的唯一权威
2. **现有代码** —— 已实现的模式和约定
3. **README.md** —— 架构和 API 说明
4. **测试用例** —— 预期行为的具体定义

如果以上四者互相冲突，以本 PRD 为准。
