# 设计文档（DESIGN.md）

本文说明代码审查 Agent 的**架构决策、关键实现与取舍理由**，配合 [README.md](README.md) 阅读。

---

## 1. 目标与范围

**目标**：实现一个具备完整 Agent 能力（LLM 调用、Prompt 设计、工具集成、上下文记忆、
错误处理）的代码审查助手，代码量可控、结构清晰、可被逐层验证。

**明确的非目标**（避免过度设计）：

- 不做代码自动修复（只给建议，改代码交给人类）
- 不做跨文件类型推断与依赖图分析（超出教学项目必要范围）
- 不引入向量数据库 / RAG（本项目没有知识库检索需求）
- 不做多 Agent 协作（单 Agent + 工具已能覆盖评审要点）

---

## 2. 为什么不用 LangChain / AutoGen / CrewAI

题目允许使用主流框架，本项目**刻意选择原生实现**，理由如下：

| 维度 | 框架方案 | 本项目方案 |
|---|---|---|
| 可解释性 | Agent 循环被封装在框架内部，评审者需要读框架源码才能确认「工具调用后如何回喂」 | 循环就是 `agent.py` 里一个 40 行的 `for`，一眼看完 |
| 依赖体积 | LangChain 会引入数十个传递依赖，版本冲突风险高 | 运行时只依赖 `openai` 一个包 |
| 可控性 | 重试策略、记忆裁剪、终止条件受框架约束，定制要绕开抽象 | 全部自己实现，行为完全可预测 |
| 学习价值 | 学会「怎么用框架」 | 理解「Agent 到底是什么」——这正是本项目的目标 |

> 结论：作为学习 Agent 基础能力的项目，**手写循环比调用框架更能体现对 Agent 设计模式的理解**。
> 框架的价值在大规模生产场景，而非理解原理。

---

## 3. 分层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│ 表现层（两个平级消费者，共用 AgentEvent 契约）                        │
│   ├ CLI 层      cli.py                                              │
│   │   职责：参数解析、终端渲染、退出码。不含任何 Agent 逻辑。          │
│   └ Web 层      web/server.py, sessions.py, events.py, static/      │
│       职责：HTTP 路由、SSE 推送、会话生命周期、浏览器渲染。            │
├─────────────────────────────────────────────────────────────────────┤
│ Agent 层        agent.py                                            │
│   职责：控制循环、维护消息序列、决定何时终止、错误分级处理。           │
│   不关心 UI，也不实现任何具体工具。                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 能力层          llm.py / memory.py / prompts.py / config.py         │
│   职责：把「一次 LLM 调用」「上下文管理」「提示词」「配置」各自封装。   │
├─────────────────────────────────────────────────────────────────────┤
│ 工具层          tools/base.py / analyzer.py / builtin.py            │
│   职责：工具协议、注册与分发、路径安全、具体能力实现。                 │
│   工具是纯函数：参数进，文本出，不感知 LLM 与 UI。                    │
└─────────────────────────────────────────────────────────────────────┘
```

**依赖方向单向向下**：表现层 → Agent → 能力层/工具层。工具层不反向依赖 Agent，
因此可以脱离 LLM 单独测试（见 `tests/test_tools.py`）。

CLI 与 Web **互不依赖**，二者唯一的共同契约是 `AgentEvent` 流。
这就是把主循环写成生成器的直接收益：新增交互方式（TUI、IDE 插件、HTTP API）
只需再写一个事件消费者，Agent 与工具层零改动。

---

## 4. Agent 主循环

### 4.1 伪代码

```python
用户输入 → 加入 memory
for step in 1..max_steps:
    reply = LLM.chat(memory.messages, tools=registry.specs())   # 推理
    memory.add(reply)
    if reply.tool_calls 为空:                                    # 收敛
        输出 reply.content; return
    for call in reply.tool_calls:                                # 行动
        result = registry.execute(call.name, call.arguments)      # 观察
        memory.add(tool 消息)
        if call.name == "finish_review" and result.ok:            # 主动终止
            输出 result.content; return
raise StepLimitExceeded                                          # 步数耗尽
```

### 4.2 关键设计点

**① 事件流而非回调**

`ReviewAgent` 的入口方法是**生成器**，逐步 `yield AgentEvent`：

```python
for event in agent.review(path="app/service.py"):
    render_event(event)     # CLI 渲染
```

- **好处 1**：UI 与 Agent 解耦。同一个 Agent 既能给 CLI 用，也能给未来的 Web/TUI 用，
  不需要改动 Agent 一行代码。
- **好处 2**：天然支持实时展示。工具调用发生在 `yield` 之间，用户能看到 Agent「正在做什么」，
  而不是干等最终结果。
- **好处 3**：可测试。测试里 `list(agent.review())` 就能断言完整事件序列。

**② 终止条件有三个**

| 条件 | 触发点 | 处理 |
|---|---|---|
| 模型不再请求工具 | `if not reply.tool_calls` | 正常收敛，输出 `content` |
| 模型显式调用 `finish_review` | 工具执行后置位标记 | 立即结束，**不再请求下一轮 LLM** |
| 达到 `max_steps` | 循环自然结束 | 抛 `StepLimitExceeded`，CLI 捕获后走兜底结论 |

第二个条件是刻意设计的：仅靠「模型不再请求工具」作为终止信号，模型容易在收尾时又去读一个文件、
陷入无意义的工具循环。提供 `finish_review` 让它有明确的「交付」动作。

实现上有一个细节：生成器内部无法向外部作用域赋值，因此用 `finished: list[AgentEvent]`
作为可变容器传递，工具执行后往里追加事件，主循环检测到非空即 `return`。

**③ 工具结果永远回喂给模型**

工具执行失败（路径被拒、文件不存在、参数错误）**不会**中断循环，而是把错误文本作为
`role="tool"` 消息回喂。这让模型有机会自我纠正——例如路径穿越被 `PathGuard` 拒绝后，
模型会改用 `search_code` 换个思路。这比直接抛异常终止更接近真实 Agent 的行为。

---

## 5. 工具层设计

### 5.1 工具协议

```python
@dataclass
class Tool:
    name: str                        # 模型看到的工具名
    description: str                 # 模型判断「何时该用」的依据
    parameters: dict[str, Any]       # JSON Schema
    handler: Callable[..., ToolResult]

    def to_spec(self) -> dict:       # 转成 OpenAI function-calling 格式
        ...
```

**约定**：

1. 工具是**纯函数**——输入参数（关键字参数），输出 `ToolResult(content: str, ok: bool, meta: dict)`。
   不读全局状态、不感知 LLM。
2. `ToolRegistry.execute()` 是**唯一的调用入口**，统一负责：
   JSON 解析 → 参数校验 → 异常捕获 → 调用日志记录。
3. 工具**不允许向外抛异常**。`execute()` 会捕获 `TypeError`（参数不匹配，模型最常犯的错）
   和其他所有异常，转成可读文本 + `ok=False`。
4. `call_log` 记录每次调用的工具名、参数、耗时、成败，供 `--json` 输出统计。

### 5.2 为什么是这 7 个工具

工具集的设计遵循「**覆盖一次完整代码审查所需的最小能力闭环**」：

| 阶段 | 工具 | 缺了会怎样 |
|---|---|---|
| 侦查 | `list_directory` | 模型只能盲猜项目结构，或被迫逐个文件试读，浪费 token |
| 取证 | `read_file`, `search_code` | 无法看到真实代码，只能凭训练记忆臆测 → 幻觉 |
| 分析 | `analyze_code` | 复杂度和坏味道靠模型肉眼数，不可靠且昂贵 |
| 验证 | `run_tests` | 结论无法被实测证伪，只能停留在静态推断 |
| 交付 | `write_report`, `finish_review` | 报告只存在于终端，无法落盘；且缺少明确终止信号 |

`run_tests` 是**唯一有副作用的工具**（执行外部进程），因此额外做了：命令由工具内部构造
（不接受任意命令字符串）、`shell=False`、超时限制、输出截断。

### 5.3 路径安全

所有文件类工具都必须先经过 `PathGuard.resolve()`：

```python
resolved = candidate.resolve()          # 展开 .. 与符号链接
if resolved != root and not resolved.is_relative_to(root):
    raise PermissionError("拒绝访问工作区之外的路径")
```

两个要点：

- 必须在 `resolve()` **之后**判断。用 `startswith` 做字符串前缀比较会被
  `/workspace_evil` 这类同前缀路径绕过。
- 自动跳过 `.git` / `node_modules` / `__pycache__` / `.venv` 等噪声目录，
  避免把无关内容塞进上下文。

---

## 6. 静态分析器设计

### 6.1 双策略：AST 优先，启发式兜底

```
analyze_file(path)
  ├─ .py  → analyze_python()     : ast.parse → NodeVisitor → 精确规则
  ├─ .js/.ts/.java/.go/... → analyze_heuristic() : 正则规则表
  └─ 其他 → 只返回行数
```

**为什么不用 flake8 / pylint / eslint？**

- 它们是**外部进程依赖**，会让「clone 下来就能跑」变成「先配一堆工具链」；
- 本项目要演示的是 Agent 架构，分析器只是「工具」的一个示例实现；
- 自己写反而能精确控制输出格式（结构化 + 严重度 + 行号），直接喂给模型。

**为什么 Python 用 AST 而其他语言用正则？**

Python 的 `ast` 是标准库，零成本获得精确的语法树：函数边界、圈复杂度、默认参数节点、
异常处理节点都能准确获取。其他语言要获得同等精度就必须引入第三方解析器，与上一条冲突。
**启发式结果会在输出的 `method` 字段标注为 `heuristic`**，提示词中也要求模型不要把
启发式结论当作精确事实——这是对「模型可能过度自信」的防御。

### 6.2 规则清单（Python）

| 规则 | 严重度 | 检测方式 |
|---|---|---|
| 语法错误 | 阻塞 | `ast.parse` 抛 `SyntaxError` |
| 硬编码凭据 | 阻塞/严重 | 赋值目标名匹配敏感词 + 字面量形状匹配 |
| 危险调用（eval/exec/pickle.loads） | 阻塞 | 调用名匹配 |
| `subprocess(shell=True)` | 严重 | 关键字参数检查 |
| 可变默认参数 | 严重 | 默认值节点是 `List`/`Dict`/`Set`/`list()`/`dict()` |
| 裸 `except:` | 严重 | `ExceptHandler.type is None` |
| 静默吞异常 | 严重 | `ExceptHandler.body` 全是 `Pass` |
| 超长函数（>80 / >150 行） | 一般/严重 | `end_lineno - lineno` |
| 圈复杂度（>12 / >20） | 一般/严重 | 统计决策节点数 |
| `assert` 作运行时校验 | 提示 | `Assert` 节点 |
| TODO/FIXME、超长行 | 提示 | 逐行正则 |

### 6.3 圈复杂度的计算

```python
COMPLEXITY_NODES = (If, For, AsyncFor, While, ExceptHandler, With, AsyncWith,
                    BoolOp, IfExp, Assert, comprehension)
complexity = 1 + count(节点，在函数子树内)
```

采用 McCabe 的常用简化定义：函数基础复杂度为 1，每个决策点 +1。
`BoolOp`（`and`/`or`）整体算一个决策点，这是简化处理，会在文档中说明，避免误导。

---

## 7. 上下文记忆设计

### 7.1 两级策略：滑窗 + 摘要压缩

```
消息数 ≤ summary_trigger(默认 1.5×window)
    → 全量保留
消息数 >  summary_trigger
    → 丢弃最旧的消息，窗口内保留最近 window 条
    → 被丢弃的部分交给 LLM 压缩成一段摘要，插在 system 之后
```

**为什么需要摘要而不是纯滑窗？**
纯滑窗会让 Agent「失忆」：一个 12 步的审查任务，早期用 `analyze_code` 发现的问题一旦滑出窗口，
模型在写最终报告时就可能漏掉它。摘要用很小的 token 成本保留了「已确认的问题清单」这个关键信息。

**摘要器的容错**：摘要失败（网络抖动、返回异常）只记录 warning，不影响主流程，
继续使用旧摘要——记忆是辅助功能，不能成为故障点。

### 7.2 一个容易踩的坑：tool 消息配对

OpenAI 协议要求：`role="tool"` 的消息必须紧跟在带有对应 `tool_calls` 的 assistant 消息之后。
如果滑窗裁剪恰好把 assistant 消息裁掉、只留下 tool 消息，**下一次请求会被服务端直接拒绝**。

因此在裁剪边界上做了修正：

```python
cut = len(messages) - window
while cut < len(messages) and messages[cut].role == "tool":
    cut += 1          # 向前推，保证不以 tool 消息开头
```

`tests/test_memory.py::test_trim_boundary_never_starts_with_tool_message` 专门覆盖这个边界。

---

## 8. 错误处理与重试

### 8.1 三级分类

```
LLM 调用
 ├─ 可恢复：APITimeoutError / APIConnectionError / RateLimitError
 │           InternalServerError / APIStatusError(429, 5xx)
 │   → 指数退避重试，最多 max_retries 次；耗尽抛 LLMRetryExhausted
 └─ 不可恢复：AuthenticationError(401) / NotFoundError(404) / BadRequestError(400)
     → 不重试，立即翻译成 LLMConfigError 并给出可操作的提示

工具执行
 └─ 全部捕获 → ToolResult(ok=False, 可读错误文本) → 回喂模型自我纠正

Agent 循环
 ├─ LLMConfigError  → 直接终止（Key 错了重试没意义）
 ├─ 其他 LLM 异常    → 产出 error 事件后抛 AgentError
 └─ 步数耗尽         → 抛 StepLimitExceeded，CLI 走兜底结论
```

### 8.2 退避策略

```python
delay = min(backoff_cap, backoff_base * 2**attempt) * (0.7 + 0.6 * random.random())
```

- **指数增长**：1.5s → 3s → 6s → 12s，避免持续冲击已过载的服务端。
- **上限封顶**：最长 30s，防止等待时间失控。
- **随机抖动**：±30% 抖动，避免多个客户端同时重试形成尖峰（thundering herd）。

SDK 自带的重试被显式关闭（`max_retries=0`），否则框架重试与业务重试会叠加，延迟被放大。

### 8.3 错误信息的可操作性

错误处理不只是「不崩」，还要告诉用户**怎么办**：

```python
if status == 401:
    return LLMConfigError("鉴权失败（401）：API Key 无效或与 base_url 不匹配。")
```

`selfcheck` 子命令把这些检查前置：Python 版本、依赖、配置、工具注册、API 连通性，
一次跑完给出问题清单和修复建议。

---

## 9. 一次完整执行的时序

以 `python -m code_review_agent review app/` 为例：

```
用户        CLI            Agent          LLM            工具
 │           │               │             │              │
 │─命令──────▶│               │             │              │
 │           │─构造 Agent────▶│             │              │
 │           │               │─system+task─▶│              │
 │           │               │◀─tool_calls──│              │
 │           │◀─tool_call 事件─│             │              │
 │           │               │───────────execute(list_directory)──▶│
 │           │               │◀──────────目录树文本───────────────│
 │           │◀─tool_result──│             │              │
 │           │               │─回喂结果─────▶│              │
 │           │               │◀─tool_calls──│ (analyze_code)│
 │           │               │───────────execute(analyze_code)───▶│
 │           │               │◀──────────结构化问题清单──────────│
 │           │               │─回喂─────────▶│              │
 │           │               │◀─最终报告文本─│              │
 │           │◀─final 事件────│             │              │
 │◀─报告─────│               │             │              │
 │           │─打印统计───────│             │              │
```

---

## 10. Web 层设计

### 10.1 为什么用标准库而不是 Flask / FastAPI

| 维度 | 框架方案 | 本项目方案 |
|---|---|---|
| 依赖 | 连带引入 WSGI/ASGI 服务器、模板引擎、路由库 | 零新增依赖，`http.server` 是标准库 |
| 与项目定位的一致性 | 与「运行时只依赖 openai」的取舍冲突 | 一致：评审者 `pip install -r requirements.txt` 后即可用 |
| 需要的功能 | 路由、JSON、SSE、静态文件——仅此四项 | 手写约 300 行即可覆盖 |

框架的价值在大型生产服务；本项目 Web 界面是**演示入口**，
用标准库反而更符合「依赖最小化」的整体设计取向。

### 10.2 三个必须解决的工程问题

**① Agent 是有状态的，HTTP 是无状态的**

`ConversationMemory` 承载完整对话历史，所以「审查完继续追问」必须复用同一个
`ReviewAgent` 实例。解法是引入带 id 的 `Session` 把状态挂在服务端：

```
POST /api/review  →  创建 Session(agent, options)  →  返回 session_id
POST /api/chat    →  按 session_id 取出同一个 Session  →  复用记忆
```

`Session` 同时承担三件事：持有 Agent、缓存事件、管理订阅者。

**② Agent 是同步阻塞的，HTTP 需要流式**

不能等整个审查跑完再返回（真实审查要几十秒到几分钟）。解法：

```
后台 worker 线程 ──emit──▶ 事件列表 ──notify──▶ SSE 生成器 ──▶ 浏览器
```

- 每个会话一个 worker 线程，**串行**执行任务（`busy` 标志拒绝并发，避免记忆被交叉写入）；
- 事件写入内存列表后通过 `threading.Condition` 广播唤醒订阅者；
- SSE 生成器按游标增量读取，**游标从 0 开始**，因此「先开跑、后连接」也不会漏事件。

**③ 单条工具结果可能很大**

一次 `analyze_code` 可返回 20KB 文本，原样推给浏览器会让 SSE 负载膨胀。
解法是在 `events.py` 统一截断工具结果（上限 4000 字符，保留首尾并标注 `truncated`），
而**推理文本不截断**——用户就是要读它。

### 10.3 会话生命周期与清理

```
created ──start_review()──▶ running ──worker 结束──▶ ready
                                     └─异常────────▶ error
```

- 同一会话 `busy` 时再提交任务返回 **409 Conflict**；
- 事件缓冲上限 4000 条，超出后追加一条一次性告警并停止记录，防止内存无界增长；
- 会话表最多 20 个、TTL 1 小时：新建时淘汰最旧的**空闲**会话（正在跑的保留）；
- SSE 在「任务结束 + 事件发完」后主动发 `event: closed` 再关闭连接。
  前端收到后**不再自动重连**——否则 EventSource 的自动重连会把历史事件重放一遍，
  界面上就会出现重复的时间线。

### 10.4 安全边界

Web 界面会把「读服务端任意文件」和「消耗你的 LLM 额度」两种能力暴露给能访问该端口的人，
因此做了五层约束：

| 层次 | 措施 |
|---|---|
| 网络 | 默认只监听 `127.0.0.1`；`--host 0.0.0.0` 时打印醒目警告 |
| 目录 | **白名单策略**（`DirectoryAccessPolicy`）：默认只允许服务启动目录，其他目录须用 `--allow-root` 显式加入。校验在 `resolve()` 之后进行，防 `..` 与符号链接绕过，也不会被 `/work` 与 `/work_evil` 的前缀相似性骗过；拒绝时返回应追加的具体参数 |
| 凭证 | Key 可来自服务端 `.env` 或页面输入；页面输入的 Key **只在内存中使用一次**，不落盘、不写日志、不回显（`/api/test-key` 的响应体有测试断言不含 Key） |
| 出站 | 请求体**不能覆盖 `base_url`**——只允许 `model`/`temperature`/`max_tokens`/`timeout` 四个白名单字段，否则界面会变成 SSRF 跳板 |
| 输入 | 请求体上限 1MB、自由文本限长、`max_steps` 限 1~50、密钥限 256 字符且禁止换行 |

**为什么目录用白名单而不是「随便填」**：审查代码意味着把文件内容发给 LLM。
如果允许任意路径，界面就等价于一个「远程文件读取器」——配合 `--host 0.0.0.0`
可以把 `~/.ssh/id_rsa` 读出来送进模型。白名单把授权动作显式化（每个项目加一次
`--allow-root`），代价是一次启动参数，收益是默认安全。

落地方式上有个细节值得记录：默认根与额外根统一收进 `DirectoryAccessPolicy`，
由它同时负责「解析」与「判定命中哪个根」。命中的根会随 `ReviewOptions.root`
一路传到界面回显，用户在选了额外目录时能确认自己到底在审查哪里。
早期实现把这条约束写成一个布尔开关（`require_root_bound`），加入白名单后
它自然退化成策略对象——这类「开关膨胀成策略」的重构是值得的。

**关于「允许前端传 Key」这个取舍**：它确实削弱了「凭证只属于服务端」的模型——
能访问该端口的人可以用**他自己的** Key，也可以用你的（若不填就走服务端配置）。
之所以接受这个让步：

1. 界面的定位是本地单机工具，默认只监听回环地址；
2. 相比「必须去编辑 `.env` 并重启服务」的可用性收益，对教学/演示场景更划算；
3. 风险被 `base_url` 白名单限制在「用你的 Key 调用你配置的那个服务商」，
   而**不能**被用来把服务端当跳板访问内网地址。

真正的加固方案（一次性 token 鉴权）列在 13 节的后续改进里。

需要审查其他项目时，用 `--allow-root D:\其他项目` 把它加入白名单，
或直接在那个目录下启动服务（`web -w D:\其他项目`）——两条路都不会放开全局限制。

### 10.5 前端的两个实现选择

- **不引入任何 CDN 资源**：内联 CSS/JS + 自写的极简 Markdown 渲染器（约 60 行）。
  这样离线环境、内网机器都能正常显示；代价是渲染能力弱于 marked.js
  （不支持表格以外的复杂语法），对本项目的报告结构足够。
- **事件类型直接映射为 SSE 事件名**：`event: tool_call` / `event: final` …
  前端用 `addEventListener` 按类型注册，避免在客户端写一套 switch 分发。

---

## 11. 可扩展性：如何新增一个工具

以新增「查询 git 最近变更」为例，只需两步：

```python
# 1) 在 tools/builtin.py 中实现纯函数
def tool_git_diff(ctx: ToolContext, ref: str = "HEAD~1") -> ToolResult:
    completed = subprocess.run(["git", "diff", "--stat", ref],
                               cwd=ctx.workspace, capture_output=True, text=True)
    if completed.returncode != 0:
        return ToolResult(content=f"[工具错误] git 执行失败：{completed.stderr}", ok=False)
    return ToolResult(content=completed.stdout or "(无变更)")

# 2) 在 build_default_registry 中注册
registry.register(
    "git_diff",
    "查看相对某个提交的代码变更，用于只审查本次改动。",
    _schema({"ref": {"type": "string", "description": "对比基准，默认 HEAD~1"}}),
    lambda ref="HEAD~1": tool_git_diff(ctx, ref),
)
```

**无需改动 Agent、CLI、Prompt**：工具 Schema 会自动出现在下一次 LLM 请求的 `tools` 参数里，
系统提示词中「工作流程」一节描述的是通用策略，不绑定具体工具名。

同理，扩展其他能力：

- **换 LLM 服务商** → 改 `.env`（`llm.py` 只依赖 OpenAI 协议）
- **换 Prompt 风格** → 改 `prompts.py`，或传 `system_prompt=` 覆盖
- **换记忆策略** → 替换 `ConversationMemory`，Agent 只依赖 `to_messages()/add()/reset()`
- **接入 Web 界面** → 消费 `AgentEvent` 流即可，Agent 层无需改动

---

## 12. 验证情况说明

**代码构建环境无法执行任何外部程序**（沙箱禁止进程派生），因此下列内容由使用者在本机执行验证，
仓库内已提供对应的自动化测试：

| 验证项 | 方式 |
|---|---|
| 静态分析规则正确性 | `pytest tests/test_analyzer.py` |
| 工具行为与路径安全 | `pytest tests/test_tools.py` |
| Agent 循环 / 终止条件 / 错误恢复 | `pytest tests/test_agent.py`（注入假 LLM，离线） |
| 记忆裁剪与摘要 | `pytest tests/test_memory.py` |
| 重试退避与异常翻译 | `pytest tests/test_llm.py` |
| 会话生命周期与 SSE 帧 | `pytest tests/test_web.py` |
| HTTP 路由与 SSE 端到端 | `pytest tests/test_web_http.py`（真实启动 http.server） |
| CLI 子命令与退出码 | `pytest tests/test_cli.py` |
| 真实 LLM 连通性 | `python -m code_review_agent selfcheck --probe` |
| 端到端审查效果 | `python -m code_review_agent review app/ --save` |
| Web 界面人工验收 | `python -m code_review_agent web` 后浏览器操作 |

测试使用 `FakeOpenAI` 注入 `LLMClient`，覆盖了「模型返回工具调用 → 执行 → 回喂 → 收敛」
的完整链路，**不依赖网络与 API Key**，因此在任何环境都能复现。
Web 层测试进一步用 `FakeAgent` 替身，但会**真实启动 HTTP 服务并发起请求**，
因此路由、状态码、SSE 分帧与连接关闭都是真实验证，而非模拟。

---

## 13. 已知取舍与后续改进

| 取舍 | 现状 | 若要改进 |
|---|---|---|
| 串行工具执行 | 同一轮多个 tool_calls 顺序执行 | 只读工具可并行（`concurrent.futures`），有副作用的保持串行 |
| 无 token 预算控制 | 只有步数上限 | 增加 token 预算，接近上限时提前触发摘要与收敛 |
| 单文件静态分析 | 不解析跨文件调用 | 引入 `mypy`/`pyright` 作为可选工具，或在 Prompt 中引导跨文件检索 |
| 无缓存 | 同一文件重复读取会重复付费 | 对 `read_file`/`analyze_code` 结果做内容哈希缓存 |
| 启发式规则误报 | JS/Java 等语言正则会误报 | 输出中已标注 `heuristic`；可接入 tree-sitter 提升精度 |
| 无人工确认 | 工具自动执行 | 对有副作用的工具（`run_tests`）增加 `--dry-run` 或确认机制 |
| Web 会话在内存中 | 重启即丢，且不能多进程部署 | 换成 SQLite 或 Redis 存事件与消息，支持横向扩展 |
| Web 无鉴权 | 仅靠监听本机、目录约束与 base_url 白名单保护 | 增加一次性 token（启动时打印在控制台），校验 `Origin` |
| 前端可传 Key | 便利性优先，Key 只在内存用一次 | 若要更严：改为服务端下发短期凭证，或强制只用 `.env` |
| SSE 单订阅者语义 | 多标签页打开同一会话会各自收到全量事件 | 按会话共享一个广播器，或引入 `Last-Event-ID` 断点续传 |
| 自写 Markdown 渲染 | 不支持表格以外的复杂语法 | 需要更完整渲染时引入本地打包的 marked.js（不引 CDN） |
