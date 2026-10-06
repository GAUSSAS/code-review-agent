# 代码审查 Agent（Code Review Agent）

一个基于 **LLM 工具调用（Function Calling）** 的自动化代码审查助手。
它不只是一次 Prompt 调用——Agent 会像人一样**先侦查项目结构、再针对性读代码、
跑静态分析、实际执行测试**，最后基于取到的证据输出结构化审查报告。

```
输入任务 → LLM 推理 → 选择工具 → 执行工具 → 结果回喂 → 再推理 → … → 输出报告
              ↑                                              │
              └────────────── 上下文记忆（滑窗 + 摘要）────────┘
```

---

## 1. 功能特性

| 特性 | 说明 |
|---|---|
| **完整 Agent 循环** | `推理 → 工具调用 → 观察结果 → 继续推理`，直到模型不再请求工具或主动交付结论 |
| **Web 界面 + CLI** | 零额外依赖的浏览器界面（标准库 `http.server` + SSE 实时推送），以及完整命令行 |
| **7 个内置工具** | 目录侦查、带行号读文件、正则搜索、静态分析、执行 pytest、写报告、结束任务 |
| **多语言静态分析** | Python 用 `ast` 做语法树分析；JS/TS/Java/Go 等用启发式规则扫描 |
| **路径安全** | 所有文件工具都经过 `PathGuard` 校验，拒绝访问工作区之外的路径 |
| **上下文记忆** | 滑窗裁剪 + LLM 摘要压缩，长会话不会撑爆上下文 |
| **错误处理与重试** | LLM 侧指数退避重试、异常分类翻译；工具侧异常转为文本回喂让模型自我纠正 |
| **同步与流式双接口** | `review()` 返回事件生成器，CLI 与 Web 共用同一事件流 |
| **离线可测** | 测试注入假 LLM，无需 API Key 和网络即可验证全部核心逻辑 |
| **多服务商兼容** | 只依赖 `openai` SDK，通过 `base_url` 适配 DeepSeek / 通义 / Kimi / 智谱 / Ollama / vLLM |
| **安全约束** | Key 可在页面填写（内存即用即弃、不落盘不回显）；被审查目录限制在服务启动目录之内；默认只监听本机 |

---

## 2. 快速开始

### 2.1 环境要求

- Python **3.10+**（代码使用了 `X | Y` 类型语法）
- 一个 OpenAI 兼容的 LLM 服务（推荐 [DeepSeek](https://platform.deepseek.com)）

### 2.2 安装

```bash
# 1) 创建虚拟环境（可选但推荐）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2) 安装依赖（只有 openai 一个运行时依赖）
pip install -r requirements.txt
```

### 2.3 配置

```bash
# 复制配置模板
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

编辑 `.env`，**至少填写 API Key**：

```ini
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-你的真实Key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
```

> 切换到其他服务商只需改这三项。例如通义千问：
> `LLM_PROVIDER=dashscope`、`LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`、`LLM_MODEL=qwen-plus`。

### 2.4 自检（建议先跑这个）

```bash
# 检查 Python 版本、依赖、配置、工具注册情况
python -m code_review_agent selfcheck

# 额外发起一次真实 LLM 调用，验证网络与 Key 是否可用
python -m code_review_agent selfcheck --probe
```

---

## 3. 使用方式

### 3.1 审查整个工作区

```bash
python -m code_review_agent review
```

运行时会实时打印 Agent 的每一步：

```
▸ 已接收任务，工作区：/path/to/project｜工具：analyze_code, finish_review, ...

🤔 我先了解一下项目结构。

🔧 调用工具 list_directory {"path": ".", "max_depth": 2}
   ✓ list_directory 返回：
   目录：.
     app/
       service.py  (1KB)
   统计：1 个子目录，3 个文件

🔧 调用工具 analyze_code {"path": "app"}
   ✓ analyze_code 返回：
   静态分析完成：分析 1 个文件，共 6 条发现 ...

========================================================================
审查报告
========================================================================
## 一、项目概览
...
```

### 3.2 审查单个文件

```bash
python -m code_review_agent review app/service.py
```

### 3.3 指定关注方向

```bash
python -m code_review_agent review --focus "并发安全与竞态条件"
python -m code_review_agent review --language "Python 3.12 / FastAPI" --focus "性能"
```

### 3.4 保存报告

```bash
python -m code_review_agent review --save                    # 保存为 review_report.md
python -m code_review_agent review --save -o docs/audit.md   # 自定义路径
```

### 3.5 审查后追问

```bash
python -m code_review_agent review -i
```

进入交互模式后可以继续追问，记忆会跨轮保留：

```
你> 第 2 个问题请给出修复后的完整代码
你> 再帮我针对这个问题写一个回归测试
你> :report        # 重新打印上一份报告
你> exit
```

### 3.6 多轮问答模式

```bash
python -m code_review_agent chat                        # 进入交互模式
python -m code_review_agent chat "这个项目有哪些安全风险？"
```

### 3.7 查看工具清单

```bash
python -m code_review_agent tools
```

### 3.8 其他实用参数

```bash
python -m code_review_agent review --no-color      # 禁用彩色输出（便于重定向到文件）
python -m code_review_agent review -v              # 显示 DEBUG 日志与完整工具输出
python -m code_review_agent review --max-steps 20  # 放宽 Agent 步数上限
python -m code_review_agent review --json          # 额外输出 token / 工具调用统计
python -m code_review_agent review -w ./src        # 指定被审查目录
python -m code_review_agent review --no-few-shot   # 系统提示词中不带 Few-shot 示例
```

> 通用参数（`-w` / `-v` / `--no-color` / `--env-file`）写在**子命令之后**，
> 例如 `python -m code_review_agent review -w ./src -v`。

### 3.9 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 参数错误 |
| 3 | 配置错误（缺 Key、Key 无效、base_url 错误） |
| 4 | 运行期错误（网络失败、LLM 重试耗尽等） |
| 5 | 达到最大步数仍未收敛 |

---

## 4. Web 界面

除了命令行，项目自带一个**零额外依赖**的浏览器界面（标准库 `http.server` + SSE），
适合演示「Agent 边推理边调用工具」的过程。

### 4.1 启动

```bash
python -m code_review_agent web
```

输出：

```
====================================================================
  代码审查 Agent —— Web 界面已启动
====================================================================
  访问地址 : http://127.0.0.1:8765/
  服务目录 : D:\huanjing
  模型     : deepseek-chat（deepseek）
  API Key  : 已配置
  按 Ctrl+C 停止服务
====================================================================
```

浏览器打开该地址即可。常用参数：

```bash
python -m code_review_agent web --port 9000        # 换端口
python -m code_review_agent web --port 0           # 自动选空闲端口
python -m code_review_agent web --no-browser       # 不自动打开浏览器
python -m code_review_agent web --host 0.0.0.0     # 监听所有网卡（见下方警告）
```

**要审查其他目录**，用 `--allow-root` 显式加入白名单（可重复指定）：

```bash
# 单目录
python -m code_review_agent web --allow-root D:\projects\demo

# 多个目录
python -m code_review_agent web --allow-root D:\projects\demo --allow-root E:\work\api
```

然后在页面上点「**浏览…**」按钮，会弹出**系统原生的文件夹选择框**（Windows 的「选择文件夹」窗口），
选好后路径自动填入，点「开始审查」即可。也可以直接从下拉框快捷选用已允许的目录，或手动粘贴路径。

> **为什么对话框由服务端弹出？** 浏览器出于安全**无法获取本地绝对路径**——
> `webkitdirectory` 与 File System Access API 都只返回相对路径。所以要让您看到
> 真实的 `D:\projects\demo`，只能由服务端在您的桌面会话里弹出原生对话框，
> 再把路径回填到页面。这在本机使用时完全成立；若是远程连接（无桌面会话），
> 对话框会打不开，此时请手动填写路径。

> **为什么还要 `--allow-root`？** 对话框本身可以浏览整个磁盘（原生行为），
> 但**选中后仍要过白名单**。不在白名单内的目录会被标红并提示该加什么参数——
> 这样既保留了「随便点着选」的顺手，又不会把界面变成「读取本机任意文件并送给 LLM」的工具。

> 启动后**会自动打开默认浏览器**。若自动打开失败（远程会话、精简系统等），
> 手动复制控制台打印的地址即可，不影响服务运行。

### 4.2 界面功能

| 区域 | 能力 |
|---|---|
| 左侧表单 | **API Key（可测试连接）**、**被审查目录（「浏览…」弹出系统原生文件夹选择框 / 下拉快捷选用 / 手动粘贴）**、单个文件模式、重点关注、技术栈提示、**最大步数（1~50，输入即时校验）**、是否保存报告、是否允许追问 |
| 实时时间线 | 逐条显示推理文本、工具调用参数、工具返回结果（成功/失败配色，超长输出自动截断并标注） |
| 最终报告 | Markdown 渲染（标题/代码块/列表/表格/严重程度着色）+ 一键复制 |
| 统计面板 | 步数、工具调用次数、输入/输出 token、总耗时（**随事件实时刷新**），并带一个手动刷新按钮与「更新于 hh:mm:ss」时间戳 |
| 刷新统计按钮 | 任务进行中也能点：直接向服务端拉取当前权威数值（含 `tools_used`）。若实时数字不动、点它能跳到正确值，说明是事件流（SSE）出了问题而非统计本身 |
| 追问输入框 | 审查结束后出现，复用**同一个 Agent 实例**，因此上下文记忆连续 |
| 工具标签 | 列出 7 个工具，点击查看用途说明 |

### 4.4 在页面上填写 API Key

界面支持**直接输入 API Key**，因此不配置 `.env` 也能使用：

1. 在左侧「API Key」输入框粘贴 Key（`password` 类型，输入时不显示明文）；
2. 点「测试连接」——会发起一次 8 token 的最小请求，直接告诉你 Key、网络、
   模型名是否都可用，**不用等到审查跑一半才发现问题**；
3. 点「开始审查」。填写的内容会覆盖服务端 `.env` 中的配置。

防误用的细节：

| 问题 | 处理方式 |
|---|---|
| 会不会被写进磁盘？ | 不会。Key 只存在于服务端内存，随请求构造 Agent 后即用即弃 |
| 会不会出现在日志里？ | 不会。服务端不打印 Key；错误信息经 `_translate()` 过滤，只暴露状态码与原因 |
| 会不会回显给浏览器？ | 不会。`/api/health` 只回 `api_key_configured` 布尔值；`/api/test-key` 的响应体不含 Key（有测试断言） |
| 刷新页面要重填吗？ | 默认要重填（最安全）。勾选「在本标签页内记住」后会存入 `sessionStorage`——**关闭标签页即清除**，且不跨标签页、不跨域名 |
| 能用它把请求转发到别处吗？ | 不能。请求体只能覆盖 `model` / `temperature` / `max_tokens` / `timeout`，**`base_url` 被白名单排除**，防止界面变成 SSRF 跳板 |
| Key 无效怎么办？ | 「测试连接」会返回具体原因（401 鉴权失败 / 404 模型名或地址错误 / SSL 连接问题） |

> 留空则回退到服务端 `.env`；两处都没有时，「开始审查」会提示先填写 Key。

### 4.5 接口一览

Web 层是纯前后端分离的 JSON/SSE 接口，也可以直接脚本调用：

| 路径 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 单页界面 |
| `/api/health` | GET | 模型、工具数、是否已配置 Key |
| `/api/tools` | GET | 7 个工具的 JSON Schema |
| `/api/stats?sid=` | GET | 主动拉取该会话的当前统计与状态（界面「刷新统计」按钮使用） |
| `/api/review` | POST | 创建会话并启动审查，返回 `session_id`（可携带 `api_key` 与模型参数覆盖） |
| `/api/test-key` | POST | 用提交的 Key 发一次 8 token 最小请求，验证连通性 |
| `/api/browse` | POST | 在服务端弹出**系统原生文件夹选择框**，返回选中路径（并标注是否在白名单内） |
| `/api/stream?sid=` | GET | SSE 事件流（`start`/`assistant`/`tool_call`/`tool_result`/`final`/`stats`/`done`/`closed`） |
| `/api/chat` | POST | 在同一会话上追问 |

```bash
# curl 示例：启动一次审查
curl -X POST http://127.0.0.1:8765/api/review \
     -H "Content-Type: application/json" \
     -d '{"workspace": ".", "focus": "安全", "max_steps": 12}'
```

### 4.6 安全约束

界面默认只监听 `127.0.0.1`，并且：

- **API Key 支持两种来源**：服务端 `.env`，或页面直接填写（见 4.4）。页面上填写的 Key
  **只在内存中使用一次**，不落盘、不写日志、不回显；
- **被审查目录采用白名单**：默认只允许服务启动目录，其他目录须用 `--allow-root` 显式加入。
  校验在 `Path.resolve()` 之后进行，因此 `..` 与符号链接无法绕过，`D:\work_evil`
  也不会被 `D:\work` 的字符串前缀骗过；未授权目录会被 400 拒绝，并在响应中给出应追加的参数；
- 文件工具本身还有一层 `PathGuard` 兜底；
- 请求体上限 1MB，自由文本字段限长，`max_steps` 限定 1~50，密钥长度上限 256 字符；
- 请求体**只能覆盖 `model` / `temperature` / `max_tokens` / `timeout`**，
  `base_url` 被白名单排除——否则界面会变成让服务端向任意地址发请求的 SSRF 跳板。

> ⚠️ **本工具没有鉴权机制**。用 `--host 0.0.0.0` 暴露到局域网或公网，会让任何能访问该端口的人
> 使用你的 API Key 审查你的代码。仅在可信网络下临时使用。

---

## 5. 内置工具

Agent 通过 function calling 自主决定调用哪个工具。所有工具的 JSON Schema 在运行时自动导出给模型。

| 工具 | 参数 | 作用 |
|---|---|---|
| `list_directory` | `path`, `max_depth` | 列出目录树与文件大小，用于了解项目布局 |
| `read_file` | `path`, `start_line`, `end_line` | 带行号读取源码，支持分段，有体积上限保护 |
| `search_code` | `pattern`, `path`, `glob`, `case_sensitive`, `max_results` | 正则跨文件搜索，返回 `文件:行号: 内容` |
| `analyze_code` | `path`, `max_files` | 静态分析，输出结构化问题清单与度量 |
| `run_tests` | `path`, `args`, `timeout` | 实际执行 pytest，把真实结果作为审查证据 |
| `write_report` | `content`, `path` | 把 Markdown 报告写入工作区 |
| `finish_review` | `report`, `summary` | 显式结束审查并交付最终结论 |

### 静态分析能发现什么

**Python（`ast` 语法树，结论精确）**

- 圈复杂度超阈值（>12 提示，>20 记为严重）
- 超长函数（>80 行提示，>150 行记为严重）
- 可变默认参数（`def f(x, cache={})` 这一类经典陷阱）
- 裸 `except:` 与静默吞异常（`except ...: pass`）
- 危险调用：`eval` / `exec` / `os.system` / `pickle.loads` / `yaml.load`
- `subprocess(..., shell=True)` 命令注入风险
- 硬编码凭据（变量名含 key/secret/token/password + 字面量）
- `assert` 用作运行时校验（`-O` 下会被移除）
- 语法错误直接标为 `blocker`
- 度量：函数数、类数、平均/最大复杂度、最长函数 Top5、docstring 覆盖率

**其他语言（正则启发式，结果中会标注 `method=heuristic`）**

- JavaScript：`var` 声明、`==` 松散比较、空 catch、`innerHTML` 赋值（XSS）、残留 `console.log`、`eval`
- TypeScript：`any` 滥用、`@ts-ignore`、空 catch、`innerHTML`
- Java：空 catch、`System.out.println`、`printStackTrace`、字符串用 `==` 比较
- Go：错误被 `_` 忽略、`fmt.Print` 直接输出
- 全语言：TODO/FIXME、疑似硬编码凭据、超长行

---

## 6. Prompt 设计

Prompt 集中放在 [`code_review_agent/prompts.py`](code_review_agent/prompts.py)，与业务逻辑分离，便于调优和对比。

**系统提示词的结构**（五段式）：

1. **角色定义**：资深代码审查专家，强调「基于证据、绝不猜测」。
2. **工作流程**：先侦查 → 再取证 → 可验证 → 后结论。明确要求「不要一上来就逐文件读」，
   这是控制 token 成本的关键约束。
3. **审查维度**：正确性 Bug、安全性、健壮性、可维护性、性能，并各给出具体检查点。
4. **输出规范**：固定 Markdown 章节 + 每条问题的固定字段（文件 / 证据 / 问题 / 建议），
   严重程度限定为 `阻塞` / `严重` / `一般` / `建议` 四档。
5. **硬性约束**：每条结论必须有文件名与行号；不确定要明说；只评价代码不评价作者；控制篇幅。

**Few-shot**：附一个完整的输出范例（路径穿越漏洞），示范「证据 → 后果 → 可落地修复代码」的粒度。
可用 `--no-few-shot` 关闭，用于对比实验。

**降级策略**：当 Agent 达到最大步数仍未收敛时，会追加一条「不要再调用任何工具，基于已有信息
直接给结论，证据不足之处标注未能验证」的指令做兜底，保证任何情况下都有输出。

---

## 7. Agent 架构说明

```
┌──────────────────────────────────────────────────────────────────┐
│  表现层（两个消费者，共用同一事件契约 AgentEvent）                 │
│   ├ cli.py           review / chat / web / tools / selfcheck     │
│   │                   终端渲染，同步阻塞                          │
│   └ web/             浏览器界面，SSE 流式推送                     │
│       ├ server.py    路由 + http.server 装配                      │
│       ├ sessions.py  会话表、后台 worker、事件缓冲与订阅           │
│       ├ events.py    AgentEvent → JSON（工具输出截断）            │
│       └ static/      单页界面（内联 CSS/JS，无 CDN）              │
├──────────────────────────────────────────────────────────────────┤
│  Agent 层 (agent.py)      ReviewAgent.run()                      │
│   循环：memory → LLM → (无工具调用? 输出 : 执行工具 → 回喂)        │
│   终止：模型不请求工具 / 调用 finish_review / 步数耗尽             │
├──────────────────────────────────────────────────────────────────┤
│  能力层                                                           │
│   ├ llm.py       OpenAI 兼容客户端：重试退避、异常翻译、用量统计   │
│   ├ memory.py    滑窗 + LLM 摘要压缩，保证 tool 消息配对完整       │
│   ├ prompts.py   系统提示 / Few-shot / 任务模板 / 摘要模板         │
│   └ config.py    环境变量 + .env 加载，多服务商预设               │
├──────────────────────────────────────────────────────────────────┤
│  工具层 (tools/)                                                  │
│   ├ base.py      Tool 协议、ToolRegistry 分发、PathGuard 路径安全  │
│   ├ analyzer.py  AST 分析器 + 启发式规则引擎                      │
│   └ builtin.py   7 个内置工具（含 pytest 执行）                   │
└──────────────────────────────────────────────────────────────────┘
```

**关键点**：CLI 与 Web 是**平级**的两个消费者，都只依赖 `AgentEvent` 这一层契约。
新增交互方式（TUI、IDE 插件、HTTP API）不需要改动 Agent 与工具层——
这正是把主循环写成事件生成器的收益。

详见 [DESIGN.md](DESIGN.md)。

---

## 8. 项目结构

```
.
├── README.md                      # 本文件：安装与使用说明
├── DESIGN.md                      # 设计文档：架构决策与取舍
├── requirements.txt               # 依赖（运行时仅 openai）
├── pytest.ini                     # 测试配置
├── .env.example                   # 配置模板
├── .gitignore
├── code_review_agent/
│   ├── __init__.py
│   ├── __main__.py                # 支持 python -m code_review_agent
│   ├── cli.py                     # 命令行入口（review / chat / web / tools / selfcheck）
│   ├── agent.py                   # Agent 主循环
│   ├── llm.py                     # LLM 客户端
│   ├── memory.py                  # 上下文记忆
│   ├── prompts.py                 # Prompt 模板
│   ├── config.py                  # 配置加载
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                # 工具协议 / 注册表 / 路径守卫
│   │   ├── analyzer.py            # 静态分析器
│   │   └── builtin.py             # 内置工具实现
│   └── web/                       # Web 界面层（零额外依赖）
│       ├── __init__.py
│       ├── server.py              # http.server 路由 + SSE
│       ├── sessions.py            # 会话管理与事件队列
│       ├── events.py              # AgentEvent → JSON 序列化
│       ├── folder_dialog.py       # 系统原生文件夹选择框（独立进程）
│       └── static/
│           └── index.html         # 单页界面（内联 CSS/JS）
└── tests/
    ├── conftest.py                # 临时工作区与假 LLM 夹具
    ├── helpers.py                 # 假 LLM 与假响应的构造器
    ├── test_agent.py              # Agent 循环、终止条件、错误恢复
    ├── test_cli.py                # CLI 子命令冒烟测试
    ├── test_tools.py              # 工具行为与路径安全
    ├── test_analyzer.py           # 静态分析规则
    ├── test_memory.py             # 记忆裁剪与摘要
    ├── test_llm.py                # 重试、退避、异常翻译
    ├── test_web.py                # 会话生命周期、SSE 帧、事件序列化
    └── test_web_http.py           # HTTP 路由与 SSE 端到端
```

---

## 9. 运行测试

测试**完全离线**（注入假 LLM），不需要 API Key 和网络：

```bash
pip install pytest
pytest
```

> `pytest.ini` 使用了 `pythonpath` 配置项，需要 **pytest ≥ 7.0**。
> 版本较低时可改用 `python -m pytest`（会把当前目录加入 `sys.path`）。

预期输出类似：

```
159 passed in 10.2s
```

> 其中 `test_memory.py` 的裁剪边界用例做了参数化（6 组窗口大小），
> `test_web*.py` 会真实启动 HTTP 服务并发起请求（Agent 用替身，仍然离线）。

测试覆盖的关键边界情况：

- 路径穿越（`../../etc/passwd`）必须被拒绝
- 工具收到非法 JSON / 未知工具名 / 缺少必填参数时，返回可读错误而非抛异常
- 模型返回格式错误的 tool_calls 时，Agent 仍能继续
- `finish_review` 之后不得再调用模型
- 达到步数上限时抛 `StepLimitExceeded`，并可走兜底结论
- 记忆裁剪后首条消息不能是 `tool`（否则请求会被服务端拒绝），且每条 `tool` 都有配对
- 鉴权错误不重试；限流错误按指数退避重试并在耗尽后抛明确异常
- 退避延迟在抖动之后仍不超过上限
- Web：会话并发拒绝、事件缓冲上限、SSE 结束后主动关闭、路径穿越返回 400、未配置 Key 返回 503
- Web：页面填写的 Key 能覆盖服务端配置，且响应体**不含** Key（有断言）
- Web：请求体无法覆盖 `base_url`（防 SSRF）、`max_steps`/`temperature` 等越界值被拒

---

## 10. 配置项参考

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | 服务商标识，决定默认 base_url 与模型 |
| `LLM_API_KEY` | — | **必填**，API Key |
| `LLM_BASE_URL` | 按 provider | OpenAI 兼容接口地址 |
| `LLM_MODEL` | 按 provider | 模型名 |
| `LLM_TEMPERATURE` | `0.2` | 采样温度，审查任务建议低值 |
| `LLM_MAX_TOKENS` | `4096` | 单次回复上限 |
| `LLM_TIMEOUT` | `60` | 单次请求超时（秒） |
| `LLM_MAX_RETRIES` | `4` | 可恢复错误重试次数 |
| `AGENT_MAX_STEPS` | `12` | Agent 最大推理-工具步数 |
| `AGENT_MEMORY_WINDOW` | `24` | 记忆滑窗大小 |
| `AGENT_MAX_FILE_BYTES` | `200000` | 单文件读取上限 |
| `AGENT_WORKSPACE` | `.` | 被审查目录 |
| `AGENT_VERBOSE` | `false` | 是否输出调试日志 |

内置服务商预设：

| provider | base_url | 默认模型 |
|---|---|---|
| `deepseek` | `https://api.deepseek.com/v1` | `deepseek-chat` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| `dashscope` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| `moonshot` | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| `zhipu` | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| `ollama` | `http://localhost:11434/v1` | `qwen2.5-coder:7b` |
| `vllm` | `http://localhost:8000/v1` | — |

> 使用 Ollama / vLLM 时可把 `LLM_API_KEY` 填任意非空字符串（如 `ollama`）。

---

## 11. 已知限制

- **静态分析器只做「单文件」分析**：不解析跨文件调用关系、不做类型推断、不解析依赖图。
  跨文件问题依赖 LLM 阅读多个文件后自行关联。
- **非 Python 语言是启发式正则**：会有误报和漏报，工具输出中已用 `method=heuristic` 标注，
  提示词中也要求模型不要把它当作精确结论。
- **`run_tests` 仅支持 pytest**：其他框架（unittest / jest 等）需要扩展 `tool_run_tests`。
- **无代码自动修复能力**：Agent 只给建议，不直接改代码（避免误改风险）。
- **成本随项目规模上升**：大仓库会消耗较多 token，建议先用 `--focus` 缩小范围，
  或只审查变更过的文件。
- **`list_directory` 最多列出 400 个文件**，超大仓库建议直接用 `search_code` 定位。
- **Web 界面无鉴权**：设计为本地单机工具，默认只监听 `127.0.0.1`。不要用 `--host 0.0.0.0`
  暴露到不可信网络。
- **Web 会话存在内存中**：服务重启后追问上下文丢失；会话默认 1 小时未活动即回收，
  同一进程最多保留 20 个会话。
- **被审查目录采用白名单**：默认只能审查服务启动目录；其他目录需用 `--allow-root` 显式加入
  （每个项目加一次，可用 `--allow-root` 重复指定）。这是防止 Web 被当作任意文件读取工具的
  安全约束，代价是一次启动参数。未授权目录会被拒绝并提示应追加的参数。

---

## 12. 常见问题

**Q：报错「未配置 API Key」**
复制 `.env.example` 为 `.env` 并填写 `LLM_API_KEY`，然后运行 `selfcheck` 确认。

**Q：报错 401 / 403**
Key 无效或与 `LLM_BASE_URL` 不匹配。注意 DeepSeek 的地址要带 `/v1`。

**Q：报错 404**
`base_url` 写错或模型名不存在。用 `selfcheck` 打印的 base_url 与模型名核对。

**Q：卡在「已达到最大步数」**
说明项目太大或任务太宽泛。用 `--focus` 或指定具体文件，也可以 `--max-steps 20` 放宽上限。

**Q：终端中文乱码 / 颜色乱码**
Windows 下先执行 `chcp 65001`，或用 `--no-color` 关闭着色。

**Q：`run_tests` 说未安装 pytest**
`pip install pytest`。若项目本身没有测试，Agent 会改为只做静态分析。

**Q：连不上 LLM，报 SSL / 连接错误 / 超时**
按下面三层逐一排查（`selfcheck --probe` 会把失败原因打出来）：

```powershell
# 第 1 层：DNS 能不能解析
Resolve-DnsName api.deepseek.com -Type A

# 第 2 层：TCP 443 能不能连（注意 PowerShell 5.1 需先启用 TLS 1.2）
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
(New-Object Net.Sockets.TcpClient).Connect("api.deepseek.com", 443); "TCP OK"

# 第 3 层：完整 HTTPS 请求
Invoke-WebRequest https://api.deepseek.com/ -TimeoutSec 15 -UseBasicParsing | Select-Object StatusCode
curl.exe -sS -o NUL -w "%{http_code}" https://api.deepseek.com/
```

分层结论：

- **DNS 就失败** → 网络/DNS 配置问题，先解决基础连通性。
- **DNS 通、TCP 443 不通** → 防火墙或公司网络策略拦截；需要放行出站 443。
- **TCP 通、HTTPS/TLS 失败**（报「基础连接已经关闭」「SSL: CERTIFICATE_VERIFY_FAILED」）
  → 常见于企业代理、SSL 中间人检查、杀毒软件的 HTTPS 扫描。处理方式：
  1. 若公司要求走代理，设置环境变量（本项目用的 `httpx` 会自动读取）：
     ```powershell
     $env:HTTPS_PROXY = "http://proxy.corp.com:8080"
     $env:HTTP_PROXY  = "http://proxy.corp.com:8080"
     python -m code_review_agent selfcheck --probe
     ```
  2. 若企业根证书未被信任，安装该根证书到系统信任库（不要用关闭校验的方式绕过）。
  3. 临时试验可换用手机热点，用于判断是否为网络策略导致。

**Q：想用本地模型（完全离线，不依赖外网）**
用 Ollama：`LLM_PROVIDER=ollama`、`LLM_BASE_URL=http://localhost:11434/v1`、
`LLM_MODEL=qwen2.5-coder:7b`，`LLM_API_KEY` 填任意非空字符串（如 `ollama`）。
这样 Agent 全程不出网，只在本机通信。

**Q：Web 界面能打开，但点「开始审查」立刻报错**
看页面顶部状态与红色错误卡片。最常见的是服务端未配置 `LLM_API_KEY`（此时「开始审查」按钮
会被直接禁用并给出提示），其次是端口被占用、或被审查目录不在服务启动目录之下。

**Q：Web 界面提示端口被占用**
`python -m code_review_agent web --port 9000`，或 `--port 0` 让系统自动分配空闲端口。

**Q：Web 页面上工具输出被截断**
这是刻意的：单条工具结果上限 4000 字符（保留首尾并标注「已截断」），避免 SSE 负载过大。
完整文本可在服务端日志或落盘报告里查看。

**Q：点「浏览…」没有弹出文件夹选择框**
对话框是**服务端**在您的桌面会话里弹出的，因此：

- **远程/SSH 连接**：服务端所在机器没有可用的桌面会话，弹不出来。请改用手动粘贴路径。
- **精简版 Python**：缺少 `tkinter`。页面上会显示「缺少 tkinter」的提示，
  此时同样手动填写路径即可（`run_tests`/`analyze_code` 等工具不受影响）。
- **窗口被挡在后面**：对话框设置了置顶，若仍未见，检查任务栏。
- 手动填写路径**始终可用**，不依赖图形环境。

**Q：统计面板的数字不动 / 不动了**
先点数字上方的**刷新按钮**：

- **点了会更新** → 说明统计本身没问题，是事件流（SSE）没推送到位。
  检查浏览器开发者工具 Network 里 `stream` 请求是否还开着；断线后刷新页面重新开始即可。
- **点了也不变** → 说明服务端任务的统计确实是这些值。任务结束后数字会定格在最终值，
  这是正常的。

统计的正常行为：每完成一个工具调用就会刷新一次（步数、工具调用数立即跳动，
token 随之增长），并在「更新于 hh:mm:ss」显示最后一次刷新时间。

**Q：读取 GBK 编码的源码出现乱码**
工具读取文件时用 `errors="replace"` 兜底，不会崩溃，但非 UTF-8 中文注释会变成替换字符。
建议把源码统一存为 UTF-8。

---

## 13. 打包与提交

### 13.1 提交前必须确认的三件事

```powershell
cd D:\huanjing

# 1) 确认没有把真实 API Key 打进去（应为「不存在」）
Test-Path .env

# 2) 确认测试全绿
python -m pytest

# 3) 确认能跑起来
python -m code_review_agent selfcheck
```

> `.env` 已在 `.gitignore` 中，但**打包前仍要肉眼确认**——一旦把真实 Key 提交到公开仓库，
> 只能立刻去控制台吊销重签。

### 13.2 上传到 GitHub（完整步骤）

#### 第 0 步：安装 git

```powershell
winget install --id Git.Git -e --source winget
```

装完**必须重开一个终端**（PATH 才会生效），然后验证：`git --version`

#### 第 1 步：设置提交身份

GitHub 靠邮箱把提交关联到你的账号，**必须用注册 GitHub 时那个邮箱**：

```powershell
git config --global user.name  "你的GitHub用户名"
git config --global user.email "你的GitHub注册邮箱"
git config --global init.defaultBranch main
git config --global core.quotepath false
```

#### 第 2 步：配置 SSH 密钥（免密码推送）

```powershell
# 一路回车即可（默认路径、空密码）
ssh-keygen -t ed25519 -C "你的GitHub注册邮箱"

# 启动 ssh-agent 并加载私钥，避免每次推送都输密码
Start-Service ssh-agent
ssh-add "$env:USERPROFILE\.ssh\id_ed25519"

# 打印公钥，整行复制
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

把复制到的内容粘贴到 GitHub：
**Settings → SSH and GPG keys → New SSH key** → Title 随便填（如 `我的笔记本`）→
Key type 选 `Authentication Key` → 粘贴 → Add SSH key。

验证：`ssh -T git@github.com`
首次会问 `Are you sure you want to continue connecting?`，输入 `yes`。
看到 `Hi <你的用户名>! You've successfully authenticated` 即成功。

#### 第 3 步：在 GitHub 上建空仓库

网页右上角 **+ → New repository**：

| 字段 | 填什么 |
|---|---|
| Repository name | `code-review-agent` |
| Description | 代码审查 Agent —— 基于 LLM 工具调用的自动化代码评审助手 |
| 可见性 | Public（作业提交通常要求公开）或 Private |
| Add a README file | **不要勾**（本地已有 README，勾了会产生冲突） |
| Add .gitignore / license | 都选 None（本地已有 `.gitignore`） |

#### 第 4 步：在本地提交并推送

```powershell
cd D:\huanjing

git init
git add .
git status                # ← 关键：确认没有 .env / __pycache__ / .pytest_cache

git commit -m "feat: 代码审查 Agent（LLM 工具调用 + CLI/Web 双界面）"

git remote add origin git@github.com:<你的用户名>/code-review-agent.git
git push -u origin main
```

`git status` 那一步**务必肉眼确认列表里没有 `.env`**。它虽已在 `.gitignore` 中，
但这是最后一道人工检查——真实 Key 一旦进了公开仓库，只能立刻去 LLM 控制台吊销重签。

#### 第 5 步：验证

刷新 GitHub 页面应能看到全部文件。本地也可确认：

```powershell
git log --oneline     # 看到刚才那次提交
git remote -v         # 确认远端地址正确
```

#### 备选：HTTPS + 令牌（SSH 配不通时用）

1. GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic)**
   → Generate new token (classic) → 勾选 `repo` 权限 → 生成后**立刻复制**（只显示一次）。
2. 远端换成 HTTPS 形式并推送：
   ```powershell
   git remote add origin https://github.com/<你的用户名>/code-review-agent.git
   git push -u origin main
   ```
3. 凭据窗口弹出时，用户名填 GitHub 用户名，**密码处粘贴令牌**（不是账号密码，这是最常见的坑）。
   首次通过 [Git Credential Manager](https://github.com/git-ecosystem/git-credential-manager)
   保存后，后续推送无需再输。

#### 常见问题

| 现象 | 原因与处理 |
|---|---|
| `git: command not found` | 装完没重开终端，PATH 未生效 |
| `Permission denied (publickey)` | 公钥没加到 GitHub，或 ssh-agent 没加载私钥（重跑 `ssh-add`） |
| `remote origin already exists` | 已加过远端：`git remote set-url origin <新地址>` |
| `failed to push some refs` | 远端非空（建仓库时勾了 README）。先 `git pull --rebase origin main` |
| 提交里出现 `__pycache__` | 先 `git rm -r --cached __pycache__`，确认 `.gitignore` 生效后再提交 |
| 推送很慢或超时 | 网络问题，可重试；或改用 Gitee 镜像仓库 |
| 想改名/删掉远端 | `git remote rename` / `git remote remove origin` |

### 13.3 方式二：打 zip 包（不使用 git 时）

用 PowerShell 内置的 `Compress-Archive`。**关键是排除缓存目录**：
`__pycache__`、`.pytest_cache`、`.env` 都不该进包。

```powershell
cd D:\huanjing
$exclude = @('__pycache__', '.pytest_cache', '.env', 'review_report.md', '.venv', '.idea', '.vscode')
Get-ChildItem -Recurse -Force -Directory |
    Where-Object { $exclude -contains $_.Name } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Compress-Archive -Path .\code_review_agent, .\tests, `
    .\README.md, .\DESIGN.md, .\requirements.txt, .\.env.example, `
    .\.gitignore, .\pytest.ini `
    -DestinationPath ..\code-review-agent.zip -Force
```

> 用**显式文件列表**而不是 `-Path .\*`，是因为后者会把 `.pytest_cache` 一起打进去，
> 而该目录可能因权限问题导致打包报错。也可直接把 zip 上传到 GitHub/Gitee 网页的
> 「上传文件」入口，跳过本地 git。

验证压缩包内容：

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
([System.IO.Compression.ZipFile]::OpenRead("D:\code-review-agent.zip")).Entries |
    Select-Object -First 20 FullName
```

### 13.5 压缩包里应该有什么

```
code-review-agent/
├── README.md              ← 提交物 2：项目说明与使用方法
├── DESIGN.md              ← 提交物 2：设计文档
├── requirements.txt       ← 依赖清单（运行时仅 openai）
├── .env.example           ← 配置模板（不含真实 Key）
├── .gitignore             ← 确保 .env / 缓存不被提交
├── pytest.ini
├── code_review_agent/     ← 提交物 1：全部源码（28 个 .py 文件）
│   ├── agent.py           主循环
│   ├── llm.py             LLM 客户端（重试/退避/异常翻译）
│   ├── memory.py          上下文记忆
│   ├── prompts.py         Prompt 模板
│   ├── config.py          配置加载
│   ├── cli.py             命令行入口
│   ├── tools/             工具层（协议/注册表/静态分析器/7 个工具）
│   └── web/               Web 界面层（含 static/index.html）
└── tests/                 ← 159 个用例，全部离线可跑
```

**不应出现**：`.env`、`__pycache__/`、`.pytest_cache/`、`review_report.md`、`.venv/`。

### 13.6 提交说明可以这样写

> **代码审查 Agent** — 基于 LLM Function Calling 的自动化代码评审助手。
> 实现完整 Agent 循环（推理→工具调用→观察→再推理），内置 7 个工具
> （目录侦查、带行号读文件、正则搜索、静态分析、执行 pytest、写报告、结束任务），
> 静态分析器对 Python 用 `ast` 语法树、其他语言用启发式规则；
> 支持上下文记忆（滑窗 + LLM 摘要压缩）与三级错误处理（指数退避重试 / 工具错误回喂 / 步数上限兜底）。
> 提供命令行与零额外依赖的 Web 界面（标准库 `http.server` + SSE 实时推送）两种交互方式。
> 运行时依赖仅 `openai` 一个包，兼容 DeepSeek / OpenAI / 通义 / Kimi / 智谱 / Ollama / vLLM。
> 159 个自动化测试全部离线可跑，无需 API Key 与网络。
