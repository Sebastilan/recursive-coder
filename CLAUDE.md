# Recursive Coder - 项目备忘

## 当前状态

L1-L5 端到端测试全部通过（qwen-plus 模型，远程机 DESKTOP-CFEOJ9J）。
验证机制已重构为 criteria-based（Agent 自写测试脚本，Framework 只检查 returncode）。

L1-L8 端到端测试全部 PASS（qwen-plus），但均以 LEAF 模式完成（Depth=0）。

L9-L10 VRP 基准测试（测试升级分解路径）：
- L9 (cvrp_solver): CVRP 最近邻 + 2-opt，5 模块，max_agent_steps=12 触发升级分解
- L10 (vrp_benchmark): 多实例 VRP 批处理，2 种格式 + 2 种算法 + 报告，max_agent_steps=15
- 两者均测试 escalation 路径：LEAF 失败 → 重新 judge → 分解
- VRP 数据来源：BCP-Lap 项目 TSPLIB 标准测试集（E-n22-k4, E-n13-k4）

**L11 CVRP Branch-and-Price PASS**（2026-02-15，qwen-plus）：
- 最佳结果：84 API 调用，681K tokens，1333s（~22min）
- Depth=1，5 个 LEAF 子任务
- 3 个独立模块（parser/master/pricing）并行执行
- branch_and_price 判为 LEAF 直接 import 已有模块（sibling context 修复生效）
- 关键修复：judge prompt 添加 sibling context，防止递归冗余分解
- Phase 6 优化：upstream module interface injection，tokens 从 931K 降至 681K（-27%）
- 已知问题：pricing ESPPRC 实现质量随机，偶尔超时需重试（stochastic）

上下文缓存监控已集成（api_caller.py 记录 cached_tokens），隐式缓存已确认生效。

Web 搜索工具已集成（web_search + fetch_page），使用 DuckDuckGo HTML 搜索。
代理配置：config.yaml `proxy` 字段 或 `HTTPS_PROXY` 环境变量。
- 本地机代理：`http://127.0.0.1:7897`
- 远程机代理：`http://127.0.0.1:7890`（Clash）
- DDG 需要 Cookie `kl=us-en` 才能从国内正常获取英文结果。

## 快速开始

1. 设置 API Key（默认模型 qwen-plus）：

```bash
export DASHSCOPE_API_KEY="sk-..."
```

其他模型对应的环境变量：
- Claude: `ANTHROPIC_API_KEY`
- GPT: `OPENAI_API_KEY`
- Gemini: `GOOGLE_API_KEY`
- DeepSeek: `DEEPSEEK_API_KEY`

2. 运行：

```bash
cd recursive-coder
python -m recursive_coder --project-dir . run "任务描述" --verbose
```

3. 切换模型：`--model qwen-plus` / `qwen-max` / `deepseek-v3` / `claude-sonnet` / `gpt-4.1-mini`

## 常用命令

```bash
# 单次运行
python -m recursive_coder run "任务描述" --verbose

# 迭代优化（多轮自动调参）
python -m recursive_coder iterate "任务描述" --rounds 3 --auto

# 查看报告
python -m recursive_coder report ./workspace/<run_id>

# 查看优化历史
python -m recursive_coder history

# 运行测试
python -m pytest tests/ -q

# 运行 eval（mock 模式，不消耗 API）
python eval/run_eval.py --mock

# 只跑递归分解测试
python eval/run_eval.py --level 6 7 --model qwen-plus

# 跑 VRP 升级分解测试
python eval/run_eval.py --level 9 --model qwen-plus
python eval/run_eval.py --level 9 10 --model qwen-plus
```

## 架构概览

递归任务分解框架，核心循环：judge -> execute -> verify -> (fix/backtrack) -> integrate

- `processor.py` — 递归主控（judge/execute/backtrack/integrate）
- `agent_loop.py` — 多步 agent 循环（tool-use）
- `api_caller.py` — 统一 LLM API 调用（支持 Anthropic / OpenAI / OpenAI-compatible）
- `prompt_builder.py` — 模板加载 + 上下文注入
- `models.py` — TaskNode / TaskTree 数据模型
- `web_tools.py` — Web 搜索 + 页面抓取（DuckDuckGo + HTML-to-text）
- `executor.py` — 沙盒命令执行
- `evaluator.py` — 评估报告生成
- `optimizer.py` — 迭代优化（分析报告 -> 调整 prompt/config）

## 配置

主配置在 `config.yaml`，关键参数：
- `max_depth: 5` — 递归最大深度
- `max_retries: 3` — 单任务最大重试
- `max_total_api_calls: 500` — API 调用总量上限
- `command_timeout: 60` — 命令执行超时（秒）
- `proxy: ""` — Web 搜索代理（也可用 `HTTPS_PROXY` 环境变量）

## 已踩过的坑

1. **Windows \r\n 问题**：Agent 写测试脚本时如果精确比较字节（`b'42\n'` vs `b'42\r\n'`）会失败。解决：system prompt 提醒用 `.strip()` 比较
2. **Judge 预测答案必错**：Judge 预测 `expected_output` 经常算错（如词频统计），导致正确的 Agent 输出被判为失败。解决：改为 criteria-based 验证，Judge 只定义验证标准，Agent 自己写测试
3. **Judge 内嵌断言**：Judge 把精确比较写进 `verification.command`（如 `python x.py | python -c "assert ..."`），Agent 无法修改。解决：约束 command 必须是 `python test_xxx.py` 形式
4. **Windows 路径大小写**：`Path.resolve()` 在 Windows 上驱动器号大小写不一致导致路径逃逸检查误报。解决：用 `.lower()` 做大小写无关比较
5. **相对路径 workspace**：Executor 用相对路径存 workspace，不同时刻 resolve 结果不同。解决：init 时立即 `.resolve()` 为绝对路径
6. **idle detection 过于激进**：只有 write_file 才重置计数器，正常的 read_file/list_dir 探索会触发 idle 停止。解决：任何成功的 tool call 都重置计数器
7. **max_retries config 未传递**：config 中设置 `max_retries=1` 但 TaskNode 始终用默认值 3。解决：processor.py 创建 TaskNode 时读取 config
8. **GBK 编码崩溃**：Windows 中文环境 `print()` 遇到 Unicode 特殊字符（如 `\u2212`）崩溃。解决：try/except UnicodeEncodeError 后 fallback `errors='replace'`
9. **Google 搜索返回 JS 渲染页**：httpx 无法获取 Google 搜索结果（全是 JS），Bing 国内版返回垃圾结果。解决：改用 DuckDuckGo HTML 版（`html.duckduckgo.com`）+ Cookie `kl=us-en`
10. **DuckDuckGo 202 问题**：DDG 从某些 IP 返回 202 空页面。解决：设置 Cookie `kl=us-en` 强制英文区域后恢复正常
11. **递归冗余分解**：Judge 评估子任务时不知道兄弟任务已实现了哪些模块（如 parser/master/pricing），导致 "integrate" 类子任务被递归分解为重新实现所有组件。解决：`prompt_builder.py` 新增 `_format_sibling_context()` 方法，在 judge prompt 中注入已完成兄弟任务的描述、产出文件和接口信息。judge.txt 添加 "禁止冗余分解" 规则
12. **依赖模块 read_file 开销**：下游模块（如 branch_and_price）需要多次 read_file 了解上游模块接口，消耗大量 tokens。解决：`prompt_builder.py` 新增 `_format_upstream_modules()` 方法，在 execute prompt 中注入已完成上游模块的接口定义。master_problem tokens 从 183K 降至 47K
