# Recursive Coder - 项目备忘

## 当前状态

L1-L5 端到端测试全部通过（qwen-plus 模型，远程机 DESKTOP-CFEOJ9J）。
验证机制已重构为 criteria-based（Agent 自写测试脚本，Framework 只检查 returncode）。

L6-L7 递归分解测试已添加：
- L6 (multi_file_stats): 多文件文本统计管道，3 模块 + 编排，强制分解
- L7 (query_engine): 迷你 CSV 查询引擎，4 模块 + 复杂数据流，强制 2 级分解

L6 首次运行结果（2026-02-14）：
- Judge 正确分解为 4 个子任务（text_parser → stats_calculator → report_generator → main）
- 子任务依赖链正确
- 因 DashScope 欠费中断，待充值后完整跑通

上下文缓存监控已集成（api_caller.py 记录 cached_tokens），隐式缓存已确认生效。

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
```

## 架构概览

递归任务分解框架，核心循环：judge -> execute -> verify -> (fix/backtrack) -> integrate

- `processor.py` — 递归主控（judge/execute/backtrack/integrate）
- `agent_loop.py` — 多步 agent 循环（tool-use）
- `api_caller.py` — 统一 LLM API 调用（支持 Anthropic / OpenAI / OpenAI-compatible）
- `prompt_builder.py` — 模板加载 + 上下文注入
- `models.py` — TaskNode / TaskTree 数据模型
- `executor.py` — 沙盒命令执行
- `evaluator.py` — 评估报告生成
- `optimizer.py` — 迭代优化（分析报告 -> 调整 prompt/config）

## 配置

主配置在 `config.yaml`，关键参数：
- `max_depth: 5` — 递归最大深度
- `max_retries: 3` — 单任务最大重试
- `max_total_api_calls: 500` — API 调用总量上限
- `command_timeout: 60` — 命令执行超时（秒）

## 已踩过的坑

1. **Windows \r\n 问题**：Agent 写测试脚本时如果精确比较字节（`b'42\n'` vs `b'42\r\n'`）会失败。解决：system prompt 提醒用 `.strip()` 比较
2. **Judge 预测答案必错**：Judge 预测 `expected_output` 经常算错（如词频统计），导致正确的 Agent 输出被判为失败。解决：改为 criteria-based 验证，Judge 只定义验证标准，Agent 自己写测试
3. **Judge 内嵌断言**：Judge 把精确比较写进 `verification.command`（如 `python x.py | python -c "assert ..."`），Agent 无法修改。解决：约束 command 必须是 `python test_xxx.py` 形式
4. **Windows 路径大小写**：`Path.resolve()` 在 Windows 上驱动器号大小写不一致导致路径逃逸检查误报。解决：用 `.lower()` 做大小写无关比较
5. **相对路径 workspace**：Executor 用相对路径存 workspace，不同时刻 resolve 结果不同。解决：init 时立即 `.resolve()` 为绝对路径
6. **idle detection 过于激进**：只有 write_file 才重置计数器，正常的 read_file/list_dir 探索会触发 idle 停止。解决：任何成功的 tool call 都重置计数器
