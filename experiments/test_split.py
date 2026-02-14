"""实验：二分分解 — 永远拆成两半。

用法：
    python experiments/test_split.py --model qwen-plus
    python experiments/test_split.py --model qwen-plus --depth 3
"""

import argparse
import asyncio
import time
import sys
sys.path.insert(0, ".")

from recursive_coder.api_caller import APICaller

# ── prompts ──────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a feasibility judge. An AI coding agent will attempt to complete the user's task.
The agent can write code, run shell commands, search the web, and read web pages.

Your ONLY job: can this task be done in ONE agent session, or does it need to be broken down?

Reply ONLY with: YES or NO, followed by a one-line reason."""

SPLIT_SYSTEM = """If two AIs were to collaborate on this task, what would each one be responsible for?

Reply ONLY with two lines:
A: <first AI's task>
B: <second AI's task>"""

# ── 核心函数 ──────────────────────────────────────────────────────────────────

async def judge(api: APICaller, task: str, model: str) -> tuple[bool, str, float]:
    """判断任务是否可一步完成。返回 (feasible, response, seconds)。"""
    t0 = time.monotonic()
    resp = await api.call(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f'"{task}"'},
        ],
        task_node_id="judge",
        phase="judge",
        model_name=model,
    )
    elapsed = time.monotonic() - t0
    content = resp["choices"][0]["message"]["content"].strip()
    feasible = content.upper().startswith("YES")
    return feasible, content, elapsed


async def split(api: APICaller, task: str, model: str) -> tuple[str, str, float]:
    """二分拆解。返回 (task_a, task_b, seconds)。"""
    t0 = time.monotonic()
    resp = await api.call(
        messages=[
            {"role": "system", "content": SPLIT_SYSTEM},
            {"role": "user", "content": f'"{task}"'},
        ],
        task_node_id="split",
        phase="split",
        model_name=model,
    )
    elapsed = time.monotonic() - t0
    content = resp["choices"][0]["message"]["content"].strip()

    # 解析 A: ... B: ...
    task_a, task_b = "", ""
    for line in content.split("\n"):
        line = line.strip()
        if line.upper().startswith("A:"):
            task_a = line[2:].strip()
        elif line.upper().startswith("B:"):
            task_b = line[2:].strip()

    # fallback: 如果解析失败，按行分
    if not task_a or not task_b:
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if len(lines) >= 2:
            task_a = lines[0]
            task_b = lines[1]
        else:
            task_a = content
            task_b = "(parse failed)"

    return task_a, task_b, elapsed


async def recursive_decompose(
    api: APICaller, task: str, model: str,
    max_depth: int = 3, depth: int = 0, prefix: str = "",
):
    """递归二分分解，直到每个子任务都被判断为 YES 或达到最大深度。"""
    indent = "  " * depth

    # 1. 判断
    feasible, judge_resp, judge_time = await judge(api, task, model)
    tag = "YES" if feasible else "NO"

    print(f"{indent}{prefix}[{tag} {judge_time:.1f}s] {task[:90]}")

    if feasible:
        return  # 叶子节点，可以执行

    if depth >= max_depth:
        print(f"{indent}  (max depth reached, would execute anyway)")
        return

    # 2. 二分
    task_a, task_b, split_time = await split(api, task, model)
    print(f"{indent}  SPLIT ({split_time:.1f}s):")

    # 3. 递归处理两半
    await recursive_decompose(api, task_a, model, max_depth, depth + 1, "A: ")
    await recursive_decompose(api, task_b, model, max_depth, depth + 1, "B: ")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--depth", type=int, default=3, help="Max recursion depth")
    parser.add_argument("--task", default=(
        "Implement a complete C++ CVRP Branch-and-Price algorithm with column generation, "
        "shortest path pricing problem (SPPRC), and branch-and-bound framework. "
        "Use TSPLIB benchmark instances for validation."
    ))
    args = parser.parse_args()

    print(f"Model: {args.model} | Max depth: {args.depth}")
    print(f"Task: {args.task[:100]}")
    print("=" * 70)

    api = APICaller()
    t0 = time.monotonic()
    await recursive_decompose(api, args.task, args.model, max_depth=args.depth)
    total = time.monotonic() - t0

    print("=" * 70)
    print(f"Total time: {total:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
