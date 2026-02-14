"""实验：二分分解 — 递归拆解直到所有叶子都是 YES。

用法：
    python experiments/test_split.py --model qwen-plus
    python experiments/test_split.py --model qwen-plus --max-depth 6
    python experiments/test_split.py --model qwen-plus --task "your task here"
"""

import argparse
import asyncio
import json
import time
import sys
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from recursive_coder.api_caller import APICaller
from tree_viz import save_tree_html, node_to_dict

# ── prompts ──────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a feasibility judge. An AI coding agent will attempt to complete the user's task.
The agent can write code, run shell commands, search the web, and read web pages.

Your ONLY job: can this task be done in ONE agent session, or does it need to be broken down?

Reply ONLY with: YES or NO, followed by a one-line reason."""

SPLIT_SYSTEM = """If two AIs were to collaborate on this task, what would each one be responsible for?

Reply ONLY with two lines:
A: <first AI's task>
B: <second AI's task>"""

# ── 树节点 ────────────────────────────────────────────────────────────────────

class Node:
    def __init__(self, task: str, depth: int = 0):
        self.task = task
        self.depth = depth
        self.feasible: bool | None = None
        self.judge_response = ""
        self.judge_time = 0.0
        self.split_time = 0.0
        self.children: list[Node] = []

    @property
    def is_leaf(self):
        return len(self.children) == 0


# ── 核心函数 ──────────────────────────────────────────────────────────────────

async def judge(api: APICaller, task: str, model: str) -> tuple[bool, str, float]:
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


async def split_task(api: APICaller, task: str, model: str) -> tuple[str, str, float]:
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

    task_a, task_b = "", ""
    for line in content.split("\n"):
        line = line.strip()
        if line.upper().startswith("A:"):
            task_a = line[2:].strip()
        elif line.upper().startswith("B:"):
            task_b = line[2:].strip()

    if not task_a or not task_b:
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if len(lines) >= 2:
            task_a = lines[0]
            task_b = lines[1]
        else:
            task_a = content
            task_b = "(parse failed)"

    return task_a, task_b, elapsed


async def build_tree(
    api: APICaller, task: str, model: str,
    max_depth: int = 8, depth: int = 0,
) -> Node:
    node = Node(task, depth)

    # 判断
    feasible, judge_resp, judge_time = await judge(api, task, model)
    node.feasible = feasible
    node.judge_response = judge_resp
    node.judge_time = judge_time

    tag = "YES" if feasible else "NO"
    indent = "  " * depth
    print(f"{indent}[{tag} {judge_time:.1f}s] {task[:100]}")

    if feasible or depth >= max_depth:
        if not feasible and depth >= max_depth:
            print(f"{indent}  ** max depth, stop splitting **")
        return node

    # 二分
    task_a, task_b, split_time = await split_task(api, task, model)
    node.split_time = split_time
    print(f"{indent}  SPLIT ({split_time:.1f}s)")

    child_a = await build_tree(api, task_a, model, max_depth, depth + 1)
    child_b = await build_tree(api, task_b, model, max_depth, depth + 1)
    node.children = [child_a, child_b]

    return node


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--task", default=(
        "Implement a complete C++ CVRP Branch-and-Price algorithm with column generation, "
        "shortest path pricing problem (SPPRC), and branch-and-bound framework. "
        "Use TSPLIB benchmark instances for validation."
    ))
    parser.add_argument("--output", default="experiments/split_tree.html")
    args = parser.parse_args()

    print(f"Model: {args.model} | Max depth: {args.max_depth}")
    print(f"Task: {args.task[:100]}")
    print("=" * 70)

    api = APICaller()
    root = None
    t0 = time.monotonic()
    try:
        root = await build_tree(api, args.task, args.model, max_depth=args.max_depth)
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        total = time.monotonic() - t0
        print("=" * 70)
        print(f"Total time: {total:.1f}s")
        if root:
            data = node_to_dict(root)
            save_tree_html(data, args.output, title="二分分解树")
            print(f"Tree saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
