"""实验：二分分解 — 递归拆解直到所有叶子都是 YES。

用法：
    python experiments/test_split.py --model qwen-plus
    python experiments/test_split.py --model qwen-plus --max-depth 6
    python experiments/test_split.py --model qwen-plus --task "your task here"
"""

import argparse
import asyncio
import html
import json
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


# ── HTML 可视化 ───────────────────────────────────────────────────────────────

def tree_to_html(root: Node) -> str:
    """生成可交互的 HTML 树。"""

    def render_node(node: Node) -> str:
        esc = html.escape(node.task)
        reason = html.escape(node.judge_response)

        if node.feasible:
            color = "#22c55e"  # green
            icon = "&#10004;"  # checkmark
            tag = "YES"
        elif node.is_leaf and not node.feasible:
            color = "#f97316"  # orange (hit max depth)
            icon = "&#9888;"   # warning
            tag = "NO*"
        else:
            color = "#ef4444"  # red
            icon = "&#10006;"  # cross
            tag = "NO"

        time_str = f"{node.judge_time:.1f}s"
        if node.split_time:
            time_str += f" + split {node.split_time:.1f}s"

        node_html = f'''
        <div class="node" style="border-left: 3px solid {color};">
          <div class="header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="icon" style="color:{color}">{icon}</span>
            <span class="tag" style="background:{color}">{tag}</span>
            <span class="task">{esc}</span>
            <span class="time">{time_str}</span>
          </div>
          <div class="reason">{reason}</div>
        '''

        if node.children:
            node_html += '<div class="children">'
            for child in node.children:
                node_html += render_node(child)
            node_html += '</div>'

        node_html += '</div>'
        return node_html

    # 统计
    def count_nodes(n: Node) -> dict:
        stats = {"total": 1, "yes": 0, "no": 0, "time": n.judge_time + n.split_time}
        if n.feasible:
            stats["yes"] = 1
        else:
            stats["no"] = 1
        for c in n.children:
            cs = count_nodes(c)
            stats["total"] += cs["total"]
            stats["yes"] += cs["yes"]
            stats["no"] += cs["no"]
            stats["time"] += cs["time"]
        return stats

    stats = count_nodes(root)

    return f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Binary Split Tree</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
  h1 {{ color: #f8fafc; font-size: 18px; }}
  .stats {{ color: #94a3b8; margin-bottom: 20px; font-size: 14px; }}
  .node {{ margin: 4px 0 4px 20px; padding: 4px 8px; }}
  .header {{ cursor: pointer; display: flex; align-items: center; gap: 8px; padding: 4px 0; }}
  .header:hover {{ background: #1e293b; border-radius: 4px; }}
  .icon {{ font-size: 14px; width: 18px; text-align: center; }}
  .tag {{ font-size: 11px; padding: 1px 6px; border-radius: 3px; color: white; font-weight: bold; }}
  .task {{ flex: 1; font-size: 13px; }}
  .time {{ font-size: 11px; color: #64748b; white-space: nowrap; }}
  .reason {{ font-size: 11px; color: #64748b; margin: 2px 0 2px 26px; }}
  .children {{ }}
  .collapsed .children {{ display: none; }}
  .collapsed .reason {{ display: none; }}
  .root-task {{ color: #93c5fd; font-size: 14px; margin-bottom: 16px; padding: 8px; background: #1e293b; border-radius: 6px; }}
</style></head><body>
<h1>Binary Split Tree</h1>
<div class="stats">Nodes: {stats["total"]} | Leaves YES: {stats["yes"]} | Leaves NO: {stats["no"]} | Total time: {stats["time"]:.1f}s</div>
<div class="root-task">{html.escape(root.task)}</div>
{render_node(root)}
</body></html>'''


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
    t0 = time.monotonic()
    root = await build_tree(api, args.task, args.model, max_depth=args.max_depth)
    total = time.monotonic() - t0

    print("=" * 70)
    print(f"Total time: {total:.1f}s")

    # 保存 HTML
    html_content = tree_to_html(root)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Tree saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
