"""实验：二分分解 — 数据感知 + 依赖排序。

用法：
    python experiments/test_split.py --model qwen-plus
    python experiments/test_split.py --model qwen-plus --max-depth 5
    python experiments/test_split.py --model qwen-plus --task "your task here"
"""

import argparse
import asyncio
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

JUDGE_SYSTEM = """You are a feasibility judge for an AI coding agent.
The agent can write code, run shell commands, search the web, and read web pages.

Given a task and its available context, decide: can ONE agent session complete this?

Rules:
- Data gathering tasks (web search, download files, read docs) → YES
- Single-module implementation with clear spec → YES
- Multi-module system requiring coordination → NO
- Task needing data that isn't available and can't be web-searched → NO

Reply with ONLY one word: YES or NO"""

SPLIT_SYSTEM_TEMPLATE = """Split this task into TWO subtasks for two AIs to collaborate.

RULES:
1. Each subtask must be DIFFERENT from the parent (not a rephrasing).
2. Specify whether B depends on A's output (SERIAL) or they're independent (PARALLEL).
3. The first step often is obtaining necessary data/resources (via web search, reading docs, etc.).

ALREADY AVAILABLE DATA:
{context}

Reply in this EXACT format:
ORDER: SERIAL or PARALLEL
A: <first AI's task>
A_PRODUCES: <what A generates>
B: <second AI's task>
B_NEEDS: <what B needs from A, or "nothing" if PARALLEL>"""

# ── 树节点 ────────────────────────────────────────────────────────────────────

class Node:
    def __init__(self, task: str, depth: int = 0, parent=None):
        self.task = task
        self.depth = depth
        self.parent = parent
        self.feasible: bool | None = None
        self.judge_response = ""
        self.judge_time = 0.0
        self.split_time = 0.0
        self.children: list['Node'] = []
        # 数据感知
        self.context = ""       # 该节点已有的输入数据描述
        self.produces = ""      # Split 解析出的 A_PRODUCES
        self.order = ""         # "serial" 或 "parallel"（split 决定）

    @property
    def is_leaf(self):
        return len(self.children) == 0


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def text_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta | tb:
        return 0
    return len(ta & tb) / len(ta | tb)


def build_ancestry(node: Node) -> str:
    """构建祖先链，让深层节点知道大局。"""
    chain = []
    cur = node.parent
    while cur:
        line = f"[D{cur.depth}] {cur.task[:80]}"
        if cur.produces:
            line += f" -> produces: {cur.produces}"
        chain.append(line)
        cur = cur.parent
    return "\n".join(reversed(chain)) if chain else "(root task)"


# ── 核心函数 ──────────────────────────────────────────────────────────────────

async def judge(api: APICaller, node: Node, model: str) -> tuple[bool, str, float]:
    """判断节点任务是否可行。返回 (feasible, full_response, elapsed)。"""
    context_str = node.context or "(nothing yet — agent starts from scratch)"
    user_msg = f'TASK: "{node.task}"\n\nALREADY AVAILABLE:\n{context_str}'

    t0 = time.monotonic()
    resp = await api.call(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        task_node_id="judge",
        phase="judge",
        model_name=model,
    )
    elapsed = time.monotonic() - t0
    content = resp["choices"][0]["message"]["content"].strip()
    feasible = content.upper().startswith("YES")
    return feasible, content, elapsed


async def split_task(api: APICaller, node: Node, model: str) -> tuple[str, str, str, str, str, float]:
    """拆分任务。返回 (order, task_a, a_produces, task_b, b_needs, elapsed)。"""
    context_str = node.context or "(nothing yet)"
    system_prompt = SPLIT_SYSTEM_TEMPLATE.format(context=context_str)

    t0 = time.monotonic()
    resp = await api.call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'"{node.task}"'},
        ],
        task_node_id="split",
        phase="split",
        model_name=model,
    )
    elapsed = time.monotonic() - t0
    content = resp["choices"][0]["message"]["content"].strip()

    # 解析
    order = "serial"
    task_a = ""
    a_produces = ""
    task_b = ""
    b_needs = ""

    for line in content.split("\n"):
        line_s = line.strip()
        up = line_s.upper()
        if up.startswith("ORDER:"):
            val = line_s[len("ORDER:"):].strip().upper()
            order = "parallel" if "PARALLEL" in val else "serial"
        elif up.startswith("A:"):
            task_a = line_s[2:].strip()
        elif up.startswith("A_PRODUCES:"):
            a_produces = line_s[len("A_PRODUCES:"):].strip()
        elif up.startswith("B:"):
            task_b = line_s[2:].strip()
        elif up.startswith("B_NEEDS:"):
            b_needs = line_s[len("B_NEEDS:"):].strip()

    # 容错：如果 A/B 解析失败
    if not task_a or not task_b:
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if len(lines) >= 2:
            task_a = task_a or lines[-2]
            task_b = task_b or lines[-1]
        else:
            task_a = task_a or content
            task_b = task_b or "(parse failed)"

    return order, task_a, a_produces, task_b, b_needs, elapsed


async def build_tree(
    api: APICaller, task: str, model: str,
    max_depth: int = 8, depth: int = 0,
    parent: Node = None, context: str = "",
) -> Node:
    node = Node(task, depth, parent=parent)
    node.context = context

    # 判断
    feasible, judge_resp, judge_time = await judge(api, node, model)
    node.feasible = feasible
    node.judge_response = judge_resp
    node.judge_time = judge_time

    tag = "YES" if feasible else "NO"
    indent = "  " * depth
    ctx_hint = " [ctx]" if node.context else ""
    print(f"{indent}[{tag} {judge_time:.1f}s]{ctx_hint} {task[:100]}")

    if feasible or depth >= max_depth:
        if not feasible and depth >= max_depth:
            print(f"{indent}  ** max depth, stop splitting **")
        return node

    # 二分
    order, task_a, a_produces, task_b, b_needs, split_time = await split_task(api, node, model)
    node.split_time = split_time
    node.order = order
    order_tag = order.upper()
    print(f"{indent}  SPLIT ({split_time:.1f}s) [{order_tag}]")

    # 相似度检测
    sim_a = text_similarity(task, task_a)
    sim_b = text_similarity(task, task_b)
    if sim_a > 0.8:
        print(f"{indent}  WARNING: A 与父任务高度相似 ({sim_a:.0%})")
    if sim_b > 0.8:
        print(f"{indent}  WARNING: B 与父任务高度相似 ({sim_b:.0%})")

    # 上下文传播
    ctx_a = context  # A 继承父节点已有数据

    if order == "serial":
        ctx_b = context
        if a_produces:
            if ctx_b:
                ctx_b += "\n"
            ctx_b += f"Output from prior task: {a_produces}"
    else:
        ctx_b = context  # PARALLEL: 各自继承父 context

    print(f"{indent}  A: {task_a[:90]}")
    if a_produces:
        print(f"{indent}     A_PRODUCES: {a_produces[:80]}")
    print(f"{indent}  B: {task_b[:90]}")
    if b_needs and b_needs.lower() != "nothing":
        print(f"{indent}     B_NEEDS: {b_needs[:80]}")

    child_a = await build_tree(api, task_a, model, max_depth, depth + 1,
                               parent=node, context=ctx_a)
    child_a.produces = child_a.produces or a_produces

    child_b = await build_tree(api, task_b, model, max_depth, depth + 1,
                               parent=node, context=ctx_b)

    node.children = [child_a, child_b]
    return node


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--max-depth", type=int, default=5)
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
        import traceback
        traceback.print_exc()
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
