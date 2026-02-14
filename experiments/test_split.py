"""实验：二分分解 — 数据感知 + 依赖排序 + 执行规划。

用法：
    python experiments/test_split.py --model qwen-plus
    python experiments/test_split.py --model qwen-plus --max-depth 3
    python experiments/test_split.py --model qwen-plus --max-depth 3 --execute
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

JUDGE_SYSTEM = """You are a feasibility judge for an AI coding agent.
The agent can write code, run shell commands, search the web, and read web pages.

Given a task and what data is ALREADY AVAILABLE, decide: can ONE agent session complete this?

Important: "search the web for X" or "find data for X" IS a feasible single-session task.
But "implement a complex multi-module system" may NOT be feasible in one session.

Reply in this exact format:
VERDICT: YES or NO
PRODUCES: <what output this task generates>
REASON: <one line>"""

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

EXEC_SYSTEM = """You are given a leaf task judged as feasible.
You have these tools: write code, run shell commands, web_search, fetch_page.

Produce a concrete execution plan. If you need data you don't have,
include "use web_search to find ..." as a step.

Reply in this EXACT format:
PLAN:
1. <step>
2. <step>
...
OUTPUT: <what concrete artifact this produces>
BLOCKERS: <missing prerequisites that can't be resolved by web search, or "none">"""

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
        self.produces = ""      # Judge 解析出的"产出什么"
        self.order = ""         # "serial" 或 "parallel"（split 决定）
        # 执行规划（YES 叶子）
        self.exec_plan: list[str] = []
        self.exec_blockers: list[str] = []
        self.exec_output = ""
        self.exec_time = 0.0

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


def collect_leaves_in_order(node: Node) -> list[Node]:
    """按依赖顺序收集 YES 叶子。SERIAL 时 A 侧先于 B 侧。"""
    if node.is_leaf:
        return [node] if node.feasible else []
    leaves = []
    for child in node.children:
        leaves.extend(collect_leaves_in_order(child))
    return leaves


# ── 核心函数 ──────────────────────────────────────────────────────────────────

async def judge(api: APICaller, node: Node, model: str) -> tuple[bool, str, str, float]:
    """判断节点任务是否可行。返回 (feasible, full_response, produces, elapsed)。"""
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

    # 解析 VERDICT
    feasible = False
    produces = ""
    for line in content.split("\n"):
        line_s = line.strip()
        if line_s.upper().startswith("VERDICT:"):
            val = line_s[len("VERDICT:"):].strip()
            feasible = val.upper().startswith("YES")
        elif line_s.upper().startswith("PRODUCES:"):
            produces = line_s[len("PRODUCES:"):].strip()

    # 兼容旧格式（直接 YES/NO 开头）
    if "VERDICT:" not in content.upper():
        feasible = content.upper().startswith("YES")

    return feasible, content, produces, elapsed


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


async def execute_leaf(api: APICaller, leaf: Node, model: str) -> float:
    """为 YES 叶子生成执行计划（干跑，不实际执行）。返回耗时。"""
    ancestry = build_ancestry(leaf)
    context_str = leaf.context or "(nothing)"
    user_msg = (
        f'TASK: "{leaf.task}"\n\n'
        f'ANCESTRY:\n{ancestry}\n\n'
        f'AVAILABLE DATA:\n{context_str}'
    )

    t0 = time.monotonic()
    resp = await api.call(
        messages=[
            {"role": "system", "content": EXEC_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        task_node_id="exec",
        phase="exec",
        model_name=model,
    )
    elapsed = time.monotonic() - t0
    content = resp["choices"][0]["message"]["content"].strip()

    # 解析 PLAN / OUTPUT / BLOCKERS
    plan_steps = []
    output = ""
    blockers = []
    section = None

    for line in content.split("\n"):
        line_s = line.strip()
        up = line_s.upper()

        if up.startswith("PLAN:"):
            section = "plan"
            continue
        elif up.startswith("OUTPUT:"):
            output = line_s[len("OUTPUT:"):].strip()
            section = "output"
            continue
        elif up.startswith("BLOCKERS:"):
            val = line_s[len("BLOCKERS:"):].strip()
            if val.lower() not in ("none", "n/a", ""):
                blockers.append(val)
            section = "blockers"
            continue

        if section == "plan" and line_s:
            # 去掉编号前缀
            step = line_s.lstrip("0123456789.-) ").strip()
            if step:
                plan_steps.append(step)
        elif section == "blockers" and line_s:
            if line_s.lower() not in ("none", "n/a"):
                blockers.append(line_s.lstrip("- ").strip())

    leaf.exec_plan = plan_steps
    leaf.exec_output = output
    leaf.exec_blockers = blockers
    leaf.exec_time = elapsed
    return elapsed


async def build_tree(
    api: APICaller, task: str, model: str,
    max_depth: int = 8, depth: int = 0,
    parent: Node = None, context: str = "",
) -> Node:
    node = Node(task, depth, parent=parent)
    node.context = context

    # 判断
    feasible, judge_resp, produces, judge_time = await judge(api, node, model)
    node.feasible = feasible
    node.judge_response = judge_resp
    node.produces = produces
    node.judge_time = judge_time

    tag = "YES" if feasible else "NO"
    indent = "  " * depth
    ctx_hint = " [ctx]" if node.context else ""
    print(f"{indent}[{tag} {judge_time:.1f}s]{ctx_hint} {task[:100]}")
    if produces:
        print(f"{indent}  -> produces: {produces[:80]}")

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
        # B 的 context = 父 context + A 的产出
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


async def execute_leaves(api: APICaller, root: Node, model: str):
    """按依赖顺序遍历 YES 叶子，生成执行计划。"""
    leaves = collect_leaves_in_order(root)
    if not leaves:
        print("\nNo YES leaves to execute.")
        return

    print(f"\n{'=' * 70}")
    print(f"EXECUTION PLANNING: {len(leaves)} YES leaves")
    print(f"{'=' * 70}")

    for i, leaf in enumerate(leaves, 1):
        print(f"\n[Leaf {i}/{len(leaves)}] {leaf.task[:80]}")
        elapsed = await execute_leaf(api, leaf, model)
        steps_str = f"{len(leaf.exec_plan)} steps"
        blockers_str = f", {len(leaf.exec_blockers)} BLOCKERS" if leaf.exec_blockers else ""
        print(f"  Plan: {steps_str}{blockers_str} ({elapsed:.1f}s)")
        for j, step in enumerate(leaf.exec_plan, 1):
            print(f"    {j}. {step[:90]}")
        if leaf.exec_blockers:
            for b in leaf.exec_blockers:
                print(f"    BLOCKER: {b[:90]}")

        # A 执行完后，模拟把产出追加到同级 B 的 context
        # （在真正执行中，A 的实际输出会传给 B）
        if leaf.parent and leaf.parent.order == "serial":
            siblings = leaf.parent.children
            if len(siblings) == 2 and leaf is siblings[0]:
                # 当前叶子是 A 侧，把 exec_output 追加到 B 侧子树所有叶子的 context
                b_leaves = collect_leaves_in_order(siblings[1])
                if leaf.exec_output:
                    for bl in b_leaves:
                        if bl.context:
                            bl.context += "\n"
                        bl.context += f"Actual output from sibling: {leaf.exec_output}"


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--execute", action="store_true",
                        help="Generate execution plans for YES leaves")
    parser.add_argument("--task", default=(
        "Implement a complete C++ CVRP Branch-and-Price algorithm with column generation, "
        "shortest path pricing problem (SPPRC), and branch-and-bound framework. "
        "Use TSPLIB benchmark instances for validation."
    ))
    parser.add_argument("--output", default="experiments/split_tree.html")
    args = parser.parse_args()

    print(f"Model: {args.model} | Max depth: {args.max_depth} | Execute: {args.execute}")
    print(f"Task: {args.task[:100]}")
    print("=" * 70)

    api = APICaller()
    root = None
    t0 = time.monotonic()
    try:
        root = await build_tree(api, args.task, args.model, max_depth=args.max_depth)

        if args.execute and root:
            await execute_leaves(api, root, args.model)

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
            title = "二分分解树"
            if args.execute:
                title += " (含执行规划)"
            save_tree_html(data, args.output, title=title)
            print(f"Tree saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
