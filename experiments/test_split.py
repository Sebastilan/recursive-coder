"""实验：N-ary 分解 — 单次 Decompose 调用 + DAG 依赖 + 拓扑排序。

用法：
    python experiments/test_split.py --model qwen-plus
    python experiments/test_split.py --model qwen-plus --max-depth 3
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

# ── prompt ──────────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = """You are a task decomposition planner for an AI coding agent.
The agent can: write code, run shell commands, search the web, and read web pages.

Given a task and its context, decide:
- LEAF: One agent session can complete this task (single module, clear spec, < 200 lines).
- DECOMPOSE: Split into 2-5 subtasks with a dependency DAG.

DECOMPOSE rules:
1. Each subtask description MUST be ≤ 20 words. Be concise and specific.
2. Subtasks must be DIFFERENT from the parent (not a rephrasing).
3. Use "dependencies" to specify which subtasks must complete first (by index, 0-based).
4. Independent subtasks should have empty dependencies [] so they can run in parallel.
5. Prefer WIDE (more parallel subtasks) over DEEP (serial chains) when possible.
6. Data gathering (web search, download) is often an independent first step.

ALREADY AVAILABLE DATA:
{context}

Reply with a JSON block inside <json> tags:

For LEAF tasks:
<json>
{{"decision": "LEAF", "reason": "brief reason"}}
</json>

For DECOMPOSE tasks:
<json>
{{
  "decision": "DECOMPOSE",
  "reason": "brief reason for splitting",
  "subtasks": [
    {{"description": "≤20 words", "produces": "what this outputs", "dependencies": []}},
    {{"description": "≤20 words", "produces": "what this outputs", "dependencies": [0]}},
    ...
  ]
}}
</json>"""

# ── 树节点 ────────────────────────────────────────────────────────────────────

class Node:
    def __init__(self, task: str, depth: int = 0, parent=None):
        self.task = task
        self.depth = depth
        self.parent = parent
        self.feasible: bool | None = None
        self.judge_response = ""
        self.judge_time = 0.0
        self.split_time = 0.0  # kept for viz compat, now part of judge_time
        self.children: list['Node'] = []
        # 数据感知
        self.context = ""       # 该节点已有的输入数据描述
        self.produces = ""      # 该节点产出什么
        self.order = ""         # "serial" | "parallel" | "dag" (for viz)
        self.dependencies: list[int] = []  # indices of sibling deps (for viz)

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


def topological_sort(subtasks: list[dict]) -> list[int]:
    """Kahn's algorithm for topological sort. Returns indices in execution order."""
    n = len(subtasks)
    in_degree = [0] * n
    adj = [[] for _ in range(n)]
    for i, st in enumerate(subtasks):
        for dep in st.get("dependencies", []):
            if isinstance(dep, int) and 0 <= dep < n:
                adj[dep].append(i)
                in_degree[i] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    result = []
    while queue:
        cur = queue.pop(0)
        result.append(cur)
        for nb in adj[cur]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    # Add any remaining (cycle detection)
    for i in range(n):
        if i not in result:
            result.append(i)
    return result


def parse_decompose_response(content: str) -> dict:
    """Extract JSON from <json>...</json> tags or raw JSON."""
    import re
    # Try <json> tags first
    m = re.search(r"<json>\s*(.*?)\s*</json>", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON block
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {"decision": "LEAF", "reason": "parse failed"}


# ── 核心函数 ──────────────────────────────────────────────────────────────────

async def decompose(api: APICaller, node: Node, model: str) -> tuple[dict, float]:
    """Single decompose call: decide LEAF or DECOMPOSE with subtasks.
    Returns (parsed_result, elapsed).
    """
    context_str = node.context or "(nothing yet — agent starts from scratch)"
    ancestry = build_ancestry(node)

    user_msg = f'TASK: "{node.task}"\n\nANCESTRY CHAIN:\n{ancestry}'

    system_prompt = DECOMPOSE_SYSTEM.format(context=context_str)

    t0 = time.monotonic()
    resp = await api.call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        task_node_id="decompose",
        phase="decompose",
        model_name=model,
    )
    elapsed = time.monotonic() - t0
    content = resp["choices"][0]["message"]["content"].strip()
    parsed = parse_decompose_response(content)
    parsed["_raw"] = content
    return parsed, elapsed


async def build_tree(
    api: APICaller, task: str, model: str,
    max_depth: int = 3, depth: int = 0,
    parent: Node = None, context: str = "",
) -> Node:
    node = Node(task, depth, parent=parent)
    node.context = context

    # 单次 decompose 调用
    result, elapsed = await decompose(api, node, model)
    node.judge_time = elapsed
    node.judge_response = result.get("_raw", "")

    decision = result.get("decision", "LEAF").upper()
    is_leaf = decision == "LEAF"
    node.feasible = is_leaf

    indent = "  " * depth
    tag = "LEAF" if is_leaf else "DECOMPOSE"
    ctx_hint = " [ctx]" if node.context else ""
    print(f"{indent}[{tag} {elapsed:.1f}s]{ctx_hint} {task[:100]}")

    if is_leaf or depth >= max_depth:
        if not is_leaf and depth >= max_depth:
            print(f"{indent}  ** max depth reached, forcing LEAF **")
            node.feasible = True
        return node

    subtasks = result.get("subtasks", [])
    if not subtasks:
        print(f"{indent}  WARNING: DECOMPOSE but no subtasks, treating as LEAF")
        node.feasible = True
        return node

    # Validate and log subtasks
    topo_order = topological_sort(subtasks)

    # Determine DAG structure for visualization
    has_deps = any(st.get("dependencies", []) for st in subtasks)
    all_have_deps = all(st.get("dependencies", []) for st in subtasks[1:])
    if not has_deps:
        node.order = "parallel"
    elif all_have_deps and len(subtasks) <= 2:
        node.order = "serial"
    else:
        node.order = "dag"

    print(f"{indent}  DECOMPOSE -> {len(subtasks)} subtasks [{node.order.upper()}]")
    print(f"{indent}  Topo order: {topo_order}")

    # Check for similarity
    for i, st in enumerate(subtasks):
        desc = st.get("description", "")
        word_count = len(desc.split())
        sim = text_similarity(task, desc)
        dep_str = f" deps={st.get('dependencies', [])}" if st.get("dependencies") else ""
        prod_str = f" -> {st.get('produces', '')[:40]}" if st.get("produces") else ""
        warn = ""
        if sim > 0.7:
            warn = f" WARNING: {sim:.0%} similar to parent!"
        if word_count > 20:
            warn += f" WARNING: {word_count} words (>20)"
        print(f"{indent}  [{i}]{dep_str} {desc[:80]}{prod_str}{warn}")

    # Build context for each subtask based on dependencies
    subtask_contexts = {}
    for i, st in enumerate(subtasks):
        ctx = context  # inherit parent context
        deps = st.get("dependencies", [])
        for dep_idx in deps:
            if isinstance(dep_idx, int) and 0 <= dep_idx < len(subtasks):
                dep_produces = subtasks[dep_idx].get("produces", "")
                if dep_produces:
                    if ctx:
                        ctx += "\n"
                    ctx += f"Output from subtask [{dep_idx}]: {dep_produces}"
        subtask_contexts[i] = ctx

    # Recursively build children in topological order
    children_nodes = {}
    for i in topo_order:
        st = subtasks[i]
        desc = st.get("description", "")
        child = await build_tree(
            api, desc, model, max_depth, depth + 1,
            parent=node, context=subtask_contexts[i],
        )
        child.produces = st.get("produces", "")
        child.dependencies = st.get("dependencies", [])
        children_nodes[i] = child

    # Store children in original order (for consistent visualization)
    node.children = [children_nodes[i] for i in range(len(subtasks))]
    return node


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--task", default=(
        "Implement a Python CVRP Branch-and-Price solver with column generation, "
        "SPPRC pricing subproblem, and branch-and-bound. "
        "Use E-n22-k4 TSPLIB instance (optimal=375) for validation."
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

        # Tree stats
        if root:
            def count_nodes(n):
                c = 1
                for ch in n.children:
                    c += count_nodes(ch)
                return c

            def max_tree_depth(n):
                if not n.children:
                    return n.depth
                return max(max_tree_depth(ch) for ch in n.children)

            def count_leaves(n):
                if not n.children:
                    return 1
                return sum(count_leaves(ch) for ch in n.children)

            def has_parallel(n):
                if n.order == "parallel" or n.order == "dag":
                    return True
                return any(has_parallel(ch) for ch in n.children)

            total_nodes = count_nodes(root)
            tree_depth = max_tree_depth(root)
            leaf_count = count_leaves(root)
            parallel = has_parallel(root)

            print(f"Total time: {total:.1f}s | Nodes: {total_nodes} | "
                  f"Depth: {tree_depth} | Leaves: {leaf_count} | "
                  f"Has parallel: {parallel}")
            print(f"API calls: {api.call_count} | Tokens: {api.total_input_tokens + api.total_output_tokens}")

            # Quality checks
            checks = []
            if tree_depth <= 3:
                checks.append("PASS: depth <= 3")
            else:
                checks.append(f"WARN: depth = {tree_depth} > 3")
            if parallel:
                checks.append("PASS: has parallel subtasks")
            else:
                checks.append("WARN: no parallel subtasks")
            if leaf_count >= 3:
                checks.append(f"PASS: {leaf_count} leaves (enough modules)")
            else:
                checks.append(f"WARN: only {leaf_count} leaves")

            print("Quality: " + " | ".join(checks))

            data = node_to_dict(root)
            save_tree_html(data, args.output, title="N-ary 分解树")
            print(f"Tree saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
