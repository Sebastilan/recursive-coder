"""实验：判断模型 — 评估任务是否可以一步完成。

用法：
    python experiments/test_judgment.py --model qwen-plus
    python experiments/test_judgment.py --model qwen-plus --task L8_hard -v
"""

import argparse
import asyncio
import json
import time
import sys
sys.path.insert(0, ".")

from recursive_coder.api_caller import APICaller

# ── 判断 prompt ──────────────────────────────────────────────────────────────

JUDGMENT_SYSTEM = """You are a feasibility judge. An AI coding agent will attempt to complete the user's task.
The agent can write code, run shell commands, search the web, and read web pages.

Your ONLY job: can this task be done in ONE agent session, or does it need to be broken down?

Reply ONLY with: YES or NO, followed by a one-line reason.
Examples:
  YES - straightforward single-file Python script
  NO - requires multiple complex modules with external dependencies"""

JUDGMENT_USER = '"{task}"'

# ── 测试任务 ─────────────────────────────────────────────────────────────────

TEST_TASKS = [
    ("L1_easy", "Write a Python script that prints 42"),
    ("L3_medium", "Write a Python program that reads a text file and outputs the top 5 most frequent words"),
    ("L5_boundary", "Implement a CVRP nearest-neighbor heuristic in Python that reads TSPLIB format and outputs routes"),
    ("L8_hard", "Implement a complete C++ CVRP Branch-and-Price algorithm with column generation, "
                "shortest path pricing problem (SPPRC), and branch-and-bound framework. "
                "Use TSPLIB benchmark instances for validation."),
    ("L7_vague", "Build a VRP solver"),
]


async def run_judgment(api: APICaller, task_name: str, task_desc: str, model: str):
    """调用判断模型，返回结果。"""
    messages = [
        {"role": "system", "content": JUDGMENT_SYSTEM},
        {"role": "user", "content": JUDGMENT_USER.format(task=task_desc)},
    ]

    t0 = time.monotonic()
    resp = await api.call(
        messages=messages,
        task_node_id=f"judgment_{task_name}",
        phase="judgment",
        model_name=model,
    )
    elapsed = time.monotonic() - t0

    content = resp["choices"][0]["message"]["content"].strip()
    usage = resp.get("usage", {})

    # 解析 YES/NO
    first_line = content.split("\n")[0].strip()
    feasible = first_line.upper().startswith("YES")

    return {
        "task_name": task_name,
        "task_desc": task_desc,
        "feasible": feasible,
        "response": content,
        "messages": messages,
        "elapsed_s": round(elapsed, 2),
        "tokens": {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        },
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--task", help="Only run specific task by name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full input/output")
    args = parser.parse_args()

    api = APICaller()

    tasks = TEST_TASKS
    if args.task:
        tasks = [(n, d) for n, d in tasks if n == args.task]
        if not tasks:
            print(f"Task '{args.task}' not found. Available: {[n for n, _ in TEST_TASKS]}")
            return

    print(f"Model: {args.model}")
    print(f"Tasks: {len(tasks)}")
    print("=" * 60)

    for name, desc in tasks:
        result = await run_judgment(api, name, desc, args.model)

        if args.verbose:
            print(f"\n--- INPUT (system) ---")
            print(result["messages"][0]["content"])
            print(f"\n--- INPUT (user) ---")
            print(result["messages"][1]["content"])
            print(f"\n--- OUTPUT ---")
            print(result["response"])
            print(f"--- END ---")

        tag = "YES" if result["feasible"] else "NO"
        print(f"[{result['elapsed_s']:>5.1f}s] {name:<14} {tag:<3}  {result['response']}")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
