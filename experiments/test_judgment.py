"""实验：判断模型 — 评估任务是否可以一步完成。

用法：
    python experiments/test_judgment.py --model qwen-plus
    python experiments/test_judgment.py --model deepseek-v3
"""

import argparse
import asyncio
import json
import time
import sys
sys.path.insert(0, ".")

from recursive_coder.api_caller import APICaller

# ── 判断 prompt ──────────────────────────────────────────────────────────────

JUDGMENT_SYSTEM = """You are a task feasibility assessor for an AI coding agent.

The agent has these capabilities:
- Write and execute code (shell, read/write files)
- Search the web and read web pages
- Work in a sandboxed directory

Your job: determine if the given task can be completed reliably in a SINGLE agent session (about 20-30 tool calls).

Respond in JSON:
{
  "feasible": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "if_feasible": {
    "estimated_steps": number,
    "approach_summary": "one-line approach"
  },
  "if_not_feasible": {
    "blockers": ["list of reasons why not"],
    "suggested_subtasks": ["list of smaller tasks that would be feasible individually"]
  }
}

Be honest. A task is NOT feasible in one session if:
- It requires writing >500 lines of complex algorithmic code
- It involves multiple independent modules that each need careful implementation and testing
- It requires deep domain knowledge that needs research first
- It requires iterative debugging across multiple components

A task IS feasible if:
- It's a single script or module (<300 lines)
- The algorithm is well-known and straightforward
- It can be implemented and tested in ~20 tool calls
"""

JUDGMENT_USER = """Task from user:
"{task}"

Can this be completed reliably in a single agent session?"""

# ── 测试任务 ─────────────────────────────────────────────────────────────────

TEST_TASKS = [
    # 应该判定为 feasible
    ("L1_easy", "Write a Python script that prints 42"),
    ("L3_medium", "Write a Python program that reads a text file and outputs the top 5 most frequent words"),

    # 边界情况
    ("L5_boundary", "Implement a CVRP nearest-neighbor heuristic in Python that reads TSPLIB format and outputs routes"),

    # 应该判定为 not feasible
    ("L8_hard", "Implement a complete C++ CVRP Branch-and-Price algorithm with column generation, "
                "shortest path pricing problem (SPPRC), and branch-and-bound framework. "
                "Use TSPLIB benchmark instances for validation."),

    # 模糊任务
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

    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})

    # 尝试解析 JSON
    try:
        # 提取 JSON（可能被 markdown 包裹）
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        result = json.loads(json_str.strip())
    except (json.JSONDecodeError, IndexError):
        result = {"raw": content, "parse_error": True}

    return {
        "task_name": task_name,
        "task_desc": task_desc[:80],
        "result": result,
        "raw_content": content,
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
        print(f"\n{'─' * 60}")
        print(f"Task: {name}")
        print(f"Desc: {desc[:100]}")
        print(f"{'─' * 60}")

        result = await run_judgment(api, name, desc, args.model)

        if args.verbose:
            print(f"\n--- INPUT (system) ---")
            print(result["messages"][0]["content"])
            print(f"\n--- INPUT (user) ---")
            print(result["messages"][1]["content"])
            print(f"\n--- OUTPUT (raw) ---")
            print(result["raw_content"])
            print(f"--- END ---\n")

        r = result["result"]
        if r.get("parse_error"):
            print(f"⚠ JSON parse failed. Raw response:")
            print(r.get("raw", "")[:500])
        else:
            feasible = r.get("feasible", "?")
            confidence = r.get("confidence", "?")
            reason = r.get("reason", "")
            print(f"Feasible: {feasible} (confidence: {confidence})")
            print(f"Reason: {reason}")

            if feasible and r.get("if_feasible"):
                f = r["if_feasible"]
                print(f"Steps: ~{f.get('estimated_steps', '?')}")
                print(f"Approach: {f.get('approach_summary', '')}")
            elif not feasible and r.get("if_not_feasible"):
                nf = r["if_not_feasible"]
                print(f"Blockers:")
                for b in nf.get("blockers", []):
                    print(f"  - {b}")
                print(f"Suggested subtasks:")
                for s in nf.get("suggested_subtasks", []):
                    print(f"  - {s}")

        print(f"Time: {result['elapsed_s']}s | Tokens: {result['tokens']}")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
