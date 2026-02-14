"""
End-to-end evaluation suite: 5 levels from simple to complex.

Each test case targets specific capabilities of the recursive-coder framework:

L1: echo_number    — Agent writes & runs trivial code, no decomposition
                     Tests: agent loop, tool use, exact verification
L2: sum_from_file  — Read data file, compute result, verify output
                     Tests: data pipeline, file I/O in agent, verification
L3: word_count     — Should decompose into read→count→sort steps
                     Tests: judge decomposition, subtask execution, integration
L4: csv_pipeline   — Multi-step: parse CSV → filter → aggregate → output
                     Tests: data flow between subtasks, dependency ordering
L5: expression_eval— Build expression evaluator (tokenize→parse→compute)
                     Tests: deeper recursion, potential backtracking
L6: multi_file_stats— Multi-file text analysis pipeline (3 modules + orchestrator)
                     Tests: forced decomposition, inter-module data flow, integration
L7: query_engine    — Mini CSV query engine (4 modules, complex data flow)
                     Tests: 2-level decomposition, dependency ordering, integration

Usage:
    DASHSCOPE_API_KEY=sk-xxx python eval/run_eval.py                    # run all (qwen-plus)
    DASHSCOPE_API_KEY=sk-xxx python eval/run_eval.py --level 1          # run only L1
    DEEPSEEK_API_KEY=sk-xxx python eval/run_eval.py --model deepseek-v3 # use deepseek
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make sure recursive_coder is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recursive_coder.api_caller import APICaller
from recursive_coder.evaluator import Evaluator
from recursive_coder.executor import Executor
from recursive_coder.logger_setup import setup_logging, get_logger
from recursive_coder.models import DataPort, TaskStatus
from recursive_coder.persistence import Persistence
from recursive_coder.processor import RecursiveProcessor
from recursive_coder.prompt_builder import PromptBuilder


logger = get_logger("eval")


@dataclass
class TestCase:
    level: int
    name: str
    description: str               # what we ask the framework to do
    task_text: str                  # the actual task description for the processor
    data_port: DataPort             # root data port
    setup_files: dict[str, str]     # files to create in workspace before run
    expected_check: str             # how to verify success
    config_overrides: dict = field(default_factory=dict)
    max_api_calls: int = 50


# ── Test case definitions ──────────────────────────────────────────────────

CASES: list[TestCase] = [
    # ── L1: Trivial leaf task ──
    TestCase(
        level=1,
        name="echo_number",
        description="Write a Python script that prints 42. No decomposition needed.",
        task_text="Write a Python script called solution.py that prints the number 42.",
        data_port=DataPort(
            input_description="No input needed. The program should simply print 42.",
        ),
        setup_files={},
        expected_check="contains:42",
        config_overrides={"max_depth": 2, "max_retries": 2, "max_agent_steps": 15},
        max_api_calls=20,
    ),

    # ── L2: Data-driven verification ──
    TestCase(
        level=2,
        name="sum_from_file",
        description="Read numbers from a file and print their sum.",
        task_text=(
            "Read the file data/numbers.txt which contains one integer per line. "
            "Write a Python script sum.py that reads the file and prints the sum of all numbers."
        ),
        data_port=DataPort(
            input_description="A text file with one integer per line",
            input_files=["data/numbers.txt"],
        ),
        setup_files={
            "data/numbers.txt": "10\n20\n30\n40\n",
        },
        expected_check="contains:100",
        config_overrides={"max_depth": 2, "max_retries": 2, "max_agent_steps": 15},
        max_api_calls=30,
    ),

    # ── L3: Should trigger decomposition ──
    TestCase(
        level=3,
        name="word_count",
        description="Count word frequencies and output sorted result. May decompose.",
        task_text=(
            "Given the text file data/article.txt, write a program that:\n"
            "1. Reads the file\n"
            "2. Counts the frequency of each word (case-insensitive, strip punctuation)\n"
            "3. Outputs the top 5 most frequent words with counts, one per line, "
            "in format 'word: count', sorted by count descending.\n"
            "Save the result to output/top_words.txt and also print it to stdout."
        ),
        data_port=DataPort(
            input_description="A text file containing an English article",
            input_files=["data/article.txt"],
            output_files=["output/top_words.txt"],
        ),
        setup_files={
            "data/article.txt": (
                "The quick brown fox jumps over the lazy dog. "
                "The dog barked at the fox. The fox ran away. "
                "The dog chased the fox through the park. "
                "The quick fox jumped over the fence. "
                "A lazy dog slept in the sun. The dog is a good dog."
            ),
        },
        expected_check="contains:the",  # "the" should be the most frequent word
        config_overrides={"max_depth": 3, "max_retries": 2, "max_agent_steps": 20},
        max_api_calls=60,
    ),

    # ── L4: Multi-step data pipeline ──
    TestCase(
        level=4,
        name="csv_pipeline",
        description="Parse CSV, filter rows, compute aggregate. Tests data flow.",
        task_text=(
            "Process the CSV file data/sales.csv:\n"
            "1. Parse the CSV (columns: product, region, amount)\n"
            "2. Filter rows where region == 'North'\n"
            "3. Sum the 'amount' column for the filtered rows\n"
            "4. Write the total to output/north_total.txt\n"
            "5. Print the total to stdout.\n"
            "Expected output for the given data: 350"
        ),
        data_port=DataPort(
            input_description="A CSV file with columns: product, region, amount",
            input_files=["data/sales.csv"],
            output_files=["output/north_total.txt"],
        ),
        setup_files={
            "data/sales.csv": (
                "product,region,amount\n"
                "Widget,North,100\n"
                "Gadget,South,200\n"
                "Widget,North,150\n"
                "Doohickey,East,50\n"
                "Gadget,North,100\n"
                "Widget,South,75\n"
            ),
        },
        expected_check="contains:350",
        config_overrides={"max_depth": 4, "max_retries": 2, "max_agent_steps": 25},
        max_api_calls=80,
    ),

    # ── L5: Complex — expression evaluator ──
    TestCase(
        level=5,
        name="expression_eval",
        description="Build an arithmetic expression evaluator. Deeper decomposition.",
        task_text=(
            "Build a Python program eval_expr.py that evaluates simple arithmetic expressions.\n"
            "The program should:\n"
            "1. Read expressions from data/expressions.txt (one per line)\n"
            "2. Evaluate each expression (support +, -, *, / and parentheses)\n"
            "3. Write the results to output/results.txt, one result per line\n"
            "4. Print each result to stdout\n"
            "The expressions use integers only. Division should use integer division (//).\n"
            "Do NOT use eval() for security reasons. Implement a proper parser."
        ),
        data_port=DataPort(
            input_description="A text file with one arithmetic expression per line",
            input_files=["data/expressions.txt"],
            output_files=["output/results.txt"],
        ),
        setup_files={
            "data/expressions.txt": (
                "2 + 3\n"
                "10 - 4 * 2\n"
                "(1 + 2) * (3 + 4)\n"
                "100 // 3\n"
            ),
        },
        expected_check="contains:5",  # first expression should eval to 5
        config_overrides={"max_depth": 5, "max_retries": 3, "max_agent_steps": 30},
        max_api_calls=120,
    ),

    # ── L6: Multi-file text analysis pipeline (force decomposition) ──
    TestCase(
        level=6,
        name="multi_file_stats",
        description="Multi-file text analysis pipeline. Forces decomposition into 3+ subtasks.",
        task_text=(
            "Build a multi-file text analysis pipeline with the following SEPARATE modules:\n\n"
            "Module 1 - text_parser.py: Read all .txt files from data/texts/ directory. "
            "For each file, extract a word list (lowercase, strip punctuation) and a sentence "
            "list (split by period/question mark/exclamation mark). "
            "Export function: parse_file(filepath) -> dict with keys 'words', 'sentences', 'filename'\n\n"
            "Module 2 - stats_calculator.py: Given parsed data from Module 1, compute per-file "
            "statistics: word_count, unique_word_count, sentence_count, avg_words_per_sentence, "
            "top_3_most_frequent_words. "
            "Export function: calculate_stats(parsed_data: dict) -> dict\n\n"
            "Module 3 - report_generator.py: Using stats from Module 2 for all files, generate:\n"
            "  - output/file_stats.csv: columns = filename, word_count, unique_words, sentences, "
            "avg_words_per_sentence, top_words (semicolon-separated)\n"
            "  - output/summary.txt: total files processed, total words across all files, "
            "total sentences, the single most common word across ALL files with its count\n"
            "Export function: generate_reports(all_stats: list[dict], output_dir: str)\n\n"
            "Module 4 - main.py: Orchestrate the pipeline: parse all files -> compute stats -> generate reports.\n\n"
            "IMPORTANT: Each module MUST be a separate .py file with clearly defined function interfaces. "
            "The modules must import from each other (main imports all three, report_generator uses stats output)."
        ),
        data_port=DataPort(
            input_description="Three text files in data/texts/ directory",
            input_files=["data/texts/animals.txt", "data/texts/nature.txt", "data/texts/food.txt"],
            output_files=["output/file_stats.csv", "output/summary.txt"],
        ),
        setup_files={
            "data/texts/animals.txt": (
                "The cat sat on the mat. The cat is very fluffy. "
                "A fluffy cat likes warm milk. The dog barked at the cat. "
                "The dog is friendly."
            ),
            "data/texts/nature.txt": (
                "The sun rises in the east every morning. Birds sing in the tall trees. "
                "The river flows to the sea. Fish swim in the river. "
                "The trees provide shade and shelter."
            ),
            "data/texts/food.txt": (
                "Pizza is a popular food around the world. "
                "Many people enjoy eating pizza with cheese. "
                "Bread and cheese make a great combination. "
                "Fresh bread smells wonderful. People love good food."
            ),
        },
        expected_check="contains:the",  # "the" is the most frequent word across all files
        config_overrides={"max_depth": 4, "max_retries": 3, "max_agent_steps": 20},
        max_api_calls=120,
    ),

    # ── L7: Mini CSV query engine (force 2-level decomposition) ──
    TestCase(
        level=7,
        name="query_engine",
        description="Mini CSV query engine with 4 modules. Forces multi-level decomposition.",
        task_text=(
            "Build a simple CSV query engine with these SEPARATE modules:\n\n"
            "Module 1 - schema_reader.py: Read CSV file, infer column types (detect if values "
            "are integers, floats, or strings by attempting conversion). Return a Schema object "
            "with column names and types. "
            "Export function: read_schema(csv_path: str) -> dict  (keys: 'columns' list of "
            "{name, type}, 'data' list of row dicts with typed values)\n\n"
            "Module 2 - query_parser.py: Parse simple SQL-like query strings. Supported syntax:\n"
            "  SELECT col1,col2 WHERE col3>value ORDER BY col1 DESC LIMIT n\n"
            "  - SELECT is required, others are optional\n"
            "  - WHERE supports: >, <, >=, <=, ==, != (for numbers compare numerically, for strings lexicographic)\n"
            "  - ORDER BY supports ASC (default) and DESC\n"
            "  - LIMIT is an integer\n"
            "Export function: parse_query(query_str: str) -> dict with keys: "
            "'select_columns', 'where_conditions', 'order_by', 'order_dir', 'limit'\n\n"
            "Module 3 - query_executor.py: Execute a parsed query against typed data. "
            "Apply WHERE filters, SELECT columns, ORDER BY sorting, and LIMIT. "
            "Export function: execute_query(parsed_query: dict, schema: dict) -> list[dict]\n\n"
            "Module 4 - main.py: Read data/employees.csv, read data/queries.txt (one query per line), "
            "execute each query, write results to output/results.txt in this format:\n"
            "  --- Query: <original query> ---\n"
            "  col1 | col2 | ...\n"
            "  val1 | val2 | ...\n"
            "  (N rows)\n"
            "  <blank line>\n\n"
            "IMPORTANT: Each module MUST be in a separate .py file. The query parser must NOT "
            "use eval() or exec(). Test with the provided queries."
        ),
        data_port=DataPort(
            input_description="A CSV file with employee data and a text file with queries",
            input_files=["data/employees.csv", "data/queries.txt"],
            output_files=["output/results.txt"],
        ),
        setup_files={
            "data/employees.csv": (
                "name,department,salary,age\n"
                "Alice,Engineering,95000,32\n"
                "Bob,Marketing,72000,28\n"
                "Charlie,Engineering,88000,35\n"
                "Diana,Marketing,82000,30\n"
                "Eve,Engineering,105000,40\n"
                "Frank,Sales,68000,26\n"
                "Grace,Engineering,91000,29\n"
                "Henry,Sales,75000,33\n"
            ),
            "data/queries.txt": (
                "SELECT name,salary WHERE department==Engineering ORDER BY salary DESC\n"
                "SELECT name,department WHERE salary>80000\n"
                "SELECT name,age WHERE age<30 ORDER BY name ASC\n"
                "SELECT name,department,salary ORDER BY salary DESC LIMIT 3\n"
            ),
        },
        expected_check="contains:Eve",  # Eve is the highest-paid in Engineering, first row of first query
        config_overrides={"max_depth": 5, "max_retries": 3, "max_agent_steps": 25},
        max_api_calls=200,
    ),
]


# ── Evaluation runner ──────────────────────────────────────────────────────

@dataclass
class EvalResult:
    level: int
    name: str
    success: bool
    root_status: str
    api_calls: int
    total_tokens: int
    duration_s: float
    tree_depth: int
    leaf_count: int
    first_pass_rate: float
    final_pass_rate: float
    error: str = ""
    tree_repr: str = ""


async def run_case(case: TestCase, eval_dir: Path, use_mock: bool = False, model: str = "qwen-plus") -> EvalResult:
    """Run a single test case and return its evaluation result."""
    ws = eval_dir / f"L{case.level}_{case.name}"
    output_dir = ws / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup files in workspace
    for rel_path, content in case.setup_files.items():
        fpath = output_dir / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")

    setup_logging(str(ws), verbose=True)

    persistence = Persistence(str(ws))

    if use_mock:
        from eval.mock_api import MockAPICaller
        api = MockAPICaller(scenario=f"L{case.level}", persistence=persistence)
    else:
        api = APICaller(default_model=model, persistence=persistence)

    executor = Executor(
        workspace_dir=str(output_dir),
        timeout=case.config_overrides.get("command_timeout", 30),
        output_truncate=10000,
    )
    prompt_builder = PromptBuilder()
    evaluator = Evaluator(str(ws))

    config = {
        "default_model": model,
        "max_total_api_calls": case.max_api_calls,
        **case.config_overrides,
    }

    processor = RecursiveProcessor(
        api_caller=api,
        prompt_builder=prompt_builder,
        executor=executor,
        persistence=persistence,
        config=config,
    )

    t0 = time.time()
    error = ""
    tree = None

    try:
        tree = await processor.run(case.task_text, data_port=case.data_port)
    except Exception as exc:
        error = str(exc)
        logger.error("Case L%d %s failed: %s", case.level, case.name, error)

    duration = time.time() - t0

    # Gather results
    if tree is None:
        return EvalResult(
            level=case.level, name=case.name, success=False,
            root_status="error", api_calls=api.call_count,
            total_tokens=api.total_input_tokens + api.total_output_tokens,
            duration_s=round(duration, 1), tree_depth=0, leaf_count=0,
            first_pass_rate=0, final_pass_rate=0, error=error,
        )

    # Generate evaluation report
    report = evaluator.generate_report(
        tree=tree, api_stats=api.get_stats(), config=config,
        backtrack_count=processor.backtrack_count,
    )
    persistence.save_report(report)

    root = tree.nodes.get(tree.root_id)
    root_status = root.status.value if root else "?"
    task_passed = root_status == "passed"

    # Additional output check
    output_ok = True
    if case.expected_check.startswith("exact:"):
        expected = case.expected_check[6:]
        # Check if any file or verification_result contains the expected output
        output_ok = _check_output(tree, expected, output_dir, "exact")
    elif case.expected_check.startswith("contains:"):
        expected = case.expected_check[9:]
        output_ok = _check_output(tree, expected, output_dir, "contains")

    # Count tree metrics
    leaves = [n for n in tree.nodes.values() if not n.children and n.id != tree.root_id]
    max_depth = max((n.depth for n in tree.nodes.values()), default=0)

    return EvalResult(
        level=case.level,
        name=case.name,
        success=task_passed and output_ok,
        root_status=root_status,
        api_calls=api.call_count,
        total_tokens=api.total_input_tokens + api.total_output_tokens,
        duration_s=round(duration, 1),
        tree_depth=max_depth,
        leaf_count=len(leaves),
        first_pass_rate=report.get("quality", {}).get("first_pass_rate", 0),
        final_pass_rate=report.get("quality", {}).get("final_pass_rate", 0),
        error=error,
        tree_repr=tree.print_tree(),
    )


def _check_output(tree, expected: str, output_dir: Path, mode: str) -> bool:
    """Check if the expected string appears in task outputs."""
    # Check verification results in leaf nodes
    for node in tree.nodes.values():
        vr = node.verification_result
        if vr:
            if mode == "exact" and expected.strip() == vr.strip():
                return True
            if mode == "contains" and expected in vr:
                return True

    # Check output files
    for f in output_dir.rglob("*"):
        if f.is_file() and f.suffix in (".txt", ".csv", ".json", ".py"):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if mode == "exact" and expected.strip() == content.strip():
                    return True
                if mode == "contains" and expected in content:
                    return True
            except Exception:
                pass

    return False


# ── Report formatting ──────────────────────────────────────────────────────

def print_report(results: list[EvalResult]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  RECURSIVE-CODER END-TO-END EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    passed = sum(1 for r in results if r.success)
    total = len(results)
    lines.append(f"  Overall: {passed}/{total} passed")
    lines.append("")

    lines.append(f"  {'Level':<6} {'Name':<20} {'Status':<10} {'API':<6} {'Tokens':<10} {'Time':<8} {'Depth':<6} {'Leaves':<7} {'FPR':<6} {'FinalPR'}")
    lines.append("  " + "-" * 95)

    for r in results:
        status = "PASS" if r.success else "FAIL"
        lines.append(
            f"  L{r.level:<5} {r.name:<20} {status:<10} {r.api_calls:<6} "
            f"{r.total_tokens:<10} {r.duration_s:<8} {r.tree_depth:<6} "
            f"{r.leaf_count:<7} {r.first_pass_rate:<6} {r.final_pass_rate}"
        )

    lines.append("")

    # Detailed results
    for r in results:
        lines.append(f"  --- L{r.level}: {r.name} ---")
        lines.append(f"  Root status: {r.root_status}")
        if r.error:
            lines.append(f"  Error: {r.error[:200]}")
        if r.tree_repr:
            for tl in r.tree_repr.split("\n"):
                lines.append(f"    {tl}")
        lines.append("")

    lines.append("=" * 72)

    report = "\n".join(lines)
    return report


# ── Main ───────────────────────────────────────────────────────────────────

async def main(levels: list[int] | None = None, use_mock: bool = False, model: str = "qwen-plus"):
    # Ensure API key (not needed for mock mode)
    if not use_mock:
        from recursive_coder.api_caller import PRESET_MODELS
        cfg = PRESET_MODELS.get(model)
        if cfg is None:
            print(f"Error: Unknown model '{model}'. Available: {', '.join(PRESET_MODELS)}")
            sys.exit(1)
        if not os.environ.get(cfg.api_key_env):
            print(f"Error: Environment variable {cfg.api_key_env} not set for model '{model}'.")
            print(f"Set it with: export {cfg.api_key_env}=your-key-here")
            sys.exit(1)

    mode_label = "MOCK" if use_mock else "LIVE"
    eval_dir = Path("eval_runs") / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode_label}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cases = CASES
    if levels:
        cases = [c for c in CASES if c.level in levels]

    print(f"\nRunning {len(cases)} test cases ({mode_label} mode)...")
    print(f"Results will be saved to: {eval_dir}\n")

    results: list[EvalResult] = []
    for case in cases:
        print(f"  [{time.strftime('%H:%M:%S')}] L{case.level}: {case.name} — {case.description}")
        result = await run_case(case, eval_dir, use_mock=use_mock, model=model)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{time.strftime('%H:%M:%S')}] L{case.level}: {status} "
              f"(api={result.api_calls}, tokens={result.total_tokens}, "
              f"time={result.duration_s}s)")
        print()

    report = print_report(results)
    print(report)

    # Save report
    report_path = eval_dir / "eval_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save structured results
    json_path = eval_dir / "eval_results.json"
    json_data = []
    for r in results:
        json_data.append({
            "level": r.level, "name": r.name, "success": r.success,
            "root_status": r.root_status, "api_calls": r.api_calls,
            "total_tokens": r.total_tokens, "duration_s": r.duration_s,
            "tree_depth": r.tree_depth, "leaf_count": r.leaf_count,
            "first_pass_rate": r.first_pass_rate,
            "final_pass_rate": r.final_pass_rate,
            "error": r.error,
        })
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults saved to: {eval_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation")
    parser.add_argument("--level", type=int, nargs="+", help="Only run specific levels (1-7)")
    parser.add_argument("--model", type=str, default="qwen-plus", help="Model to use (default: qwen-max)")
    parser.add_argument("--mock", action="store_true", help="Use mock API (no network needed)")
    args = parser.parse_args()
    asyncio.run(main(args.level, use_mock=args.mock, model=args.model))
