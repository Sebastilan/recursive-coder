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
L8: cutting_stock_cg— Column generation for cutting stock (knapsack+LP+CG loop)
                     Tests: genuine complexity forcing decomposition, algorithm correctness
L9: cvrp_solver     — CVRP nearest-neighbor + 2-opt (5 modules, low max_agent_steps)
                     Tests: escalation decomposition path (LEAF fail → re-judge → decompose)
L10: vrp_benchmark  — Multi-instance VRP batch runner (2 formats, 2 algorithms, report)
                     Tests: complex instruction following, escalation decomposition

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

    # ── L8: Column Generation for Cutting Stock (genuinely complex) ──
    TestCase(
        level=8,
        name="cutting_stock_cg",
        description="Column generation for cutting stock problem. Genuinely complex, should trigger decomposition.",
        task_text=(
            "Implement a Column Generation solver for the Cutting Stock Problem.\n\n"
            "Problem: Given stock rolls of fixed length L, and a set of piece types each with\n"
            "a required length and demand quantity, find the minimum number of stock rolls\n"
            "needed to satisfy all demands. Each roll can be cut into multiple pieces as long\n"
            "as their total length does not exceed L.\n\n"
            "Algorithm overview (Column Generation):\n"
            "1. Start with initial cutting patterns (one piece type per pattern, packed maximally)\n"
            "2. Solve the LP relaxation of the master problem to get dual prices\n"
            "3. Solve the pricing subproblem (unbounded knapsack) to find a new pattern with\n"
            "   negative reduced cost (i.e., knapsack objective value > 1)\n"
            "4. If found, add the new pattern and go to step 2\n"
            "5. If not found, the LP relaxation is optimal — output the result\n\n"
            "You MUST implement these as SEPARATE Python modules:\n\n"
            "Module 1 - knapsack.py:\n"
            "  Solve the unbounded knapsack problem using dynamic programming.\n"
            "  Function: solve_knapsack(values: list[float], weights: list[int], capacity: int)\n"
            "            -> tuple[float, list[int]]\n"
            "  Returns (optimal_value, item_counts) where item_counts[i] is how many of item i to take.\n"
            "  This is used as the pricing subproblem: values = dual prices, weights = piece lengths,\n"
            "  capacity = stock length.\n\n"
            "Module 2 - lp_master.py:\n"
            "  Solve the master LP relaxation using scipy.optimize.linprog.\n"
            "  Function: solve_master(patterns: list[list[int]], demands: list[int])\n"
            "            -> tuple[float, list[float], list[float]]\n"
            "  Input: patterns is a list of cutting patterns (each pattern is a list of counts per piece type),\n"
            "         demands is the demand for each piece type.\n"
            "  Returns (objective_value, solution_x, dual_prices).\n"
            "  The LP is: minimize sum(x_j) subject to A @ x >= demands, x >= 0,\n"
            "  where column j of A is pattern j.\n"
            "  Use scipy.optimize.linprog with A_ub=-A^T (transposed and negated) and b_ub=-demands.\n"
            "  Extract dual prices from result.ineqlin.marginals (negate them since constraints were flipped).\n\n"
            "Module 3 - column_generation.py:\n"
            "  The main column generation loop.\n"
            "  Function: solve_cutting_stock(stock_length: int, pieces: list[dict]) -> dict\n"
            "  Input: stock_length and pieces (each with 'length' and 'demand' keys).\n"
            "  Steps:\n"
            "    a. Generate initial patterns: for each piece type i, create a pattern with\n"
            "       floor(stock_length / piece_length_i) copies of piece i and 0 for others.\n"
            "    b. Loop: solve master LP -> get duals -> solve knapsack pricing -> if value > 1+eps,\n"
            "       add new pattern and repeat; else break.\n"
            "    c. Return dict with: 'lp_value', 'patterns', 'solution', 'iterations', 'converged'.\n\n"
            "Module 4 - main.py:\n"
            "  Read input from data/problem.json, call solve_cutting_stock, write results\n"
            "  to output/result.txt AND print to stdout.\n"
            "  Output format must include lines like:\n"
            "    LP Optimal Value: <value>\n"
            "    Converged after <N> iterations\n"
            "    Pattern [a1, a2, ...]: <usage> rolls\n\n"
            "IMPORTANT:\n"
            "- Each module MUST be in a separate .py file\n"
            "- scipy is available for LP solving\n"
            "- The knapsack solver must use DP, not brute force\n"
            "- Use tolerance eps=1e-6 for convergence check\n"
        ),
        data_port=DataPort(
            input_description="JSON file with stock_length and pieces array",
            input_files=["data/problem.json"],
            output_files=["output/result.txt"],
        ),
        setup_files={
            "data/problem.json": (
                '{\n'
                '    "stock_length": 100,\n'
                '    "pieces": [\n'
                '        {"length": 45, "demand": 10},\n'
                '        {"length": 36, "demand": 20},\n'
                '        {"length": 31, "demand": 15}\n'
                '    ]\n'
                '}\n'
            ),
        },
        # LP optimal = 18.75 (hand-verified):
        # Initial patterns: [2,0,0],[0,2,0],[0,0,3] -> LP=20, duals=[0.5,0.5,0.333]
        # Pricing finds [0,1,2] (value=1.167>1) -> add
        # New LP=18.75, duals=[0.5,0.5,0.25] -> no improving pattern -> converged
        expected_check="contains:18.75",
        config_overrides={"max_depth": 5, "max_retries": 3, "max_agent_steps": 25},
        max_api_calls=200,
    ),

    # ── L9: CVRP solver — test escalation decomposition path ──
    TestCase(
        level=9,
        name="cvrp_solver",
        description="CVRP nearest-neighbor + 2-opt. Low max_agent_steps to trigger escalation decomposition.",
        task_text=(
            "Solve the Capacitated Vehicle Routing Problem (CVRP) using a Nearest Neighbor\n"
            "heuristic followed by 2-opt local search improvement.\n\n"
            "Input: data/instance.vrp in TSPLIB format (EUC_2D type).\n"
            "The file contains: NODE_COORD_SECTION (node coordinates), DEMAND_SECTION\n"
            "(customer demands), CAPACITY (vehicle capacity), depot is node 1.\n\n"
            "Implement these SEPARATE modules:\n\n"
            "Module 1 - vrp_parser.py:\n"
            "  Parse TSPLIB .vrp file format.\n"
            "  Function: parse_vrp(filepath) -> dict with keys:\n"
            "    'name', 'dimension', 'capacity', 'coords' (dict: node_id -> (x,y)),\n"
            "    'demands' (dict: node_id -> demand), 'depot' (int)\n"
            "  Distances are Euclidean (round to nearest integer as per TSPLIB convention).\n\n"
            "Module 2 - distance.py:\n"
            "  Compute and cache the distance matrix.\n"
            "  Function: compute_distance_matrix(coords: dict) -> dict[tuple, int]\n"
            "  Use Euclidean distance rounded to nearest integer: round(sqrt((x1-x2)^2 + (y1-y2)^2))\n\n"
            "Module 3 - nn_solver.py:\n"
            "  Nearest Neighbor construction heuristic for CVRP.\n"
            "  Function: nearest_neighbor(dist_matrix, demands, capacity, depot, n_nodes) -> list[list[int]]\n"
            "  Returns list of routes (each route is a list of customer node IDs, NOT including depot).\n"
            "  Algorithm: repeatedly pick the nearest unvisited customer that fits in current vehicle capacity.\n"
            "  When no more customers fit, start a new route.\n\n"
            "Module 4 - two_opt.py:\n"
            "  2-opt local search improvement for each route.\n"
            "  Function: improve_routes(routes, dist_matrix, depot) -> list[list[int]]\n"
            "  For each route, try all 2-opt swaps within the route. Accept if distance improves.\n"
            "  Repeat until no improvement found.\n\n"
            "Module 5 - main.py:\n"
            "  Read data/instance.vrp, build distance matrix, run NN + 2-opt, validate solution\n"
            "  (all customers visited exactly once, each route respects capacity), compute total distance.\n"
            "  Write to output/solution.txt AND print to stdout:\n"
            "    Instance: <name>\n"
            "    Known optimal: 375\n"
            "    Heuristic solution: <total_distance>\n"
            "    Gap: <percentage>%\n"
            "    Number of routes: <k>\n"
            "    Routes:\n"
            "      Route 1: depot -> c1 -> c2 -> ... -> depot (distance: X, load: Y/CAP)\n"
            "      ...\n\n"
            "IMPORTANT: Each module in a separate .py file. Use integer distances (TSPLIB convention)."
        ),
        data_port=DataPort(
            input_description="A TSPLIB .vrp file with EUC_2D coordinates for 22-node CVRP instance",
            input_files=["data/instance.vrp"],
            output_files=["output/solution.txt"],
        ),
        setup_files={
            "data/instance.vrp": (
                "NAME : E-n22-k4\n"
                "COMMENT : (Christophides and Eilon, Min no of trucks: 4, Optimal value: 375)\n"
                "TYPE : CVRP\n"
                "DIMENSION : 22\n"
                "EDGE_WEIGHT_TYPE : EUC_2D\n"
                "CAPACITY : 6000\n"
                "NODE_COORD_SECTION\n"
                "1 145 215\n"
                "2 151 264\n"
                "3 159 261\n"
                "4 130 254\n"
                "5 128 252\n"
                "6 163 247\n"
                "7 146 246\n"
                "8 161 242\n"
                "9 142 239\n"
                "10 163 236\n"
                "11 148 232\n"
                "12 128 231\n"
                "13 156 217\n"
                "14 129 214\n"
                "15 146 208\n"
                "16 164 208\n"
                "17 141 206\n"
                "18 147 193\n"
                "19 164 193\n"
                "20 129 189\n"
                "21 155 185\n"
                "22 139 182\n"
                "DEMAND_SECTION\n"
                "1 0\n"
                "2 1100\n"
                "3 700\n"
                "4 800\n"
                "5 1400\n"
                "6 2100\n"
                "7 400\n"
                "8 800\n"
                "9 100\n"
                "10 500\n"
                "11 600\n"
                "12 1200\n"
                "13 1300\n"
                "14 1300\n"
                "15 300\n"
                "16 900\n"
                "17 2100\n"
                "18 1000\n"
                "19 900\n"
                "20 2500\n"
                "21 1800\n"
                "22 700\n"
                "DEPOT_SECTION\n"
                " 1\n"
                " -1\n"
                "EOF\n"
            ),
        },
        expected_check="contains:375",  # output must reference the known optimal value
        config_overrides={
            "max_depth": 5,
            "max_retries": 1,        # only 1 retry before escalation
            "max_agent_steps": 12,   # 12 steps not enough for 5 modules → fail → escalate
        },
        max_api_calls=200,
    ),

    # ── L10: Multi-instance VRP benchmark runner ──
    TestCase(
        level=10,
        name="vrp_benchmark",
        description="Multi-instance VRP batch runner. 2 formats, 2 algorithms, comparison report.",
        task_text=(
            "Build a VRP benchmark runner that processes multiple CVRP instances and generates\n"
            "a comparison report.\n\n"
            "Input files:\n"
            "  - data/instances/E-n13-k4.vrp (EXPLICIT distance format, 12 customers, optimal=290)\n"
            "  - data/instances/E-n22-k4.vrp (EUC_2D format, 21 customers, optimal=375)\n"
            "  - data/config.json (specifies which algorithm to run and report format)\n\n"
            "Implement these modules:\n\n"
            "Module 1 - vrp_parser.py:\n"
            "  Parse TSPLIB .vrp files. Must handle BOTH EUC_2D (coordinate-based) and EXPLICIT\n"
            "  (lower-triangular distance matrix) formats.\n"
            "  Function: parse_vrp(filepath) -> dict\n"
            "  For EUC_2D: compute integer Euclidean distances from coordinates.\n"
            "  For EXPLICIT: read the lower-triangular matrix directly.\n\n"
            "Module 2 - solvers.py:\n"
            "  Implement two CVRP heuristics:\n"
            "  a) nearest_neighbor(instance) -> solution dict\n"
            "  b) savings_algorithm(instance) -> solution dict (Clarke-Wright parallel savings)\n"
            "  Each returns: {'routes': [...], 'total_distance': int, 'n_vehicles': int}\n\n"
            "Module 3 - reporter.py:\n"
            "  Generate comparison report.\n"
            "  Function: generate_report(results: list[dict], output_dir: str)\n"
            "  Write output/report.csv with columns:\n"
            "    instance, algorithm, total_distance, n_vehicles, optimal, gap_percent\n"
            "  Write output/summary.txt with:\n"
            "    - Best algorithm for each instance (lowest gap)\n"
            "    - Overall average gap per algorithm\n"
            "    - Recommendation: which algorithm to use\n\n"
            "Module 4 - main.py:\n"
            "  Read config.json, for each instance run all specified algorithms,\n"
            "  collect results, generate report.\n\n"
            "IMPORTANT: Each module in a separate .py file.\n"
            "The savings algorithm must correctly compute savings s(i,j) = d(0,i) + d(0,j) - d(i,j)\n"
            "and merge routes respecting capacity constraints."
        ),
        data_port=DataPort(
            input_description="Two TSPLIB .vrp files (EUC_2D and EXPLICIT formats) plus a config.json",
            input_files=[
                "data/instances/E-n13-k4.vrp",
                "data/instances/E-n22-k4.vrp",
                "data/config.json",
            ],
            output_files=["output/report.csv", "output/summary.txt"],
        ),
        setup_files={
            "data/instances/E-n13-k4.vrp": (
                "NAME : eil13\n"
                "COMMENT : (Eilon et al, Min no of trucks: 4, Best value: 290)\n"
                "TYPE : CVRP\n"
                "DIMENSION : 13\n"
                "EDGE_WEIGHT_TYPE : EXPLICIT\n"
                "EDGE_WEIGHT_FORMAT: LOWER_ROW \n"
                "DISPLAY_DATA_TYPE: NO_DISPLAY\n"
                "CAPACITY : 6000\n"
                "EDGE_WEIGHT_SECTION\n"
                "     9    14    21    23    22    25    32    36    38    42\n"
                "    50    52     5    12    22    21    24    31    35    37\n"
                "    41    49    51     7    17    16    23    26    30    36\n"
                "    36    44    46    10    21    30    27    37    43    31\n"
                "    37    39    19    28    25    35    41    29    31    29\n"
                "     9    10    16    22    20    28    30     7    11    13\n"
                "    17    25    27    10    16    10    18    20     6     6\n"
                "    14    16    12    12    20     8    10    10\n"
                "DEMAND_SECTION\n"
                "1 0\n"
                "2 1200\n"
                "3 1700\n"
                "4 1500\n"
                "5 1400\n"
                "6 1700\n"
                "7 1400\n"
                "8 1200\n"
                "9 1900\n"
                "10 1800\n"
                "11 1600\n"
                "12 1700\n"
                "13 1100\n"
                "DEPOT_SECTION\n"
                "1\n"
                "-1\n"
                "EOF\n"
            ),
            "data/instances/E-n22-k4.vrp": (
                "NAME : E-n22-k4\n"
                "COMMENT : (Christophides and Eilon, Min no of trucks: 4, Optimal value: 375)\n"
                "TYPE : CVRP\n"
                "DIMENSION : 22\n"
                "EDGE_WEIGHT_TYPE : EUC_2D\n"
                "CAPACITY : 6000\n"
                "NODE_COORD_SECTION\n"
                "1 145 215\n"
                "2 151 264\n"
                "3 159 261\n"
                "4 130 254\n"
                "5 128 252\n"
                "6 163 247\n"
                "7 146 246\n"
                "8 161 242\n"
                "9 142 239\n"
                "10 163 236\n"
                "11 148 232\n"
                "12 128 231\n"
                "13 156 217\n"
                "14 129 214\n"
                "15 146 208\n"
                "16 164 208\n"
                "17 141 206\n"
                "18 147 193\n"
                "19 164 193\n"
                "20 129 189\n"
                "21 155 185\n"
                "22 139 182\n"
                "DEMAND_SECTION\n"
                "1 0\n"
                "2 1100\n"
                "3 700\n"
                "4 800\n"
                "5 1400\n"
                "6 2100\n"
                "7 400\n"
                "8 800\n"
                "9 100\n"
                "10 500\n"
                "11 600\n"
                "12 1200\n"
                "13 1300\n"
                "14 1300\n"
                "15 300\n"
                "16 900\n"
                "17 2100\n"
                "18 1000\n"
                "19 900\n"
                "20 2500\n"
                "21 1800\n"
                "22 700\n"
                "DEPOT_SECTION\n"
                " 1\n"
                " -1\n"
                "EOF\n"
            ),
            "data/config.json": (
                '{\n'
                '    "algorithms": ["nearest_neighbor", "savings"],\n'
                '    "instances_dir": "data/instances",\n'
                '    "output_dir": "output",\n'
                '    "optimals": {"E-n13-k4": 290, "E-n22-k4": 375}\n'
                '}\n'
            ),
        },
        expected_check="contains:290",  # E-n13-k4 optimal must appear in report
        config_overrides={
            "max_depth": 5,
            "max_retries": 1,
            "max_agent_steps": 15,   # 2 instances + 2 algorithms + report → not enough
        },
        max_api_calls=250,
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
    parser.add_argument("--level", type=int, nargs="+", help="Only run specific levels (1-10)")
    parser.add_argument("--model", type=str, default="qwen-plus", help="Model to use (default: qwen-max)")
    parser.add_argument("--mock", action="store_true", help="Use mock API (no network needed)")
    args = parser.parse_args()
    asyncio.run(main(args.level, use_mock=args.mock, model=args.model))
