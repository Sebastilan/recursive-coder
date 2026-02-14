"""Tool definitions (OpenAI function-calling format) and tool dispatch."""

from __future__ import annotations

import json
import time
from typing import Any

from .executor import Executor
from .logger_setup import get_logger
from .models import ToolCallRecord
from . import web_tools

logger = get_logger("tools")

# ── OpenAI-compatible tool definitions ──────────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a shell command in the working directory. Use for compiling, running tests, installing packages, inspecting files, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the working directory",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the working directory",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the working directory (default: '.')",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web (Google) for information. Use this to find documentation, "
                "algorithms, code examples, API references, research papers, or any knowledge "
                "needed to complete the task. Returns titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (use English for best results)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5, max: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch a web page and return its text content. Use this to read documentation, "
                "code examples, tutorials, or any web resource found via web_search. "
                "HTML is automatically converted to readable plain text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (must start with http:// or https://)",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default: 8000)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_done",
            "description": "Declare that the current task is finished. Call this when the verification passes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was completed successfully",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was done",
                    },
                },
                "required": ["success", "summary"],
            },
        },
    },
]


class ToolExecutor:
    """Dispatch tool calls to the underlying Executor."""

    def __init__(self, executor: Executor, proxy: str | None = None) -> None:
        self.executor = executor
        self.proxy = proxy  # for web_search / fetch_page

    async def execute(self, tool_name: str, arguments: dict) -> ToolCallRecord:
        t0 = time.monotonic()
        try:
            if tool_name == "shell":
                result = await self.executor.run(arguments["command"])
                summary = result.stdout or result.stderr or "(no output)"
                if result.blocked:
                    summary = f"BLOCKED: {result.block_reason}"
                success = result.returncode == 0 and not result.blocked
            elif tool_name == "write_file":
                summary = await self.executor.write_file(
                    arguments["path"], arguments["content"],
                )
                success = summary.startswith("OK")
            elif tool_name == "read_file":
                summary = await self.executor.read_file(arguments["path"])
                success = not summary.startswith("ERROR")
            elif tool_name == "list_dir":
                summary = await self.executor.list_dir(arguments.get("path", "."))
                success = not summary.startswith("ERROR")
            elif tool_name == "web_search":
                query = arguments.get("query", "")
                max_results = min(arguments.get("max_results", 5), 10)
                summary = await web_tools.web_search(query, max_results, proxy=self.proxy)
                success = not summary.startswith("ERROR")
            elif tool_name == "fetch_page":
                url = arguments.get("url", "")
                max_chars = arguments.get("max_chars", 8000)
                summary = await web_tools.fetch_page(url, max_chars, proxy=self.proxy)
                success = not summary.startswith("ERROR")
            elif tool_name == "task_done":
                summary = arguments.get("summary", "")
                success = arguments.get("success", False)
            else:
                summary = f"Unknown tool: {tool_name}"
                success = False
        except Exception as exc:
            summary = f"Tool execution error: {exc}"
            success = False

        duration = int((time.monotonic() - t0) * 1000)
        # Log arguments for debugging (truncate long values like file content)
        log_args = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                    for k, v in arguments.items()}
        logger.debug("tool=%s args=%s success=%s duration=%dms", tool_name, log_args, success, duration)
        return ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result_summary=summary,
            success=success,
            duration_ms=duration,
        )
