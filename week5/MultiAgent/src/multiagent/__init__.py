from __future__ import annotations

import ast
import json
import operator as op
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

MODEL = "openai/gpt-oss-120b"
MAX_ATTEMPTS = 5


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your environment or .env file.")
    return value


client = Groq(api_key=_require_env("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=_require_env("TAVILY_API_KEY"))


@dataclass
class TokenUsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add_usage(self, usage: Dict[str, Any]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)

    def snapshot(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _normalize_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


# --- Tool implementations ---------------------------------------------------

def web_search(query: str) -> str:
    result = tavily_client.search(query=query, max_results=5)
    items = result.get("results", [])
    if not items:
        return "No results found."
    lines = [f"{item.get('title', 'Untitled')}: {item.get('content', '')} ({item.get('url', '')})" for item in items]
    return "\n".join(lines)


_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def _requires_float_output(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, ast.Div) or _requires_float_output(node.left) or _requires_float_output(node.right)
    if isinstance(node, ast.UnaryOp):
        return _requires_float_output(node.operand)
    return False


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
        if _requires_float_output(tree.body) or isinstance(value, float) and not value.is_integer():
            return str(value)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def detect_task_mix(query: str) -> List[str]:
    text = query.lower()
    search_keywords = [
        "search",
        "find",
        "look up",
        "latest",
        "current",
        "news",
        "price",
        "stock",
        "weather",
        "who is",
        "what is",
        "where",
        "when",
        "how much",
        "today",
    ]
    math_keywords = [
        "calculate",
        "math",
        "add",
        "subtract",
        "multiply",
        "divide",
        "sum",
        "total",
        "tax",
        "discount",
        "%",
        "plus",
        "minus",
        "cost",
        "profit",
    ]
    matches: List[str] = []
    if any(keyword in text for keyword in search_keywords):
        matches.append("search")
    if any(keyword in text for keyword in math_keywords):
        matches.append("math")
    if not matches:
        return ["search", "math"]
    return matches


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information and return a summary of relevant results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to use on the web."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression like 2 + 3 * 4 or (10 - 2) / 4.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate."}
                },
                "required": ["expression"],
            },
        },
    },
]

SEARCH_ONLY_TOOLS = [tool for tool in TOOLS if tool["function"]["name"] == "web_search"]
MATH_ONLY_TOOLS = [tool for tool in TOOLS if tool["function"]["name"] == "calculate"]

AVAILABLE_TOOLS = {"web_search": web_search, "calculate": calculate}


# --- LLM helper -------------------------------------------------------------


def ask_llm(
    messages: List[Dict[str, Any]],
    *,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = MODEL,
    usage_tracker: Optional[TokenUsageTracker] = None,
) -> Any:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages if system_prompt is None else [{"role": "system", "content": system_prompt}, *messages],
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = client.chat.completions.create(**payload)
    if usage_tracker is not None:
        usage_tracker.add_usage(_normalize_usage(response))
    return response


# --- Single-agent system ----------------------------------------------------


def run_single_agent(user_query: str, max_attempts: int = MAX_ATTEMPTS) -> Dict[str, Any]:
    messages = [{"role": "user", "content": user_query}]
    usage = TokenUsageTracker()

    for attempt in range(1, max_attempts + 1):
        response = ask_llm(messages, tools=TOOLS, usage_tracker=usage)
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return {
                "answer": message.content or "",
                "attempts_used": attempt,
                "usage": usage.snapshot(),
            }

        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": tool_calls})

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = AVAILABLE_TOOLS[function_name]
            function_args = json.loads(tool_call.function.arguments)
            result = function_to_call(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(result),
                }
            )

    return {
        "answer": "Max attempts reached without a final answer.",
        "attempts_used": max_attempts,
        "usage": usage.snapshot(),
    }


# --- Multi-agent system -----------------------------------------------------


MANAGER_PROMPT = """
You are the manager agent. Your only job is to decide which helper agent(s) should respond to the user request.
Return only valid JSON in this format:
{"helpers": ["search", "math"]}
Choose only from: search, math.
If the user request includes both search and math, include both in the order you want them to execute.
Keep your reasoning minimal.
"""

SEARCH_PROMPT = """
You are the search helper agent. Use only the web_search tool. Answer the user's question using the results from the web search. If a result is not found, say so honestly.
"""

MATH_PROMPT = """
You are the math helper agent. Use only the calculate tool and perform the arithmetic accurately. Return the final result clearly.
"""


class MultiAgentSystem:
    def __init__(self, notes_path: str | Path | None = None):
        self.notes_path = Path(notes_path) if notes_path else Path("manager_notes.txt")
        self.notes: List[str] = []
        self.manager_usage = TokenUsageTracker()
        self.helper_usage = TokenUsageTracker()

    def _save_notes(self) -> None:
        self.notes_path.write_text("\n\n".join(self.notes), encoding="utf-8")

    def _parse_helper_decision(self, content: str) -> List[str]:
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and isinstance(payload.get("helpers"), list):
                helpers = [str(item).strip().lower() for item in payload["helpers"]]
                valid = [item for item in helpers if item in {"search", "math"}]
                if valid:
                    return valid
        except json.JSONDecodeError:
            pass
        return detect_task_mix(content)

    def _run_helper(self, helper_name: str, user_query: str) -> str:
        if helper_name == "search":
            tools = SEARCH_ONLY_TOOLS
            system_prompt = SEARCH_PROMPT
        elif helper_name == "math":
            tools = MATH_ONLY_TOOLS
            system_prompt = MATH_PROMPT
        else:
            raise ValueError(f"Unsupported helper: {helper_name}")

        messages = [{"role": "user", "content": user_query}]
        response = ask_llm(messages, system_prompt=system_prompt, tools=tools, usage_tracker=self.helper_usage)
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return message.content or "No output returned by the helper."

        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": tool_calls})
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = AVAILABLE_TOOLS[function_name]
            function_args = json.loads(tool_call.function.arguments)
            result = function_to_call(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(result),
                }
            )

        second_response = ask_llm(messages, usage_tracker=self.helper_usage)
        content = second_response.choices[0].message.content or "No output returned by the helper."
        return content

    def run(self, user_query: str) -> Dict[str, Any]:
        manager_messages = [{"role": "user", "content": user_query}]
        manager_response = ask_llm(manager_messages, system_prompt=MANAGER_PROMPT, usage_tracker=self.manager_usage)
        manager_content = manager_response.choices[0].message.content or "{}"
        helper_order = self._parse_helper_decision(manager_content)

        helper_outputs: List[str] = []
        for helper in helper_order:
            helper_output = self._run_helper(helper, user_query)
            helper_outputs.append(f"[{helper}] {helper_output}")
            self.notes.append(f"Helper: {helper}\nResponse:\n{helper_output}")

        self._save_notes()

        synthesis_prompt = (
            "You are the manager agent. You have collected responses from helper agents. "
            "Now synthesize them into a final answer for the user. Keep it clear and concise.\n\n"
            f"Helper notes:\n{chr(10).join(self.notes)}"
        )
        synthesis_response = ask_llm(
            [{"role": "user", "content": user_query}],
            system_prompt=synthesis_prompt,
            usage_tracker=self.manager_usage,
        )
        answer = synthesis_response.choices[0].message.content or "No final answer created."
        return {
            "answer": answer,
            "helper_order": helper_order,
            "helper_outputs": helper_outputs,
            "notes_path": str(self.notes_path),
            "usage": {
                "manager": self.manager_usage.snapshot(),
                "helpers": self.helper_usage.snapshot(),
            },
        }


# --- Comparison --------------------------------------------------------------


def compare_systems(user_query: str) -> Dict[str, Any]:
    single_result = run_single_agent(user_query)
    multi_result = MultiAgentSystem(notes_path="manager_notes.txt").run(user_query)
    single_tokens = single_result["usage"]["total_tokens"]
    multi_tokens = multi_result["usage"]["manager"]["total_tokens"] + multi_result["usage"]["helpers"]["total_tokens"]
    return {
        "single_agent": single_result,
        "multi_agent": multi_result,
        "comparison": {
            "single_total_tokens": single_tokens,
            "multi_total_tokens": multi_tokens,
            "more_tokens": "multi_agent" if multi_tokens > single_tokens else "single_agent",
            "difference": abs(multi_tokens - single_tokens),
        },
    }


def main() -> None:
    sample_query = (
        "Look up the current price of the iPhone 16 and calculate the total cost after a 12% sales tax "
        "and a 5% discount. Then calculate how much is left after paying 8% GST on the discounted price."
    )
    result = compare_systems(sample_query)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
