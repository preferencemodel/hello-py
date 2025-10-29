from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict

from anthropic.types import ToolUnionParam


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace: dict[str, Any] = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


# Tool schemas
PYTHON_EXPRESSION_SCHEMA: ToolUnionParam = {
    "name": "python_expression",
    "description": "Evaluates a Python expression",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
            }
        },
        "required": ["expression"],
    },
}

SUBMIT_ANSWER_SCHEMA: ToolUnionParam = {
    "name": "submit_answer",
    "description": "Submit the final answer",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"description": "The final answer to submit"}},
        "required": ["answer"],
    },
}


class BasicToolset:
    """Common tools for math/logic problems."""

    @staticmethod
    def get_tools() -> list[ToolUnionParam]:
        """Get tool schemas for the toolset."""
        return [PYTHON_EXPRESSION_SCHEMA, SUBMIT_ANSWER_SCHEMA]

    @staticmethod
    def get_handlers() -> dict[str, Callable[..., Any]]:
        """Get tool handlers for the toolset."""
        return {
            "python_expression": python_expression_tool,
            "submit_answer": submit_answer_tool,
        }
