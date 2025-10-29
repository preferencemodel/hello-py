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


class ReadFileToolResult(TypedDict):
    content: str | None
    error: str | None


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


def read_file_tool(file_path: str) -> ReadFileToolResult:
    """
    Tool that reads a file from the problem_data directory.
    Only allows reading files from the problem_data folder for security.
    """
    from pathlib import Path

    try:
        # Get the base directory (where tools.py is located)
        base_dir = Path(__file__).parent
        problem_data_dir = base_dir / "problem_data"

        # Normalize the file path
        requested_path = Path(file_path)

        # If it's a relative path, treat it as relative to problem_data
        if not requested_path.is_absolute():
            full_path = problem_data_dir / requested_path
        else:
            full_path = requested_path

        # Resolve to canonical path
        full_path = full_path.resolve()

        # Security check: ensure the path is within problem_data directory
        if not full_path.is_relative_to(problem_data_dir.resolve()):
            return {
                "content": None,
                "error": f"Access denied: {file_path} is outside problem_data directory",
            }

        # Check if file exists
        if not full_path.exists():
            return {"content": None, "error": f"File not found: {file_path}"}

        # Read the file
        with open(full_path, encoding="utf-8") as f:
            content = f.read()

        return {"content": content, "error": None}

    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"content": None, "error": f"Error reading file: {str(e)}"}


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

READ_FILE_SCHEMA: ToolUnionParam = {
    "name": "read_file",
    "description": "Read a file from the problem_data directory. Provide a relative path like 'slow_ml_training.py'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file relative to the problem_data directory",
            }
        },
        "required": ["file_path"],
    },
}


class BasicToolset:
    """Common tools for math/logic problems."""

    @staticmethod
    def get_tools() -> list[ToolUnionParam]:
        """Get tool schemas for the toolset."""
        return [PYTHON_EXPRESSION_SCHEMA, SUBMIT_ANSWER_SCHEMA, READ_FILE_SCHEMA]

    @staticmethod
    def get_handlers() -> dict[str, Callable[..., Any]]:
        """Get tool handlers for the toolset."""
        return {
            "python_expression": python_expression_tool,
            "submit_answer": submit_answer_tool,
            "read_file": read_file_tool,
        }
