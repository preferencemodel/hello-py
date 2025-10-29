from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from anthropic.types import ToolUnionParam


class Problem(ABC):
    """Abstract base class for agent tasks/problems."""

    def __init__(
        self,
        prompt: str,
        tools: list[ToolUnionParam],
        tool_handlers: dict[str, Callable[..., Any]],
        expected_answer: Any,
    ):
        self.prompt = prompt
        self.tools = tools
        self.tool_handlers = tool_handlers
        self.expected_answer = expected_answer

    @abstractmethod
    def grade(self, result: Any) -> bool:
        """
        Grade the result against the expected answer.
        Subclasses must implement this method.

        Args:
            result: The agent's submitted answer

        Returns:
            bool: True if the result is correct, False otherwise
        """
        pass
