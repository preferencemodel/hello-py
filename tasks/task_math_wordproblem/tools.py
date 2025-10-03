import os
import math

class TaskTools:
    """
    Defines tools for task_boats_ratio.
    """

    def __init__(self, task_base: str):
        self.task_base = task_base

    def get_tools(self):
        return [
            {
                "name": "read_file",
                "description": "Read a text file from the repo",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "python_calc",
                "description": "Evaluate a Python expression safely with math functions",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
            {
                "name": "submit_answer",
                "description": "Submit the final numeric result",
                "input_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        ]

    def get_tool_handlers(self):
        return {
            "read_file": self.read_file_tool,
            "python_calc": self.python_calc_tool,
            "submit_answer": self.submit_answer_tool,
        }

    # ========== TOOL IMPLEMENTATIONS ==========

    def read_file_tool(self, path: str):
        full_path = os.path.join(self.task_base, path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return {"content": f.read()}
        except Exception as e:
            return {"error": str(e)}

    def python_calc_tool(self, expression: str):
        try:
            safe_globals = {"__builtins__": None, "math": math}
            result = eval(expression, safe_globals, {})
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def submit_answer_tool(self, answer: str):
        return {"answer": answer, "submitted": True}
