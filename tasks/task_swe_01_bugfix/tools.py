import os
import subprocess

class TaskTools:
    """
    Defines tools for task_swe_01_bugfix.
    """

    def __init__(self, task_base: str):
        self.task_base = task_base

    def get_tools(self):
        """
        Returns the tool specifications for the LLM.
        """
        return [
            {
                "name": "read_file",
                "description": "Read a file from the repo",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write/overwrite a file in the repo",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "run_pytest",
                "description": "Run pytest on the repository",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "submit_answer",
                "description": "Submit the final answer",
                "input_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        ]

    def get_tool_handlers(self):
        """
        Returns a dict mapping tool names to handler functions.
        """
        return {
            "read_file": self.read_file_tool,
            "write_file": self.write_file_tool,
            "run_pytest": self.run_pytest_tool,
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

    def write_file_tool(self, path: str, content: str):
        full_path = os.path.join(self.task_base, path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "ok", "path": path}
        except Exception as e:
            return {"error": str(e)}

    def run_pytest_tool(self):
        try:
            proc = subprocess.run(
                ["pytest", "-q"],
                cwd=self.task_base,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
        except Exception as e:
            return {"error": str(e)}

    def submit_answer_tool(self, answer: str):
        return {"answer": answer, "submitted": True}
