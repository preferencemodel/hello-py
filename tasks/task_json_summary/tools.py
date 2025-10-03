import os
import json

class TaskTools:

    def __init__(self, task_base: str):
        self.task_base = task_base

    def get_tools(self):
        return [
            {
                "name": "read_file",
                "description": "Read a text or JSON file from the repo",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "parse_json",
                "description": "Parse a JSON file and return structured data",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "submit_answer",
                "description": "Submit the final JSON answer",
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
            "parse_json": self.parse_json_tool,
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

    def parse_json_tool(self, path: str):
        full_path = os.path.join(self.task_base, path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"data": data}
        except Exception as e:
            return {"error": str(e)}

    def submit_answer_tool(self, answer: str):
        return {"answer": answer, "submitted": True}
