import os
import csv

class TaskTools:
    
    def __init__(self, task_base: str):
        self.task_base = task_base

    def get_tools(self):
        return [
            {
                "name": "read_file",
                "description": "Read a text or CSV file from the repo",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "filter_csv",
                "description": "Filter a CSV file by column and condition, return matching rows",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "column": {"type": "string"},
                        "condition": {"type": "string"}
                    },
                    "required": ["path", "column", "condition"],
                },
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
        return {
            "read_file": self.read_file_tool,
            "filter_csv": self.filter_csv_tool,
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

    def filter_csv_tool(self, path: str, column: str, condition: str):
        full_path = os.path.join(self.task_base, path)
        try:
            with open(full_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    try:
                        # naive eval-based condition (simple only)
                        if eval(f"int(row[column]) {condition}"):
                            rows.append(row)
                    except Exception:
                        pass
            return {"rows": rows}
        except Exception as e:
            return {"error": str(e)}

    def submit_answer_tool(self, answer: str):
        return {"answer": answer, "submitted": True}
