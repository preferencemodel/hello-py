import json
import os

def grade(model_output: str, repo_template=None):
    if repo_template is None:
        repo_template = os.path.dirname(__file__)

    json_path = os.path.join(repo_template, "data/students.json")
    with open(json_path, "r", encoding="utf-8") as f:
        students = json.load(f)

    passed_students = [s for s in students if s["passed"]]
    avg = round(sum(s["grade"] for s in passed_students) / len(passed_students))

    top_students = sorted(passed_students, key=lambda s: (-s["grade"], s["name"]))[:2]
    expected = {
        "average": avg,
        "top_students": [s["name"] for s in top_students],
    }

    try:
        submitted = json.loads(model_output)
    except Exception as e:
        return False, {"error": f"Invalid JSON: {e}", "submitted": model_output}

    return (submitted == expected), {"expected": expected, "submitted": submitted}
