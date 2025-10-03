import csv
import os

def grade(model_output: str, repo_template="tasks/task_csv_filtering"):
    csv_path = os.path.join(repo_template, "data/employees.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        filtered = [
            int(row["salary"])
            for row in reader
            if row["department"] == "Engineering" and int(row["experience"]) > 3
        ]

    expected = round(sum(filtered) / len(filtered))

    try:
        submitted = int(model_output)
    except Exception:
        return False, {"error": f"Could not parse answer: {model_output}"}

    return (submitted == expected), {"expected": expected, "submitted": submitted}
