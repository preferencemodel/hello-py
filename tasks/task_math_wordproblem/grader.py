import os
import math

def grade(model_output: str, repo_template=None):
    if repo_template is None:
        repo_template = os.path.dirname(__file__)

    eta = 1.2
    expected = round(eta / math.sqrt(eta**2 - 1), 1)  # -> 1.8

    try:
        submitted = round(float(model_output), 1)
    except Exception:
        return False, {"error": f"Invalid numeric output: {model_output}"}

    return (submitted == expected), {"expected": expected, "submitted": submitted}
