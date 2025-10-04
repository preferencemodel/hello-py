import os

def grade(model_output: str, repo_template=None):
    if repo_template is None:
        repo_template = os.path.dirname(__file__)

    expected = 0.747  # theoretical optimum
    try:
        submitted = float(model_output)
    except Exception:
        return False, {"error": f"Invalid numeric output: {model_output}"}

    # Allow small tolerance (within 0.005)
    success = abs(submitted - expected) < 0.005

    return success, {"expected": expected, "submitted": submitted}
