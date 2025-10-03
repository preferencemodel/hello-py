import subprocess
import os

def grade(model_output: str, repo_template="tasks/task_swe_01_bugfix"):
    """
    Grader: ignores model output content.
    Checks that pytest passes in the final repo state.
    """
    proc = subprocess.run(
        ["pytest", "-q"],
        cwd=repo_template,
        capture_output=True,
        text=True,
        timeout=10,
    )
    passed = (proc.returncode == 0)
    return passed, {"stdout": proc.stdout, "stderr": proc.stderr}
