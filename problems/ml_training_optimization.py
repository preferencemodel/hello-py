"""
ML Training Optimization Problem

This problem contains intentionally slow ML training code with performance bottlenecks.
The task is to profile and optimize the training pipeline.
"""

import os
import re
import subprocess
from typing import Any

from problem import Problem
from tools import BasicToolset

# Target speedup ratio required to pass
TARGET_SPEEDUP = 2.0


class MLTrainingOptimizationProblem(Problem):
    """
    A problem that requires profiling and optimizing a slow ML training pipeline.

    The task involves:
    1. Running the slow training code to establish a baseline
    2. Profiling to identify bottlenecks
    3. Optimizing the code (vectorization, better algorithms, etc.)
    4. Achieving at least TARGET_SPEEDUP speedup
    """

    def __init__(self):
        # Store the slow training filename
        self.slow_training_file = "problem_data/slow_ml_training_1.py"

        prompt = f"""You have an ML training pipeline that needs optimization for a {TARGET_SPEEDUP}x speedup.

Your task is to:
1. Read the training code from '{self.slow_training_file}' using read_file
2. Analyze the code and identify performance bottlenecks
3. Create an optimized version of the code (use vectorization, better algorithms, etc.)
4. Write the optimized code to a new file (e.g., 'optimized_ml_training.py')
5. Submit the path to your optimized file using submit_answer(filepath)

IMPORTANT:
- You must achieve at least {TARGET_SPEEDUP}x speedup (ratio >= {TARGET_SPEEDUP})
- Ensure the optimized code produces the same results (same accuracy)
- The grader will run both versions and compare runtime and correctness
- Submit the filepath to your optimized code using submit_answer(filepath)

Example workflow:
1. Read and analyze {self.slow_training_file}
2. Create optimized_ml_training.py with improvements
3. Submit "optimized_ml_training.py" using submit_answer("optimized_ml_training.py")
"""

        super().__init__(
            prompt=prompt,
            tools=BasicToolset.get_tools(),
            tool_handlers=BasicToolset.get_handlers(),
            expected_answer=f"{TARGET_SPEEDUP}x or better",  # Human-readable expected speedup
        )

        # Run the baseline code and store results
        print(f"Running baseline training from {self.slow_training_file}...")
        self.baseline_time, self.baseline_accuracy = self._run_training_file(
            self.slow_training_file
        )
        print(
            f"Baseline: time={self.baseline_time:.2f}s, accuracy={self.baseline_accuracy:.4f}"
        )

    def _run_training_file(self, filepath: str) -> tuple[float, float]:
        """
        Run a training file and extract the runtime and accuracy.

        Args:
            filepath: Path to the Python file to run

        Returns:
            tuple of (runtime, accuracy)
        """
        try:
            # Run the training file
            result = subprocess.run(
                ["python", filepath],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Training file failed with error:\n{result.stderr}")

            output = result.stdout

            # Extract FINAL_TIME
            time_match = re.search(r"FINAL_TIME:\s+([\d.]+)", output)
            if not time_match:
                raise ValueError(f"Could not find FINAL_TIME in output:\n{output}")
            runtime = float(time_match.group(1))

            # Extract test accuracy
            accuracy_match = re.search(r"Test accuracy:\s+([\d.]+)", output)
            if not accuracy_match:
                raise ValueError(f"Could not find test accuracy in output:\n{output}")
            accuracy = float(accuracy_match.group(1))

            return runtime, accuracy

        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Training file {filepath} timed out after 5 minutes"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error running {filepath}: {e}") from e

    def grade(self, artifacts: dict[str, Any]) -> bool:
        """
        Grade the optimized training file.

        The agent should submit the path to their optimized file in artifacts.
        We will run it and check:
        1. Runtime speedup (must be >= 2.0x)
        2. Correctness (accuracy must match baseline within tolerance)
        """
        optimized_file = artifacts.get("result")

        if optimized_file is None:
            print("  ⚠ Agent did not submit an optimized file path")
            return False

        if not isinstance(optimized_file, str):
            print(f"  ⚠ Expected file path string, got {type(optimized_file)}")
            return False

        if not os.path.exists(optimized_file):
            print(f"  ⚠ Optimized file does not exist: {optimized_file}")
            return False

        try:
            # Run the optimized file
            print(f"Running optimized training from {optimized_file}...")
            optimized_time, optimized_accuracy = self._run_training_file(optimized_file)
            print(
                f"Optimized: time={optimized_time:.2f}s, accuracy={optimized_accuracy:.4f}"
            )

            # Calculate speedup
            speedup_ratio = self.baseline_time / optimized_time

            # Check correctness (accuracy should be within 0.01 tolerance)
            accuracy_diff = abs(optimized_accuracy - self.baseline_accuracy)
            if accuracy_diff > 0.01:
                print(
                    f"  ✗ Accuracy mismatch: baseline={self.baseline_accuracy:.4f}, "
                    f"optimized={optimized_accuracy:.4f} (diff={accuracy_diff:.4f} > 0.01)"
                )
                return False

            # Check if speedup is reasonable (between 0.1x and 100x)
            if not (0.1 <= speedup_ratio <= 100.0):
                print(
                    f"  ⚠ Speedup ratio {speedup_ratio:.2f}x seems unrealistic (expected 0.1-100x)"
                )
                return False

            # Check if we achieved the target speedup
            if speedup_ratio >= TARGET_SPEEDUP:
                print(
                    f"  ✓ Agent achieved {speedup_ratio:.2f}x speedup with correct accuracy (target: {TARGET_SPEEDUP}x)"
                )
                return True
            else:
                print(
                    f"  ✗ Speedup {speedup_ratio:.2f}x is below target (need: {TARGET_SPEEDUP}x, achieved: {speedup_ratio:.2f}x)"
                )
                return False

        except Exception as e:
            print(f"  ⚠ Error grading optimized file: {e}")
            return False
