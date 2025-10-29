"""
ML Training Optimization Problem

This problem contains intentionally slow ML training code with performance bottlenecks.
The task is to profile and optimize the training pipeline.
"""

from typing import Any

from problem import Problem
from tools import BasicToolset


class MLTrainingOptimizationProblem(Problem):
    """
    A problem that requires profiling and optimizing a slow ML training pipeline.

    The task involves:
    1. Running the slow training code to establish a baseline
    2. Profiling to identify bottlenecks
    3. Optimizing the code (vectorization, better algorithms, etc.)
    4. Achieving at least 2x speedup
    """

    def __init__(self):
        prompt = """You have an ML training pipeline that needs optimization for a 2x speedup.

Your task is to:
1. Read the training code from 'slow_ml_training_1.py' using read_file
2. Run the original code to measure the baseline time (look for "FINAL_TIME: X.XX" in output)
3. Analyze the code and identify performance bottlenecks
4. Create an optimized version of the code (use vectorization, better algorithms, etc.)
5. Run your optimized code and measure the new execution time
6. Calculate the speedup ratio: baseline_time / optimized_time
7. Submit the speedup ratio as a float (e.g., if baseline is 1.0s and optimized is 0.4s, submit 2.5)

IMPORTANT:
- The code prints "FINAL_TIME: X.XX" at the end - extract this for both runs
- You must achieve at least 2x speedup (ratio >= 2.0)
- Ensure the optimized code produces the same results (same accuracy)
- Submit the speedup ratio using submit_answer(ratio)

Example workflow:
1. Baseline run shows "FINAL_TIME: 1.25"
2. Your optimized version shows "FINAL_TIME: 0.50"
3. Speedup = 1.25 / 0.50 = 2.5
4. Submit 2.5 using submit_answer(2.5)

The training code file is located at: slow_ml_training_1.py
"""

        super().__init__(
            prompt=prompt,
            tools=BasicToolset.get_tools(),
            tool_handlers=BasicToolset.get_handlers(),
            expected_answer="2.0x or better",  # Human-readable expected speedup
        )

        # Store baseline for comparison
        self.baseline_time: float | None = None
        self.optimized_time: float | None = None

    def grade(self, result: Any) -> bool:
        """
        Check if the speedup ratio meets the 2x target.

        The agent should submit a speedup ratio (baseline_time / optimized_time).
        We require at least 2.0x speedup to pass.
        """
        if result is None:
            print("  ⚠ Agent did not submit a speedup ratio")
            return False

        try:
            speedup_ratio = float(result)

            # Check if speedup is reasonable (between 0.1x and 100x)
            if not (0.1 <= speedup_ratio <= 100.0):
                print(
                    f"  ⚠ Speedup ratio {speedup_ratio:.2f}x seems unrealistic (expected 0.1-100x)"
                )
                return False

            # Check if we achieved the 2x target
            if speedup_ratio >= 2.0:
                print(f"  ✓ Agent achieved {speedup_ratio:.2f}x speedup (target: 2.0x)")
                return True
            else:
                print(
                    f"  ✗ Speedup {speedup_ratio:.2f}x is below target (need: 2.0x, achieved: {speedup_ratio:.2f}x)"
                )
                return False

        except (ValueError, TypeError) as e:
            print(f"  ⚠ Could not parse result as speedup ratio: {result} (error: {e})")
            return False
