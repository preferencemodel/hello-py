"""
ML Training Optimization Problem

This problem contains intentionally slow ML training code with performance bottlenecks.
The task is to profile and optimize the training pipeline.
"""

import os
import re
import subprocess
from typing import Any, Literal, overload

from problem import Problem
from tools import BasicToolset

# Target speedup ratio required to pass
TARGET_SPEEDUP = 2.0


class MLTrainingOptimizationProblem(Problem):
    """
    Base class for ML training optimization problems.

    This is an abstract base class that provides the infrastructure for grading
    ML optimization tasks. Subclasses should call super().__init__() with
    appropriate configuration parameters.

    The task involves:
    1. Running the slow training code to establish a baseline
    2. Profiling to identify bottlenecks
    3. Optimizing the code (vectorization, better algorithms, etc.)
    4. Achieving at least TARGET_SPEEDUP speedup
    """

    def __init__(
        self,
        slow_training_file: str,
        required_imports: list[str],
        required_components: list[str],
        expected_output_patterns: dict[str, Any],
        min_file_length: int = 500,
        accuracy_tolerance: float = 0.01,
    ):
        """
        Initialize the ML Training Optimization Problem.

        Args:
            slow_training_file: Path to the baseline slow training file
            required_imports: List of required import strings (e.g., ["numpy", "sklearn"])
            required_components: List of required code components (e.g., ["RandomForestClassifier"])
            expected_output_patterns: Dict of expected output patterns for validation
                Example: {"train_size": 800, "test_size": 200, "n_samples": 1000, "n_features": 8}
            min_file_length: Minimum acceptable file length in characters
            accuracy_tolerance: Maximum allowed difference in accuracy between baseline and optimized
        """
        # Store configuration for safety checks
        self.slow_training_file = slow_training_file
        self.required_imports = required_imports
        self.required_components = required_components
        self.expected_output_patterns = expected_output_patterns
        self.min_file_length = min_file_length
        self.accuracy_tolerance = accuracy_tolerance

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

    @overload
    def _run_training_file(
        self, filepath: str, return_output: Literal[False] = False
    ) -> tuple[float, float]: ...

    @overload
    def _run_training_file(
        self, filepath: str, return_output: Literal[True]
    ) -> tuple[float, float, str]: ...

    def _run_training_file(
        self, filepath: str, return_output: bool = False
    ) -> tuple[float, float] | tuple[float, float, str]:
        """
        Run a training file and extract the runtime and accuracy.

        Args:
            filepath: Path to the Python file to run
            return_output: If True, also return the stdout output

        Returns:
            tuple of (runtime, accuracy) or (runtime, accuracy, output) if return_output=True
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

            if return_output:
                return runtime, accuracy, output
            return runtime, accuracy

        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Training file {filepath} timed out after 5 minutes"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error running {filepath}: {e}") from e

    def _safety_checks(self, filepath: str, output: str) -> tuple[bool, str]:
        """
        Perform safety checks to ensure the optimized file is legitimate
        and not just cheating by printing hardcoded values.

        Args:
            filepath: Path to the optimized training file
            output: The stdout output from running the file

        Returns:
            tuple of (passed: bool, message: str)
        """
        # Check 1: Verify file contains actual ML code
        try:
            with open(filepath) as f:
                file_content = f.read()

            # Check for required imports (configurable)
            for imp in self.required_imports:
                if imp not in file_content:
                    return False, f"Missing required import: {imp}"

            # Check for key components (configurable)
            for component in self.required_components:
                if component not in file_content:
                    return False, f"Missing required component: {component}"

            # Check that file is not suspiciously short (configurable)
            if len(file_content) < self.min_file_length:
                return (
                    False,
                    f"File suspiciously short ({len(file_content)} chars). Possible cheating.",
                )

        except Exception as e:
            return False, f"Error reading file for safety checks: {e}"

        # Check 2: Verify intermediate outputs match expected patterns (if configured)
        if (
            "train_size" in self.expected_output_patterns
            and "test_size" in self.expected_output_patterns
        ):
            train_size_match = re.search(r"Training set size:\s+(\d+)", output)
            test_size_match = re.search(r"Test set size:\s+(\d+)", output)

            if train_size_match and test_size_match:
                train_size = int(train_size_match.group(1))
                test_size = int(test_size_match.group(1))

                expected_train = self.expected_output_patterns["train_size"]
                expected_test = self.expected_output_patterns["test_size"]

                if train_size != expected_train or test_size != expected_test:
                    return (
                        False,
                        f"Unexpected train/test split sizes: {train_size}/{test_size} "
                        f"(expected {expected_train}/{expected_test})",
                    )

        # Check 3: Verify deterministic outputs with same seed
        # Run the file twice and check if outputs are consistent
        try:
            result1 = self._run_training_file(filepath, return_output=True)
            assert len(result1) == 3, "Expected 3 return values when return_output=True"
            _, accuracy1, output1 = result1

            result2 = self._run_training_file(filepath, return_output=True)
            assert len(result2) == 3, "Expected 3 return values when return_output=True"
            _, accuracy2, output2 = result2

            # With random_state=42, accuracy should be identical
            if abs(accuracy1 - accuracy2) > 1e-6:
                return (
                    False,
                    f"Non-deterministic results detected: {accuracy1:.6f} vs {accuracy2:.6f}. "
                    "Code may not be using proper random seeds.",
                )

            # Check that the actual computation is happening (not just cached values)
            # by verifying the feature shape is mentioned (if configured)
            if (
                "n_samples" in self.expected_output_patterns
                and "n_features" in self.expected_output_patterns
            ):
                feature_shape_match = re.search(
                    r"Feature shape:\s+\((\d+),\s+(\d+)\)", output1
                )
                if feature_shape_match:
                    n_samples = int(feature_shape_match.group(1))
                    n_features = int(feature_shape_match.group(2))

                    expected_samples = self.expected_output_patterns["n_samples"]
                    expected_features = self.expected_output_patterns["n_features"]

                    if n_samples != expected_samples:
                        return (
                            False,
                            f"Unexpected number of samples in features: {n_samples} (expected {expected_samples})",
                        )

                    if n_features != expected_features:
                        return (
                            False,
                            f"Unexpected number of features: {n_features} (expected {expected_features})",
                        )

        except Exception as e:
            return False, f"Error during deterministic check: {e}"

        # All checks passed
        return True, "All safety checks passed"

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
            result = self._run_training_file(optimized_file, return_output=True)
            assert len(result) == 3, "Expected 3 return values when return_output=True"
            optimized_time, optimized_accuracy, output = result
            print(
                f"Optimized: time={optimized_time:.2f}s, accuracy={optimized_accuracy:.4f}"
            )

            # Perform safety checks to prevent cheating
            print("Running safety checks...")
            safety_passed, safety_message = self._safety_checks(optimized_file, output)
            if not safety_passed:
                print(f"  ✗ Safety check failed: {safety_message}")
                return False
            print(f"  ✓ {safety_message}")

            # Calculate speedup
            speedup_ratio = self.baseline_time / optimized_time

            # Check correctness (accuracy should be within tolerance)
            accuracy_diff = abs(optimized_accuracy - self.baseline_accuracy)
            if accuracy_diff > self.accuracy_tolerance:
                print(
                    f"  ✗ Accuracy mismatch: baseline={self.baseline_accuracy:.4f}, "
                    f"optimized={optimized_accuracy:.4f} (diff={accuracy_diff:.4f} > {self.accuracy_tolerance})"
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


class MLTrainingOptimizationNumPy(MLTrainingOptimizationProblem):
    """
    ML training optimization problem using sklearn Random Forest (NumPy-based).

    This problem uses the slow_ml_training_numpy.py baseline which contains:
    - Inefficient feature extraction loops
    - Slow normalization implementation
    - 1000 samples with 28x28 images
    - 8 features extracted per image
    """

    def __init__(self):
        super().__init__(
            slow_training_file="problem_data/slow_ml_training_numpy.py",
            required_imports=["numpy", "sklearn"],
            required_components=["RandomForestClassifier", "train_test_split"],
            expected_output_patterns={
                "train_size": 800,
                "test_size": 200,
                "n_samples": 1000,
                "n_features": 8,
            },
            min_file_length=500,
            accuracy_tolerance=0.01,
        )


class MLTrainingOptimizationPyTorch(MLTrainingOptimizationProblem):
    """
    ML training optimization problem using PyTorch neural network.

    This problem uses the slow_ml_training_pytorch.py baseline which contains:
    - PyTorch neural network training with inefficient single-sample updates
    - Larger dataset (2000 samples)
    - 8 features extracted per image
    - Inefficient feature extraction and normalization loops
    """

    def __init__(self):
        super().__init__(
            slow_training_file="problem_data/slow_ml_training_pytorch.py",
            required_imports=["torch", "nn"],
            required_components=["SimpleNN", "Adam", "CrossEntropyLoss"],
            expected_output_patterns={
                "train_size": 1600,
                "test_size": 400,
                "n_samples": 2000,
                "n_features": 8,
            },
            min_file_length=700,
            accuracy_tolerance=0.02,
        )
