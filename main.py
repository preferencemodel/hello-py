import asyncio

from problem_runner import main
from problems import (
    MLTrainingOptimizationPyTorch,
)

# Uncomment to use MathCalculationProblem:
# from problems import MathCalculationProblem

if __name__ == "__main__":
    # Create the problem instance
    # problem = MathCalculationProblem()
    # problem = MLTrainingOptimizationNumPy()
    problem = MLTrainingOptimizationPyTorch()

    # Configuration
    NUM_RUNS = 1
    CONCURRENT = False
    VERBOSE = False  # Set to True to see agent's reasoning and tool usage

    # Run the test suite
    asyncio.run(
        main(problem=problem, num_runs=NUM_RUNS, concurrent=CONCURRENT, verbose=VERBOSE)
    )
