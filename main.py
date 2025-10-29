import asyncio

from problem_runner import main

# from problems import MathCalculationProblem
from problems import MLTrainingOptimizationProblem

if __name__ == "__main__":
    # Create the problem instance
    # problem = MathCalculationProblem()
    problem = MLTrainingOptimizationProblem()

    # Configuration
    NUM_RUNS = 1
    CONCURRENT = False
    VERBOSE = False  # Set to True to see agent's reasoning and tool usage

    # Run the test suite
    asyncio.run(
        main(problem=problem, num_runs=NUM_RUNS, concurrent=CONCURRENT, verbose=VERBOSE)
    )
