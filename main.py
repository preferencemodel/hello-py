import asyncio

from problem_runner import main
from problems import MathCalculationProblem

if __name__ == "__main__":
    # Create the problem instance
    problem = MathCalculationProblem()

    # Run the test suite
    asyncio.run(main(problem=problem, num_runs=1, concurrent=False))
