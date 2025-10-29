hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution.

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

# Connor Soohoo's solution

## Task Brainstorming

Here are 3 potential tasks that ML researchers commonly encounter:

### Option 1: **Reproduce Experimental Results from Paper with Verification**
**Task Description**: Given a paper's methodology section and baseline results, implement the experiment and verify the results match within statistical tolerance.

**Why it's interesting**:
- Core research skill: reproducing published results
- Requires reading comprehension + implementation + numerical validation
- Multiple valid implementation approaches
- Tests attention to detail in methodology

**Expected failure modes**:
- Misunderstanding experimental setup
- Wrong hyperparameters or initialization
- Not using the same evaluation metric
- Missing preprocessing steps
- Statistical comparison issues (not accounting for randomness)

**Grading approach**: Check if reproduced result is within confidence interval of reported result

**Difficulty tuning**: Adjust complexity of method, clarity of methodology description, strictness of tolerance

---

### Option 2: **Refactor Messy Research Notebook into Production-Ready Module**
**Task Description**: Convert a Jupyter notebook with exploratory code into a clean, documented, tested Python package with proper structure.

**Why it's interesting**:
- Critical transition from research to production
- Tests software engineering skills (structure, testing, documentation)
- Multiple design decisions to make
- Common real-world task

**Expected failure modes**:
- Missing tests or low test coverage
- Poor code organization (everything in one file)
- Missing or incomplete docstrings
- No type hints
- Linting issues (unused imports, formatting)
- Package not importable

**Grading approach**:
```python
checks = {
    "has_pyproject_toml": check_file_exists("pyproject.toml"),
    "has_tests": check_tests_exist(),
    "passes_linting": run_ruff_check(),
    "has_docstrings": check_docstring_coverage() > 0.8,
    "has_type_hints": run_mypy_check(),
    "imports_work": test_package_import()
}
```

**Difficulty tuning**: Adjust messiness of notebook, number of required checks, strictness of thresholds

---

### Option 3: **Optimize Slow Training Loop with Profiling**
**Task Description**: Given a slow training loop implementation, use profiling to identify bottlenecks and optimize performance while maintaining equivalent model performance.

**Why it's interesting**:
- Critical skill for efficient research
- Requires understanding of profiling tools
- Multiple optimization strategies (vectorization, caching, algorithmic improvements)
- Tests ability to maintain correctness while improving speed

**Expected failure modes**:
- Not using profiler to identify bottleneck
- Optimizing wrong part of code
- Breaking model correctness/equivalence
- Premature optimization without measurement
- Over-optimizing at cost of readability

**Grading approach**:
- Performance improvement threshold (e.g., >2x speedup)
- Model performance maintained (results match within tolerance)
- Could check profiler was actually used

**Difficulty tuning**: Adjust target speedup, subtlety of bottleneck, complexity of code

---

### Selected Task: [TBD]
