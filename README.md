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

## Task: **Optimize Slow Training Loop with Profiling**

**Task Description**: Given a slow training loop implementation, use profiling to identify bottlenecks and optimize performance while maintaining equivalent model performance.

**Why it's interesting**:
- Critical skill for efficient research
- Requires understanding of profiling tools
- Multiple optimization strategies (vectorization, caching, algorithmic improvements).
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

## ML Training Optimization Problem - Performance Bottlenecks

The `slow_ml_training_1.py` file in `problem_data/` contains several intentional performance bottlenecks:

### Data Generation Bottleneck
- **Function**: `generate_synthetic_images()`
- **Issue**: Returns Python lists instead of numpy arrays
- **Impact**: Forces conversion overhead in downstream functions

### Feature Extraction Bottlenecks
- **Function**: `extract_features_slow()`
- **Issues**:
  1. Uses nested Python loops instead of vectorized NumPy operations
  2. Recalculates the mean value multiple times (for std dev calculation)
  3. Manually iterates through pixels for basic statistics (mean, std, max, min)
  4. Calculates quadrant means using nested loops instead of array slicing
- **Impact**: This is typically the slowest part of the pipeline

### Feature Normalization Bottleneck
- **Function**: `normalize_features_slow()`
- **Issue**: Uses triple-nested loops to normalize features instead of vectorized operations
- **Impact**: Could be replaced with simple NumPy broadcasting

### Training Configuration
- **Function**: `train_model()`
- **Issue**: RandomForest configured with `n_jobs=1` (single-threaded)
- **Impact**: Could benefit from parallel processing

### Optimization Strategies
Agents should:
1. Use profiling to identify which functions take the most time
2. Replace Python loops with vectorized NumPy operations
3. Eliminate redundant calculations
4. Consider parallelization where appropriate
5. Verify correctness is maintained after optimization

---
