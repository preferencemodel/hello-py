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

### Available Problem Variants

The codebase includes two ML optimization problems with different frameworks:

| Feature | Problem 1 (NumPy) | Problem 2 (PyTorch) |
|---------|-------------------|---------------------|
| Framework | sklearn | PyTorch |
| Model | RandomForest | Neural Network (SimpleNN) |
| Baseline File | `slow_ml_training_numpy.py` | `slow_ml_training_pytorch.py` |
| Samples | 1000 | 2000 |
| Features | 8 | 8 |
| Main bottleneck | Feature extraction loops | Single-sample training loop |
| Required imports | `numpy`, `sklearn` | `torch`, `nn` |
| Key components | `RandomForestClassifier`, `train_test_split` | `SimpleNN`, `Adam`, `CrossEntropyLoss` |
| Accuracy tolerance | 0.01 | 0.02 |
| Target speedup | 2.0x | 2.0x |

**To switch between problems**, edit `main.py`:
```python
# Use NumPy/sklearn version:
problem = MLTrainingOptimizationNumPy()

# Use PyTorch version (default):
problem = MLTrainingOptimizationPyTorch()
```

### Safety Checks Implemented

The grader includes comprehensive safety checks to prevent cheating:

1. **Code Content Validation**
   - Verifies required imports are present (framework-specific)
   - Checks for required ML components (models, optimizers, etc.)
   - Ensures file is not suspiciously short (min 500-700 chars)

2. **Output Pattern Validation**
   - Validates train/test split sizes match expected values
   - Checks feature shape matches baseline
   - Ensures actual computation is happening (not just print statements)

3. **Deterministic Behavior Check**
   - Runs optimized code twice with same random seed
   - Verifies results are identical (proves proper seeding, not random outputs)

4. **Correctness Validation**
   - Compares accuracy between baseline and optimized versions
   - Requires accuracy to match within tolerance


  Based on the complex training pipeline analysis:
  - Fix O(n²) normalization: ~3x speedup (30-35% of runtime)
  - Vectorize custom activation: ~2x additional (25-30% of runtime)
  - Optimize gradient extraction: ~1.2x additional (8-10% of runtime)
  - Increase batch size: ~1.5x additional (8-10% overhead)
  - Total achievable: ~9-10x speedup

Failure modes:
1. tries to completely rewrite ComplexFeatureExtractor or ComplexNN into something non-sensical and tries to cheat the system, instead of using proper profiling tools or optimizations


2. Doesn't adhere to the prompt requirements (don't rewrite the function names because of validation, or instead of outputting artifacts under output/run_X, it just outputs to the root directory)

Gets between 32 and 34x optimized.
3.
# Line 81-101: Replace O(n²) normalization
  def custom_normalize(self, features):
      """Efficient z-score normalization - O(n)."""
      mean = np.mean(features, axis=0, keepdims=True)
      std = np.std(features, axis=0, keepdims=True) + 1e-8
      return (features - mean) / std
  Result: ~15-16s runtime (1.7-1.8x speedup) ✅

4. Doesn't actually generate a optimized_ml_training.py file (model instability)
