
---

## Running Tasks

This project contains **5 reinforcement learning (RL) tasks** designed for model evaluation.  
When you run `main.py`, you will be prompted to select which task to execute.

### Task Menu

- **1 - CSV Filtering**  
  Reads employee data from CSV and computes an average salary with filtering.  

- **2 - JSON Summary**  
  Parses student performance data from JSON and computes average + top scorers.  

- **3 - Math Word Problem**  
  Solves a distance/speed/time algebraic problem using tools.  

- **4 - Logic Puzzle**  
  Constraint-satisfaction puzzle where the agent must assign pets to people.  

- **5 - SWE Bugfix**  
  A software engineering task where the agent inspects, edits, and tests Python code.  


## Setup instructions:

1. Clone the repository:
   ```bash
   git clone https://github.com/trive-coder/Devesh-RL-Tasks/tree/devesh-RL-Tasks
````

2. Navigate to the project directory:

   ```bash
   cd Devesh-RL-Tasks
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   or if you are using `uv`:

   ```bash
   uv pip install -r requirements.txt
   ```

4. Set up `ANTHROPIC_API_KEY` environment variable:

   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

5. Run the agent:

   ```bash
   uv run main.py


### How to Run a Specific Task

When you execute:


uv run main.py


You’ll see:

```
Select a task to run:
1 - CSV Filtering
2 - JSON Summary
3 - Math Word Problem
4 - Logic Puzzle
5 - SWE Bugfix
Enter task number (1–5):
```

* Enter `1` to run **CSV Filtering**.
* Enter `3` to run **Math Word Problem**.
* Enter `5` to run **SWE Bugfix**, etc.

If you type an invalid number (or press Enter), the program defaults to **Task 1 (CSV Filtering)**.

---


