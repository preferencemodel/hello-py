# import json
# import math
# from contextlib import redirect_stdout
# from io import StringIO
# from typing import Any, Callable, TypedDict

# from anthropic import Anthropic
# from anthropic.types import MessageParam, ToolUnionParam


# class PythonExpressionToolResult(TypedDict):
#     result: Any
#     error: str | None


# class SubmitAnswerToolResult(TypedDict):
#     answer: Any
#     submitted: bool


# def python_expression_tool(expression: str) -> PythonExpressionToolResult:
#     """
#     Tool that evaluates Python expressions using eval.
#     """
#     try:
#         # Make common modules available
#         safe_globals = {
#             "__builtins__": {},
#             "math": math,
#             "abs": abs,
#             "min": min,
#             "max": max,
#             "sum": sum,
#             "len": len,
#             "range": range,
#             "round": round,
#             "int": int,
#             "float": float,
#             "str": str,
#             "list": list,
#             "dict": dict,
#             "tuple": tuple,
#             "set": set,
#             "print": print,
#         }
#         stdout = StringIO()
#         with redirect_stdout(stdout):
#             exec(expression, safe_globals, {})
#         return {"result": stdout.getvalue(), "error": None}
#     except KeyboardInterrupt:
#         raise
#     except Exception as e:
#         return {"result": None, "error": str(e)}


# def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
#     """
#     Tool for submitting the final answer.
#     """
#     return {"answer": answer, "submitted": True}


# def run_agent_loop(
#     prompt: str,
#     tools: list[ToolUnionParam],
#     tool_handlers: dict[str, Callable],
#     max_steps: int = 5,
#     model: str = "claude-3-5-haiku-latest",
#     verbose: bool = True,
# ) -> Any | None:
#     """
#     Runs an agent loop with the given prompt and tools.

#     Args:
#         prompt: The initial prompt for the agent
#         tools: List of tool definitions for Anthropic API
#         tool_handlers: Dictionary mapping tool names to their handler functions
#         max_steps: Maximum number of steps before stopping (default 5)
#         model: The Anthropic model to use
#         verbose: Whether to print detailed output (default True)

#     Returns:
#         The submitted answer if submit_answer was called, otherwise None
#     """
#     client = Anthropic()
#     messages: list[MessageParam] = [{"role": "user", "content": prompt}]

#     for step in range(max_steps):
#         if verbose:
#             print(f"\n=== Step {step + 1}/{max_steps} ===")

#         response = client.messages.create(
#             model=model, max_tokens=1000, tools=tools, messages=messages
#         )

#         # Track if we need to continue
#         has_tool_use = False
#         tool_results = []
#         submitted_answer = None

#         # Process the response
#         for content in response.content:
#             if content.type == "text":
#                 if verbose:
#                     print(f"Assistant: {content.text}")
#             elif content.type == "tool_use":
#                 has_tool_use = True
#                 tool_name = content.name

#                 if tool_name in tool_handlers:
#                     if verbose:
#                         print(f"Using tool: {tool_name}")

#                     # Extract arguments based on tool
#                     handler = tool_handlers[tool_name]
#                     tool_input = content.input

#                     # Call the appropriate tool handler
#                     if tool_name == "python_expression":
#                         assert (
#                             isinstance(tool_input, dict) and "expression" in tool_input
#                         )
#                         if verbose:
#                             print("\nInput:")
#                             print("```")
#                             for line in tool_input["expression"].split("\n"):
#                                 print(f"{line}")
#                             print("```")
#                         result = handler(tool_input["expression"])
#                         if verbose:
#                             print("\nOutput:")
#                             print("```")
#                             print(result)
#                             print("```")
#                     elif tool_name == "submit_answer":
#                         assert isinstance(tool_input, dict) and "answer" in tool_input
#                         result = handler(tool_input["answer"])
#                         submitted_answer = result["answer"]
#                     else:
#                         # Generic handler call
#                         result = (
#                             handler(**tool_input)
#                             if isinstance(tool_input, dict)
#                             else handler(tool_input)
#                         )

#                     tool_results.append(
#                         {
#                             "type": "tool_result",
#                             "tool_use_id": content.id,
#                             "content": json.dumps(result),
#                         }
#                     )

#         # If we have tool uses, add them to the conversation
#         if has_tool_use:
#             messages.append({"role": "assistant", "content": response.content})

#             messages.append({"role": "user", "content": tool_results})

#             # If an answer was submitted, return it
#             if submitted_answer is not None:
#                 if verbose:
#                     print(f"\nAgent submitted answer: {submitted_answer}")
#                 return submitted_answer
#         else:
#             # No tool use, conversation might be complete
#             if verbose:
#                 print("\nNo tool use in response, ending loop.")
#             break

#     if verbose:
#         print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
#     return None


# def main():
#     tools: list[ToolUnionParam] = [
#         {
#             "name": "python_expression",
#             "description": "Evaluates a Python expression",
#             "input_schema": {
#                 "type": "object",
#                 "properties": {
#                     "expression": {
#                         "type": "string",
#                         "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
#                     }
#                 },
#                 "required": ["expression"],
#             },
#         },
#         {
#             "name": "submit_answer",
#             "description": "Submit the final answer",
#             "input_schema": {
#                 "type": "object",
#                 "properties": {"answer": {"description": "The final answer to submit"}},
#                 "required": ["answer"],
#             },
#         },
#     ]

#     tool_handlers = {
#         "python_expression": python_expression_tool,
#         "submit_answer": submit_answer_tool,
#     }

#     # Run the test 10 times and track success rate
#     num_runs = 10
#     expected_answer = 8769
#     successes = 0

#     print(f"Running {num_runs} test iterations...")
#     print("=" * 60)

#     for i in range(num_runs):
#         print(f"\n\n{'=' * 20} RUN {i + 1}/{num_runs} {'=' * 20}")

#         result = run_agent_loop(
#             prompt="Calculate (2^10 + 3^5) * 7 - 100. Use the python_expression tool and then submit the answer.",
#             tools=tools,
#             tool_handlers=tool_handlers,
#             max_steps=5,
#             verbose=False,  # Set to False for cleaner output during multiple runs
#         )

#         if result == expected_answer:
#             print(f"✓ Run {i + 1}: SUCCESS - Got {result}")
#             successes += 1
#         else:
#             print(f"✗ Run {i + 1}: FAILURE - Got {result}, expected {expected_answer}")

#     # Calculate and display pass rate
#     pass_rate = (successes / num_runs) * 100
#     print(f"\n{'=' * 60}")
#     print("Test Results:")
#     print(f"  Passed: {successes}/{num_runs}")
#     print(f"  Failed: {num_runs - successes}/{num_runs}")
#     print(f"  Pass Rate: {pass_rate:.1f}%")
#     print(f"{'=' * 60}")


# if __name__ == "__main__":
#     main()

import json
import importlib.util
import os
import time
from typing import Any, Callable

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolUnionParam


# =====================
# Agent Loop
# =====================
def run_agent_loop(prompt, tools, tool_handlers, max_steps=4, model="claude-3-5-haiku-latest", verbose=True):
    client = Anthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    submitted_answer = None

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = client.messages.create(model=model, max_tokens=1000, tools=tools, messages=messages)

        has_tool_use = False
        tool_results = []

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")

            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                tool_input = content.input

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")
                        print("Input:", tool_input)

                    handler = tool_handlers[tool_name]
                    result = handler(**tool_input) if isinstance(tool_input, dict) else handler(tool_input)

                    if verbose:
                        print("Output:", result)

                    if isinstance(result, dict) and result.get("submitted"):
                        submitted_answer = result["answer"]

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result),
                    })

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\n✅ Agent submitted: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("No tool use. Ending loop.")
            break

    if verbose:
        print("Reached max steps without submitting an answer.")
    return submitted_answer


# =====================
# Task Runner
# =====================
def load_grader(task_name: str):
    grader_path = os.path.join("tasks", task_name, "grader.py")
    spec = importlib.util.spec_from_file_location("grader", grader_path)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    return grader

def load_tools(task_name: str):
    tools_path = os.path.join("tasks", task_name, "tools.py")
    spec = importlib.util.spec_from_file_location("tools", tools_path)
    tools_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools_module)

    task_base = os.path.join("tasks", task_name)
    tool_class = tools_module.TaskTools(task_base)
    return tool_class.get_tools(), tool_class.get_tool_handlers()


def run_task(task_name: str, num_runs=3):
    prompt_path = os.path.join("tasks", task_name, "prompt.txt")
    with open(prompt_path, "r") as f:
        prompt = f.read()

    grader = load_grader(task_name)
    tools, tool_handlers = load_tools(task_name)

    print(f"\n{'='*20} Running Task: {task_name} {'='*20}")
    successes = 0
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        start = time.time()
        result = run_agent_loop(prompt, tools, tool_handlers, verbose=True)
        elapsed = time.time() - start

        passed, info = grader.grade(str(result))
        if passed:
            print(f"✓ Success: {info}")
            successes += 1
        else:
            print(f"✗ Fail: {info}")

        print(f"Run {i+1} finished in {elapsed:.2f}s")

    pass_rate = successes / num_runs * 100
    print(f"\nTask {task_name} Results: {successes}/{num_runs} ({pass_rate:.1f}%)")


# =====================
# Entry Point
# =====================
def main():
    run_task("task_swe_01_bugfix", num_runs=10)
    # later: run_task("task2_xyz"), run_task("task3_xyz"), ...


if __name__ == "__main__":
    main()
