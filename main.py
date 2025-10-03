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
    run_task("task_csv_filtering", num_runs=10)
    run_task("task_json_summary", num_runs=10)
    run_task("task_math_wordproblem", num_runs=10)


if __name__ == "__main__":
    main()
