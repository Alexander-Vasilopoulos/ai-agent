"""Dataset Analysis Agent

Answers natural-language questions about a CSV file by letting Claude write and
run pandas code, then feeding the result back until it can give a final answer.

Usage:
    python main.py path/to/data.csv
    python main.py path/to/data.csv --reset        # clear saved conversation
    python main.py path/to/data.csv --max-turns 8

WARNING: the model's code runs with exec() and NO sandbox. Only use trusted,
non-sensitive data. Sample rows are also sent to the Anthropic API.
"""

import argparse
import contextlib
import io
import json
import os
import sys

import pandas as pd
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1000
MAX_TOOL_OUTPUT_CHARS = 3000  # cap what a single tool call can send back to the model
HISTORY_FILE = "conversation.json"  # stored next to where you run the script

TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python code against the dataframe `df` (already loaded; `pd` is also available). "
            "Use print() to see output. Only printed output is returned to you, and it is truncated if "
            "very long, so print summaries (e.g. .head(), .value_counts()) rather than whole tables. "
            "Each call gets a fresh copy of `df`: changes you make do not persist between calls, "
            "so recompute anything you need inside the same code block."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run. Use pandas on `df`. Print results you want to see.",
                }
            },
            "required": ["code"],
        },
    }
]


# --------------------------------------------------------------------------
# Data and tool execution
# --------------------------------------------------------------------------

def build_data_summary(df: pd.DataFrame) -> str:
    """A short description of the data so the model knows what it is working with."""
    return (
        f"Columns: {list(df.columns)}\n"
        f"Dtypes: {df.dtypes.astype(str).to_dict()}\n"
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"First rows:\n{df.head().to_string()}"
    )


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n... (output truncated: {len(text)} characters in total. "
        "Print less, e.g. use .head() or aggregate first.)"
    )


def run_python(code: str, df: pd.DataFrame) -> str:
    """Run model-written code on a COPY of the dataframe and return captured output."""
    namespace = {"df": df.copy(), "pd": pd}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, namespace)  # noqa: S102 - unsandboxed on purpose, see warning at top
        return truncate(output.getvalue() or "(no output - did you forget to print()?)")
    except Exception as e:  # the model's code may be wrong; report it back so it can retry
        return truncate(f"Error: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# Conversation history
# --------------------------------------------------------------------------

def repair_messages(messages: list) -> list:
    """Drop a trailing assistant message that contains an unanswered tool_use."""
    if not messages:
        return messages
    last = messages[-1]
    if last["role"] == "assistant" and isinstance(last["content"], list):
        if any(block.get("type") == "tool_use" for block in last["content"]):
            print("Warning: found an incomplete tool call in saved history - dropping it.")
            return messages[:-1]
    return messages


def load_history(dataset_name: str) -> list:
    """Load saved messages, but only if they belong to the same dataset."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Warning: could not read saved history - starting fresh.")
        return []

    if not isinstance(data, dict):  # old format (a bare list) has no dataset info
        print("Saved history is in an old format - starting a fresh conversation.")
        return []
    if data.get("dataset") != dataset_name:
        print(f"Saved history is for '{data.get('dataset')}', not '{dataset_name}' - starting fresh.")
        return []
    return repair_messages(data.get("messages", []))


def save_history(dataset_name: str, messages: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "messages": messages}, f, indent=2)


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------

def print_usage(input_tokens: int, output_tokens: int) -> None:
    print(f"\n[tokens used this run - input: {input_tokens}, output: {output_tokens}]")


def ask_agent(
    client: Anthropic,
    df: pd.DataFrame,
    dataset_name: str,
    question: str,
    model: str = MODEL,
    max_turns: int = 5,
) -> str:
    system_prompt = (
        "You are a data analysis assistant. Answer questions by running pandas code with "
        f"the run_python tool.\nDataset file: {dataset_name}\n{build_data_summary(df)}"
    )

    messages = load_history(dataset_name)
    history_length = len(messages)  # used to roll back if this run fails
    messages.append({"role": "user", "content": f"Question: {question}"})

    total_input_tokens = 0
    total_output_tokens = 0

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Typed blocks -> plain dicts so they can be saved as JSON
        assistant_content = [block.model_dump(exclude_none=True) for block in response.content]
        messages.append({"role": "assistant", "content": assistant_content})

        # The model wants to run code: execute EVERY tool call and send all results back
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "run_python":
                    code = block.input["code"]
                    print(f"\n--- running code ---\n{code}\n--------------------")
                    result = run_python(code, df)
                else:
                    result = f"Error: unknown tool '{block.name}'"
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        # Anything else (end_turn, max_tokens, ...) ends the loop, so the history can never
        # be left ending on an assistant message that the loop then tries to continue from.
        was_cut_off = response.stop_reason == "max_tokens"
        if was_cut_off and any(block.type == "tool_use" for block in response.content):
            # A tool call cut off mid-way is unusable: drop it and don't save a broken history.
            messages.pop()
            save_history(dataset_name, messages)
            print_usage(total_input_tokens, total_output_tokens)
            return (
                "The response was cut off in the middle of a tool call. "
                "Try a simpler question or raise MAX_TOKENS."
            )

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            text = "(The model returned no text.)"
        if was_cut_off:
            text += "\n\n[Note: this answer was cut off at the token limit.]"

        save_history(dataset_name, messages)
        print_usage(total_input_tokens, total_output_tokens)
        return text

    # Ran out of turns: don't save this run, so history stays valid for the next question
    del messages[history_length:]
    save_history(dataset_name, messages)
    print_usage(total_input_tokens, total_output_tokens)
    return "Stopped after max turns without a final answer. (This question was not saved to history.)"


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a CSV file using Claude.")
    parser.add_argument("csv_path", help="path to the CSV file to analyse")
    parser.add_argument("--reset", action="store_true", help="clear saved conversation history first")
    parser.add_argument("--max-turns", type=int, default=5, help="max tool-use rounds per question")
    parser.add_argument("--model", default=MODEL, help=f"model name (default: {MODEL})")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        sys.exit(f"File not found: {args.csv_path}")

    if args.reset and os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        print("Conversation history cleared.\n")

    df = pd.read_csv(args.csv_path)
    dataset_name = os.path.basename(args.csv_path)
    client = Anthropic()  # picks up ANTHROPIC_API_KEY automatically

    question = input("Ask a question about the data: ")
    answer = ask_agent(client, df, dataset_name, question, model=args.model, max_turns=args.max_turns)
    print("\n=== FINAL ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
