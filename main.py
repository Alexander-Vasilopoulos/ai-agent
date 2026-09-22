import json
import os
import sys
import pandas as pd
from anthropic import Anthropic


client = Anthropic()  # picks up ANTHROPIC_API_KEY automatically

df = pd.read_csv("olist_customers_dataset.csv")  # use your own CSV here. It must be placed in the same folder as main.py

# Give the model a quick summary of the data so it knows what it's working with
data_summary = f"""
Columns: {list(df.columns)}
Dtypes: {df.dtypes.to_dict()}
First rows:
{df.head().to_string()}
"""

#test message
# response = client.messages.create(
#     model="claude-sonnet-4-6",
#     max_tokens=500,
#     messages=[{
#         "role": "user",
#         "content": f"Here is a dataset:\n{data_summary}\n\nWhat's an interesting question I could ask about this data?"
#     }]
# )

# print(response.content[0].text)

# define tools/methods so the agent knows how to call them. they are used as a parameter in the llm response 
tools = [
    {
        "name": "run_python",
        "description": "Execute Python code against the dataframe `df` (already loaded). Use print() to see output. Only the last exec's printed output is returned to you.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run. Use pandas on `df`. Print results you want to see."
                }
            },
            "required": ["code"]
        }
    }
]

def run_python(code: str) -> str:
    import io
    import contextlib

    namespace = {"df": df, "pd": pd}
    output = io.StringIO()
    try: # code from ai may not be entirely correct can be seen in conversation json
        with contextlib.redirect_stdout(output):
            exec(code, namespace)
        return output.getvalue() or "(no output — did you forget to print()?)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

#store file in the same folder as main.py
HISTORY_FILE = "conversation.json"

def load_messages():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_messages(messages):
    '''saves conversation into a json in the same folder as main.py'''
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)


def ask_agent(question: str, max_turns: int = 5):
    messages = load_messages()
    messages = repair_messages(messages)
    messages.append({
        "role": "user",
        "content": f"Question: {question}"
    })
    total_input_tokens = 0 #tracks tokenusage
    total_output_tokens = 0

# loop request until agent gets the answer
    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=f"You are a data analysis assistant. Here is the dataset you're working with:\n{data_summary}",
            tools=tools,
            messages=messages
        )
         # Track token usage for this call
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Convert the response's typed content blocks into plain dicts so they're JSON-serializable
        assistant_content = [block.model_dump() for block in response.content]
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "run_python":
                    code = block.input["code"]
                    print(f"\n--- running code ---\n{code}\n--------------------")
                    result = run_python(code)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            for block in response.content:
                if block.type == "text":
                    save_messages(messages)
                    print(f"\n[tokens used this run — input: {total_input_tokens}, output: {total_output_tokens}]")
                    return block.text

    save_messages(messages)
    print(f"\n[tokens used this run — input: {total_input_tokens}, output: {total_output_tokens}]")
    return "Stopped after max turns without a final answer."

def repair_messages(messages):
    """Remove a trailing assistant message if it contains an unanswered tool_use."""
    if not messages:
        return messages
    last = messages[-1]
    if last["role"] == "assistant":
        content = last["content"]
        if any(block.get("type") == "tool_use" for block in content):
            print("Warning: found an incomplete tool call in saved history — dropping it.")
            return messages[:-1]
    return messages

if __name__ == "__main__":
    # typing python main.py --reset in console clears old conversation
    if "--reset" in sys.argv:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        print("Conversation history cleared.\n")
    question = input("Ask a question about the data: ")
    answer = ask_agent(question)
    print("\n=== FINAL ANSWER ===")
    print(answer)

# run in terminal    python main.py --reset in console to clear old conversation