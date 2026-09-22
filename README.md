# Dataset Analysis Agent

A small AI agent that answers natural-language questions about a dataset by writing and running its own Python/pandas code. Built to learn the core tool-use loop behind AI agents: the model decides what to run, the code executes locally, and the result is fed back so the model can refine its answer or dig deeper.

## What it does

- Ask a question in plain English about a CSV dataset (e.g. *"Which state has the most customers?"*)
- The agent (powered by Claude, via the Anthropic API) inspects the data and writes pandas code to answer it
- The code runs locally against the dataset, and the result is sent back to the model
- If the code fails or needs refining, the agent retries automatically, up to a set number of turns
- Conversations persist between runs, so you can ask follow-up questions that reference earlier answers
- Token usage is tracked and printed after each run

## Example

```
Ask a question about the data: Which state has the most customers?

--- running code ---
print(df['customer_state'].value_counts().head())
--------------------

=== FINAL ANSWER ===
São Paulo (SP) has the most customers, followed by Rio de Janeiro (RJ) and Minas Gerais (MG).

[tokens used this run — input: 612, output: 148]
```

## How it works

This project doesn't use any agent framework — the tool-use loop is written by hand to understand the mechanics:

1. The dataset and a summary of its structure (columns, types, sample rows) are given to the model via the system prompt.
2. A single tool, `run_python`, is described to the model: it can submit Python code to be executed against the dataframe.
3. If the model requests the tool, the code is executed locally (with output and errors captured), and the result is sent back.
4. This repeats until the model has enough information to give a final answer in plain text, or a turn limit is reached.
5. The full conversation is saved to `conversation.json` after each run, so context carries over to the next session.

## Setup

```bash
pip install pandas anthropic
```

Set your Anthropic API key as an environment variable:

```powershell
# PowerShell
$env:ANTHROPIC_API_KEY="your-key-here"
```

```bash
# macOS/Linux
export ANTHROPIC_API_KEY="your-key-here"
```

Update the CSV path in `main.py` to point to your dataset.

## Usage

```bash
python main.py
```

You'll be prompted to type a question. To start a fresh conversation and clear saved history:

```bash
python main.py --reset
```

## Dataset

This project was tested on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), specifically the customers table, but it works with any tabular CSV dataset.

## Notes and limitations

- The model's generated code runs via Python's `exec()` with no sandboxing. This is fine for local use with trusted, non-sensitive data, but would need a sandboxed execution environment (e.g. a container or restricted subprocess) before handling untrusted input or running in production.
- Conversation history grows indefinitely across runs; for long-term use this would benefit from summarization or a cap on stored turns.
- Currently supports a single CSV file at a time; extending to multiple tables or a database connection would be a natural next step.
- If the script is interrupted mid tool-call (e.g. closed early), the saved conversation can end up with an incomplete tool request. Use python main.py --reset in your console to repair. 

## Possible next steps

- Add support for querying a SQL database directly, alongside CSV files
- Generate and save charts, not just text answers
- Add a simple web interface (e.g. Streamlit) instead of a terminal prompt
- Explore Anthropic's managed Agents API as a higher-level alternative to the hand-rolled loop
