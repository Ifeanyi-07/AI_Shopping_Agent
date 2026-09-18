"""
Response quality eval for the shopping agent, using an LLM as judge.

Scores each agent response on:
  - relevance: does it address what the user asked?
  - correctness: do the products shown plausibly match the stated filters?
  - format_compliance: does it follow the required numbered list format?

Run with:
    python eval/response_quality_eval.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from shopping_agent import agent

judge_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

TEST_QUERIES = [
    "organic honey under $20",
    "I want almonds with a rating above 4",
    "show me non-organic snacks under $10",
]

JUDGE_SYSTEM_PROMPT = """
You are grading a shopping assistant's response for quality. Score each of
the following dimensions from 1 (poor) to 5 (excellent):

- relevance: does the response directly address the user's request?
- correctness: do the products shown plausibly match the stated filters
  (price, organic status, rating) based on the user's query? Judge based on
  internal consistency, since you don't have direct database access.
- format_compliance: does it use the required plain-text numbered list format
  "#<number>. <name> (ID:<id>) — $<price> stars<rating> — <organic/non-organic>"
  with a blank line between entries — or, if no products qualify, a clear
  plain-text statement of that?

Respond with ONLY a JSON object, no other text, like:
{"relevance": 1-5, "correctness": 1-5, "format_compliance": 1-5, "notes": "short explanation"}
"""


def judge_response(query: str, response_text: str) -> dict:
    judge_input = f"User query: {query}\n\nAgent response:\n{response_text}"
    result = judge_llm.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_input},
        ]
    )
    raw = result.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "relevance": None,
            "correctness": None,
            "format_compliance": None,
            "notes": f"Could not parse judge output: {raw}",
        }


def run_eval() -> None:
    scores = []
    for i, query in enumerate(TEST_QUERIES, start=1):
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        response_text = result["messages"][-1].content

        verdict = judge_response(query, response_text)
        scores.append(verdict)

        preview = response_text[:200] + ("..." if len(response_text) > 200 else "")
        print(f'Test {i}: "{query}"')
        print(f"  Response: {preview}")
        print(f"  Scores: {verdict}")
        print()

    numeric_scores = [s for s in scores if isinstance(s.get("relevance"), (int, float))]
    if numeric_scores:
        for dim in ("relevance", "correctness", "format_compliance"):
            vals = [s[dim] for s in numeric_scores if isinstance(s.get(dim), (int, float))]
            if vals:
                print(f"Average {dim}: {sum(vals) / len(vals):.2f}")


if __name__ == "__main__":
    run_eval()
