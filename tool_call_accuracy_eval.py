"""
Tool call accuracy eval for the shopping agent.

Checks that the agent calls the expected tool with the expected arguments for
a small set of representative queries (e.g. "organic honey under $20" should
call search_products with is_organic=True, max_price=20).

Run with:
    python eval/tool_call_accuracy_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from shopping_agent import agent, get_tool_calls

# Each case: a query, the tool we expect to see called, and the arguments we
# expect it to be called with (subset match — extra args in the actual call
# are fine, missing/mismatched expected args are not).
TEST_CASES = [
    {
        "query": "organic honey under $20",
        "expected_tool": "search_products",
        "expected_args": {"query": "honey", "is_organic": True, "max_price": 20},
    },
    {
        "query": "show me almonds, I don't care about the price",
        "expected_tool": "search_products",
        "expected_args": {"query": "almonds"},
    },
    {
        "query": "non-organic olive oil under $15",
        "expected_tool": "search_products",
        "expected_args": {"query": "olive oil", "is_organic": False, "max_price": 15},
    },
    {
        "query": "what have I ordered before?",
        "expected_tool": "get_order_summary",
        "expected_args": {},
    },
    {
        "query": "list every order I've ever placed",
        "expected_tool": "view_orders",
        "expected_args": {},
    },
]


def args_match(actual: dict, expected: dict) -> bool:
    """Subset match: every key/value in `expected` must appear in `actual`.
    String values are compared case-insensitively and allow substring match
    (since the model may phrase the search query slightly differently)."""
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            if (
                expected_value.lower() not in actual_value.lower()
                and actual_value.lower() not in expected_value.lower()
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def run_eval() -> None:
    passed = 0
    for i, case in enumerate(TEST_CASES, start=1):
        result = agent.invoke({"messages": [HumanMessage(content=case["query"])]})
        calls = get_tool_calls(result["messages"])

        matched = any(
            name == case["expected_tool"] and args_match(args, case["expected_args"])
            for name, args in calls
        )

        status = "PASS" if matched else "FAIL"
        if matched:
            passed += 1

        print(f'[{status}] Test {i}: "{case["query"]}"')
        print(f"    Expected: {case['expected_tool']}({case['expected_args']})")
        print(f"    Actual tool calls: {calls}")
        print()

    print(f"Result: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    run_eval()
