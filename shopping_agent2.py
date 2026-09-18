import base64
import json
import os
import sqlite3
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
vision_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)


# ---------------------------------------------------------------------------
# Schema — add a preferences table if it doesn't exist yet (idempotent)
# ---------------------------------------------------------------------------

def ensure_schema() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS preferences (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    conn.close()


ensure_schema()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params: list = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id":          row[0],
            "name":        row[1],
            "category":    row[2],
            "price":       row[3],
            "description": row[4],
            "is_organic":  bool(row[5]),
        }
        for row in rows
    ]
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and returns
    a confirmation message with the order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


@tool
def view_orders() -> str:
    """
    Retrieve the full order history: every order placed so far, including order ID,
    product name, price, and the date/time it was ordered. Use this when the user
    wants the FULL itemized list (e.g. "list all my orders"). For a quick overview
    instead, use get_order_summary.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, product_name, price, ordered_at FROM orders ORDER BY ordered_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    orders = [
        {
            "order_id":     row[0],
            "product_name": row[1],
            "price":        row[2],
            "ordered_at":   row[3],
        }
        for row in rows
    ]
    return json.dumps(orders)


@tool
def get_order_summary() -> str:
    """
    Get a quick summary of the user's order history: total number of orders,
    total amount spent, and the most recently ordered product. Use this for
    general questions like "what have I ordered before?" or "what's my order
    history?". For the full itemized list, use view_orders instead.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), COALESCE(SUM(price), 0) FROM orders")
    count, total_spent = cursor.fetchone()
    cursor.execute(
        "SELECT product_name, price, ordered_at FROM orders ORDER BY ordered_at DESC LIMIT 1"
    )
    last = cursor.fetchone()
    conn.close()

    summary = {
        "total_orders": count,
        "total_spent": round(total_spent, 2),
        "most_recent_order": (
            {"product_name": last[0], "price": last[1], "ordered_at": last[2]}
            if last else None
        ),
    }
    return json.dumps(summary)


@tool
def get_preferences() -> str:
    """
    Retrieve all saved standing shopping preferences for this user (e.g.
    prefers_organic, max_price, favorite_category). Call this at the start
    of a browsing request so you can apply the user's standing preferences
    as default filters, unless the current message explicitly overrides them.
    Returns a JSON object of preference key/value pairs (may be empty).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM preferences")
    rows = cursor.fetchall()
    conn.close()
    return json.dumps({key: value for key, value in rows})


@tool
def save_preference(key: str, value: str) -> str:
    """
    Save or update a standing user preference so it persists across sessions.
    Use this ONLY when the user states a lasting preference (words like
    "always", "never", "from now on", "remember that..."), for example:
      - "I always want organic" -> save_preference("prefers_organic", "true")
      - "never show me anything over $20" -> save_preference("max_price", "20")
      - "I like snacks" -> save_preference("favorite_category", "snacks")
    Do NOT call this for one-off requests that only apply to the current message.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO preferences (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value),
    )
    conn.commit()
    conn.close()
    return f"Saved: {key} = {value}. I'll apply this automatically from now on."


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract its key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
                "- search_query: a short keyword to search for it (e.g. 'honey', 'olive oil')\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return response.content


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_agent(
    tools=[
        search_products,
        get_rating,
        checkout,
        view_orders,
        get_order_summary,
        get_preferences,
        save_preference,
        describe_product_image,
    ],
    model=llm,
    system_prompt=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "PREFERENCES — at the start of any BROWSING request:\n"
        "1. Call get_preferences first.\n"
        "2. Apply saved preferences (e.g. prefers_organic, max_price) as default filters "
        "   for search_products, UNLESS the user's current message explicitly overrides them "
        "   for this one request.\n"
        "3. If the user states a standing preference (using words like 'always', 'never', "
        "   'from now on', 'remember that...'), call save_preference to store it, then continue "
        "   with their request using the updated preference. Do not save one-off requests.\n\n"
        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products (after applying "
        "   preferences per the PREFERENCES rule above).\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        "BROWSING — when the user describes what they want to buy:\n"
        "1. Call search_products to find matching items (apply any price/organic filters given, "
        "   falling back to saved preferences where the user didn't specify).\n"
        "2. For each candidate, call get_rating to retrieve its average rating.\n"
        "3. Filter by the user's minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use this exact format "
        "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability. "
        "   Always include (ID:X) so you can reference it later.\n"
        "5. If only one product qualifies, still show it in the list and ask: "
        "   'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
        "'order number 2', 'the first one', 'get me #3', or just a bare number like '2'):\n"
        "1. Look back at your own previous message in this conversation to find the (ID:X) "
        "   for the chosen product (if only one was listed and the user says 'yes', use that "
        "   product's ID; if the user gives a list position like 'number 2' or '#2', map it to "
        "   the ID of the 2nd item you listed — the number and the ID are NOT the same thing).\n"
        "2. Call checkout with that product_id (the number from (ID:X)).\n"
        "3. Confirm the order to the user in plain text, and mention they can ask 'what have I "
        "   ordered before?' at any time to see their order history.\n\n"
        "ORDER HISTORY — when the user asks about past orders:\n"
        "1. For a general question ('what have I ordered before?', 'what's my order history?'), "
        "   call get_order_summary.\n"
        "2. For a request for the full list ('list all my orders', 'show every order'), call "
        "   view_orders instead and present it as a numbered plain-text list.\n"
        "3. If there are no orders yet, say so plainly.\n\n"
        "Never place an order unless the user explicitly confirms. "
        "Never guess a product_id — always take it from the (ID:X) in your own previous message, "
        "not from the list position the user mentions."
        "And importantly, do not handle out of scope questions"
    ),
)


# ---------------------------------------------------------------------------
# Input guardrail — keep the agent on-topic
# ---------------------------------------------------------------------------

GUARDRAIL_SYSTEM_PROMPT = """
You are a strict topic classifier for a shopping assistant. Decide whether the
LATEST user message is something the shopping assistant should handle.

Answer YES if the message is:
- About browsing, searching, comparing, or asking about products, prices, ratings, or categories
- About placing, confirming, or canceling an order
- About the user's order history, cart, or standing shopping preferences
- A short reply that continues an ongoing shopping conversation (e.g. "yes",
  "no", a number, "the first one", a product ID) — especially if the previous
  assistant message below asked a question or listed products

Answer NO if the message is unrelated to shopping — e.g. general knowledge
questions, creative writing requests, coding help, weather, small talk with
no shopping context, etc.

Respond with ONLY one word: YES or NO.
"""

OFF_TOPIC_REPLY = (
    "I'm your shopping assistant, so I can only help with browsing products, "
    "checking prices and ratings, placing orders, or reviewing your order "
    "history and preferences. Is there something you'd like to shop for?"
)


def is_shopping_related(user_message: str, last_assistant_message: str = "") -> bool:
    """Lightweight LLM-based input guardrail, run BEFORE the agent sees the
    message. Returns True if the agent should handle it, False if it should
    be redirected with OFF_TOPIC_REPLY instead."""
    context = (
        f"Previous assistant message (may be empty): {last_assistant_message}\n\n"
        f"Latest user message: {user_message}"
    )
    response = llm.invoke(
        [
            {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
    )
    verdict = response.content.strip().upper()
    return verdict.startswith("YES")


def get_tool_calls(result_messages: list) -> list:
    """Utility for evals: extract every (tool_name, args) pair the agent
    issued during a run, in order."""
    calls = []
    for msg in result_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                calls.append((call["name"], call["args"]))
    return calls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_chat() -> None:
    """Interactive command-line chat loop that preserves conversation history
    (so 'yes' / '#2' resolve against the product list just shown) and applies
    the input guardrail before every agent call."""
    print("🛒 Shopping Assistant — type 'quit' or 'exit' to leave.\n")
    messages: list = []
    last_assistant_text = ""

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not is_shopping_related(user_input, last_assistant_text):
            print(f"\nAssistant: {OFF_TOPIC_REPLY}\n")
            continue

        messages.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke({"messages": messages})
        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}\n")
            continue

        messages = result["messages"]
        last_assistant_text = messages[-1].content
        print(f"\nAssistant: {last_assistant_text}\n")


if __name__ == "__main__":
    run_chat()
