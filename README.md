# 🛒 Shopping Assistant

An AI shopping agent built with LangChain and Groq. It can search a product catalog, check ratings, place orders, remember your standing preferences, summarize your order history, and even look up products from a photo — available both as a command-line chatbot and a Streamlit web app.

---

## Features

- **Conversational shopping** — describe what you want ("organic honey under $20, 4.5+ rating") and get a numbered list of matching products.
- **Follow-up ordering** — reply "yes", a number, or "order #2" and the agent places the order against the product it just showed you, no need to repeat details.
- **Image search** — upload a product photo and the agent identifies it and searches the catalog for matches.
- **Order history & summary** — ask "what have I ordered before?" for a quick summary (total orders, total spent, most recent order), or "list all my orders" for the full itemized history.
- **Standing preferences** — tell it once ("I always want organic", "never show me anything over $20") and it's applied automatically to future searches, across sessions.
- **Input guardrail** — off-topic requests (general trivia, coding help, poems, etc.) are politely redirected before they ever reach the agent.
- **Evals included** — a tool-call accuracy test and an LLM-as-judge response quality test, so you can check the agent still behaves correctly after changes.

> ⚠️ This is a demo/learning project. Orders aren't fulfilled by a real store — they're recorded locally in `store.db`.

---

## Project structure

```
.
├── shopping_agent.py               # Core agent: tools, system prompt, guardrail, CLI chat loop
├── streamlit_app.py                # Streamlit web UI for the same agent
├── reviews_api.py                  # Your existing ratings/reviews lookup module
├── store.db                        # SQLite database: products, reviews, orders, preferences
├── requirements_shopping_app.txt   # Python dependencies
├── eval/
│   ├── tool_call_accuracy_eval.py  # Checks the agent calls the right tool with the right args
│   └── response_quality_eval.py    # LLM-as-judge scoring of response quality
└── README.md                       # This file
```

---

## Requirements

- Python 3.9+
- A [Groq API key](https://console.groq.com/keys)
- `store.db` with `products`, `reviews`, and `orders` tables already populated (the `preferences` table is created automatically on first run)

---

## Setup

1. **Clone the repo and enter the project folder**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_shopping_app.txt
   ```

4. **Add your API key**

   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```
   (For the Streamlit app, you can alternatively paste the key into the sidebar at runtime instead of using `.env`.)

---

## Running the app

### Command line
```bash
python shopping_agent.py
```
Chat directly in the terminal. Type `quit` or `exit` to leave.

### Streamlit web app
```bash
streamlit run streamlit_app.py
```
Opens in your browser, usually at `http://localhost:8501`. Includes:
- A chat window with full conversation memory
- A sidebar image uploader for product-photo search
- **Orders** and **Prefs** buttons to inspect order history and saved preferences at any time
- A **New conversation** button to reset the chat (preferences and order history persist regardless — only the chat memory resets)

---

## Example interactions

```
You: organic honey under $20
Assistant:
#1. Organic Raw Honey (ID:1) — $14.99 ★4.62 — organic
#2. Organic Buckwheat Honey (ID:5) — $18.99 ★4.62 — organic

Would you like to order any of these? Just say yes or give me the number.

You: 2
Assistant: Order #7 confirmed! 'Organic Buckwheat Honey' has been successfully
ordered for $18.99. Your order will arrive in 3-5 business days...

You: I always want organic from now on
Assistant: Saved: prefers_organic = true. I'll apply this automatically from now on.

You: what have I ordered before?
Assistant: You've placed 1 order totaling $18.99. Your most recent order was
'Organic Buckwheat Honey' for $18.99.

You: write me a poem about the sea
Assistant: I'm your shopping assistant, so I can only help with browsing
products, checking prices and ratings, placing orders, or reviewing your
order history and preferences. Is there something you'd like to shop for?
```

---

## Running the evals

```bash
python eval/tool_call_accuracy_eval.py
python eval/response_quality_eval.py
```

- **Tool call accuracy** — runs a fixed set of queries and checks the agent called the expected tool with the expected arguments (e.g. `"organic honey under $20"` → `search_products(is_organic=True, max_price=20)`).
- **Response quality** — uses an LLM judge to score each response 1–5 on relevance, correctness, and format compliance, then prints per-test and average scores.

Re-run these after changing the system prompt, tools, or model to catch regressions.

---

## How memory & preferences work

- **Order history** lives in the `orders` table and is queried directly — no separate memory store needed.
- **Preferences** live in a `preferences` key/value table, created automatically the first time you run the app. The agent checks it before every search and only writes to it when you state a *standing* preference (words like "always", "never", "from now on") — one-off requests aren't saved.

---

## Security notes

- Your API key is **never hardcoded** in the source code — it's read from `.env` or entered at runtime.
- Add a `.gitignore` entry before pushing to GitHub so your key is never committed:
  ```
  .env
  ```
- If sharing this repo publicly, include a `.env.example` with a placeholder value instead of a real key.

---

## License

Add your license of choice here (e.g. MIT).
