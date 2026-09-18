"""
Shopping Assistant - Streamlit App
-----------------------------------
A chat interface for the LangChain shopping agent defined in shopping_agent.py.

Features:
- Full conversation memory across turns (so "yes" / "#2" replies work)
- Product photo upload for image-based search
- Order history + order summary, read straight from store.db
- Standing preferences (e.g. "always organic", "never over $20") shown and
  persisted across sessions
- An input guardrail that politely redirects off-topic messages before they
  ever reach the agent

Run with:
    streamlit run streamlit_app.py

Requires shopping_agent.py, reviews_api.py, and store.db in the same folder,
plus a GROQ_API_KEY set as an environment variable (a local .env file, or a
Railway/host environment variable in production). The key is never shown or
editable in the UI.
"""

import os
import sqlite3
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Shopping Assistant", page_icon="🛒", layout="wide")

# ----------------------------------------------------------------------
# API key check — read purely from the environment, never shown or
# editable in the UI. Checked BEFORE the agent module is imported, since
# ChatGroq reads the key at construction time.
# ----------------------------------------------------------------------
if not os.getenv("GROQ_API_KEY"):
    st.title("🛒 Shopping Assistant")
    st.error(
        "GROQ_API_KEY is not set. Add it as an environment variable "
        "(a .env file locally, or a Railway/host variable in production)."
    )
    st.stop()

# Import only after the key is confirmed present.
from shopping_agent2 import agent, DB_PATH, is_shopping_related, OFF_TOPIC_REPLY  # noqa: E402

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []  # full LangChain message history sent to the agent
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # [{"role": "user"/"assistant", "content": str}]

# ----------------------------------------------------------------------
# Sidebar: orders, preferences, reset
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Shopping Assistant")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧾 Orders", use_container_width=True):
            st.session_state.show_orders = True
    with col2:
        if st.button("⭐ Prefs", use_container_width=True):
            st.session_state.show_preferences = True

    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.agent_messages = []
        st.session_state.display_messages = []
        st.rerun()

    st.divider()
    uploaded_image = st.file_uploader(
        "Have a photo of a product?",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload it, then just send your message — the agent will look at it.",
    )

if st.session_state.get("show_orders"):
    st.subheader("🧾 Your order history")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), COALESCE(SUM(price), 0) FROM orders")
    count, total_spent = cursor.fetchone()
    cursor.execute(
        "SELECT id, product_name, price, ordered_at FROM orders ORDER BY ordered_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    st.caption(f"{count} order(s) · ${total_spent:.2f} spent total")
    if rows:
        st.table(
            [
                {"Order ID": r[0], "Product": r[1], "Price": f"${r[2]:.2f}", "Ordered At": r[3]}
                for r in rows
            ]
        )
    else:
        st.info("No orders yet.")

    if st.button("Close", key="close_orders"):
        st.session_state.show_orders = False
        st.rerun()
    st.divider()

if st.session_state.get("show_preferences"):
    st.subheader("⭐ Your standing preferences")
    st.caption(
        "These are applied automatically as default filters, e.g. tell the agent "
        "\"I always want organic\" or \"never show me anything over $20\"."
    )
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value, updated_at FROM preferences ORDER BY updated_at DESC")
    prefs = cursor.fetchall()
    conn.close()

    if prefs:
        st.table([{"Preference": p[0], "Value": p[1], "Last updated": p[2]} for p in prefs])
    else:
        st.info("No standing preferences saved yet.")

    if st.button("Close", key="close_prefs"):
        st.session_state.show_preferences = False
        st.rerun()
    st.divider()

# ----------------------------------------------------------------------
# Main chat UI
# ----------------------------------------------------------------------
st.title("🛒 Shopping Assistant")
st.caption("Ask for products, filter by price/organic/rating, then say 'yes' or a number to order.")

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("\n", "  \n"))

user_text = st.chat_input("What are you looking for?")

if user_text:
    from langchain_core.messages import HumanMessage

    # If an image was uploaded, save it locally and point the agent at the path.
    full_input = user_text
    if uploaded_image is not None:
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploaded_images")
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join(uploads_dir, f"{uuid.uuid4().hex}_{uploaded_image.name}")
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        full_input = f"{user_text}\n\n[Uploaded product image at: {image_path}]"

    st.session_state.display_messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
        if uploaded_image is not None:
            st.image(uploaded_image, width=200)

    # Last assistant text, for guardrail context (so short replies like "yes"
    # or "2" are recognized as continuations of an ongoing shopping chat).
    last_assistant_text = next(
        (m["content"] for m in reversed(st.session_state.display_messages[:-1]) if m["role"] == "assistant"),
        "",
    )

    with st.chat_message("assistant"):
        if not is_shopping_related(user_text, last_assistant_text):
            reply = OFF_TOPIC_REPLY
            st.markdown(reply)
        else:
            st.session_state.agent_messages.append(HumanMessage(content=full_input))
            with st.spinner("Thinking..."):
                try:
                    result = agent.invoke({"messages": st.session_state.agent_messages})
                    st.session_state.agent_messages = result["messages"]
                    reply = st.session_state.agent_messages[-1].content
                except Exception as e:
                    reply = f"Sorry, something went wrong: {e}"
            st.markdown(reply.replace("\n", "  \n"))

    st.session_state.display_messages.append({"role": "assistant", "content": reply})
