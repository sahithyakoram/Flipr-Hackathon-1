# app.py
import streamlit as st
import csv
import os
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from transformers import pipeline
import pandas as pd

# Load vectorstore & model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token="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",
    temperature=0.5,
    max_new_tokens=512
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Load Sentiment Analyzer
sentiment_analyzer = pipeline("sentiment-analysis")

# Helper: Detect sentiment
def detect_sentiment(text):
    try:
        res = sentiment_analyzer(text)[0]
        return res["label"].lower()
    except Exception:
        return "neutral"

# Helper: Create ticket
def create_ticket(query, sentiment, priority="Normal"):
    ticket_id = f"TCK{datetime.now().strftime('%Y%m%d%H%M%S')}"
    with open("support_tickets.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "timestamp", "user_query", "sentiment", "priority", "status"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "ticket_id": ticket_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": query,
            "sentiment": sentiment,
            "priority": priority,
            "status": "Open"
        })

# Load tickets into DataFrame
def load_tickets():
    if not os.path.exists("support_tickets.csv"):
        return pd.DataFrame(columns=["ticket_id", "timestamp", "user_query", "sentiment", "priority", "status"])
    return pd.read_csv("support_tickets.csv")

# Streamlit UI
st.set_page_config(page_title="SwiftTech Support", layout="wide")
st.title("💬 SwiftTech AI Support")

tab1, tab2, tab3 = st.tabs(["💁 Customer Chat", "🧑‍💻 Agent Tickets", "📊 Admin Summary"])

# 💁 Customer
with tab1:
    st.subheader("Ask SwiftTech a question")
    user_query = st.text_input("Your Question:")
    if st.button("Get Answer"):
        if user_query:
            result = qa_chain.invoke({"query": user_query})
            answer = result.get('result', '').strip()
            
            if not answer or "I'm not sure" in answer.lower() or "I don't know" in answer.lower():
                sentiment = detect_sentiment(user_query)
                priority = "High" if sentiment == "negative" else "Normal"
                create_ticket(user_query, sentiment, priority)
                st.error("🚨 We couldn't confidently answer. A support ticket has been created.")
            else:
                st.success(answer)

# 🧑‍💻 Agent View
with tab2:
    st.subheader("View Open Support Tickets")
    df = load_tickets()
    if not df.empty:
        open_tickets = df[df["status"] == "Open"]
        for idx, row in open_tickets.iterrows():
            with st.expander(f"{row['ticket_id']} | {row['priority']} | {row['sentiment']}"):
                st.write(f"📅 {row['timestamp']}")
                st.write(f"📝 {row['user_query']}")
                if st.button(f"Generate AI Reply for {row['ticket_id']}"):
                    reply = qa_chain.invoke({"query": row['user_query']}).get('result', '').strip()
                    st.success(reply)
    else:
        st.info("✅ No open tickets.")

# 📊 Admin Summary
with tab3:
    st.subheader("Summary Overview")
    df = load_tickets()
    if df.empty:
        st.write("No tickets yet.")
    else:
        st.metric("Total Tickets", len(df))
        st.metric("Open Tickets", len(df[df["status"] == "Open"]))
        st.metric("High Priority", len(df[df["priority"] == "High"]))
        st.bar_chart(df["sentiment"].value_counts())
