from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import csv
import uuid
from datetime import datetime
from utils.classify import classify_query
from feedback_logger import collect_feedback

# ... inside the chat loop after `print("\nBot:", answer)`

def create_support_ticket(user_query, sentiment='neutral', priority='Normal'):
    import re

    # Simple keyword-based categorization
    category_keywords = {
        'Account': ["login", "password", "reset", "username", "account"],
        'Billing': ["payment", "invoice", "billing", "charged", "refund"],
        'Technical': ["error", "issue", "bug", "not working", "crash", "fail"],
        'General': ["info", "information", "support", "help", "question"]
    }

    category = "Uncategorized"
    for cat, keywords in category_keywords.items():
        if any(re.search(rf"\b{kw}\b", user_query.lower()) for kw in keywords):
            category = cat
            break

    ticket_id = f"TCK{str(uuid.uuid4())[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ticket_data = {
        'ticket_id': ticket_id,
        'timestamp': timestamp,
        'user_query': user_query,
        'sentiment': sentiment,
        'priority': priority,
        'category': category,
        'status': 'Open'
    }

    with open("support_tickets.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=ticket_data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(ticket_data)

    print(f"\n🚨 No confident answer found. Ticket {ticket_id} created and assigned to the support team. (Category: {category}, Priority: {priority})")

# # Support ticket creation
# def create_support_ticket(user_query, sentiment='neutral', priority='Normal'):
#     ticket_id = f"TCK{str(uuid.uuid4())[:8].upper()}"
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     ticket_data = {
#         'ticket_id': ticket_id,
#         'timestamp': timestamp,
#         'user_query': user_query,
#         'sentiment': sentiment,
#         'priority': priority,
#         'status': 'Open'
#     }

#     with open("support_tickets.csv", mode='a', newline='', encoding='utf-8') as file:
#         writer = csv.DictWriter(file, fieldnames=ticket_data.keys())
#         if file.tell() == 0:
#             writer.writeheader()
#         writer.writerow(ticket_data)

#     print(f"\n🚨 No confident answer found. Ticket {ticket_id} created and assigned to the support team.")

# Load embedding model and vectorstore
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

# Load QA chain with local model
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

# # Chat loop
# print("🤖 Ask me anything about SwiftTech! Type 'exit' to quit.")
# while True:
#     query = input("You: ")
#     if query.lower() in ["exit", "quit"]:
#         break

#     try:
#         result = qa_chain.invoke({"query": query})
#         answer = result.get('result', '').strip()

#         if not answer or "I'm not sure" in answer or "I don't know" in answer:
#             sentiment = "negative" if any(word in query.lower() for word in ["angry", "upset", "frustrated", "bad", "disappointed"]) else "neutral"
#             priority = "High" if sentiment == "negative" else "Normal"
#             create_support_ticket(user_query=query, sentiment=sentiment, priority=priority)
#         else:
#             print("\nBot:", answer)
#     except Exception as e:
#         print("⚠️ Error occurred while processing your query:", str(e))

# Chat loop
print("🤖 Ask me anything about SwiftTech! Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    try:
        result = qa_chain.invoke({"query": query})
        answer = result.get('result', '').strip()

        if not answer or "I'm not sure" in answer.lower() or "i don't know" in answer.lower():
            # ✨ Use classifier instead of keyword-based sentiment
            intent, sentiment = classify_query(query)
            priority = "High" if sentiment == "negative" else "Normal"

            create_support_ticket(user_query=query, sentiment=sentiment, priority=priority)
        else:
            print("\nBot:", answer)
            collect_feedback(query)

            # Ask for feedback
            feedback = input("\n🤔 Was this helpful? (yes/no): ").strip().lower()

            if feedback == "no":
                sentiment = "negative"
                priority = "High"
                create_support_ticket(user_query=query, sentiment=sentiment, priority=priority)
            elif feedback == "yes":
                print("✅ Great! Let me know if you have more questions.")
            else:
                print("ℹ️ Skipping feedback. Let's move on.")


    except Exception as e:
        print("⚠️ Error occurred while processing your query:", str(e))

