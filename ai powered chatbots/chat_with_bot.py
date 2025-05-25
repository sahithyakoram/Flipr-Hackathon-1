from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import csv
import uuid
from datetime import datetime

# Support ticket creation
def create_support_ticket(user_query, sentiment='neutral', priority='Normal'):
    ticket_id = f"TCK{str(uuid.uuid4())[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ticket_data = {
        'ticket_id': ticket_id,
        'timestamp': timestamp,
        'user_query': user_query,
        'sentiment': sentiment,
        'priority': priority,
        'status': 'Open'
    }

    with open("support_tickets.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=ticket_data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(ticket_data)

    print(f"\n🚨 No confident answer found. Ticket {ticket_id} created and assigned to the support team.")

# Load embedding model and vectorstore
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

# Load QA chain with local model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token="YOUR API KEY",
    temperature=0.5,
    max_new_tokens=512
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Chat loop
print("🤖 Ask me anything about SwiftTech! Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    try:
        result = qa_chain.invoke({"query": query})
        answer = result.get('result', '').strip()

        if not answer or "I'm not sure" in answer or "I don't know" in answer:
            sentiment = "negative" if any(word in query.lower() for word in ["angry", "upset", "frustrated", "bad", "disappointed"]) else "neutral"
            priority = "High" if sentiment == "negative" else "Normal"
            create_support_ticket(user_query=query, sentiment=sentiment, priority=priority)
        else:
            print("\nBot:", answer)
    except Exception as e:
        print("⚠️ Error occurred while processing your query:", str(e))
