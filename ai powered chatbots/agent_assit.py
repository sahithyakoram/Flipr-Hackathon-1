from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

loader = TextLoader("support_faq.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 256})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

tickets = [
    {"id": "TCK001", "summary": "The LED on my SmartHome Hub is blinking orange."},
    {"id": "TCK002", "summary": "I tried resetting my hub but it's still not connecting to Wi-Fi."},
    {"id": "TCK003", "summary": "How can I change the Wi-Fi network on my SwiftTech device?"},
]

for ticket in tickets:
    print(f"\nTicket ID: {ticket['id']}")
    print(f"Summary: Customer issue: {ticket['summary']}")
    result = qa_chain.invoke({"query": ticket["summary"]})
    print(f"Suggested Reply: {result['result']}")
