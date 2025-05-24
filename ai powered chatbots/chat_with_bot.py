from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

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

# Console chatbot loop
print("🤖 Ask me anything about SwiftTech! Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.invoke({"query": query})
    print("\nBot:", result['result'])
    print("-" * 50)
