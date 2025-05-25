# from langchain_community.llms import HuggingFaceHub
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import Chroma

# loader = TextLoader(r"/Users/manojsunuguri/Desktop/Flipr_Hackathon/Flipr-Hackathon/ai powered chatbots/support_faq.txt")
# documents = loader.load()


# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
# retriever = vectorstore.as_retriever()




# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)

# # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # vectorstore = FAISS.from_documents(docs, embeddings)
# # retriever = vectorstore.as_retriever()

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     huggingfacehub_api_token="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",
#     temperature=0.5,
#     max_new_tokens=512
# )

# #
# # llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 256})
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# tickets = [
#     {"id": "TCK001", "summary": "The LED on my SmartHome Hub is blinking orange."},
#     {"id": "TCK002", "summary": "I tried resetting my hub but it's still not connecting to Wi-Fi."},
#     {"id": "TCK003", "summary": "How can I change the Wi-Fi network on my SwiftTech device?"},
# ]

# for ticket in tickets:
#     print(f"\nTicket ID: {ticket['id']}")
#     print(f"Summary: Customer issue: {ticket['summary']}")
#     result = qa_chain.invoke({"query": ticket["summary"]})
#     print(f"Suggested Reply: {result['result']}")
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
import os


# Load and split docs
loader = TextLoader("ai powered chatbots/support_faq.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Embed and build vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Setup LLM
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
    huggingfacehub_api_token= "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",
    temperature=0.5,
    max_new_tokens=512
)

# Prompt template to guide answers
template = """
You are a helpful support agent. Use the following context to answer the question.
If the answer is not found in the context, say 'Sorry, I don't have that info.'

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# QA chain with sources
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"
    }
)

def get_support_reply(ticket_summary: str) -> str:
    try:
        # Use invoke() instead of deprecated run()
        result = qa_chain.invoke({"question": ticket_summary})
        return result['answer']
    except Exception as e:
        print(f"Error during QA chain invoke: {e}")
        return "Sorry, I'm unable to provide an answer right now."

tickets = [
    {"id": "TCK001", "summary": "The LED on my SmartHome Hub is blinking orange."},
    {"id": "TCK002", "summary": "I tried resetting my hub but it's still not connecting to Wi-Fi."},
    {"id": "TCK003", "summary": "How can I change the Wi-Fi network on my SwiftTech device?"},
]

for ticket in tickets:
    print(f"\nTicket ID: {ticket['id']}")
    print(f"Summary: {ticket['summary']}")
    reply = get_support_reply(ticket['summary'])
    print("Suggested Reply:", reply)