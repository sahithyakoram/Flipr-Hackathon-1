# import os

# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma


# loader = TextLoader("swifttech_kb.txt", encoding="utf-8")
# documents = loader.load()


# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = splitter.split_documents(documents)

# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma.from_documents(
#     docs,
#     embedding,
#     persist_directory="chroma_db/"
# )

# print("✅ Vector database created and saved to 'chroma_db/'")

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Choose the appropriate file
loader = TextLoader("support_faq.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="chroma_db/"
)

vectorstore.persist()
print("✅ Vector database created and saved to 'chroma_db/'")
