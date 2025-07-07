from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def store_chunks_in_chroma(chunks):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        collection_name="codeqa",
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vector_store


