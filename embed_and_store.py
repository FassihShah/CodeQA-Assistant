from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def store_chunks_in_chroma(chunks, persist_directory="./chroma_code_store"):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=persist_directory,
    )

    vector_store.persist()
    print(f"Stored {len(chunks)} chunks in ChromaDB at {persist_directory}")
    return vector_store


