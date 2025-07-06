from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory

from load_github_repo import load_all_github_files
from text_splitting import chunk_code_documents
from embed_and_store import store_chunks_in_chroma

import streamlit as st
import atexit
import shutil
import os
import re
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
            repo_id='deepseek-ai/DeepSeek-R1-0528',
            task='text-generation',
            max_new_tokens=512,
            temperature=0.3,
        )


# Streamlit Config
st.set_page_config(page_title="CodeQA Chat", layout="wide", page_icon="🤖")
st.title("🤖 CodeQA — Chat with Your GitHub Codebase")

# Session state
if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# GitHub Repo Loading Section
if not st.session_state.repo_loaded:
    with st.expander("📂 Load GitHub Repository", expanded=True):
        st.subheader("Load a Public GitHub Repository")
        st.write("Enter the repository URL and branch (optional) to begin indexing.")

        with st.form("repo_form"):
            repo_url = st.text_input("🔗 Repository URL", placeholder="e.g. https://github.com/openai/whisper")
            branch = st.text_input("🌿 Branch (optional)", value="main")
            submitted = st.form_submit_button("Load Repository")

            if submitted and repo_url:
                with st.spinner("📥 Cloning and indexing the repository..."):
                    try:
                        docs = load_all_github_files(repo_url, branch)
                        chunks = chunk_code_documents(docs)
                        store_chunks_in_chroma(chunks)
                        st.session_state.repo_loaded = True
                        st.success("✅ Repository indexed successfully! You can now start chatting.")
                    except Exception as e:
                        st.error(f"❌ Failed to load repository: {e}")

# Chat Interface
if st.session_state.repo_loaded:
    st.subheader("💬 Ask Anything About the Codebase")

    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        vector_store = Chroma(
            persist_directory="./chroma_code_store",
            embedding_function=embedding,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        chat_model = ChatHuggingFace(llm=llm, memory=st.session_state.memory)

        docs = retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = PromptTemplate(
            template = """
        You are an expert assistant helping developers understand the following GitHub codebase.

        Code Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables = ['context', 'question']
        )

        final_prompt = prompt.format(context=context,question=user_input)

        with st.spinner("🤖 Thinking..."):

            # Remove Extra thinking when using DeepSeek
            response = chat_model.invoke([HumanMessage(content=final_prompt)])
            match = re.search(r'Answer:\s*(.*)', response.content, re.DOTALL | re.IGNORECASE)
            answer = match.group(1).strip() if match else response.content.strip()

            st.session_state.chat_history.append(("ai", answer))

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    # Reset button
    if st.button("Reset & Load New Repo"):
        try:
            # First: initialize the vector store to access collection name
            vector_store = Chroma(
                persist_directory="./chroma_code_store",
                embedding_function=embedding,
            )

            # Step 1: delete the entire collection explicitly
            vector_store._client.delete_collection(name=vector_store._collection.name)

            # Step 2: Close/reset the Chroma client
            vector_store._client.reset()

            # Step 3: Delete the folder from disk
            del vector_store  # Ensure it's closed
            shutil.rmtree("./chroma_code_store", ignore_errors=True)

        except Exception as e:
            st.warning(f"⚠️ Cleanup failed: {e}")

        # Clear session state
        st.session_state.repo_loaded = False
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()


def clean_chroma_store():
    try:
        if os.path.exists("./chroma_code_store"):
            vector_store = Chroma(
                persist_directory="./chroma_code_store",
                embedding_function=embedding
            )
            vector_store._client.delete_collection(name=vector_store._collection.name)
            vector_store._client.reset()
            del vector_store
            shutil.rmtree("./chroma_code_store", ignore_errors=True)
    except Exception as e:
        print(f"[Exit Cleanup Warning] {e}")


atexit.register(clean_chroma_store)
