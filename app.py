from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory

from load_github_repo import load_all_github_files
from text_splitting import chunk_code_documents
from embed_and_store import store_chunks_in_chroma

import streamlit as st
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
                with st.spinner("📥 Cloning and indexing the repository... Please wait, this may take a while depending on the repository size."):
                    try:
                        docs = load_all_github_files(repo_url, branch)
                        chunks = chunk_code_documents(docs)
                        vector_store = store_chunks_in_chroma(chunks)
                        st.session_state.vector_store = vector_store
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

        # In-memory Chroma
        vector_store = st.session_state.vector_store
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        chat_model = ChatHuggingFace(llm=llm, memory=st.session_state.memory)

        docs = retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = PromptTemplate(
            template="""
                You are an expert assistant helping developers understand the following GitHub codebase.

                Code Context:
                {context}

                Question:
                {question}

                Answer:
            """,
            input_variables=["context", "question"]
        )

        final_prompt = prompt.format(context=context, question=user_input)

        with st.spinner("🤖 Thinking..."):
            response = chat_model.invoke([HumanMessage(content=final_prompt)])
            cleaned_response = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)
            match = re.search(r'Answer:\s*(.*)', cleaned_response, re.DOTALL | re.IGNORECASE)
            answer = match.group(1).strip() if match else cleaned_response.strip()

            st.session_state.chat_history.append(("ai", answer))

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    # Reset button
    if st.button("Reset & Load New Repo"):
        st.session_state.repo_loaded = False
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()
