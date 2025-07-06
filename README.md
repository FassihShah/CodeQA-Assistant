# 🤖 CodeQA — Chat with Your GitHub Codebase

**CodeQA** is an intelligent Streamlit-based assistant that lets you ask natural language questions about any public GitHub repository and get AI-generated answers with context from the codebase. It uses RAG (Retrieval-Augmented Generation) powered by LangChain, HuggingFace models, and Chroma for vector storage.

---

## 🚀 Features

- 🔗 **Load any public GitHub repository** by providing its URL and branch
- 🧠 **RAG-powered question answering** over the codebase
- 🧾 **Semantic chunking** of code using language-aware text splitters
- 💬 **Conversational memory** for multi-turn questions
- 💾 **Persistent vector store** using ChromaDB
- 🧠 **HuggingFace-based LLM** for accurate answers from code
- 📘 Supports code files like `.py`, `.cpp`, `.js`, `.ipynb`, `.md`, and more

---

## 🛠️ Tech Stack

| Layer            | Library/Tool                      |
|------------------|-----------------------------------|
| Frontend         | [Streamlit](https://streamlit.io) |
| RAG Framework    | [LangChain](https://www.langchain.com) |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace |
| LLM              | `deepseek-ai/DeepSeek-R1-0528` via HuggingFace |
| Vector Database  | [ChromaDB](https://www.trychroma.com) |
| GitHub Access    | LangChain's `GithubFileLoader` |

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```
3. **Install dependencies**

```bash
pip install -r requirements.txt
```
4. **Set up environment variables**

Create a .env file and add your GitHub token:

```bash
GITHUB_PERSONAL_ACCESS_TOKEN = your_github_token
HUGGINGFACEHUB_API_TOKEN = your_huggingface_token
```
---

## ▶️ Running the App

```bash
streamlit run app.py
```
---

## 📂 Supported File Types

CodeQA currently loads the following file types from the repo:
-  .py, .ipynb, .js, .ts, .cpp, .c, .cs, .html, .md, .java, .h, .cshtml, .kt

Notebooks (.ipynb) are processed using nbformat, extracting markdown and code cells into plain text.

---

## 🧠 How It Works

The application uses a Retrieval-Augmented Generation (RAG) pipeline to allow natural language interaction with GitHub codebases:

---

### 1. 🔗 Load GitHub Repo

- The GitHub API is used to fetch the full repository tree.
- Only relevant files (e.g., `.py`, `.js`, `.ipynb`, `.cpp`, `.md`, etc.) are selected based on their extensions.
- Files are loaded using LangChain's `GithubFileLoader`, which supports batching and filtering.

---

### 2. ✂️ Chunking and Embedding

- Loaded files are split into smaller text chunks using `RecursiveCharacterTextSplitter` or language-specific splitters.
- Each chunk is embedded into a high-dimensional vector using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`.
- All embeddings are stored persistently in a Chroma vector database (`chroma_code_store`).

---

### 3. 💬 Conversational Retrieval

- When a user asks a question, the top-k most relevant chunks are retrieved from ChromaDB using semantic similarity search.
- These chunks, along with the user's query, are formatted into a natural language prompt.
- The prompt is passed to a HuggingFace-hosted LLM (e.g., DeepSeek) using `ChatHuggingFace` to generate a contextual answer.
- Conversation memory is maintained using `ConversationBufferMemory` to support follow-up questions.

---



