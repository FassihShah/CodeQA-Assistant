import json
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language
from langchain_core.documents import Document


EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".ipynb": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".java": Language.JAVA,
    ".html": Language.HTML,
    ".cshtml": Language.HTML,
    ".kt": Language.CPP,
}


# Extract markdown/code cells from notebook
def extract_ipynb_content(doc):
    try:
        notebook = json.loads(doc.page_content)
        cells = notebook.get("cells", [])
        content = ""

        for cell in cells:
            cell_type = cell.get("cell_type")
            source = cell.get("source", [])

            if cell_type == "markdown":
                for line in source:
                    content += f"# {line}" if not line.startswith("#") else line
                content += "\n\n"
            elif cell_type == "code":
                content += "\n" + "".join(source) + "\n\n"

        return Document(page_content=content.strip(), metadata=doc.metadata)

    except Exception as e:
        print(f"Failed to parse .ipynb: {doc.metadata.get('source')} | {e}")
        return doc



# Get code splitter
def get_code_splitter(file_path, chunk_size=300, chunk_overlap=50):
    for ext, lang in EXTENSION_TO_LANGUAGE.items():
        if file_path.endswith(ext):
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)



def chunk_code_documents(documents: list[Document], chunk_size=800, chunk_overlap=50):
    all_chunks = []

    for doc in documents:
        file_path = doc.metadata.get("source", "").lower()

        # Preprocess notebooks
        if file_path.endswith(".ipynb"):
            doc = extract_ipynb_content(doc)

        # Get splitter for code type
        splitter = get_code_splitter(file_path, chunk_size, chunk_overlap)
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    return all_chunks

