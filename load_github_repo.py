from langchain_community.document_loaders.github import GithubFileLoader
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
import os

load_dotenv()


token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


def get_repo_file_paths(repo_url, branch="main", extensions=None):

    if extensions is None:
        extensions = [".py",".ipynb", ".cpp", ".js", ".ts", ".cs", ".java", ".html", ".md", ".h", ".c", ".cs", "cshtml", ".kt"]

    # Parse owner and repo name from URL
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repo URL")

    owner, repo = parts[0], parts[1]

    # GitHub API URL to get repo tree recursively
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    headers = {"Authorization": f"token {token}"} if token else {}

    response = requests.get(api_url, headers=headers)

    
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code} - {response.text}")

    tree = response.json().get("tree", [])
    file_paths = [
        item["path"]
        for item in tree
        if item["type"] == "blob" and any(item["path"].endswith(ext) for ext in extensions)
    ]
    return file_paths, owner, repo


def load_all_github_files(repo_url, branch="main", batch_size=15):

    file_paths, owner, repo = get_repo_file_paths(repo_url, branch)

    if not file_paths:
        print("No matching files found.")
        return []

    print(f"Found {len(file_paths)} code files to load in batches of {batch_size}")

    all_documents = []

    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        print(f"Loading batch {i // batch_size + 1} with {len(batch_paths)} files")

        loader = GithubFileLoader(
            repo=f"{owner}/{repo}",
            branch=branch,
            access_token=token,
            file_filter=lambda file_path, batch=batch_paths: file_path in batch
        )

        try:
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading batch {i // batch_size + 1}: {e}")
            continue

    print(f"Finished loading. Total documents loaded: {len(all_documents)}")
    return all_documents



