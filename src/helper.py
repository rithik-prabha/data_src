import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_text_documents(folder_path: str) -> List[Document]:
    """
    Loads all .txt files from the given folder using UTF-8 encoding with fallback.
    """
    def custom_loader(path: str):
        try:
            return TextLoader(path, encoding="utf-8").load()
        except UnicodeDecodeError:
            return TextLoader(path, encoding="ISO-8859-1").load()

    loader = DirectoryLoader(
        path=folder_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True
    )

    try:
        return loader.load()
    except Exception as e:
        print(f"[ERROR] Failed to load documents: {e}")
        return []


def clean_whitespace(docs: List[Document]) -> List[Document]:
    for doc in docs:
        doc.page_content = ' '.join(doc.page_content.split())
    return docs


def remove_noise(docs: List[Document]) -> List[Document]:
    cleaned_docs = []
    for doc in docs:
        text = re.sub(r"Page \d+ of \d+", "", doc.page_content)  # Example: remove footer page numbers
        cleaned_docs.append(Document(page_content=text.strip(), metadata=doc.metadata))
    return cleaned_docs


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)
    return unique_docs


def text_split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_documents(docs)


def download_hugging_face_embeddings():
    """
    Loads the BAAI BGE embedding model from HuggingFace.
    """
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}  # Change to "cuda" for GPU usage
    )
    return embedding


