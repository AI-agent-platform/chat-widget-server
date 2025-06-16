from langchain_community.document_loaders import (
    CSVLoader, UnstructuredExcelLoader, PyPDFLoader, Docx2txtLoader, TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def extract_text_chunks_from_file(file_path, filename, chunk_size=1000, chunk_overlap=200):
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path, autodetect_encoding=True)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return [chunk.page_content for chunk in chunks]