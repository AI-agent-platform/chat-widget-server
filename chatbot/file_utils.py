import os
from langchain_community.document_loaders import (
    CSVLoader, UnstructuredExcelLoader, PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_chunks_from_file(file_path, filename, chunk_size=1000, chunk_overlap=200):
    """
    Extracts and splits file contents into small RAG chunks for all file types (CSV, PDF, DOCX, TXT, etc).
    Each chunk is a dict:
      - chunk_type: "file_chunk"
      - file_name: original filename
      - file_type: file extension (without '.')
      - chunk_index: index of the chunk
      - content: the text content of the chunk (string)
    """
    ext = os.path.splitext(filename)[-1].lower()
    try:
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
            loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
    except Exception as e:
        raise ValueError(f"Unsupported or unreadable file extension/type: {ext}. Details: {str(e)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    chunk_dicts = []
    for idx, chunk in enumerate(chunks):
        text = str(chunk.page_content).strip() if hasattr(chunk, "page_content") else str(chunk).strip()
        chunk_dicts.append({
            "chunk_type": "file_chunk",
            "file_name": filename,
            "file_type": ext.replace('.', ''),
            "chunk_index": idx,
            "content": text
        })
    return chunk_dicts