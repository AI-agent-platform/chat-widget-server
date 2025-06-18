import os
import json
import re
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ FIXED

BASE_FAISS_DIR = r"E:\RAGDB"
os.makedirs(BASE_FAISS_DIR, exist_ok=True)
EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # or "all-MiniLM-L6-v2" for speed

def sanitize_folder_name(name):
    if not name:
        return "unknown"
    return re.sub(r'[^A-Za-z0-9_\-]', '', name.replace(" ", "_"))

class UserRAGDBManager:
    def __init__(self, base_dir=BASE_FAISS_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.cache = {}

    def get_user_folder(self, company_name, uid, field=None):
        field_folder = sanitize_folder_name(field) if field else "general"
        user_folder = f"{sanitize_folder_name(company_name)}_{uid}"
        folder_path = os.path.join(self.base_dir, field_folder, user_folder)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def get_user_db(self, company_name, uid, field=None):
        key = (company_name, uid, field)
        if key not in self.cache:
            folder = self.get_user_folder(company_name, uid, field)
            self.cache[key] = UserRAGVectorDB(folder)
        return self.cache[key]

class UserRAGVectorDB:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.db_path = os.path.join(folder_path, "faiss_index")
        self.meta_path = os.path.join(folder_path, "faiss_meta.json")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.index = None
        self.meta = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            self.index = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
        else:
            self.index = None
            self.meta = []

    def save(self):
        if self.index:
            self.index.save_local(self.db_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

    def add_records(self, records):
        """
        records: list of dicts with 'uid', 'meta', 'chunks'
        Each chunk is stored as a Document with full per-chunk metadata for traceability.
        """
        from langchain.schema import Document  # local import to avoid cyclic
        docs = []
        meta_list = []
        for record in records:
            chunks = record.get("chunks")
            if chunks is None and "chunk" in record:
                chunks = [record["chunk"]]
            if not chunks:
                continue
            for i, chunk in enumerate(chunks):
                if chunk.get("chunk_type") == "qa":
                    text = f"Q: {chunk.get('question', '')}\nA: {chunk.get('answer', '')}"
                elif chunk.get("chunk_type") == "file_chunk":
                    content = chunk.get("content", "")
                    if isinstance(content, dict):
                        content = content.get("content", "")
                    text = content if isinstance(content, str) else str(content)
                else:
                    text = json.dumps(chunk, ensure_ascii=False)
                if not isinstance(text, str):
                    text = str(text)
                doc = Document(
                    page_content=text,
                    metadata={
                        "uid": record["uid"],
                        **record["meta"],
                        **chunk,
                        "chunk_global_index": len(self.meta) + i,
                    }
                )
                docs.append(doc)
                meta_list.append({
                    "uid": record["uid"],
                    "meta": record["meta"],
                    "chunk": chunk,
                })
        if docs:
            if not self.index:
                self.index = FAISS.from_documents(docs, self.embeddings)
            else:
                self.index.add_documents(docs)
            self.meta.extend(meta_list)
        self.save()

    def search(self, query, top_k=3):
        if not self.index:
            return []
        results = self.index.similarity_search(query, k=top_k)
        output = []
        for r in results:
            result = dict(r.metadata)
            result["content"] = getattr(r, "page_content", None) or result.get("chunk", {}).get("content", "") or ""
            if "name" not in result:
                result["name"] = result.get("meta", {}).get("name")
            if "field" not in result:
                result["field"] = result.get("meta", {}).get("field")
            output.append(result)
        return output

rag_db_manager = UserRAGDBManager()
