import faiss
import numpy as np
import os
import json
import re
from sentence_transformers import SentenceTransformer

BASE_FAISS_DIR = r"E:\RAGDB"
os.makedirs(BASE_FAISS_DIR, exist_ok=True)
EMBED_MODEL = "all-MiniLM-L6-v2"

def sanitize_folder_name(name):
    return re.sub(r'[^A-Za-z0-9_\-]', '', name.replace(" ", "_"))

class UserRAGVectorDB:
    """
    Stores one meta per user, with all chunks (QA and file) in a chunks list. 
    Vectors are stored for each chunk; meta is stored once for all.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.db_path = os.path.join(folder_path, "faiss_index.bin")
        self.meta_path = os.path.join(folder_path, "faiss_meta.json")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = None
        self.meta = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            self.index = faiss.read_index(self.db_path)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(384)
            self.meta = []

    def save(self):
        faiss.write_index(self.index, self.db_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

    def save_record(self, record):
        """
        record: dict with 'uid', 'meta', 'chunks'
        Save all chunks as vectors, with one meta per user.
        """
        self.index = faiss.IndexFlatL2(384)
        self.meta = []
        chunk_vectors = []
        for chunk in record["chunks"]:
            if chunk.get("chunk_type") == "qa":
                text = f"Q: {chunk['question']}\nA: {chunk['answer']}"
            elif chunk.get("chunk_type") == "file_chunk":
                text = chunk["content"]
            else:
                text = json.dumps(chunk, ensure_ascii=False)
            chunk_vectors.append(text)
        if chunk_vectors:
            embeds = self.model.encode(chunk_vectors)
            embeds = np.array(embeds).astype("float32")
            self.index.add(embeds)
        self.meta.append({
            "uid": record["uid"],
            "meta": record["meta"],
            "chunks": record["chunks"]
        })
        self.save()

    def search(self, query, top_k=3):
        if not self.meta or self.index.ntotal == 0:
            return []
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), min(top_k, self.index.ntotal))
        all_chunks = self.meta[0]["chunks"] if self.meta else []
        results = []
        for idx in I[0]:
            if idx < len(all_chunks):
                result = dict(all_chunks[idx])
                result["meta"] = self.meta[0]["meta"]
                results.append(result)
        return results

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        if not self.meta or self.index.ntotal == 0:
            return []
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), min(top_k * 2, self.index.ntotal))
        all_chunks = self.meta[0]["chunks"] if self.meta else []
        semantic_scores = {i: (1 - D[0][rank]) for rank, i in enumerate(I[0]) if i < len(all_chunks)}

        query_lower = query.lower()
        keyword_scores = {}
        for idx, chunk in enumerate(all_chunks):
            text = ""
            if chunk.get("chunk_type") == "qa":
                text = f"Q: {chunk['question']}\nA: {chunk['answer']}".lower()
            elif chunk.get("chunk_type") == "file_chunk":
                text = chunk["content"].lower()
            kw_score = sum(1 for word in query_lower.split() if word in text)
            if kw_score > 0:
                keyword_scores[idx] = kw_score

        if keyword_scores:
            max_kw = max(keyword_scores.values())
            keyword_scores = {idx: score / max_kw for idx, score in keyword_scores.items()}

        combined_scores = {}
        for idx in set(semantic_scores) | set(keyword_scores):
            sem = semantic_scores.get(idx, 0)
            kw = keyword_scores.get(idx, 0)
            combined_scores[idx] = alpha * sem + (1 - alpha) * kw

        sorted_idxs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, _ in sorted_idxs[:top_k]:
            if idx < len(all_chunks):
                result = dict(all_chunks[idx])
                result["meta"] = self.meta[0]["meta"]
                results.append(result)
        return results

class UserRAGDBManager:
    def __init__(self, base_dir=BASE_FAISS_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.cache = {}

    def get_user_folder(self, company_name, uid, field=None):
        if not field:
            field_folder = "general"
        else:
            field_folder = sanitize_folder_name(field)
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

rag_db_manager = UserRAGDBManager()