import faiss
import numpy as np
import os
import json
import re
from sentence_transformers import SentenceTransformer

# Custom directory for saving FAISS index and metadata
FAISS_DIR = r"E:\RAGDB"
os.makedirs(FAISS_DIR, exist_ok=True)

DB_PATH = os.path.join(FAISS_DIR, "faiss_index.bin")
META_PATH = os.path.join(FAISS_DIR, "faiss_meta.json")
EMBED_MODEL = "all-MiniLM-L6-v2"  # Small, fast model

class RAGVectorDB:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = None
        self.meta = []
        self.load()

    def load(self):
        if os.path.exists(DB_PATH):
            self.index = faiss.read_index(DB_PATH)
            if os.path.exists(META_PATH):
                with open(META_PATH, "r") as f:
                    self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(384)  # 384 = embedding dimension of MiniLM
            self.meta = []

    def save(self):
        faiss.write_index(self.index, DB_PATH)
        with open(META_PATH, "w") as f:
            json.dump(self.meta, f, indent=2)

    def add_json(self, user_json):
        doc_str = json.dumps(user_json, ensure_ascii=False)
        embedding = self.model.encode([doc_str])
        self.index.add(np.array(embedding).astype("float32"))
        self.meta.append({"uid": user_json["uid"], "json": user_json})
        self.save()

    def search(self, query, top_k=3):
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.meta):
                results.append(self.meta[idx])
        return results

    def keyword_search(self, query, top_k=3):
        query_lower = query.lower()
        scored = []
        for item in self.meta:
            text = json.dumps(item["json"]).lower()
            match_count = sum(1 for word in query_lower.split() if word in text)
            if match_count > 0:
                scored.append((match_count, item))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in scored[:top_k]]

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), top_k * 2)
        semantic_scores = {self.meta[idx]["uid"]: (1 - D[0][i]) for i, idx in enumerate(I[0]) if idx < len(self.meta)}

        query_lower = query.lower()
        keyword_scores = {}
        for item in self.meta:
            uid = item["uid"]
            text = json.dumps(item["json"]).lower()
            kw_score = sum(1 for word in query_lower.split() if word in text)
            if kw_score > 0:
                keyword_scores[uid] = kw_score

        if keyword_scores:
            max_kw = max(keyword_scores.values())
            keyword_scores = {uid: score / max_kw for uid, score in keyword_scores.items()}

        combined_scores = {}
        for uid in set(semantic_scores) | set(keyword_scores):
            sem = semantic_scores.get(uid, 0)
            kw = keyword_scores.get(uid, 0)
            combined_scores[uid] = alpha * sem + (1 - alpha) * kw

        sorted_uids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for uid, _ in sorted_uids[:top_k]:
            for item in self.meta:
                if item["uid"] == uid:
                    results.append(item)
                    break

        return results

# Singleton instance
rag_db = RAGVectorDB()
