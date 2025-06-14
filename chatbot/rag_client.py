import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
META_PATH = os.path.join(os.path.dirname(__file__), "faiss_meta.json")
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
            self.index = faiss.IndexFlatL2(384)  # Depends on embedding dimension
            self.meta = []

    def save(self):
        faiss.write_index(self.index, DB_PATH)
        with open(META_PATH, "w") as f:
            json.dump(self.meta, f, indent=2)

    def add_json(self, user_json):
        # You can store as: {"uid": ..., "json": user_json}
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

# Singleton instance
rag_db = RAGVectorDB()