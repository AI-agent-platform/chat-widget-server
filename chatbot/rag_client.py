import faiss
import numpy as np
import os
import json
import re
from sentence_transformers import SentenceTransformer

BASE_FAISS_DIR = r"E:\RAGDB"
os.makedirs(BASE_FAISS_DIR, exist_ok=True)
EMBED_MODEL = "all-MiniLM-L6-v2"  # Small, fast model

def sanitize_folder_name(name):
    # Remove unsafe chars for folder names
    return re.sub(r'[^A-Za-z0-9_\-]', '', name.replace(" ", "_"))

class UserRAGVectorDB:
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
                with open(self.meta_path, "r") as f:
                    self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(384)  # 384 = embedding dimension of MiniLM
            self.meta = []

    def save(self):
        faiss.write_index(self.index, self.db_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

    def add_json(self, user_json):
        # Overwrites previous entry (one vector per user)
        self.index = faiss.IndexFlatL2(384)
        self.meta = []
        doc_str = json.dumps(user_json, ensure_ascii=False)
        embedding = self.model.encode([doc_str])
        self.index.add(np.array(embedding).astype("float32"))
        self.meta.append({"uid": user_json["uid"], "json": user_json})
        self.save()

    def update_json(self, user_json):
        # Alias for add_json, as we always overwrite
        self.add_json(user_json)

    def search(self, query, top_k=1):
        if len(self.meta) == 0:
            return []
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), min(top_k, len(self.meta)))
        results = []
        for idx in I[0]:
            if idx < len(self.meta):
                results.append(self.meta[idx])
        return results

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        if not self.meta:
            return []
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), min(top_k * 2, len(self.meta)))
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

# Manager for loading user DBs
class UserRAGDBManager:
    def __init__(self, base_dir=BASE_FAISS_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.cache = {}

    def get_user_folder(self, company_name, uid, field=None):
        # NEW: accept field, and make path BASE/field/company_uid
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