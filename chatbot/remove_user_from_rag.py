import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
META_PATH = os.path.join(os.path.dirname(__file__), "faiss_meta.json")
EMBED_MODEL = "all-MiniLM-L6-v2"  # or whatever you use

uid_to_delete = "d7e6b28f-3bfa-480a-b8a7-96b50ed5adc9"  # <-- change to your target UID

# 1. Load metadata
with open(META_PATH, "r") as f:
    meta = json.load(f)

# 2. Filter out the entry to delete
new_meta = [entry for entry in meta if entry["uid"] != uid_to_delete]
deleted = len(meta) - len(new_meta)
if not deleted:
    print(f"UID {uid_to_delete} not found in FAISS meta.")
else:
    print(f"Deleted {deleted} entry with UID {uid_to_delete}")

# 3. Recompute embeddings for remaining entries
model = SentenceTransformer(EMBED_MODEL)
all_jsons = [entry["json"] for entry in new_meta]
doc_strs = [json.dumps(j, ensure_ascii=False) for j in all_jsons]
if doc_strs:
    embeddings = model.encode(doc_strs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
else:
    index = faiss.IndexFlatL2(384)  # Empty index

# 4. Save new index and metadata
faiss.write_index(index, DB_PATH)
with open(META_PATH, "w") as f:
    json.dump(new_meta, f, indent=2)

print("Rebuilt and saved new FAISS index and metadata.")