import json
from sentence_transformers import SentenceTransformer, util

# Load your metadata
with open("meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)[0]

model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare texts and metadata for search
texts = []
answer_refs = []
for chunk in meta["chunks"]:
    if chunk["chunk_type"] == "qa":
        texts.append(f"Q: {chunk['question']} A: {chunk['answer']}")
        answer_refs.append(chunk['answer'])
    elif chunk["chunk_type"] == "file_chunk":
        # For file chunks, treat each line as a possible answer
        for line in chunk["content"].splitlines():
            if line.strip():
                texts.append(line.strip())
                answer_refs.append(line.strip())

# Pre-compute embeddings
embeddings = model.encode(texts, convert_to_tensor=True)

def answer_only_search(query):
    query_emb = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    best_idx = int(cos_scores.argmax())
    answer = answer_refs[best_idx]

    # Optional: If price is requested, try to pick out the price value
    if "price" in query.lower() or "cost" in query.lower():
        import re
        match = re.search(r'(Rs\.\s*\d+\.?\d*)|(\d+\.?\d*)', answer)
        if match:
            return match.group(0)
    return answer

if __name__ == "__main__":
    q = input("Ask: ")
    print(answer_only_search(q))