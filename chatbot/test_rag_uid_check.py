from rag_client import rag_db

# Print number of vectors in Faiss
print("Number of vectors in Faiss:", rag_db.index.ntotal)

# Print all UIDs and corresponding names
print("--- Entries in FAISS RAG DB ---")
found = False
for entry in rag_db.meta:
    print(f"UID: {entry['uid']} | Name: {entry['json'].get('name')} | Field: {entry['json'].get('field')}")
    if entry['uid'] == "8fc0194e-5478-480b-b5d6-3608fb7ca81d":
        found = True

if found:
    print("\n✅  Your UID is present in the FAISS vector DB!")
else:
    print("\n❌  Your UID was NOT found in the FAISS vector DB.")