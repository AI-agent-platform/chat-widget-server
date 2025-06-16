from rag_client import rag_db_manager

def query_wants_whole_table(query):
    # Add more patterns as needed
    patterns = [
        "show prices", "list prices", "show table", "show milk product prices",
        "list milk product prices", "show price table", "all milk product prices"
    ]
    return any(p in query.lower() for p in patterns)

def extract_exact_line(chunk_content, query):
    if query_wants_whole_table(query):
        return chunk_content  # Return the whole table
    
    lines = chunk_content.splitlines()
    query_words = set(query.lower().split())
    best_line = None
    best_score = 0
    for line in lines:
        line_words = set(line.lower().split())
        score = len(query_words & line_words)
        if score > best_score:
            best_score = score
            best_line = line
    if best_line and best_score > 0:
        return best_line
    matching_lines = [line for line in lines if any(word in line.lower() for word in query_words)]
    if matching_lines:
        return "\n".join(matching_lines)
    return chunk_content

def main():
    company_name = "Food city"
    uid = "83e31dcf-6d06-400f-a771-b3ade5cc311d"
    field = "agriculture"

    user_db = rag_db_manager.get_user_db(company_name, uid, field)
    record = user_db.meta[0] if user_db.meta else None
    if not record:
        return
    query = input("Question: ").strip().lower()

    # 1. Try to match exact QA first
    for chunk in record["chunks"]:
        if chunk.get("chunk_type") == "qa":
            if query == chunk["question"].strip().lower():
                print(chunk['answer'])
                return

    # 2. Try vector search for best chunk (QA or file)
    results = user_db.hybrid_search(query, top_k=1, alpha=0.5)
    if not results:
        print("No relevant answer found.")
        return

    best_chunk = results[0]
    if best_chunk.get("chunk_type") == "qa":
        print(best_chunk['answer'])
    elif best_chunk.get("chunk_type") == "file_chunk":
        exact = extract_exact_line(best_chunk["content"], query)
        print(exact)
    else:
        print("No relevant answer found.")

if __name__ == "__main__":
    main()