from rag_client import rag_db_manager

def find_best_answer(query, user_answers):
    # Try exact match first
    for q in user_answers:
        if query.strip().lower() == q.strip().lower():
            return user_answers[q]
    # Fallback: match by presence of keywords (word overlap)
    query_words = set(query.lower().split())
    best_q = None
    best_score = 0
    for q in user_answers:
        q_words = set(q.lower().split())
        score = len(query_words & q_words)
        if score > best_score:
            best_score = score
            best_q = q
    if best_q and best_score > 0:
        return user_answers[best_q]
    return None

def main():
    company_name = "Kavinda"
    uid = "5e2db1f6-486c-42c1-aca9-d1d09262a14d"
    field = "agriculture"

    # User input for question
    query = input("Ask your question: ")

    user_db = rag_db_manager.get_user_db(company_name, uid, field)
    results = user_db.hybrid_search(query, top_k=1, alpha=0.5)
    if not results:
        print("No results found for this user.")
        return

    user = results[0].get("json", results[0])
    answers = user.get("answers", {})
    answer = find_best_answer(query, answers)
    if answer:
        print(answer)
    else:
        print("No relevant answer found.")

if __name__ == "__main__":
    main()