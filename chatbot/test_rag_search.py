from rag_client import rag_db_manager

def main():
    company_name = "Kavinda" # Company name for the RAG database
    uid = "5e2db1f6-486c-42c1-aca9-d1d09262a14d"  # User's UID

    field = "agriculture"  #  field value

    query = "Who are the main target customers?"
    top_k = 2
    alpha = 0.5  # Blend semantic and keyword

    # Get the user's vector DB
    user_db = rag_db_manager.get_user_db(company_name, uid, field)
    results = user_db.hybrid_search(query, top_k=top_k, alpha=alpha)
    print(f"\nQuery: {query}\n{'='*60}")
    if not results:
        print("No results found for this user.")
        return

    for idx, entry in enumerate(results, 1):
        user = entry.get("json", entry)
        print(f"Result {idx}:")
        print(f"  Name: {user.get('name')}")
        print(f"  Email: {user.get('email')}")
        print(f"  Field: {user.get('field')}")
        print("  Answers:")
        for q, a in user.get("answers", {}).items():
            print(f"    Q: {q}\n    A: {a}")
        print("-" * 40)

if __name__ == "__main__":
    main()