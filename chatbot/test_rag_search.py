from rag_client import rag_db

def main():
    query = "Looking for transport businesses in Kandy and tourism services in Mannar that offer guided tours or accommodation"
    top_k = 2
    alpha = 0.5  # Blend 60% semantic, 40% keyword

    results = rag_db.hybrid_search(query, top_k=top_k, alpha=alpha)
    print(f"\nQuery: {query}\n{'='*60}")
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
