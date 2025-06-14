from rag_client import rag_db  # Adjust import if needed

def main():
    query = "I need goods transport with trucks in Kandy for my business"
    top_k = 2

    results = rag_db.search(query, top_k=top_k)
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