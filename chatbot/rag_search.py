from rag_pipeline import create_rag_pipeline

def test_rag_query():
    print("Enter company name:")
    company_name = input().strip()
    print("Enter UID:")
    uid = input().strip()
    print("Enter field (or leave blank):")
    field = input().strip() or None

    # Create the RAG pipeline
    rag_pipeline = create_rag_pipeline(company_name, uid, field)
    
    # Test with questions
    while True:
        print("\nEnter your question (or type 'exit' to quit):")
        query = input().strip()
        if query.lower() == 'exit':
            break
        
        result = rag_pipeline(query)
        print("\n=== ANSWER ===")
        print(result["answer"])
        
        print("\n=== SOURCES ===")
        for i, source in enumerate(result["sources"]):
            print(f"Source {i+1}:")
            print(f"Content: {source['content'][:150]}...")
            print(f"Metadata: {source['metadata']}")
            print()

if __name__ == "__main__":
    test_rag_query()