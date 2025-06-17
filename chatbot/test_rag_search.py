import re
from rag_client import UserRAGDBManager

def display_result(result, query):
    """
    Print the most relevant result in a readable format.
    If possible, extract specific info (like price or location) mentioned in the query.
    """
    print("\n--- Top Match ---")
    print(f"Name : {result.get('name') or result.get('meta', {}).get('name')}")
    print(f"Field: {result.get('field') or result.get('meta', {}).get('field')}")
    print(f"UID  : {result.get('uid')}")
    content = (
        result.get('content') or
        result.get('page_content') or
        result.get('chunk', {}).get('content', '') or
        ''
    )
    print(f"\nContent:\n{content}")

    # Try to extract a keyword (like "price", "location", or something from the query)
    q_lower = query.lower()
    # Try to extract a target word from the query, e.g., "How much is X", "What is the Y"
    m_target = re.search(r"(?:how much is|what(?: is| are)?|show|give me|find|tell me) ([a-zA-Z0-9 \-/]+)", q_lower)
    if m_target:
        target = m_target.group(1).strip()
        # Search for the target keyword in content, show the matching line(s)
        matches = []
        for line in content.splitlines():
            if target.lower() in line.lower():
                matches.append(line.strip())
        if matches:
            print(f"\n[Extracted lines for '{target}':]")
            for match in matches:
                print(match)
            return

    # Fallback for common fields
    common_fields = ["price", "cost", "location", "rating", "available rooms", "product", "crop"]
    for field in common_fields:
        if field in q_lower:
            # Print all lines containing that field
            matches = []
            for line in content.splitlines():
                if field in line.lower():
                    matches.append(line.strip())
            if matches:
                print(f"\n[Extracted lines for '{field}':]")
                for match in matches:
                    print(match)
                return

def general_rag_search(company_name=None, uid=None, field=None, query="", top_k=3):
    rag_db_manager = UserRAGDBManager()
    db = rag_db_manager.get_user_db(company_name, uid, field)
    results = db.search(query, top_k=top_k)
    if not results:
        print("No results found for your query.")
        return
    display_result(results[0], query)

if __name__ == "__main__":
    print("Enter company name (or leave blank to search all):")
    company_name = input().strip() or None
    print("Enter UID (or leave blank to search all):")
    uid = input().strip() or None
    print("Enter field (or leave blank to search all):")
    field = input().strip() or None

    while True:
        try:
            q = input("\nAsk your question: ").strip()
            if not q:
                continue
            general_rag_search(company_name, uid, field, q)
        except KeyboardInterrupt:
            print("\nExiting.")
            break