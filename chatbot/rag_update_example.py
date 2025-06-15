from rag_client import rag_db_manager

def update_user_answer(company_name, uid, question_to_update, new_answer, field=None):
    # Load the user's vector DB
    user_db = rag_db_manager.get_user_db(company_name, uid, field)
    if not user_db.meta or not user_db.meta[0]["json"].get("answers"):
        print("No data found for this user.")
        return False

    # Get the user's current JSON data
    user_data = user_db.meta[0]["json"]
    answers = user_data.get("answers", {})
    if question_to_update not in answers:
        print(f"Question not found in user's answers: {question_to_update}")
        return False

    # Update the specific answer
    print(f"Old answer: {answers[question_to_update]}")
    answers[question_to_update] = new_answer
    print(f"New answer: {answers[question_to_update]}")

    # Save updated data and re-encode embedding
    user_db.update_json(user_data)
    print("Update complete.")

    # (Optional) Show the new answers
    print("\nUpdated Answers:")
    for q, a in answers.items():
        print(f"  Q: {q}\n  A: {a}")
    return True

def main():
    # ====== CHANGE BELOW FOR YOUR TEST ======
    company_name = "Kavinda"  # Company name for the RAG database
    uid = "5e2db1f6-486c-42c1-aca9-d1d09262a14d"  # User's UID

    field = "agriculture"  #field value
    question_to_update = "What are your main products or crops?"  # The question to update
    new_answer = "Milk"  # The new answer you want to set
    # ========================================

    updated = update_user_answer(company_name, uid, question_to_update, new_answer, field=field)
    if updated:
        print("\nUser answer updated successfully.")
    else:
        print("\nUser answer was NOT updated.")

if __name__ == "__main__":
    main()