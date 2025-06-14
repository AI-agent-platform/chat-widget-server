from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .mongo_client import collection
import json
import re
import uuid
import os

# Load question tree JSON
TREE_PATH = os.path.join(os.path.dirname(__file__), 'question_tree.json')
with open(TREE_PATH, 'r') as f:
    QUESTION_TREE = json.load(f)

def chatbot_form(request):
    return render(request, "chat.html")

@csrf_exempt
def chatbot_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    action = data.get("action", "")
    uid = data.get("uid")
    name = data.get("name", "").strip()
    contact = data.get("contact", "").strip()
    email = data.get("email", "").strip()
    field_selected = data.get("field", "").lower()
    current_index = data.get("question_index", 0)
    # answer and question_id are no longer needed for saving

    # Step 1: Save Name and generate UID
    if action == "name":
        if not name:
            return JsonResponse({"error": "Name is required"}, status=400)
        generated_uid = str(uuid.uuid4())
        user_data = {"uid": generated_uid, "name": name}
        collection.insert_one(user_data)
        return JsonResponse({"message": f"Hi {name}!", "uid": generated_uid})

    # Step 2: Save Contact
    elif action == "contact":
        if not (uid and contact):
            return JsonResponse({"error": "Missing uid or contact"}, status=400)
        if not re.match(r'^(\+94\d{9}|0\d{9})$', contact):
            return JsonResponse({"error": "Invalid contact number"}, status=400)
        collection.update_one({"uid": uid}, {"$set": {"contact": contact}})
        return JsonResponse({"message": "Contact saved!"})

    # Step 3: Save Email
    elif action == "email":
        if not (uid and email):
            return JsonResponse({"error": "Missing uid or email"}, status=400)
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            return JsonResponse({"error": "Invalid email address"}, status=400)
        collection.update_one({"uid": uid}, {"$set": {"email": email}})
        buttons = [
            {"title": "Agriculture", "payload": "agriculture"},
            {"title": "Transport", "payload": "transport"},
            {"title": "Tourism", "payload": "tourism"}
        ]
        return JsonResponse({"message": "What is your field?", "buttons": buttons})

    # Step 4: Save Field & Immediately Ask First Question
    elif action == "field":
        if not (uid and field_selected):
            return JsonResponse({"error": "Missing uid or field"}, status=400)
        if field_selected not in QUESTION_TREE:
            return JsonResponse({"error": "Invalid field selected"}, status=400)

        collection.update_one({"uid": uid}, {"$set": {"field": field_selected}})
        first_question = QUESTION_TREE[field_selected][0]

        return JsonResponse({
            "message": f"Thank you! You selected '{field_selected.capitalize()}'. Let's begin.",
            "field": field_selected,
            "uid": uid,
            "next_action": "field_questions",
            "question_index": 1,
            "question": first_question["question"],
            "question_id": first_question["id"],
            "type": first_question["type"],
            "options": first_question.get("options", [])
        })

    # Step 5: Continue Asking Field Questions (DO NOT SAVE ANSWERS)
    elif action == "field_questions":
        if not field_selected or field_selected not in QUESTION_TREE:
            return JsonResponse({"error": "Invalid or missing field"}, status=400)

        questions = QUESTION_TREE[field_selected]
        if current_index >= len(questions):
            return JsonResponse({
                "message": "All questions have been completed. Thank you!"
            })

        question = questions[current_index]
        return JsonResponse({
            "question_index": current_index + 1,
            "question": question["question"],
            "question_id": question["id"],
            "type": question["type"],
            "options": question.get("options", []),
            "next_action": "field_questions"
        })

    else:
        return JsonResponse({"error": "Unknown action"}, status=400)