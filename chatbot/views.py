from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .mongo_client import collection
import json
import re
import uuid
from bson import ObjectId

def chatbot_form(request):
    return render(request, "chat.html")

@csrf_exempt
def chatbot_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    # Handle form-data and raw JSON inputs
    if request.content_type == "application/json":
        try:
            data = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        action = data.get("action", "")
        user_id = data.get("user_id")
        field_selected = data.get("field")
        name = data.get("name", "").strip()
        contact = data.get("contact", "").strip()
        email = data.get("email", "").strip()
    else:
        action = request.POST.get("action", "")
        user_id = request.POST.get("user_id")
        field_selected = request.POST.get("field")
        name = request.POST.get("name", "").strip()
        contact = request.POST.get("contact", "").strip()
        email = request.POST.get("email", "").strip()

    # ✅ Handle form submission
    if action == "form":
        if not name:
            return JsonResponse({"error": "Name is required"}, status=400)
        if not re.match(r'^(\+94\d{9}|0\d{9})$', contact):
            return JsonResponse({"error": "Invalid contact number"}, status=400)
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            return JsonResponse({"error": "Invalid email address"}, status=400)

        # ✅ Generate UUID for the user
        generated_uid = str(uuid.uuid4())

        user_data = {
            "uid": generated_uid,
            "name": name,
            "contact": contact,
            "email": email,
            "field": None
        }

        inserted = collection.insert_one(user_data)
        user_id = str(inserted.inserted_id)

        buttons = [
            {"title": "Agriculture", "payload": "agriculture"},
            {"title": "Transport", "payload": "transport"},
            {"title": "Tourism", "payload": "tourism"}
        ]

        return JsonResponse({
            "message": "What is your field?",
            "buttons": buttons,
            "user_id": user_id,
            "uid": generated_uid  # Optional: can use this in future integrations
        })

    # ✅ Handle field selection
    elif action == "field":
        if not (user_id and field_selected):
            return JsonResponse({"error": "Missing user_id or field"}, status=400)
        try:
            collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"field": field_selected}}
            )
        except Exception:
            return JsonResponse({"error": "Invalid user ID"}, status=400)

        return JsonResponse({
            "message": f"Thank you! You selected '{field_selected.capitalize()}'. Your data has been saved.",
            "buttons": []
        })

    else:
        return JsonResponse({"error": "Unknown action"}, status=400)
