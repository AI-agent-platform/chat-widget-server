from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .mongo_client import collection
import json

def chatbot_form(request):
    return render(request, "chat.html")

@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        # Support both form-data and raw JSON input
        if request.content_type == "application/json":
            try:
                data = json.loads(request.body.decode("utf-8"))
            except Exception:
                return JsonResponse({"error": "Invalid JSON"}, status=400)
            action = data.get("action", "")
            user_id = data.get("user_id")
            field_selected = data.get("field")
            name = data.get("name")
            contact = data.get("contact")
            email = data.get("email")
        else:
            action = request.POST.get("action", "")
            user_id = request.POST.get("user_id")
            field_selected = request.POST.get("field")
            name = request.POST.get("name")
            contact = request.POST.get("contact")
            email = request.POST.get("email")

        # Step 4: Field selection
        if action == "field":
            if not (user_id and field_selected):
                return JsonResponse({"error": "Missing user_id or field"}, status=400)
            from bson import ObjectId
            collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"field": field_selected}}
            )
            return JsonResponse({
                "message": f"Thank you! You selected '{field_selected.capitalize()}'. Your data has been saved.",
                "buttons": []
            })

        # Step 1 to 3: Collect name, contact, email and create user
        elif action == "form":
            if not (name and contact and email):
                return JsonResponse({"error": "All fields are required"}, status=400)
            user_data = {
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
                "user_id": user_id
            })
        else:
            return JsonResponse({"error": "Unknown action"}, status=400)
    return JsonResponse({"error": "Invalid method"}, status=405)