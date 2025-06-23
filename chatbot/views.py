import os
import re
import json
import uuid
import tempfile
import markdown
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .mongo_client import collection
from .rag_client import rag_db_manager
from .file_utils import extract_text_chunks_from_file
from .rag_pipeline import create_rag_pipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

FIELD_LLM_PATHS = {
    "agriculture": r"E:\Finetuned LLMs\fine_tuned_agriculture_model\final_model",
    "tourism": r"E:\Finetuned LLMs\fine_tuned_tourism_model\final_model",
    "transport": r"E:\Finetuned LLMs\fine_tuned_transport_model\final_model",
}
FIELD_LLM_TOKENIZERS = {
    "agriculture": r"E:\Finetuned LLMs\fine_tuned_agriculture_model\final_tokenizer",
    "tourism": r"E:\Finetuned LLMs\fine_tuned_tourism_model\final_tokenizer",
    "transport": r"E:\Finetuned LLMs\fine_tuned_transport_model\final_tokenizer",
}

TREE_PATH = os.path.join(os.path.dirname(__file__), 'question_tree.json')
with open(TREE_PATH, 'r') as f:
    QUESTION_TREE = json.load(f)

def chatbot_form(request):
    return render(request, "chat.html")

def get_user_meta(session):
    return {
        "name": session.get("name"),
        "contact": session.get("contact"),
        "email": session.get("email"),
        "field": session.get("field"),
    }

def get_finetuned_llm(field):
    path = FIELD_LLM_PATHS.get(field)
    tokenizer_path = FIELD_LLM_TOKENIZERS.get(field)
    if not path or not tokenizer_path:
        return None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(path)
    llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
    return llm_pipe

@csrf_exempt
def chatbot_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    action = data.get("action", "")
    uid = data.get("uid")
    name = data.get("name", "").strip()
    contact = data.get("contact", "").strip()
    email = data.get("email", "").strip()
    field_selected = data.get("field", "").lower()
    current_index = data.get("question_index", 0)
    session = request.session

    if action == "name":
        generated_uid = str(uuid.uuid4())
        user_data = {"uid": generated_uid, "name": name}
        collection.insert_one(user_data)
        session["uid"] = generated_uid
        session["name"] = name
        session["company_name"] = name
        session["answers"] = {}
        session["field"] = None
        session["contact"] = None
        session["email"] = None
        session["dual_agents_ready"] = False
        session.modified = True
        return JsonResponse({"message": f"Hi {name}!", "uid": generated_uid, "company_name": name})

    elif action == "contact":
        if not (uid and contact):
            return JsonResponse({"error": "Missing uid or contact"}, status=400)
        if not re.match(r'^(\+94\d{9}|0\d{9})$', contact):
            return JsonResponse({"error": "Invalid contact number"}, status=400)
        collection.update_one({"uid": uid}, {"$set": {"contact": contact}})
        session["contact"] = contact
        session.modified = True
        return JsonResponse({"message": "Contact saved!"})

    elif action == "email":
        if not (uid and email):
            return JsonResponse({"error": "Missing uid or email"}, status=400)
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            return JsonResponse({"error": "Invalid email address"}, status=400)
        collection.update_one({"uid": uid}, {"$set": {"email": email}})
        session["email"] = email
        session.modified = True
        buttons = [
            {"title": "Agriculture", "payload": "agriculture"},
            {"title": "Transport", "payload": "transport"},
            {"title": "Tourism", "payload": "tourism"}
        ]
        return JsonResponse({"message": "What is your field?", "buttons": buttons})

    elif action == "field":
        if not (uid and field_selected):
            return JsonResponse({"error": "Missing uid or field"}, status=400)
        if field_selected not in QUESTION_TREE:
            return JsonResponse({"error": "Invalid field selected"}, status=400)
        collection.update_one({"uid": uid}, {"$set": {"field": field_selected}})
        session["field"] = field_selected
        session.modified = True
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

    elif action == "field_questions":
        answer = data.get("answer")
        question_id = data.get("question_id")
        field_selected = session.get("field") or field_selected
        questions = QUESTION_TREE.get(field_selected, [])
        if uid != session.get("uid"):
            return JsonResponse({"error": "Session/uid mismatch"}, status=400)
        if answer is not None and question_id is not None:
            question_obj = next((q for q in questions if q["id"] == question_id), None)
            if question_obj:
                question_text = question_obj["question"]
                answers = session.get("answers", {})
                answers[question_text] = answer
                session["answers"] = answers
                session.modified = True

        if current_index >= len(questions):
            company_name = session.get("company_name") or session.get("name") or "Unknown"
            field_val = session.get("field")
            user_db = rag_db_manager.get_user_db(company_name, session["uid"], field_val)
            meta = get_user_meta(session)
            chunks = []
            for q, a in session.get("answers", {}).items():
                chunks.append({
                    "chunk_type": "qa",
                    "question": q,
                    "answer": a
                })
            user_db.add_records([{
                "uid": session["uid"],
                "meta": meta,
                "chunks": chunks
            }])
            if "answers" in session:
                del session["answers"]
            session.modified = True
            return JsonResponse({
                "message": "Do you want to add company data in files?",
                "show_file_upload": True
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

    elif action == "file_uploaded":
        return JsonResponse({
            "message": "🔥 Would you like to add more data about your business? (You can add advanced details powered by our Smart Assistant!)",
            "more_data_prompt": True
        })

    elif action == "add_more_data":
        choice = data.get("choice", "")
        if choice == "no":
            session["dual_agents_prompt"] = True
            session.modified = True
            return JsonResponse({
                "message": "Would you like me to create your dual AI agents now? One for you (business owner) and one for your customers, each with their own privileges and endpoints!",
                "dual_agents_prompt": True
            })
        else:
            field = session.get("field")
            llm_hint = {
                "agriculture": "Now you can add more information about your agriculture business. Ask or describe anything you want! When you're done, type 'exit' to finish.",
                "tourism": "Now you can add more information about your tourism business. You can type anything about your business, and our Smart Assistant will help! When finished, type 'exit'.",
                "transport": "Now you can add more information about your transport business. You can describe services, routes, vehicles, etc. Type 'exit' when done.",
            }
            session['llm_data'] = []
            session['llm_data_active'] = True
            session.modified = True
            return JsonResponse({
                "message": llm_hint.get(field, "Now you can add more business info! Type 'exit' to finish."),
                "llm_mode": True
            })

    elif action == "llm_data_entry":
        user_message = data.get("message", "")
        field = session.get("field")
        company_name = session.get("company_name") or session.get("name") or "Unknown"
        uid = session.get("uid")
        if not field or not uid:
            return JsonResponse({"error": "Session expired or field missing"}, status=400)
        if user_message.lower().strip() == "exit":
            user_db = rag_db_manager.get_user_db(company_name, uid, field)
            meta = get_user_meta(session)
            llm_pairs = session.get("llm_data", [])
            if llm_pairs:
                user_db.add_records([{
                    "uid": uid,
                    "meta": meta,
                    "chunks": [
                        {"chunk_type": "qa", "question": q, "answer": a} for q, a in llm_pairs
                    ]
                }])
            session['llm_data'] = []
            session['llm_data_active'] = False
            session["dual_agents_prompt"] = True
            session.modified = True
            return JsonResponse({
                "message": "Thanks for your responses! Your advanced data has been saved.",
                "dual_agents_prompt": True,
            })
        llm_pipe = get_finetuned_llm(field)
        if not llm_pipe:
            return JsonResponse({"error": f"No LLM found for field '{field}'"}, status=500)
        llm_output = llm_pipe(user_message, max_new_tokens=128)[0]['generated_text']
        answer_only = llm_output
        if user_message in llm_output:
            answer_only = llm_output.split(user_message, 1)[-1].strip(" :\n")
        elif "?" in llm_output:
            answer_only = llm_output.split("?", 1)[-1].strip(" :\n")
        llm_pairs = session.get("llm_data", [])
        llm_pairs.append((user_message, answer_only))
        session["llm_data"] = llm_pairs
        session.modified = True
        return JsonResponse({
            "message": answer_only,
            "llm_mode": True
        })

    elif action == "dual_agents_confirm":
        company_name = session.get("company_name")
        uid = session.get("uid")
        session["dual_agents_ready"] = True
        session.modified = True
        base_url = "http://127.0.0.1:8000/chat"
        admin_url = f"{base_url}/admin/{company_name}/{uid}/"
        client_url = f"{base_url}/client/{company_name}/{uid}/"
        docx_url = f"/chat/download-instructions-docx/{company_name}/{uid}/"
        html_url = f"/chat/download-instructions-html/{company_name}/{uid}/"
        md_path = os.path.join(os.path.dirname(__file__), "DualAgent_Instruction.md")
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        md_content = md_content.replace("{company_name}", company_name).replace("{uid}", uid)
        html_content = markdown.markdown(md_content, extensions=['extra', 'smarty'])
        return JsonResponse({
            "message": "🎉 Your dual AI agents have been created and are ready to use!",
            "admin_url": admin_url,
            "client_url": client_url,
            "instructions": html_content,
            "docx_url": docx_url,
            "html_url": html_url,
            "thank_you": "Thank you for using our platform!"
        })
    else:
        return JsonResponse({"error": "Unknown action"}, status=400)

@csrf_exempt
def chatbot_file_upload(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    uid = request.POST.get("uid")
    company_name = request.session.get("company_name") or request.POST.get("company_name")
    field_val = request.POST.get("field")

    if not uid or not company_name or not field_val:
        return JsonResponse({"error": "Missing uid, company_name, or field"}, status=400)

    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    uploaded_file = request.FILES["file"]
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        file_path = tmp.name

    try:
        file_chunks = extract_text_chunks_from_file(file_path, uploaded_file.name)
    except Exception as e:
        os.remove(file_path)
        return JsonResponse({"error": f"File parsing error: {str(e)}"}, status=500)
    os.remove(file_path)

    user_db = rag_db_manager.get_user_db(company_name, uid, field_val)
    meta = get_user_meta(request.session)

    prev_chunks = []
    if user_db.meta and user_db.meta[0].get("chunks"):
        prev_chunks = user_db.meta[0]["chunks"]

    prev_chunks.extend(file_chunks)

    user_db.add_records([{
        "uid": uid,
        "meta": meta,
        "chunks": prev_chunks
    }])

    return JsonResponse({"next_action": "file_uploaded"})

@csrf_exempt
def chatbot_rag_query(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    company_name = data.get("company_name")
    uid = data.get("uid")
    field = data.get("field")
    query = data.get("query")

    if not company_name or not uid or not query:
        return JsonResponse({"error": "Missing company_name, uid, or query"}, status=400)

    rag_pipeline = create_rag_pipeline(company_name, uid, field)
    result = rag_pipeline(query)
    return JsonResponse({
        "answer": result["answer"],
        "sources": result["sources"]
    })

@csrf_exempt
def business_owner_agent_api(request, company_name, uid):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    action = data.get("action", "query")
    field = data.get("field")
    if action == "query":
        query = data.get("query")
        if not query:
            return JsonResponse({"error": "Missing query"}, status=400)
        rag_pipeline = create_rag_pipeline(company_name, uid, field)
        result = rag_pipeline(query)
        return JsonResponse({"answer": result["answer"], "sources": result["sources"]})
    elif action == "update":
        field_name = data.get("field")
        value = data.get("value")
        if not (field and field_name and value):
            return JsonResponse({"error": "Missing update parameters"}, status=400)
        user_db = rag_db_manager.get_user_db(company_name, uid, field)
        user_db.add_records([{
            "uid": uid,
            "meta": {},
            "chunks": [{
                "chunk_type": "qa",
                "question": f"UPDATE FIELD: {field_name}",
                "answer": value
            }]
        }])
        return JsonResponse({"message": f"Field '{field_name}' updated with value: {value}"})
    else:
        return JsonResponse({"error": "Unknown action"}, status=400)

@csrf_exempt
def client_agent_api(request, company_name, uid):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    query = data.get("query")
    field = data.get("field")
    if not query:
        return JsonResponse({"error": "Missing query"}, status=400)
    rag_pipeline = create_rag_pipeline(company_name, uid, field)
    result = rag_pipeline(query)
    return JsonResponse({"answer": result["answer"]})

# ---- HTML DOWNLOAD VIEW ----
def download_dual_agent_html(request, company_name, uid):
    md_path = os.path.join(os.path.dirname(__file__), "DualAgent_Instruction.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    md_content = md_content.replace("{company_name}", company_name).replace("{uid}", uid)
    html_content = markdown.markdown(md_content, extensions=['extra', 'smarty'])
    response = HttpResponse(html_content, content_type='text/html')
    response['Content-Disposition'] = f'attachment; filename=DualAgent_Instruction_{company_name}_{uid}.html'
    return response