import os
import re
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .rag_client import rag_db_manager
from langchain_openai import ChatOpenAI

load_dotenv()

def clean_answer(text):
    if not isinstance(text, str):
        return text
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(Questions?:|Answers?:)', '', text, flags=re.I)
    text = re.sub(r'["“”]', '', text)
    text = text.strip()
    return text

def create_rag_pipeline(company_name, uid, field=None):
    all_dbs = rag_db_manager.get_all_user_dbs(company_name, uid)
    if field:
        field_db = rag_db_manager.get_user_db(company_name, uid, field)
        all_dbs = [field_db] + [db for db in all_dbs if db.folder_path != field_db.folder_path]

    has_data = any(db.index for db in all_dbs)
    if not has_data:
        return lambda q: {"answer": "No data found for this user", "sources": []}

    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        temperature=0.7,
    )

    def hybrid_retrieve(query, top_k=3):
        results = []
        for db in all_dbs:
            results.extend(db.hybrid_search(query, top_k=top_k))
        seen = set()
        deduped = []
        for r in results:
            key = r.get("content", "")
            if key not in seen:
                deduped.append(r)
                seen.add(key)
        return deduped[:top_k]

    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    def answer_question(question):
        retrieved_docs = hybrid_retrieve(question, top_k=3)
        if not retrieved_docs:
            return {"answer": "No data found for this user", "sources": []}

        context = "\n\n".join([r["content"] for r in retrieved_docs if r.get("content")])
        sources = [
            {
                "content": r["content"],
                "metadata": r.get("meta", {})
            }
            for r in retrieved_docs
        ]

        llm_input = {
            "context": context,
            "question": question
        }

        prompt_str = PROMPT.format(**llm_input)
        result = llm.invoke(prompt_str)

        if isinstance(result, str):
            answer = result
        elif hasattr(result, "content"):
            answer = result.content
        else:
            answer = ""
        answer = clean_answer(answer)

        return {
            "answer": answer,
            "sources": sources
        }

    return answer_question