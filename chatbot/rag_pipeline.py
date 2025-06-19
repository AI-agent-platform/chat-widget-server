import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from .rag_client import rag_db_manager
import re

# UPDATED imports for HuggingFace
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

def clean_answer(text):
    # Remove unwanted patterns
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
    """
    Hybrid RAG pipeline: searches all vector DBs for this user (all fields) and combines results.
    """
    # Get all DBs for this user (if field is specified, prioritize it)
    all_dbs = rag_db_manager.get_all_user_dbs(company_name, uid)
    if field:
        # Place the selected field's db first
        field_db = rag_db_manager.get_user_db(company_name, uid, field)
        all_dbs = [field_db] + [db for db in all_dbs if db.folder_path != field_db.folder_path]

    # Check if user has any data
    has_data = any(db.index for db in all_dbs)
    if not has_data:
        return lambda q: {"answer": "No data found for this user", "sources": []}
   
    # Initialize the LLM
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    )

    # Create a retriever-like function using hybrid search across all DBs
    def hybrid_retrieve(query, top_k=3):
        results = []
        for db in all_dbs:
            results.extend(db.hybrid_search(query, top_k=top_k))
        # Deduplicate by content
        seen = set()
        deduped = []
        for r in results:
            key = r.get("content", "")
            if key not in seen:
                deduped.append(r)
                seen.add(key)
        # Limit to top_k
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
        # Hybrid search
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
        # Format for LLM
        llm_input = {
            "context": context,
            "question": question
        }
        # Call LLM
        result = llm.invoke(PROMPT.format(**llm_input))
        answer = result if isinstance(result, str) else result.get("result", "")
        answer = clean_answer(answer)
        return {
            "answer": answer,
            "sources": sources
        }
    return answer_question