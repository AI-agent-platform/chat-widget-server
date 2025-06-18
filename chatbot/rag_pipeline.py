import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from .rag_client import rag_db_manager

# UPDATED imports for new HuggingFace integration:
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

def create_rag_pipeline(company_name, uid, field=None):
    """
    Creates a RAG pipeline that can answer questions based on the user's data.

    Args:
        company_name (str): Name of the company
        uid (str): User ID
        field (str, optional): Field/domain of the user. Defaults to None.

    Returns:
        callable: A function that takes a question and returns an answer
    """
    # Get the vector DB for the user
    user_db = rag_db_manager.get_user_db(company_name, uid, field)

    # Check if the user has data
    if not user_db.index:
        return lambda q: {"answer": "No data found for this user", "sources": []}
   
    # Initialize the LLM with the new HuggingFaceEndpoint class
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    )

    # Create a retriever from the user's vector DB
    retriever = user_db.index.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Create a prompt template
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

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Return a function that can be called with a question
    def answer_question(question):
        result = qa_chain.invoke({"query": question})
        sources = []
        if hasattr(result, "source_documents"):
            for doc in result.source_documents:
                source = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source)
        return {
            "answer": result["result"],
            "sources": sources
        }

    return answer_question