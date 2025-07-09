from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from embedding import embedding_function
import streamlit as st

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful assistant for university handbook questions.
Use ONLY the context provided.

<context>
{context}
</context>

Question: {question}

If you don't know, say "I don't know."
"""

@st.cache_resource(show_spinner=False)
def load_vector_db():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function())

@st.cache_resource(show_spinner=False)
def load_llm():
    return OllamaLLM(model="llama3", temperature=0.1, max_tokens=256)

def query_rag(query_text: str):
    db = load_vector_db()
    results = db.similarity_search_with_score(query_text, k=5)

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, question=query_text
    )

    model = load_llm()
    answer = model.invoke(prompt)
    return answer
