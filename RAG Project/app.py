import streamlit as st
from query import query_rag

st.set_page_config(page_title="University Handbook Q&A")
st.title("🎓 University Handbook Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        response = query_rag(query)
        st.session_state.chat_history.insert(0, (query, response))

for q, r in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {q}")
    with st.chat_message("assistant"):
        st.markdown(f"**Assistant:** {r}")