import streamlit as st
import uuid
from chatbot import load_pdf_and_create_chain, get_response

st.set_page_config(page_title="ChatBot", layout="centered")
st.title("ChatBot 💬")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "pdf_chain" not in st.session_state:
    st.session_state.pdf_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []  

uploaded_pdf = st.file_uploader("", type="pdf")

if uploaded_pdf and st.session_state.pdf_chain is None:
    with st.spinner("Processing PDF..."):
        st.session_state.pdf_chain = load_pdf_and_create_chain(uploaded_pdf)
    st.success("PDF loaded successfully!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(user_query, st.session_state.pdf_chain, st.session_state.session_id)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
