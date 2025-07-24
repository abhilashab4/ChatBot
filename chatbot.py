import os

from dotenv import load_dotenv

load_dotenv()

import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda



llm = ChatGroq(model="llama3-8b-8192",  max_tokens=150)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

store = {}

def get_by_session_id(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="input",
    history_messages_key="history",
)


def load_pdf_and_create_chain(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load_and_split()

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for answering questions about the uploaded PDF.
    If the user's question is not about the content of the PDF or if the answer cannot be found in the PDF, respond with: NOT RELATED.

    Context:
    {context}
    """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = (
    RunnableLambda(lambda x: {
        "context": retriever.invoke(x["input"]),
        "input": x["input"],
        "history": x["history"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

    pdf_chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="input",   
        history_messages_key="history"
    )

    return pdf_chain_with_memory


def get_pdf_response(pdf_chain, user_input, session_id):
    return pdf_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
def get_general_response(user_input, session_id):
    general_response = chat_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    return general_response.content.strip()

def get_response(user_input, pdf_chain=None, session_id="default"):

    general_only_prompts = [
        "who are you", "what can you do", "hello", "hi","hai"
        "how can you help", "help", "introduce yourself"
    ]

    if any(phrase in user_input.lower() for phrase in general_only_prompts):
        general_result = get_general_response(user_input=user_input, session_id=session_id)
        return general_result
    

    if pdf_chain:
        
            pdf_response = get_pdf_response(pdf_chain=pdf_chain, user_input=user_input,session_id=session_id)
            if "not related" not in pdf_response.strip().lower():
                return pdf_response.strip()
            else:
                return get_general_response(user_input=user_input,session_id=session_id)


    general_result = get_general_response(user_input=user_input, session_id=session_id)
    return general_result
