import os

from dotenv import load_dotenv

load_dotenv()


from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


llm = ChatGroq(model="llama3-8b-8192")

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

import tempfile
def load_pdf_and_create_chain(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load_and_split()

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant for answering questions about a PDF.
        
        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def get_response(user_input, pdf_chain=None, session_id="default"):
    pdf_response = ""

    IRRELEVANT_PHRASES = [
        "i'm sorry", 
        "i don't know", 
        "not in", 
        "unable to find", 
        "cannot determine", 
        "not sure", 
        "no relevant information", 
        "information is missing",
        "based on the information provided",
        "doesn't seem",
    ]

    if pdf_chain:
        try:
            pdf_response = pdf_chain.invoke(user_input)
        except Exception as e:
            print("PDF chain error:", e)
            pdf_response = ""

    def is_irrelevant(response):
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in IRRELEVANT_PHRASES)

    if pdf_response and len(pdf_response.strip()) > 30 and not is_irrelevant(pdf_response):
        return pdf_response.strip()

    general_result = chat_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    return general_result.content.strip()
