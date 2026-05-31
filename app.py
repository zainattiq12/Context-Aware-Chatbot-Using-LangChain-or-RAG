import streamlit as st

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Context-Aware RAG Chatbot")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=200
)

llm = HuggingFacePipeline(
    pipeline=generator
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(
        msg["content"]
    )

question = st.chat_input(
    "Ask a question..."
)

if question:

    st.chat_message("user").write(question)

    response = qa_chain.invoke(
        {"question": question}
    )

    answer = response["answer"]

    st.chat_message("assistant").write(
        answer
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )