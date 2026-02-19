# -*- coding: utf-8 -*-
import streamlit as st
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from datetime import datetime
from openai import OpenAI

# RAG imports
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Trustworthy AI Explainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
@st.cache_resource
def load_llm(api_key: str):
    return OpenAI(api_key=api_key)

# =============================================================================
# PDF PROCESSING FOR RAG
# =============================================================================
@st.cache_resource
def process_pdf(uploaded_file, api_key):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(docs, embeddings)

    return vectordb

# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================
@st.cache_data(ttl=3600)
def load_feedback_data() -> pd.DataFrame:
    if 'feedback_db' in st.session_state:
        return pd.DataFrame(st.session_state.feedback_db)
    return pd.DataFrame(columns=["timestamp", "message", "response", "rating", "comment"])

@st.cache_data
def compute_explanation(_input_text: str, _response: str) -> Dict:
    return {
        "input_tokens": len(_input_text.split()),
        "response_tokens": len(_response.split()),
        "confidence": 0.85,
        "top_features": ["Input length","Semantic similarity","Context relevance","Prompt clarity"],
        "explanation": "Replace with SHAP / LIME analysis"
    }

# =============================================================================
# SESSION STATE
# =============================================================================
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_db" not in st.session_state:
        st.session_state.feedback_db = []
    if "preferences" not in st.session_state:
        st.session_state.preferences = {"temperature":0.7,"max_tokens":500,"system_prompt":"You are a helpful, transparent AI assistant."}
    if "current_explanation" not in st.session_state:
        st.session_state.current_explanation = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = {"total_messages":0,"avg_response_time":0.0,"total_feedback":0}
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

# =============================================================================
# CORE LOGIC
# =============================================================================
def generate_response(message: str, temperature: float, api_key: str):
    start = time.time()
    client = load_llm(api_key)

    # Recuperar contexto del PDF si existe
    context = ""
    if st.session_state.vectordb:
        docs = st.session_state.vectordb.similarity_search(message, k=3)
        context = "\n".join([d.page_content for d in docs])

    messages = [
        {"role": "system", "content": "Responde solo con base en el siguiente contexto:\n" + context},
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=st.session_state.preferences["max_tokens"]
    )

    output = response.choices[0].message.content
    explanation = compute_explanation(message, output)
    elapsed = time.time() - start

    st.session_state.metrics["total_messages"] += 1
    n = st.session_state.metrics["total_messages"]
    st.session_state.metrics["avg_response_time"] = (((n-1)*st.session_state.metrics["avg_response_time"]+elapsed)/n)

    return output, explanation, elapsed

def save_feedback(message, response, rating, comment):
    st.session_state.feedback_db.append({"timestamp":datetime.now(),"message":message,"response":response,"rating":rating,"comment":comment})
    st.session_state.metrics["total_feedback"] += 1
    load_feedback_data.clear()

# =============================================================================
# PAGE: CHAT
# =============================================================================
def page_chat(api_key):
    st.title("💬 AI Chat with Explainability")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.preferences["temperature"] = st.slider("Temperature",0.0,2.0,st.session_state.preferences["temperature"],0.1)
        st.session_state.preferences["max_tokens"] = st.slider("Max tokens",50,2000,st.session_state.preferences["max_tokens"],50)
        st.session_state.preferences["system_prompt"] = st.text_area("System Prompt",st.session_state.preferences["system_prompt"],height=120)

        uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
        if uploaded_file and not st.session_state.vectordb:
            st.session_state.vectordb = process_pdf(uploaded_file, api_key)
            st.success("PDF procesado correctamente ✅")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.current_explanation = None
            st.rerun()

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Conversation")
        chat_box = st.container(height=420)
        with chat_box:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role":"user","content":prompt})
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(prompt)
            with st.spinner("Thinking..."):
                response, explanation, rt = generate_response(prompt, st.session_state.preferences["temperature"], api_key)
            st.session_state.messages.append({"role":"assistant","content":response})
            st.session_state.current_explanation = {"input":prompt,"output":response,"details":explanation,"response_time":rt}
            with chat_box:
                with st.chat_message("assistant"):
                    st.markdown(response)
            st.rerun()

    with col2:
        st.subheader("Explainability")
        if st.session_state.current_explanation:
            exp = st.session_state.current_explanation
            st.metric("Confidence", f"{exp['details']['confidence']:.2f}")
            st.metric("Response Time", f"{exp['response_time']:.2f}s")
            st.markdown("**Key Factors:**")
            for f in exp["details"]["top_features"]:
                st.markdown(f"- {f}")
            st.divider()
            rating = st.radio("Rate response", ["👍 Helpful","👎 Not Helpful"])
            comment = st.text_area("Comment (optional)")
            if st.button("Submit Feedback"):
                save_feedback(exp["input"],exp["output"],rating,comment)
                st.success("Feedback saved!")
        else:
            st.info("Sube un PDF y haz una pregunta para ver explicabilidad.")

# =============================================================================
# OTHER PAGES
# =============================================================================
def page_feedback():
    st.title("📊 Feedback Dashboard")
    df = load_feedback_data()
    if df.empty:
        st.info("No feedback yet.")
        return
    st.dataframe(df, use_container_width=True)

def page_monitoring():
    st.title("📈 Monitoring")
    m = st.session_state.metrics
    st.metric("Total Messages", m["total_messages"])
    st.metric("Avg Response Time", f"{m['avg_response_time']:.2f}s")
    st.metric("Total Feedback", m["total_feedback"])

def page_documentation():
    st.title("📚 Documentation")
    st.markdown("Trustworthy AI Explainer – Module 15")

# =============================================================================
# MAIN
# =============================================================================
def main():
    initialize_session_state()
    api_key = st.sidebar.text_input("Introduce tu OpenAI API Key", type="password")
    if not api_key:
        st.warning("Por favor introduce tu API Key en la barra lateral.")
        return

    with st.sidebar:
        page = st.radio("Navigation", ["💬 Chat","📊 Feedback","📈 Monitoring","📚 Documentation"])

    if page == "💬 Chat":
        page_chat(api_key)
    elif page == "📊 Feedback":
        page_feedback()
    elif page == "📈 Monitoring":
        page_monitoring()
    elif page == "📚 Documentation":
        page_documentation()

if __name__ == "__main__":
    main()
