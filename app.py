import streamlit as st
import requests
import pandas as pd
from datetime import datetime as dt

st.set_page_config(
    page_title="ApplyEdge AI",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #1a1a2e; border-right: 1px solid #667eea; }
    p, span, label, div { color: #ffffff !important; }
    h1, h2, h3, h4 { color: #667eea !important; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: bold !important; font-size: 16px !important;
    }
    .stTextArea textarea {
        background-color: #16213e !important; color: #ffffff !important;
        border: 1px solid #667eea !important; border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] {
        background-color: #16213e !important;
        border: 2px dashed #667eea !important; border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] * { color: #ffffff !important; }
    [data-testid="stMetric"] {
        background-color: #16213e !important; border: 1px solid #667eea !important;
        border-radius: 10px !important; padding: 1rem !important;
    }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e !important; color: #ffffff !important;
        border-radius: 10px 10px 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    .stRadio label { color: #ffffff !important; }
    .stCaption { color: #aaaaaa !important; }
    .badge {
        display: inline-block; background: #16213e; border: 1px solid #667eea;
        color: #667eea !important; padding: 3px 10px;
        border-radius: 20px; font-size: 12px; margin: 3px;
    }
</style>
""", unsafe_allow_html=True)

BACKEND = "http://127.0.0.1:8000"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 ApplyEdge AI")
    st.markdown("**Smart Job Application Assistant**")
    st.markdown("---")
    st.markdown("**TECH STACK**")
    st.markdown("""
    <span class='badge'>Pinecone</span>
    <span class='badge'>Groq LLM</span>
    <span class='badge'>FastAPI</span>
    <span class='badge'>LLaMA</span>
    <span class='badge'>HuggingFace</span>
    <span class='badge'>Python</span>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Session History")
    if st.button("🔄 Refresh History"):
        try:
            r = requests.get(f"{BACKEND}/history", timeout=10)
            if r.status_code == 200:
                history = r.json().get("history", [])
                if history:
                    df = pd.DataFrame(history)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No sessions yet.")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("---")
    st.markdown("<p style='color:#00ff88;'>● Backend Connected</p>", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0;'>
    <h1 style='font-size:3rem; background:linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    🎯 ApplyEdge AI
    </h1>
    <p style='color:#aaaaaa; font-size:1.2rem;'>
    Smart Job Application Assistant using RAG
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Upload Resume Only", "🔍 Analyze Resume vs Job"])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📄 Upload Resume to Pinecone")
    st.info("Store your resume embeddings in vector database without running analysis.")

    upload_file = st.file_uploader(
        "Upload Resume (PDF or TXT)",
        type=["pdf", "txt"],
        key="upload_only"
    )

    if st.button("📤 Upload Resume", key="btn_upload"):
        if upload_file is None:
            st.error("❌ Please upload a resume file!")
        else:
            with st.spinner("⚡ Uploading and storing embeddings in Pinecone..."):
                try:
                    files = {
                        "resume": (
                            upload_file.name,
                            upload_file.getvalue(),
                            "application/pdf" if upload_file.name.endswith(".pdf") else "text/plain"
                        )
                    }
                    r = requests.post(f"{BACKEND}/upload-resume", files=files, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"✅ {data['message']}")
                        col1, col2 = st.columns(2)
                        col1.metric("📦 Chunks Stored", data["chunks_stored"])
                        col2.metric("🗄️ Pinecone", "✅ Stored")
                    else:
                        st.error(f"❌ Error: {r.text}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📄 Upload Resume")
    resume_file = st.file_uploader(
        "Upload Resume (PDF or TXT)",
        type=["pdf", "txt"],
        key="analyze_resume"
    )

    st.markdown("### 📝 Job Description")
    jd_mode = st.radio(
        "How to provide job description?",
        ["📋 Paste Text", "📁 Upload TXT File"],
        horizontal=True
    )

    job_description = ""
    if jd_mode == "📋 Paste Text":
        job_description = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="Copy and paste the job description here..."
        )
    else:
        jd_file = st.file_uploader(
            "Upload Job Description (.txt)",
            type=["txt"],
            key="jd_file"
        )
        if jd_file:
            job_description = jd_file.getvalue().decode("utf-8")
            st.success(f"✅ Loaded: {jd_file.name}")
            with st.expander("Preview"):
                st.text(job_description[:500])

    st.markdown("---")

    if st.button("🚀 Analyze My Resume", key="btn_analyze"):
        if resume_file is None:
            st.error("❌ Please upload your resume!")
        elif not job_description.strip():
            st.error("❌ Please provide a job description!")
        else:
            with st.spinner("🤖 AI is analyzing your resume using RAG..."):
                try:
                    mime = "application/pdf" if resume_file.name.endswith(".pdf") else "text/plain"
                    files = {"resume": (resume_file.name, resume_file.getvalue(), mime)}
                    data = {"job_description": job_description}

                    r = requests.post(
                        f"{BACKEND}/analyze-match",
                        files=files,
                        data=data,
                        timeout=90
                    )

                    if r.status_code == 200:
                        result = r.json()
                        analysis = result["analysis"]

                        st.markdown("---")
                        st.success("✅ Analysis Complete!")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("🎯 ATS Score", f"{result.get('ats_score','N/A')}/100")
                        col2.metric("📦 Chunks Stored", result.get("chunks_stored", "-"))
                        col3.metric("🔍 RAG Used", "✅ Yes" if result.get("rag_used") else "❌ No")

                        st.markdown("---")
                        st.markdown("### 📊 Full ATS Analysis Report")
                        st.markdown(analysis)

                        st.markdown("---")
                        st.download_button(
                            label="⬇️ Download Report (.txt)",
                            data=analysis,
                            file_name=f"applyedge_report_{dt.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(f"❌ Backend Error: {r.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend se connect nahi ho raha!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("Made by Romana Anwar | ApplyEdge AI 🎯")
 