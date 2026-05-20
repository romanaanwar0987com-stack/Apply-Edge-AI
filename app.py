import streamlit as st
import pandas as pd
from datetime import datetime as dt
import PyPDF2
import io
import os
import csv
import re
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

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
        border-radius: 10px !important; font-weight: bold !important;
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
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e !important; color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    .badge {
        display: inline-block; background: #16213e; border: 1px solid #667eea;
        color: #667eea !important; padding: 3px 10px;
        border-radius: 20px; font-size: 12px; margin: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialize clients ────────────────────────────────────────────────────────
@st.cache_resource
def load_clients():
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("applyedge")
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    return groq_client, index, model

groq_client, index, embedding_model = load_clients()

# ── Helper functions ──────────────────────────────────────────────────────────
def extract_text(file_bytes, filename):
    try:
        if filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        elif filename.endswith(".txt"):
            return file_bytes.decode("utf-8")
        else:
            return "Unsupported format"
    except Exception as e:
        return f"Error: {str(e)}"

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks if chunks else [text]

def store_in_pinecone(chunks, doc_id):
    try:
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            vectors.append({
                "id": f"{doc_id}_{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            })
        index.upsert(vectors=vectors)
        return True
    except Exception as e:
        st.error(f"Pinecone Error: {str(e)}")
        return False

def retrieve_chunks(query, top_k=5):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return "\n\n".join([f"[Score: {m.score:.2f}] {m.metadata['text']}" for m in matches])
    except Exception as e:
        return ""

def parse_ats_score(text):
    match = re.search(r'ATS\s+Match\s+Score[^\d]*(\d{1,3})', text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'(\d{1,3})\s*/\s*100', text)
    if match:
        return match.group(1)
    return "N/A"

def save_to_csv(resume_name, job_desc, chunks_count, ats_score):
    csv_file = "session_history.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Resume", "Job Description", "Chunks", "ATS Score"])
        writer.writerow([
            dt.now().strftime("%Y-%m-%d"),
            dt.now().strftime("%H:%M:%S"),
            resume_name, job_desc[:50], chunks_count, ats_score
        ])

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
        if os.path.exists("session_history.csv"):
            df = pd.read_csv("session_history.csv")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No sessions yet.")
    st.markdown("---")
    st.markdown("<p style='color:#00ff88;'>● Backend Connected</p>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0;'>
    <h1 style='font-size:3rem; background:linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    🎯 ApplyEdge AI
    </h1>
    <p style='color:#aaaaaa; font-size:1.2rem;'>Smart Job Application Assistant using RAG</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2 = st.tabs(["📤 Upload Resume Only", "🔍 Analyze Resume vs Job"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📄 Upload Resume to Pinecone")
    upload_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"], key="upload_only")
    if st.button("📤 Upload Resume", key="btn_upload"):
        if upload_file is None:
            st.error("❌ Please upload a resume file!")
        else:
            with st.spinner("⚡ Storing embeddings in Pinecone..."):
                file_bytes = upload_file.getvalue()
                resume_text = extract_text(file_bytes, upload_file.name)
                chunks = chunk_text(resume_text)
                doc_id = "resume_" + upload_file.name.replace(".pdf","").replace(".txt","").replace(" ","_")
                stored = store_in_pinecone(chunks, doc_id)
                if stored:
                    st.success("✅ Resume uploaded successfully!")
                    col1, col2 = st.columns(2)
                    col1.metric("📦 Chunks Stored", len(chunks))
                    col2.metric("🗄️ Pinecone", "✅ Stored")

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📄 Upload Resume")
    resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"], key="analyze_resume")

    st.markdown("### 📝 Job Description")
    jd_mode = st.radio("How to provide job description?", ["📋 Paste Text", "📁 Upload TXT File"], horizontal=True)

    job_description = ""
    if jd_mode == "📋 Paste Text":
        job_description = st.text_area("Paste job description here", height=200, placeholder="Paste here...")
    else:
        jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"], key="jd_file")
        if jd_file:
            job_description = jd_file.getvalue().decode("utf-8")
            st.success(f"✅ Loaded: {jd_file.name}")

    st.markdown("---")

    if st.button("🚀 Analyze My Resume", key="btn_analyze"):
        if resume_file is None:
            st.error("❌ Please upload your resume!")
        elif not job_description.strip():
            st.error("❌ Please provide a job description!")
        else:
            with st.spinner("🤖 AI is analyzing your resume using RAG..."):
                try:
                    file_bytes = resume_file.getvalue()
                    resume_text = extract_text(file_bytes, resume_file.name)
                    chunks = chunk_text(resume_text)
                    doc_id = "resume_" + resume_file.name.replace(".pdf","").replace(".txt","").replace(" ","_")
                    store_in_pinecone(chunks, doc_id)
                    context = retrieve_chunks(job_description)
                    resume_content = context if context else resume_text[:2000]

                    prompt = f"""
You are an expert ATS analyzer and career coach.

RESUME CONTENT (via RAG):
{resume_content}

JOB DESCRIPTION:
{job_description[:2000]}

Please provide a structured response with EXACTLY these 5 sections:

1. **ATS Match Score**: X/100
2. **Missing Keywords** (list 5-10)
3. **Strengths** (list 3-5)
4. **Improvement Suggestions** (list 3-5)
5. **Rewritten Resume Bullet Points**:
   ❌ Original: [original bullet]
   ✅ Rewritten: [improved bullet]
   (Provide at least 3 rewritten bullets)
"""
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    analysis = response.choices[0].message.content
                    ats_score = parse_ats_score(analysis)
                    save_to_csv(resume_file.name, job_description, len(chunks), ats_score)

                    st.success("✅ Analysis Complete!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🎯 ATS Score", f"{ats_score}/100")
                    col2.metric("📦 Chunks Stored", len(chunks))
                    col3.metric("🔍 RAG Used", "✅ Yes" if context else "❌ No")

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
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("Made by Romana Anwar | ApplyEdge AI 🎯")