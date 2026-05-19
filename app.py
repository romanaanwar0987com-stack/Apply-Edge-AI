import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="ApplyEdge AI",
    page_icon="🎯",
    layout="centered"
)

BACKEND = "http://127.0.0.1:8000"

# ── Sidebar: Session History ────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Session History")
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
            else:
                st.error("Could not load history.")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("---")
    st.caption("History auto-saves after every analysis ✅")

# ── Main UI ─────────────────────────────────────────────────────────────────
st.title("🎯 ApplyEdge AI")
st.subheader("Smart Job Application Assistant using RAG")
st.markdown("---")

# ── Tabs: Upload-only  vs  Full Analyze ─────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Upload Resume Only", "🔍 Analyze Resume vs Job"])

# ─── TAB 1: /upload-resume ──────────────────────────────────────────────────
with tab1:
    st.markdown("### 📄 Upload Resume to Pinecone")
    st.info("Use this to store your resume embeddings without running a full analysis.")

    upload_file = st.file_uploader(
        "Upload Resume (PDF or TXT)",   # ✅ FIX 1: TXT allowed
        type=["pdf", "txt"],
        key="upload_only"
    )

    if st.button("📤 Upload Resume", use_container_width=True, key="btn_upload"):
        if upload_file is None:
            st.error("❌ Please upload a resume file!")
        else:
            with st.spinner("Uploading and storing embeddings..."):
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
                        st.json({
                            "Filename": data["filename"],
                            "Chunks Stored": data["chunks_stored"],
                            "Doc ID": data["doc_id"],
                            "Pinecone": "✅ Stored" if data["pinecone_stored"] else "❌ Failed",
                        })
                    else:
                        st.error(f"❌ Error {r.status_code}: {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend se connect nahi ho raha! Backend chal raha hai?")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ─── TAB 2: /analyze-match ──────────────────────────────────────────────────
with tab2:
    st.markdown("### 📄 Upload Resume")
    resume_file = st.file_uploader(
        "Upload Resume (PDF or TXT)",   # ✅ FIX 1: TXT allowed
        type=["pdf", "txt"],
        key="analyze_resume"
    )

    st.markdown("### 📝 Job Description")

    # ✅ FIX 1: Job description bhi file se upload ho sakti hai
    jd_input_mode = st.radio(
        "How do you want to provide the job description?",
        ["📋 Paste Text", "📁 Upload TXT File"],
        horizontal=True
    )

    job_description = ""

    if jd_input_mode == "📋 Paste Text":
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="Copy and paste the job description here..."
        )
    else:
        jd_file = st.file_uploader(
            "Upload Job Description (.txt)",
            type=["txt"],
            key="jd_file"
        )
        if jd_file is not None:
            job_description = jd_file.getvalue().decode("utf-8")
            st.success(f"✅ Loaded: {jd_file.name}")
            with st.expander("Preview Job Description"):
                st.text(job_description[:500] + ("..." if len(job_description) > 500 else ""))

    st.markdown("---")

    if st.button("🚀 Analyze My Resume", use_container_width=True, key="btn_analyze"):
        if resume_file is None:
            st.error("❌ Please upload your resume!")
        elif not job_description.strip():
            st.error("❌ Please provide a job description!")
        else:
            with st.spinner("🤖 AI is analyzing your resume..."):
                try:
                    mime = "application/pdf" if resume_file.name.endswith(".pdf") else "text/plain"
                    files = {"resume": (resume_file.name, resume_file.getvalue(), mime)}
                    data  = {"job_description": job_description}

                    # ✅ FIX 4: Calls /analyze-match route
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

                        # ── Score badge ───────────────────────────────────
                        score = result.get("ats_score", "N/A")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("🎯 ATS Score", f"{score}/100")
                        col2.metric("📦 Chunks Stored", result.get("chunks_stored", "-"))
                        col3.metric("🔍 RAG Used", "✅ Yes" if result.get("rag_used") else "❌ No")

                        st.markdown("---")

                        # ── Full analysis ─────────────────────────────────
                        st.markdown("### 📊 Full ATS Analysis Report")
                        st.markdown(analysis)

                        # ── Bullet Rewrites section highlight ─────────────
                        # ✅ FIX 3: Visual separation for rewrites
                        if "Rewritten Resume Bullet" in analysis:
                            st.markdown("---")
                            st.markdown("### ✏️ Resume Bullet Point Rewrites")
                            lines = analysis.split("\n")
                            in_rewrite = False
                            rewrite_lines = []
                            for line in lines:
                                if "Rewritten Resume Bullet" in line:
                                    in_rewrite = True
                                if in_rewrite:
                                    rewrite_lines.append(line)
                            if rewrite_lines:
                                st.markdown("\n".join(rewrite_lines[1:]))  # skip the header (already shown)

                        # ── Download report ───────────────────────────────
                        from datetime import datetime as dt
                        st.download_button(
                            label="⬇️ Download Report (.txt)",
                            data=analysis,
                            file_name=f"applyedge_report_{dt.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )

                    else:
                        st.error(f"❌ Backend Error {r.status_code}: {r.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend se connect nahi ho raha! Check karo ke backend chal raha hai.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("Made by Romana Anwar | ApplyEdge AI 🎯")