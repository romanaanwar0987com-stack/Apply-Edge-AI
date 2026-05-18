import streamlit as st
import requests

st.set_page_config(
    page_title="ApplyEdge AI",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 ApplyEdge AI")
st.subheader("Smart Job Application Assistant")
st.markdown("---")

st.markdown("### 📄 Upload Your Resume")
resume_file = st.file_uploader("Choose your resume (PDF only)", type=["pdf"])

st.markdown("### 📝 Paste Job Description")
job_description = st.text_area(
    "Copy and paste the job description here",
    height=200,
    placeholder="Paste the job description here..."
)

st.markdown("---")

if st.button("🚀 Analyze My Resume", use_container_width=True):
    if resume_file is None:
        st.error("❌ Please upload your resume!")
    elif not job_description:
        st.error("❌ Please paste a job description!")
    else:
        with st.spinner("🤖 AI is analyzing your resume..."):
            try:
                files = {"resume": (resume_file.name, resume_file.getvalue(), "application/pdf")}
                data = {"job_description": job_description}
                
                response = requests.post(
                    "http://127.0.0.1:8000/analyze",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.markdown("---")
                    st.success("✅ Analysis Complete!")
                    st.markdown("### 📊 Your ATS Analysis Report")
                    st.markdown(result["analysis"])
                else:
                    st.error(f"❌ Backend Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend se connect nahi ho raha! Check karo ke backend chal raha hai.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("Made by Romana Anwar | ApplyEdge AI")