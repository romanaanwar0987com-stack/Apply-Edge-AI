from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
from groq import Groq
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import csv
from datetime import datetime
import re

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("applyedge")
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

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
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
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
        print(f"Pinecone Error: {str(e)}")
        return False

def retrieve_chunks(query, top_k=5):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        matches = sorted(results.matches, key=lambda x: x.score, reverse=True)
        chunks = [f"[Score: {m.score:.2f}] {m.metadata['text']}" for m in matches]
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"Retrieve Error: {str(e)}")
        return ""

def save_to_csv(resume_name, job_desc_snippet, chunks_count, ats_score):
    csv_file = "session_history.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Resume", "Job Description", "Chunks", "ATS Score"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            resume_name,
            job_desc_snippet[:50],
            chunks_count,
            ats_score,
        ])

def parse_ats_score(analysis_text):
    match = re.search(r'ATS\s+Match\s+Score[^\d]*(\d{1,3})', analysis_text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'(\d{1,3})\s*/\s*100', analysis_text)
    if match:
        return match.group(1)
    return "N/A"

@app.get("/")
def home():
    return {"message": "ApplyEdge AI with RAG is running!"}

@app.post("/upload-resume")
async def upload_resume(resume: UploadFile = File(...)):
    try:
        file_bytes = await resume.read()
        resume_text = extract_text(file_bytes, resume.filename)
        chunks = chunk_text(resume_text)
        doc_id = "resume_" + resume.filename.replace(".pdf","").replace(".txt","").replace(" ","_")
        stored = store_in_pinecone(chunks, doc_id)
        return {
            "message": "Resume uploaded successfully!",
            "filename": resume.filename,
            "chunks_stored": len(chunks),
            "doc_id": doc_id,
            "pinecone_stored": stored,
        }
    except Exception as e:
        print(f"UPLOAD ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-match")
async def analyze_match(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    try:
        file_bytes = await resume.read()
        resume_text = extract_text(file_bytes, resume.filename)
        print(f"Resume text length: {len(resume_text)}")

        chunks = chunk_text(resume_text)
        print(f"Total chunks: {len(chunks)}")

        doc_id = "resume_" + resume.filename.replace(".pdf","").replace(".txt","").replace(" ","_")
        stored = store_in_pinecone(chunks, doc_id)
        print(f"Stored in Pinecone: {stored}")

        context = retrieve_chunks(job_description)
        print(f"Context retrieved: {len(context)} chars")

        resume_content = context if context else resume_text[:2000]

        prompt = f"""
You are an expert ATS analyzer and career coach.

RESUME CONTENT (via RAG):
{resume_content}

JOB DESCRIPTION:
{job_description[:2000]}

Please provide a structured response with EXACTLY these 5 sections:

1. **ATS Match Score**: X/100
   (One sentence explaining the score)

2. **Missing Keywords** (list 5-10):
   - keyword1
   - keyword2

3. **Strengths** (list 3-5):
   - strength1
   - strength2

4. **Improvement Suggestions** (list 3-5):
   - suggestion1
   - suggestion2

5. **Rewritten Resume Bullet Points**:
   For each weak bullet in the resume, provide a stronger version.
   Format:
   ❌ Original: [original bullet]
   ✅ Rewritten: [improved bullet using keywords from job description]
   (Provide at least 3 rewritten bullets)
"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        analysis_text = response.choices[0].message.content
        ats_score = parse_ats_score(analysis_text)
        save_to_csv(resume.filename, job_description, len(chunks), ats_score)

        return {
            "analysis": analysis_text,
            "chunks_stored": len(chunks),
            "rag_used": bool(context),
            "ats_score": ats_score,
        }

    except Exception as e:
        print(f"MAIN ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    return await analyze_match(resume=resume, job_description=job_description)

@app.get("/history")
def get_history():
    csv_file = "session_history.csv"
    if not os.path.exists(csv_file):
        return {"history": [], "message": "No sessions yet."}
    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return {"history": rows}