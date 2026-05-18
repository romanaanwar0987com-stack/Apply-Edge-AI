from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
from groq import Groq
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"PDF Error: {str(e)}"

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

def retrieve_chunks(query, top_k=3):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        chunks = [match.metadata["text"] for match in results.matches]
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"Retrieve Error: {str(e)}")
        return ""

@app.get("/")
def home():
    return {"message": "ApplyEdge AI with RAG is running!"}

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        # Step 1: PDF text nikalo
        pdf_bytes = await resume.read()
        resume_text = extract_text_from_pdf(pdf_bytes)
        print(f"Resume text length: {len(resume_text)}")

        # Step 2: Chunks banao
        chunks = chunk_text(resume_text)
        print(f"Total chunks: {len(chunks)}")

        # Step 3: Pinecone mein store karo
        doc_id = "resume_" + resume.filename.replace(".pdf", "").replace(" ", "_")
        stored = store_in_pinecone(chunks, doc_id)
        print(f"Stored in Pinecone: {stored}")

        # Step 4: Relevant chunks retrieve karo
        context = retrieve_chunks(job_description)
        print(f"Context retrieved: {len(context)} chars")

        # Step 5: AI se analyze karo
        prompt = f"""
You are an expert ATS analyzer.

RESUME CONTENT (via RAG):
{context if context else resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:2000]}

Provide:
1. ATS Match Score (out of 100)
2. Missing Keywords (5-10)
3. Strengths (3-5)
4. Improvement Suggestions (3-5)
"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "analysis": response.choices[0].message.content,
            "chunks_stored": len(chunks),
            "rag_used": bool(context)
        }

    except Exception as e:
        print(f"MAIN ERROR: {str(e)}")
        raise