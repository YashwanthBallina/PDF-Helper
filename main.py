from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google import genai 
import PyPDF2
import io

app = FastAPI()

# Enable CORS so your frontend can talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize the Gemini Client
# IMPORTANT: Paste your active API key here!
client = genai.Client(api_key="AIzaSyAHpB55CiU24FyxGQkdNlMCO406BkyztiU")

def extract_text_from_pdf(file_bytes):
    """Helper function to read the PDF and pull out all the text."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

@app.post("/api/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pdf_text = extract_text_from_pdf(content)
        prompt = f"You are an expert tutor. Summarize the following academic text, explaining every major concept clearly so a student can understand it without reading the original document:\n\n{pdf_text}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"status": "success", "message": "Summary generated successfully!", "result": response.text}
    except Exception as e:
        return {"status": "error", "message": f"AI Error: {str(e)}"}

@app.post("/api/shortern")
async def shortern_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pdf_text = extract_text_from_pdf(content)
        prompt = f"Extract only the essential elements from this text: key formulas, fundamental definitions, and crucial brief takeaways. Format it as a concise 1-page cheat sheet using bullet points:\n\n{pdf_text}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"status": "success", "message": "Cheat sheet generated successfully!", "result": response.text}
    except Exception as e:
        return {"status": "error", "message": f"AI Error: {str(e)}"}

@app.post("/api/explain")
async def explain_topic(topic: str = Form(...), file: UploadFile = File(...)):
    try:
        content = await file.read()
        pdf_text = extract_text_from_pdf(content)
        prompt = f"Based ONLY on the provided document text below, explain the topic '{topic}' in deep detail. If the topic is not in the text, say so.\n\nText:\n{pdf_text}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"status": "success", "topic": topic, "message": "Explanation generated.", "result": response.text}
    except Exception as e:
        return {"status": "error", "message": f"AI Error: {str(e)}"}

# === NEW FEATURE: QUIZ GENERATOR ===
@app.post("/api/quiz")
async def generate_quiz(file: UploadFile = File(...), num_questions: int = Form(10)):
    try:
        content = await file.read()
        pdf_text = extract_text_from_pdf(content)
        
        # We tell the AI exactly how many questions to make using the num_questions variable
        prompt = f"Based on the following text, generate exactly {num_questions} multiple-choice questions. Each question must have exactly 4 options (A, B, C, D) and you must clearly indicate the correct answer at the bottom of each question. Format the output neatly using Markdown.\n\nText:\n{pdf_text}"
        
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"status": "success", "message": f"{num_questions} questions generated.", "result": response.text}
        
    except Exception as e:
        return {"status": "error", "message": f"AI Error: {str(e)}"}