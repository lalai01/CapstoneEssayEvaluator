import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import ocr_utils
import evaluator
from supabase_client import supabase

app = FastAPI(title="AI Essay Evaluator API")

# Allow all origins for production (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    text: str
    evaluation_type: str = "analytic"

class EvaluationResponse(BaseModel):
    scores: dict
    feedback: str

class OCRResponse(BaseModel):
    text: str
    confidence: float
    method: str
    page_count: Optional[int] = None

class KnowledgeEntry(BaseModel):
    essay: str
    grammar: int
    coherence: int
    content: int
    feedback: str
    eval_type: str
    accepted: bool = False
    satisfaction: int = 5
    teacher_feedback: Optional[str] = None

@app.post("/ocr", response_model=OCRResponse)
async def ocr_from_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']:
        raise HTTPException(400, "Unsupported file type")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        if suffix == '.pdf':
            if not ocr_utils.tesseract_found:
                raise HTTPException(500, "Tesseract OCR not installed")
            text, page_count = ocr_utils.process_pdf_document(tmp_path)
            confidence = 75.0
            method = f"Tesseract OCR (PDF, {page_count} pages)"
            return OCRResponse(text=text, confidence=confidence, method=method, page_count=page_count)
        else:
            if not ocr_utils.tesseract_found:
                raise HTTPException(500, "Tesseract OCR not installed")
            text, confidence = ocr_utils.extract_text_from_image(tmp_path, preprocessing=True)
            method = "Tesseract OCR (Image)"
            return OCRResponse(text=text, confidence=confidence, method=method)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_essay(req: EvaluationRequest):
    try:
        scores, feedback = evaluator.evaluate_essay(req.text, req.evaluation_type)
        return EvaluationResponse(scores=scores, feedback=feedback)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/knowledge")
def save_knowledge(entry: KnowledgeEntry):
    try:
        data = entry.dict()
        result = supabase.table("knowledge_base").insert(data).execute()
        return {"id": result.data[0]["id"]}
    except Exception as e:
        raise HTTPException(500, f"Supabase error: {str(e)}")

@app.get("/knowledge")
def list_knowledge(limit: int = 50):
    try:
        result = supabase.table("knowledge_base").select("*").order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/knowledge/{id}")
def get_knowledge(id: int):
    result = supabase.table("knowledge_base").select("*").eq("id", id).execute()
    if result.data:
        return result.data[0]
    raise HTTPException(404, "Not found")

@app.get("/rubric")
def get_rubric():
    return evaluator.RUBRIC

@app.get("/suggestions")
def get_suggestions():
    return evaluator.SUGGESTION_GUIDE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)