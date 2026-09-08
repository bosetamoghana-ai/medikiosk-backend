from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import json

from pipeline_coordinator import Member2Pipeline

# Initialize FastAPI
app = FastAPI(
    title="Member 2: Multimodal AI Pipeline",
    description="Voice and OCR processing for clinical triage",
    version="1.0.0"
)

# Initialize the pipeline (loads Whisper, EasyOCR, spaCy)
print("🚀 Starting Member 2 API Server...")
pipeline = Member2Pipeline()

# Create uploads folder
UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Member 2: Multimodal AI Pipeline Ready",
        "endpoints": {
            "voice": "/api/process-voice (POST)",
            "document": "/api/process-document (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
def health_check():
    """Check if all modules are loaded"""
    return {
        "status": "healthy",
        "voice_processor": "ready",
        "ocr_reader": "ready",
        "entity_extractor": "ready",
        "tts": "ready"
    }

@app.post("/api/process-voice")
async def process_voice(file: UploadFile = File(...)):
    """
    Upload an audio file and get transcription + LLM response + audio back
    
    Accepts: .wav, .mp3, .m4a
    Returns: {patient_input, llm_response, audio_file}
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📁 Saved voice file: {file_path}")
        
        # Process through pipeline
        result = pipeline.process_patient_voice(file_path)
        
        # Clean up
        os.remove(file_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": result
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/api/process-document")
async def process_document(file: UploadFile = File(...)):
    """
    Upload a prescription/discharge summary image and extract entities
    
    Accepts: .jpg, .png, .pdf
    Returns: {patient_info, symptoms, medications, vitals}
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📁 Saved document file: {file_path}")
        
        # Process through pipeline
        result = pipeline.process_prescription(file_path)
        
        # Clean up
        os.remove(file_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": result
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/api/chat")
async def chat(message: dict):
    """
    Send text directly to Member 1 LLM
    
    Body: {"text": "patient says..."}
    Returns: {"response": "doctor says..."}
    """
    try:
        patient_text = message.get("text", "")
        if not patient_text:
            raise ValueError("Empty message")
        
        # Call Member 1 LLM
        llm_response = pipeline._call_member1_llm(patient_text)
        
        # Convert to speech
        audio_result = pipeline.tts.text_to_speech(llm_response)
        
        return {
            "status": "success",
            "patient_input": patient_text,
            "llm_response": llm_response,
            "audio_file": audio_result.get("file")
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Member 2 API Server Starting...")
    print("="*60)
    print("📍 Open your browser: http://localhost:8000")
    print("📍 API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)