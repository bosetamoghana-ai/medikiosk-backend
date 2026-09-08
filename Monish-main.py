# main.py
import json
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import shared database
from database import get_db, ConsultationRecord

# Import your department routers
from allopathy_api import router as allopathy_router
from ayurveda_api import router as ayurveda_router

app = FastAPI(
    title="MediKiosk Unified API",
    description="Backend API for Smart India Hackathon PS 47",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount the Department APIs
app.include_router(allopathy_router, tags=["Allopathy"])
app.include_router(ayurveda_router, tags=["Ayurveda"])

# 2. Unified Doctor Portal Endpoints
class DoctorReviewRequest(BaseModel):
    doctor_notes: str

@app.get("/api/doctor/queue", tags=["Doctor Portal"])
async def get_doctor_active_queue(department: Optional[str] = None, db: Session = Depends(get_db)):
    """Loads all waiting patients in FIFO order, with optional department filtering."""
    query = db.query(ConsultationRecord).filter(ConsultationRecord.status == "pending_review")
    
    if department:
        query = query.filter(ConsultationRecord.department == department)
        
    records = query.order_by(ConsultationRecord.created_at.asc()).all()
    
    return [
        {
            "record_id": r.id,
            "patient_id": r.patient_id,
            "session_id": r.session_id,
            "department": r.department,
            "timestamp": r.created_at.isoformat(),
            "clinical_summary": json.loads(r.clinical_summary) if r.clinical_summary else {}
        }
        for r in records
    ]

@app.get("/api/doctor/record/{record_id}", tags=["Doctor Portal"])
async def get_detailed_record(record_id: int, db: Session = Depends(get_db)):
    """Loads full intake details when a doctor opens a patient file."""
    record = db.query(ConsultationRecord).filter(ConsultationRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Consultation record not found.")

    return {
        "record_id": record.id,
        "patient_id": record.patient_id,
        "session_id": record.session_id,
        "department": record.department,
        "created_at": record.created_at.isoformat(),
        "status": record.status,
        "raw_transcript": record.raw_transcript,
        "clinical_summary": json.loads(record.clinical_summary) if record.clinical_summary else {},
        "doctor_notes": record.doctor_notes
    }

@app.post("/api/doctor/record/{record_id}/complete", tags=["Doctor Portal"])
async def complete_consultation(record_id: int, req: DoctorReviewRequest, db: Session = Depends(get_db)):
    """Doctor marks the consultation as finished and appends prescription/notes."""
    record = db.query(ConsultationRecord).filter(ConsultationRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Consultation record not found.")

    record.doctor_notes = req.doctor_notes
    record.status = "completed"
    db.commit()

    return {"status": "success", "message": "Consultation marked as completed."}


if __name__ == "__main__":
    import uvicorn
    # Run this single file to launch the entire application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)