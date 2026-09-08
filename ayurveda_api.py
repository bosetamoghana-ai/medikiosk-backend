import os
import json
import random
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from safety import scan_for_red_flags

# Import shared database models (Ensure database.py exists)
from database import get_db, ConsultationRecord
from ayurveda_engine import ayurveda_prompt, summary_prompt, json_parser

# ---------------------------------------------------------------------------
# 1. Router & LLM Engine Setup
# ---------------------------------------------------------------------------
router = APIRouter()

llm = ChatOllama(model="phi3", temperature=0.3) 

chat_chain = ayurveda_prompt | llm
summary_chain = summary_prompt | llm | json_parser

# --- Department Routing Chain ---
ROUTING_PROMPT = """
You are an expert hospital routing assistant. Based on the patient's transcript, determine the most appropriate medical or Ayurvedic department for their consultation.

Return ONLY a valid JSON object with this exact structure:
{{
    "assigned_department": "Name of department (e.g., Kayachikitsa (General Medicine), Shalya Tantra (Surgery), Panchakarma, Orthopedics, Dermatology)",
    "reasoning": "Brief 1-sentence reason"
}}

Transcript:
{conversation_transcript}
"""
routing_prompt_template = ChatPromptTemplate.from_template(ROUTING_PROMPT)
routing_parser = JsonOutputParser()
routing_chain = routing_prompt_template | llm | routing_parser
# -------------------------------------

# Ephemeral in-memory stores
pending_otps: Dict[str, dict] = {} 
active_kiosk_sessions: Dict[str, List] = {}

# ---------------------------------------------------------------------------
# 2. Pydantic Schemas for Requests
# ---------------------------------------------------------------------------
class OTPRequest(BaseModel):
    patient_id: str

class OTPVerify(BaseModel):
    patient_id: str
    otp: str

class ChatTurnRequest(BaseModel):
    session_id: str
    message: str

class FinalizeIntakeRequest(BaseModel):
    patient_id: str
    session_id: str

def extract_text(content):
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return str(content)

# ---------------------------------------------------------------------------
# 3. Authentication Endpoints 
# ---------------------------------------------------------------------------
@router.post("/api/ayurveda/request-otp")
async def request_otp(req: OTPRequest):
    """Step 1: Patient enters ID to generate OTP."""
    generated_otp = str(random.randint(100000, 999999)) 
    
    pending_otps[req.patient_id] = {
        "otp": generated_otp,
        "department": "Ayurveda"
    }
    
    print(f"[SYSTEM] OTP for {req.patient_id} is {generated_otp}")
    return {"status": "success", "message": "OTP sent successfully."}

@router.post("/api/ayurveda/verify-otp")
async def verify_otp_and_start(req: OTPVerify):
    """Step 2: Patient enters OTP to initialize chat."""
    if req.patient_id not in pending_otps:
        raise HTTPException(status_code=404, detail="No pending OTP request found.")
        
    stored_data = pending_otps[req.patient_id]
    
    if stored_data["otp"] != req.otp:
        raise HTTPException(status_code=401, detail="Invalid OTP.")
        
    session_id = datetime.now().strftime(f"ayur_{req.patient_id}_%Y%m%d_%H%M%S")
    greeting = "Namaste. I am the Ayurvedic triage assistant. Could you tell me what symptoms or dosha imbalances brought you to the clinic today?"
    
    active_kiosk_sessions[session_id] = [AIMessage(content=greeting)]
    del pending_otps[req.patient_id]
    
    return {
        "status": "success",
        "session_id": session_id,
        "first_message": greeting
    }

# ---------------------------------------------------------------------------
# 4. Patient Portal Endpoints
# ---------------------------------------------------------------------------
@router.post("/api/ayurveda/chat")
async def handle_ayurveda_chat(req: ChatTurnRequest, db: Session = Depends(get_db)):
    if req.session_id not in active_kiosk_sessions:
        raise HTTPException(status_code=404, detail="Active intake session not found.")

    # 1. RUN 2-TIER SAFETY GUARDIAN
    red_flag = scan_for_red_flags(req.message)
    if red_flag:
        is_hard_emergency = red_flag.get("status") != "PROBE"

        if is_hard_emergency:
            # Escalate Ayurvedic intake directly to ER status
            emergency_record = ConsultationRecord(
                patient_id=req.session_id.split("_")[1] if "_" in req.session_id else "UNKNOWN",
                session_id=req.session_id,
                department="Emergency Room",
                raw_transcript=f"EMERGENCY TRIGGER: {red_flag.get('matched_trigger', 'Unknown')}\nPATIENT: {req.message}",
                clinical_summary=json.dumps(red_flag),
                status="EMERGENCY_ALERT"
            )
            db.add(emergency_record)
            db.commit()

            return {
                "red_flag_alert": True,
                "category": red_flag.get("category", "Medical Emergency"),
                "reply": red_flag["patient_instruction"]
            }
        else:
            # Probe Scenario
            history = active_kiosk_sessions[req.session_id]
            history.append(HumanMessage(content=req.message))
            history.append(AIMessage(content=red_flag["patient_instruction"]))
            
            return {
                "red_flag_alert": False, 
                "reply": red_flag["patient_instruction"]
            }

    # 2. NORMAL LLM FLOW (If no red flags)
    history = active_kiosk_sessions[req.session_id]
    
    try:
        response = chat_chain.invoke({
            "chat_history": history,
            "user_input": req.message
        })
        ai_reply = extract_text(response.content)

        history.append(HumanMessage(content=req.message))
        history.append(response)

        return {"red_flag_alert": False, "reply": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/ayurveda/finalize")
async def finalize_patient_intake(req: FinalizeIntakeRequest, db: Session = Depends(get_db)):
    """
    Summarizes the interview, assigns a department, generates a token, 
    and stores it in the patient's database history.
    """
    if req.session_id not in active_kiosk_sessions:
        raise HTTPException(status_code=404, detail="Active intake session not found.")

    history = active_kiosk_sessions[req.session_id]
    transcript = "\n".join([f"{msg.type.upper()}: {extract_text(msg.content)}" for msg in history])

    try:
        # 1. Generate Dashavidha Pariksha summary JSON
        summary_data = summary_chain.invoke({"conversation_transcript": transcript})
        
        # 2. Determine the correct department based on the transcript
        routing_data = routing_chain.invoke({"conversation_transcript": transcript})
        assigned_dept = routing_data.get("assigned_department", "Kayachikitsa (Internal Medicine)")
        
        # 3. Generate a dynamic queue token
        token_number = f"TKN-{random.randint(100, 999)}"

        # 4. Save to SQLite database using the dynamically assigned department
        new_record = ConsultationRecord(
            patient_id=req.patient_id,
            session_id=req.session_id,
            department=assigned_dept,
            raw_transcript=transcript,
            clinical_summary=json.dumps(summary_data),
            status="pending_review"
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        # Clear ephemeral kiosk memory
        del active_kiosk_sessions[req.session_id]

        # 5. Return the department and token to the frontend
        return {
            "status": "success",
            "record_id": new_record.id,
            "assigned_department": assigned_dept,
            "token_number": token_number,
            "patient_instructions": f"Assessment complete. Please proceed to the {assigned_dept} department desk. Your queue token is {token_number}.",
            "message": "Intake completed and routed to Doctor Portal."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to finalize intake: {str(e)}")

@router.get("/api/ayurveda/patient/{patient_id}/history")
async def get_patient_history(patient_id: str, db: Session = Depends(get_db)):
    """Retrieves all past consultations for a given patient."""
    records = db.query(ConsultationRecord).filter(
        ConsultationRecord.patient_id == patient_id
    ).order_by(ConsultationRecord.created_at.desc()).all()

    return [
        {
            "record_id": r.id,
            "session_id": r.session_id,
            "created_at": r.created_at.isoformat(),
            "department": r.department,
            "status": r.status,
            "clinical_summary": json.loads(r.clinical_summary) if r.clinical_summary else {},
            "doctor_notes": r.doctor_notes
        }
        for r in records
    ]