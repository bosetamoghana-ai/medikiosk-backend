import json
import random
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

# Import shared database models (Ensure database.py exists)
from database import get_db, ConsultationRecord

# Import your summary tools and the updated safety guardian
from allopathy_engine import summary_prompt, json_parser
from safety import scan_for_red_flags

# --- 1. Router & AI Engine Setup ---
router = APIRouter()

ALLOPATHIC_SOCRATES_SYSTEM_PROMPT = """
You are an empathetic clinical intake assistant at an outpatient department (OPD). 
Your job is to gather a patient's medical history using the SOCRATES method.

*** CRITICAL RULE: YOU MUST ASK EXACTLY ONE QUESTION PER RESPONSE. ***
Never output a numbered list. Never ask multiple questions in the same message. 

INSTRUCTIONS:
1. Read the chat history.
2. Identify the FIRST missing piece of information from the SOCRATES checklist below.
3. Ask ONE simple, conversational question to get that specific missing information. Stop typing after asking.

SOCRATES CHECKLIST (Find the first missing item):
- Site: Where exactly is the pain or issue?
- Onset: When did it start, and was it sudden or gradual?
- Character: What does it feel like? (e.g., sharp, dull, aching)
- Radiation: Does it move or spread anywhere else?
- Associated Symptoms: Are there any other symptoms alongside this?
- Time Course: Is it constant, or does it come and go?
- Exacerbating/Relieving Factors: Does anything make it better or worse?
- Severity: On a scale of 1 to 10, how severe is it?

Once the patient has provided information covering all these points, conclude by asking: 
"Thank you. I have noted down all these details for your doctor. Is there anything else you would like the doctor to know?"
"""

allopathy_prompt = ChatPromptTemplate.from_messages([
    ("system", ALLOPATHIC_SOCRATES_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

llm = ChatOllama(model="phi3", temperature=0.2)
chat_chain = allopathy_prompt | llm
summary_chain = summary_prompt | llm | json_parser

# --- Department Routing Chain ---
ROUTING_PROMPT = """
You are an expert hospital routing assistant. Based on the patient's transcript, determine the most appropriate medical department for their consultation.

Return ONLY a valid JSON object with this exact structure:
{{
    "assigned_department": "Name of department (e.g., Cardiology, Orthopedics, Gastroenterology, General Medicine, Dermatology)",
    "reasoning": "Brief 1-sentence reason"
}}

Transcript:
{conversation_transcript}
"""
routing_prompt_template = ChatPromptTemplate.from_template(ROUTING_PROMPT)
routing_parser = JsonOutputParser()
routing_chain = routing_prompt_template | llm | routing_parser

# Memory dictionaries
pending_otps: Dict[str, dict] = {}
active_kiosk_sessions: Dict[str, List] = {}

# --- 2. Schemas ---
class OTPRequest(BaseModel):
    patient_id: str

class OTPVerify(BaseModel):
    patient_id: str
    otp: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

def extract_text(content):
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return str(content)

# --- 3. Authentication Endpoints ---
@router.post("/api/auth/request-otp")
async def request_otp(req: OTPRequest):
    """Step 1: Patient enters ID to generate OTP."""
    generated_otp = str(random.randint(100000, 999999)) 
    
    pending_otps[req.patient_id] = {
        "otp": generated_otp,
        "department": "Allopathy"
    }
    
    print(f"[SYSTEM] OTP for {req.patient_id} is {generated_otp}")
    return {"status": "success", "message": "OTP sent successfully."}

@router.post("/api/auth/verify-otp")
async def verify_otp_and_start(req: OTPVerify):
    """Step 2: Patient enters OTP to initialize chat."""
    if req.patient_id not in pending_otps:
        raise HTTPException(status_code=404, detail="No pending OTP request found.")
        
    stored_data = pending_otps[req.patient_id]
    
    if stored_data["otp"] != req.otp:
        raise HTTPException(status_code=401, detail="Invalid OTP.")
        
    session_id = datetime.now().strftime(f"allo_{req.patient_id}_%Y%m%d_%H%M%S")
    greeting = "Hello. I am the triage assistant. Could you tell me what symptoms brought you to the clinic today?"
    
    active_kiosk_sessions[session_id] = [AIMessage(content=greeting)]
    del pending_otps[req.patient_id]
    
    return {
        "status": "success",
        "session_id": session_id,
        "first_message": greeting
    }

# --- 4. Clinical Endpoints ---
@router.post("/api/allopathy/chat")
async def handle_chat(req: ChatRequest, db: Session = Depends(get_db)):
    if req.session_id not in active_kiosk_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # --- 2-TIER SAFETY GUARDIAN CHECK ---
    red_flag = scan_for_red_flags(req.message)
    if red_flag:
        if red_flag.get("status") == "EMERGENCY":
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
                "reply_text": red_flag["patient_instruction"]
            }
        elif red_flag.get("status") == "PROBE":
            # Intercept the LLM and ask the hardcoded probe question
            history = active_kiosk_sessions[req.session_id]
            history.append(HumanMessage(content=req.message))
            history.append(AIMessage(content=red_flag["patient_instruction"]))
            
            return {
                "red_flag_alert": False, 
                "reply_text": red_flag["patient_instruction"]
            }

    # Normal LangChain Chat Flow
    history = active_kiosk_sessions[req.session_id]
    response = chat_chain.invoke({"chat_history": history, "user_input": req.message})
    ai_reply = extract_text(response.content)
    
    history.append(HumanMessage(content=req.message))
    history.append(response)
    
    return {"red_flag_alert": False, "reply_text": ai_reply}

@router.post("/api/allopathy/finalize")
async def finalize_session(req: ChatRequest, db: Session = Depends(get_db)):
    if req.session_id not in active_kiosk_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    history = active_kiosk_sessions[req.session_id]
    transcript = "\n".join([f"{msg.type.upper()}: {extract_text(msg.content)}" for msg in history])
    
    try:
        # 1. Generate the JSON summary
        summary_data = summary_chain.invoke({"conversation_transcript": transcript})
        
        # 2. Determine the department
        routing_data = routing_chain.invoke({"conversation_transcript": transcript})
        assigned_dept = routing_data.get("assigned_department", "General Medicine")
        
        # 3. Generate Token
        token_number = f"TKN-{random.randint(100, 999)}"
        
        # 4. Save to Database
        new_record = ConsultationRecord(
            patient_id=req.session_id.split("_")[1],
            session_id=req.session_id,
            department=assigned_dept,
            raw_transcript=transcript,
            clinical_summary=json.dumps(summary_data),
            status="pending_review"
        )
        db.add(new_record)
        db.commit()
        
        del active_kiosk_sessions[req.session_id]
        
        return {
            "status": "success",
            "record_id": new_record.id,
            "assigned_department": assigned_dept,
            "token_number": token_number,
            "patient_instructions": f"Assessment complete. Please proceed to the {assigned_dept} department desk. Your queue token is {token_number}.",
            "message": "SOCRATES summary saved to database."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to finalize intake: {str(e)}")