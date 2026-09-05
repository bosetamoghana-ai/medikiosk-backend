from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import ChatRequest, ChatResponse, FinalizeRequest, FinalizeResponse

app = FastAPI(
    title="MediKiosk AI Triage API",
    description="Backend API for Smart India Hackathon PS 47",
    version="1.0.0"
)

# Crucial: Allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with the frontend's exact URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """
    Receives patient text, checks for emergencies, and generates the next LLM question.
    """
    # Phase 3/4: We will inject the Red Flag monitor and LangChain logic here
    
    # Temporary mock logic for frontend testing
    mock_reply = f"Mock AI: Tell me more about your symptoms. (Path: {request.treatment_pathway})"
    is_emergency = "chest pain" in request.user_input.lower()
    
    return ChatResponse(
        session_id=request.session_id,
        ai_reply_text=mock_reply,
        is_emergency=is_emergency,
        interview_complete=False
    )

@app.post("/api/finalize-history", response_model=FinalizeResponse)
async def finalize_history(request: FinalizeRequest):
    """
    Triggers when the interview is complete. Extracts the final JSON from LangChain memory.
    """
    # Phase 4: We will pull the memory and force the LLM to output the structured JSON here
    
    # Temporary mock response
    mock_json = {
        "chief_complaint": "Headache",
        "onset": "2 days ago",
        "severity": "Moderate"
    }
    
    return FinalizeResponse(
        status="success",
        structured_history=mock_json
    )