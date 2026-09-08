from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Define the Allopathic Clinical Schema (SOCRATES framework)
class AllopathicSummary(BaseModel):
    chief_complaint: str = Field(description="The primary symptom or reason for visit")
    site: str = Field(description="Where exactly the pain or symptom is located")
    onset: str = Field(description="When the symptom started and whether it was sudden or gradual")
    character: str = Field(description="Description of the symptom (e.g., aching, stabbing, burning)")
    radiation: str = Field(description="Whether the pain or symptom radiates or spreads to other areas")
    associated_symptoms: List[str] = Field(description="Any other signs or symptoms accompanying the primary issue")
    time_course: str = Field(description="Pattern of the symptom over time (e.g., constant, comes and goes)")
    exacerbating_relieving_factors: str = Field(description="What makes the symptom better or worse")
    severity: str = Field(description="Severity of the pain or symptom, usually on a scale of 0 to 10")
    triage_level: str = Field(description="Classify as: Routine, Urgent, or Emergency")
    recommended_specialist: str = Field(description="E.g., General Medicine, Cardiology, Orthopedics, ER")
    red_flags: Optional[str] = Field(description="Any critical warning signs (e.g., chest pain, severe bleeding, loss of consciousness)")

json_parser = JsonOutputParser(pydantic_object=AllopathicSummary)

# 2. Define the Chat Prompt for the Intake Assistant
allopathy_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an empathetic allopathic triage nurse at MediKiosk. 
    Your goal is to gather the patient's symptoms systematically to complete a SOCRATES assessment.
    
    CRITICAL RULES FOR CHAT:
    1. You MUST ask ONLY ONE question per response. 
    2. NEVER ask a list of questions. 
    3. Wait for the patient to answer before moving to the next part of the SOCRATES framework.
    4. Keep your responses to a maximum of 2 short sentences.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])


# 3. Define the Summary Prompt for the Doctor's JSON
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior medical triage officer. Review the following patient intake transcript and extract the clinical details into a structured JSON format according to the SOCRATES triage protocol.
    
    Format Instructions:
    {format_instructions}
    """),
    ("human", "Here is the conversation transcript:\n\n{conversation_transcript}")
]).partial(format_instructions=json_parser.get_format_instructions())