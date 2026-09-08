import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List

# ---------------------------------------------------------------------------
# 1. Pydantic Schema: The Final Ayurvedic Clinical Summary
# ---------------------------------------------------------------------------
class DashavidhaParikshaSummary(BaseModel):
    chief_complaint: str = Field(description="Primary symptom/complaint and duration")
    associated_symptoms: List[str] = Field(description="Other complaints")
    
    # Core Functional Assessment
    agni_status: str = Field(description="Digestive fire: Manda (sluggish), Tikshna (intense), Vishama (irregular), or Sama (balanced)")
    koshtha_type: str = Field(description="Bowel nature: Krura (hard/constipated), Mridu (soft/loose), or Madhyama (normal)")
    nidra_sleep: str = Field(description="Sleep quality, duration, daytime sleepiness")
    
    # Ahara & Vihara (Diet and Lifestyle)
    ahara_dietary_habits: str = Field(description="Appetite, taste preferences (Rasa), meal timings, water intake")
    vihara_lifestyle: str = Field(description="Physical activity level, stress levels, daily routine")
    
    # Clinical Impressions
    suspected_dosha_involvement: List[str] = Field(description="Vata, Pitta, and/or Kapha clinical manifestations noted")
    prakriti_vikriti_notes: str = Field(description="Observations on baseline constitution vs current acute imbalance")
    red_flag_alert: Optional[str] = Field(None, description="Any critical/acute emergency symptoms requiring immediate allopathic triage")


# ---------------------------------------------------------------------------
# 2. System Prompt: The Conversational Ayurvedic Intake Assistant
# ---------------------------------------------------------------------------
AYURVEDA_SYSTEM_PROMPT = """
You are an expert Ayurvedic clinical assistant at a hospital intake kiosk.
Your objective is to collect a thorough patient history following Ayurvedic principles (Trividha, Ashtavidha, and Dashavidha Pariksha) before the doctor sees the patient.

RULES:
1. ASK ONLY ONE QUESTION AT A TIME. Keep questions short, warm, and accessible to non-medical patients.
2. DO NOT use obscure Sanskrit terms with the patient. Instead of asking "What is your Agni?", ask: "How is your appetite and digestion throughout the day? Do you feel heavy, bloated, or acidic after meals?"
3. Follow this clinical sequence:
   a. Chief Complaint (What brings you in? How long has it been happening?)
   b. Agni & Koshtha (Appetite, digestion, acidity, bowel movements, regularity)
   c. Ahara-Vihara (Diet type, meal regularity, water intake, sleep quality, stress)
   d. Environmental & Aggravating factors (Does weather, cold food, or stress worsen it?)
4. If the patient reports an acute medical emergency (sudden severe chest pain, extreme breathlessness, sudden paralysis, severe trauma), immediately respond:
   "RED_FLAG: Please proceed directly to the Emergency Room."
5. When you have gathered enough information across complaint, digestion/bowels, and lifestyle, conclude the interview by asking:
   "Thank you. Is there any other symptom or detail you would like the Ayurvedic physician to know?"
"""

# ---------------------------------------------------------------------------
# 3. Dynamic Prompt Template with History Injection
# ---------------------------------------------------------------------------
ayurveda_prompt = ChatPromptTemplate.from_messages([
    ("system", AYURVEDA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# ---------------------------------------------------------------------------
# 4. Final Summarization Prompt (Converts History into Doctor's JSON)
# ---------------------------------------------------------------------------
json_parser = JsonOutputParser(pydantic_object=DashavidhaParikshaSummary)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an Ayurvedic physician's documentation assistant. Analyze the entire "
     "conversation transcript and extract the structured clinical intake note.\n"
     "Format instructions:\n{format_instructions}"),
    ("human", "Here is the full conversation log:\n\n{conversation_transcript}\n\nGenerate the clinical JSON.")
]).partial(format_instructions=json_parser.get_format_instructions())