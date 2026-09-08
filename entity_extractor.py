import spacy
import re
import json
from typing import List, Dict
from pydantic import BaseModel

# ===== DATA MODELS =====
class PatientInfo(BaseModel):
    name: str = None
    age: int = None
    gender: str = None
    phone: str = None

class Medication(BaseModel):
    name: str
    dosage: str = None
    frequency: str = None

class Vital(BaseModel):
    type: str  # "BP", "Temperature", "Glucose"
    value: str
    unit: str = None

class Symptom(BaseModel):
    symptom: str
    duration: str = None
    severity: str = None


class EntityExtractor:
    """
    Parse messy OCR/voice text into structured JSON
    """
    def __init__(self):
        print("🧠 Loading NLP model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.symptoms_list = [
            "fever", "cough", "headache", "chest pain", "shortness of breath",
            "nausea", "vomiting", "diarrhea", "body ache", "fatigue",
            "sore throat", "runny nose", "rash", "dizziness", "weakness"
        ]
    
    def extract_patient_info(self, text: str) -> dict:
        """Extract patient demographics"""
        try:
            # Age
            age_match = re.search(r'age[:\s]+(\d+)', text, re.IGNORECASE)
            age = int(age_match.group(1)) if age_match else None
            
            # Phone
            phone_match = re.search(r'\b(\d{10})\b', text)
            phone = phone_match.group(1) if phone_match else None
            
            # Gender
            gender = None
            if re.search(r'\b(male|m|mr)\b', text, re.IGNORECASE):
                gender = "Male"
            elif re.search(r'\b(female|f|ms|mrs)\b', text, re.IGNORECASE):
                gender = "Female"
            
            # Name using spaCy
            doc = self.nlp(text)
            name = None
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text
                    break
            
            return PatientInfo(name=name, age=age, gender=gender, phone=phone).dict()
        except Exception as e:
            print(f"Error extracting patient info: {e}")
            return PatientInfo().dict()
    
    def extract_medications(self, text: str) -> List[dict]:
        """Extract medications and dosages"""
        medications = []
        med_pattern = r'([A-Za-z]+(?:\s[A-Za-z]+)?)\s+(\d+(?:mg|ml)?)\s+(\w+\s+\w+)'
        
        try:
            for match in re.finditer(med_pattern, text, re.IGNORECASE):
                med = Medication(
                    name=match.group(1).strip(),
                    dosage=match.group(2).strip(),
                    frequency=match.group(3).strip()
                )
                medications.append(med.dict())
        except Exception as e:
            print(f"Error extracting medications: {e}")
        
        return medications
    
    def extract_vitals(self, text: str) -> List[dict]:
        """Extract vital signs"""
        vitals = []
        
        try:
            # Blood Pressure
            bp_pattern = r'BP[:\s]+(\d+)/(\d+)'
            for match in re.finditer(bp_pattern, text, re.IGNORECASE):
                vital = Vital(
                    type="BP",
                    value=f"{match.group(1)}/{match.group(2)}",
                    unit="mmHg"
                )
                vitals.append(vital.dict())
            
            # Temperature
            temp_pattern = r'(\d+\.?\d*)\s*°?[CF]'
            for match in re.finditer(temp_pattern, text):
                unit = "°C" if "C" in text[match.start():match.end()] else "°F"
                vital = Vital(
                    type="Temperature",
                    value=match.group(1),
                    unit=unit
                )
                vitals.append(vital.dict())
            
            # Glucose
            glucose_pattern = r'glucose[:\s]+(\d+)'
            for match in re.finditer(glucose_pattern, text, re.IGNORECASE):
                vital = Vital(
                    type="Glucose",
                    value=match.group(1),
                    unit="mg/dL"
                )
                vitals.append(vital.dict())
        except Exception as e:
            print(f"Error extracting vitals: {e}")
        
        return vitals
    
    def extract_symptoms(self, text: str) -> List[dict]:
        """Extract symptoms mentioned"""
        symptoms = []
        text_lower = text.lower()
        
        try:
            for symptom in self.symptoms_list:
                if symptom in text_lower:
                    duration_pattern = f'{symptom}.*?(\\d+\\s+(?:days|weeks|hours))'
                    duration_match = re.search(duration_pattern, text_lower, re.IGNORECASE)
                    
                    sym = Symptom(
                        symptom=symptom,
                        duration=duration_match.group(1) if duration_match else None
                    )
                    symptoms.append(sym.dict())
        except Exception as e:
            print(f"Error extracting symptoms: {e}")
        
        return symptoms
    
    def extract_all(self, text: str) -> dict:
        """Extract everything and return as structured JSON"""
        return {
            "patient": self.extract_patient_info(text),
            "symptoms": self.extract_symptoms(text),
            "medications": self.extract_medications(text),
            "vitals": self.extract_vitals(text)
        }


# ===== TEST =====
if __name__ == "__main__":
    extractor = EntityExtractor()
    
    sample_text = """
    Patient: Mr. Rajesh Kumar
    Age: 45 years | Gender: Male | Phone: 9876543210
    
    Chief Complaints: Fever for 3 days, headache for 2 weeks, cough
    
    Vitals: BP 120/80, Temperature 38.5°C, Glucose 95 mg/dL
    
    Medications:
    1. Paracetamol 500mg twice daily
    2. Cough syrup 10ml thrice daily
    """
    
    result = extractor.extract_all(sample_text)
    print(json.dumps(result, indent=2))