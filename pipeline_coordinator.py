import json
import requests
from voice_handler import VoiceProcessor
from ocr_handler import DocumentOCR
from entity_extractor import EntityExtractor
from speech_synthesis import TextToSpeech

class Member2Pipeline:
    """
    Master coordinator for all Member 2 functions
    Connects to Member 1 (LLM) and Member 3 (API)
    """
    def __init__(self, member1_url: str = "http://localhost:8001", 
                 member3_url: str = "http://localhost:8000"):
        print("🚀 Initializing Member 2 Pipeline...")
        self.voice = VoiceProcessor(model_size="base")
        self.ocr = DocumentOCR()
        self.entity = EntityExtractor()
        self.tts = TextToSpeech()
        self.member1_url = member1_url
        self.member3_url = member3_url
        print("✓ All modules loaded!")
    
    # ===== VOICE INPUT PATH =====
    def process_patient_voice(self, audio_file: str) -> dict:
        """
        Step 1: Audio → Text
        Step 2: Send to Member 1 LLM
        Step 3: Response → Audio
        """
        print("\n" + "="*60)
        print("VOICE INPUT PIPELINE")
        print("="*60)
        
        # Step 1: Audio to Text
        voice_result = self.voice.audio_to_text(audio_file)
        if not voice_result["success"]:
            return voice_result
        
        patient_text = voice_result["text"]
        print(f"[MEMBER 2] Patient said: {patient_text}")
        
        # Step 2: Send to Member 1 (LLM)
        llm_response = self._call_member1_llm(patient_text)
        print(f"[MEMBER 2] LLM response: {llm_response}")
        
        # Step 3: Convert response to speech
        tts_result = self.tts.text_to_speech(llm_response)
        
        return {
            "patient_input": patient_text,
            "llm_response": llm_response,
            "audio_response": tts_result.get("file")
        }
    
    # ===== DOCUMENT INPUT PATH =====
    def process_prescription(self, image_path: str) -> dict:
        """
        Step 1: Image → OCR Text
        Step 2: Text → Structured Entities
        Step 3: Send to Member 3 API
        """
        print("\n" + "="*60)
        print("PRESCRIPTION DOCUMENT PIPELINE")
        print("="*60)
        
        # Step 1: OCR
        ocr_result = self.ocr.extract_from_image(image_path)
        if not ocr_result["success"]:
            return ocr_result
        
        ocr_text = ocr_result["text"]
        print(f"[MEMBER 2] OCR extracted {ocr_result['lines']} lines")
        print(f"[MEMBER 2] Confidence: {ocr_result['confidence']}")
        
        # Step 2: Entity Extraction
        entities = self.entity.extract_all(ocr_text)
        print(f"[MEMBER 2] Extracted structured data:")
        print(json.dumps(entities, indent=2))
        
        # Step 3: Send to Member 3 API
        api_result = self._call_member3_api(
            endpoint="/api/upload-prescription",
            data=entities
        )
        
        return {
            "raw_ocr": ocr_text,
            "structured_data": entities,
            "api_response": api_result
        }
    
    # ===== CONNECT TO MEMBER 1 (LLM) =====
    def _call_member1_llm(self, patient_input: str) -> str:
        """
        Send text to Member 1's LLM and get response
        """
        print(f"[MEMBER 2→1] Sending to LLM: '{patient_input}'")
        
        try:
            response = requests.post(
                f"{self.member1_url}/api/chat",
                json={"user_input": patient_input},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response from LLM")
            else:
                return f"LLM Error: {response.status_code}"
        
        except requests.exceptions.ConnectionError:
            print(f"[MEMBER 2→1] ⚠️ Could not connect to Member 1 at {self.member1_url}")
            # Fallback response for testing
            if "fever" in patient_input.lower():
                return "I see you have fever. How high is your temperature?"
            return "Can you describe your symptoms in detail?"
    
    # ===== CONNECT TO MEMBER 3 (API/DATABASE) =====
    def _call_member3_api(self, endpoint: str, data: dict) -> dict:
        """
        Send structured data to Member 3's FastAPI server
        """
        print(f"[MEMBER 2→3] Sending to API: {self.member3_url}{endpoint}")
        
        try:
            response = requests.post(
                f"{self.member3_url}{endpoint}",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"[MEMBER 2→3] ✓ API accepted data")
                return {"success": True, "response": response.json()}
            else:
                print(f"[MEMBER 2→3] ✗ API error: {response.status_code}")
                return {"success": False, "error": response.text}
        
        except requests.exceptions.ConnectionError:
            print(f"[MEMBER 2→3] ⚠️ Could not connect to Member 3 at {self.member3_url}")
            return {"success": False, "error": "Connection failed"}


# ===== TEST =====
if __name__ == "__main__":
    pipeline = Member2Pipeline()
    
    # Test 1: Voice processing (simulated)
    print("\n[TEST] Simulated voice input:")
    pipeline._call_member1_llm("I have fever for 3 days and headache")
    
    # Test 2: Document processing (simulated)
    print("\n[TEST] Document entity extraction:")
    sample_prescription = """
    Date: 15-Dec-2024
    Patient: Rajesh Kumar, Age 45, Male
    BP: 125/80 mmHg, Temperature: 37.5°C
    Medication: Aspirin 75mg once daily, Paracetamol 500mg thrice daily
    Symptoms: Chest pain, shortness of breath for 2 days
    """
    
    entities = pipeline.entity.extract_all(sample_prescription)
    print(json.dumps(entities, indent=2))