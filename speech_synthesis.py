from gtts import gTTS
import os

class TextToSpeech:
    """
    Convert text response back to speech for patient
    """
    def text_to_speech(self, text: str, language: str = "en", output_file: str = "response.mp3") -> dict:
        """
        Convert text to audio file
        Returns: {"success": True/False, "file": "response.mp3"}
        """
        print(f"🔊 Converting to speech: {text[:50]}...")
        
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(output_file)
            print(f"✓ Audio saved: {output_file}")
            return {"success": True, "file": output_file}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ===== TEST =====
if __name__ == "__main__":
    tts = TextToSpeech()
    result = tts.text_to_speech("Hello patient, please describe your symptoms")
    print(result)