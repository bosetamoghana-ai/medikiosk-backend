import whisper
import os
from pathlib import Path

class VoiceProcessor:
    """
    Convert patient audio to text using OpenAI Whisper
    """
    def __init__(self, model_size="base"):
        print(f"🔊 Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
    
    def audio_to_text(self, audio_file_path: str) -> dict:
        """
        Convert .wav, .mp3, .m4a to text
        Returns: {"success": True/False, "text": "...", "language": "en"}
        """
        if not os.path.exists(audio_file_path):
            return {"success": False, "error": f"File not found: {audio_file_path}"}
        
        print(f"🎙️ Transcribing: {audio_file_path}")
        
        try:
            result = self.model.transcribe(audio_file_path, language="en")
            
            return {
                "success": True,
                "text": result["text"],
                "language": result.get("language", "en")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ===== TEST =====
if __name__ == "__main__":
    processor = VoiceProcessor(model_size="base")
    print("✓ Voice processor ready!")