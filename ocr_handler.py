import easyocr
import cv2
from pathlib import Path

class DocumentOCR:
    """
    Extract text from prescription images and PDFs
    """
    def __init__(self):
        print("📄 Initializing OCR reader (downloads models first time)...")
        self.reader = easyocr.Reader(['en', 'hi'])
    
    def extract_from_image(self, image_path: str) -> dict:
        """
        Read text from JPG/PNG prescription image
        """
        print(f"📸 Reading image: {image_path}")
        
        if not Path(image_path).exists():
            return {"success": False, "error": f"File not found: {image_path}"}
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": "Could not read image"}
            
            # Run OCR
            results = self.reader.readtext(image)
            
            # Extract text
            extracted_text = "\n".join([text[1] for text in results])
            
            # Calculate confidence
            confidences = [text[2] for text in results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "success": True,
                "text": extracted_text,
                "confidence": f"{avg_confidence:.2%}",
                "lines": len(results)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ===== TEST =====
if __name__ == "__main__":
    ocr = DocumentOCR()
    print("✓ OCR reader ready!")