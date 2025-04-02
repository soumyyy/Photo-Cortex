import os
import easyocr
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict

# Set the environment variable to disable the MPS memory upper limit.
# Use with caution—this may lead to system instability.
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

class TextRecognizer:
    _instance = None
    _reader = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextRecognizer, cls).__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance

    def initialize(self, use_gpu=True):
        """Initialize the EasyOCR reader with English language support."""
        try:
            if self._reader is None:
                self.logger.info("Initializing EasyOCR reader (use_gpu=%s)...", use_gpu)
                self._reader = easyocr.Reader(['en'], gpu=use_gpu)
                # Warm up the model with a small dummy image to reduce first-call latency.
                warm_up_img = np.zeros((32, 32, 3), dtype=np.uint8)
                self._reader.readtext(warm_up_img)
                self.logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize EasyOCR reader", exc_info=True)
            self._reader = None
            raise

    def detect_text(self, image_path: str) -> Dict:
        """Detect and recognize text in an image using EasyOCR."""
        try:
            # Ensure the reader is initialized (using GPU by default).
            if self._reader is None:
                self.initialize(use_gpu=True)

            # Read the image using OpenCV.
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            # Run OCR synchronously.
            results = self._reader.readtext(image)

            # Process results into a structured dictionary.
            text_blocks = []
            raw_text = []
            for bbox, text, conf in results:
                text_blocks.append({
                    'text': text,
                    'confidence': float(conf),
                    'bbox': bbox
                })
                raw_text.append(text)

            return {
                'text_detected': bool(text_blocks),
                'text_blocks': text_blocks,
                'raw_text': ' '.join(raw_text)
            }

        except RuntimeError as e:
            # Check for the MPS out-of-memory error.
            if "MPS backend out of memory" in str(e):
                self.logger.error("MPS backend out of memory, falling back to CPU mode.", exc_info=True)
                # Reset reader and reinitialize in CPU mode.
                self._reader = None
                self.initialize(use_gpu=False)
                try:
                    results = self._reader.readtext(image)
                    text_blocks = []
                    raw_text = []
                    for bbox, text, conf in results:
                        text_blocks.append({
                            'text': text,
                            'confidence': float(conf),
                            'bbox': bbox
                        })
                        raw_text.append(text)
                    return {
                        'text_detected': bool(text_blocks),
                        'text_blocks': text_blocks,
                        'raw_text': ' '.join(raw_text)
                    }
                except Exception as e2:
                    self.logger.error("Error after falling back to CPU: %s", str(e2), exc_info=True)
                    return {
                        'text_detected': False,
                        'text_blocks': [],
                        'raw_text': '',
                        'error': str(e2)
                    }
            else:
                self.logger.error("Runtime error during text detection: %s", str(e), exc_info=True)
                return {
                    'text_detected': False,
                    'text_blocks': [],
                    'raw_text': '',
                    'error': str(e)
                }
        except Exception as e:
            self.logger.error("Error in text detection for %s: %s", image_path, str(e), exc_info=True)
            return {
                'text_detected': False,
                'text_blocks': [],
                'raw_text': '',
                'error': str(e)
            }

    def __del__(self):
        """Cleanup resources."""
        self._reader = None

# Example usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    recognizer = TextRecognizer()
    image_path = "/Users/soumya/Desktop/Projects/cortexV2/backend/image/IMG_9205.jpeg"  # Replace with your image path.
    result = recognizer.detect_text(image_path)
    if result.get('text_detected'):
        print("Detected text:")
        for block in result['text_blocks']:
            print(f"Text: {block['text']} (Confidence: {block['confidence']:.2f})")
    else:
        print("No text detected or error:", result.get('error', 'Unknown error'))