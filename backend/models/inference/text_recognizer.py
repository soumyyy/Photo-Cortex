import os
import easyocr
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Union

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

    def detect_text(self, image_input: Union[str, Path, np.ndarray]) -> Dict:
        """Detect and recognize text in an image using EasyOCR."""
        try:
            # Ensure the reader is initialized (using GPU by default).
            if self._reader is None:
                self.initialize(use_gpu=True)

            # Handle different input types
            if isinstance(image_input, (str, Path)):
                image = cv2.imread(str(image_input))
                if image is None:
                    raise ValueError(f"Could not read image at {image_input}")
            elif isinstance(image_input, np.ndarray):
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Run OCR synchronously.
            results = self._reader.readtext(image)

            # Process results into a structured dictionary.
            text_blocks = []
            raw_text = []
            for detection in results:
                bbox = detection[0]  # List of points [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                text = detection[1]
                conf = float(detection[2])  # Ensure confidence is float
                
                # Convert numpy arrays to lists for JSON serialization
                bbox = [[float(x) for x in point] for point in bbox]
                
                text_blocks.append({
                    'text': text,
                    'confidence': conf,
                    'bounding_box': bbox
                })
                raw_text.append(text)

            # Calculate average confidence if there are results
            avg_confidence = sum(conf for _, _, conf in results) / len(results) if results else 0.0

            return {
                'text_detected': bool(results),
                'text_blocks': text_blocks,
                'raw_text': ' '.join(raw_text),
                'total_confidence': avg_confidence
            }

        except RuntimeError as e:
            # Handle specific runtime errors (like CUDA out of memory)
            try:
                self.initialize(use_gpu=False)  # Reinitialize without GPU
                return self.detect_text(image_input)  # Retry detection
            except Exception as e2:
                self.logger.error("Failed to recover from runtime error: %s", str(e2))
                return {
                    'text_detected': False,
                    'text_blocks': [],
                    'raw_text': '',
                    'error': str(e2)
                }
        except Exception as e:
            self.logger.error("Error in text detection for %s: %s", image_input, str(e), exc_info=True)
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