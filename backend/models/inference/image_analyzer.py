import cv2
import numpy as np
from pathlib import Path
import logging
from .text_recognizer import TextRecognizer
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        """Initialize the image analyzer with various models."""
        self.text_recognizer = TextRecognizer()
        logger.info("ImageAnalyzer initialized successfully")

    async def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze an image using various models for different features.
        """
        try:
            # Ensure image path is absolute
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise ValueError(f"Image file not found: {image_path}")
                
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            # Validate image dimensions
            if image.shape[0] == 0 or image.shape[1] == 0:
                raise ValueError(f"Invalid image dimensions: {image.shape}")

            # Perform text recognition
            text_analysis = self.text_recognizer.detect_text(str(image_path))

            # Return combined analysis results
            return {
                "filename": image_path.name,
                "text_recognition": {
                    "text_detected": text_analysis["text_detected"],
                    "text_blocks": text_analysis["text_blocks"],
                    "total_confidence": text_analysis["total_confidence"],
                    "categories": text_analysis["categories"],
                    "raw_text": text_analysis["raw_text"]
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return {
                "filename": image_path.name,
                "error": str(e),
                "text_recognition": {
                    "text_detected": False,
                    "text_blocks": [],
                    "total_confidence": 0.0,
                    "categories": [],
                    "raw_text": ""
                }
            }