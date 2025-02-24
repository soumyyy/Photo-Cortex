import numpy as np
from typing import List, Dict, Any, Optional
import logging
import cv2
from pathlib import Path
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

class TextRecognizer:
    def __init__(self):
        """Initialize text recognition model."""
        self.initialized = False
        try:
            # Test if tesseract is installed
            pytesseract.get_tesseract_version()
            self.initialized = True
            logger.info("Tesseract initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Tesseract: {str(e)}")
            logger.warning("Make sure Tesseract is installed on your system")

    def detect_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect and recognize text in an image."""
        if not self.initialized:
            return {
                "text_detected": False,
                "text_blocks": [],
                "total_confidence": 0.0
            }

        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)

            # Get paragraph-level OCR data
            ocr_data = pytesseract.image_to_data(
                pil_image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 1'  # Automatic page segmentation with OSD
            )
            
            text_blocks = []
            total_confidence = 0.0
            valid_blocks = 0
            current_block = None

            # Process each detected text element
            for idx in range(len(ocr_data['text'])):
                # Skip empty text
                if not ocr_data['text'][idx].strip():
                    continue

                # Get confidence and convert to 0-1 range
                confidence = float(ocr_data['conf'][idx])
                if confidence < 0:  # Skip negative confidence values
                    continue
                confidence = confidence / 100.0

                # Filter very low confidence detections
                if confidence < 0.3:
                    continue

                # Get bounding box coordinates
                x = ocr_data['left'][idx]
                y = ocr_data['top'][idx]
                w = ocr_data['width'][idx]
                h = ocr_data['height'][idx]

                # If this is a new block or far from previous block
                if current_block is None:
                    current_block = {
                        'x_min': x, 'y_min': y,
                        'x_max': x + w, 'y_max': y + h,
                        'text': ocr_data['text'][idx],
                        'confidence': confidence
                    }
                else:
                    # Check if this text is close to current block
                    x_overlap = (x <= current_block['x_max'] + 20 and 
                               x + w >= current_block['x_min'] - 20)
                    y_overlap = (y <= current_block['y_max'] + 20 and 
                               y + h >= current_block['y_min'] - 20)

                    if x_overlap or y_overlap:
                        # Expand current block
                        current_block['x_min'] = min(current_block['x_min'], x)
                        current_block['y_min'] = min(current_block['y_min'], y)
                        current_block['x_max'] = max(current_block['x_max'], x + w)
                        current_block['y_max'] = max(current_block['y_max'], y + h)
                        current_block['text'] += ' ' + ocr_data['text'][idx]
                        current_block['confidence'] = (current_block['confidence'] + confidence) / 2
                    else:
                        # Add current block to results and start new block
                        points = [
                            [current_block['x_min'], current_block['y_min']],
                            [current_block['x_max'], current_block['y_min']],
                            [current_block['x_max'], current_block['y_max']],
                            [current_block['x_min'], current_block['y_max']]
                        ]
                        
                        text_blocks.append({
                            "id": valid_blocks,
                            "text": current_block['text'],
                            "confidence": current_block['confidence'],
                            "bbox": {
                                "x_min": current_block['x_min'],
                                "y_min": current_block['y_min'],
                                "x_max": current_block['x_max'],
                                "y_max": current_block['y_max']
                            },
                            "points": points
                        })
                        
                        total_confidence += current_block['confidence']
                        valid_blocks += 1
                        
                        current_block = {
                            'x_min': x, 'y_min': y,
                            'x_max': x + w, 'y_max': y + h,
                            'text': ocr_data['text'][idx],
                            'confidence': confidence
                        }

            # Add the last block if it exists
            if current_block is not None:
                points = [
                    [current_block['x_min'], current_block['y_min']],
                    [current_block['x_max'], current_block['y_min']],
                    [current_block['x_max'], current_block['y_max']],
                    [current_block['x_min'], current_block['y_max']]
                ]
                
                text_blocks.append({
                    "id": valid_blocks,
                    "text": current_block['text'],
                    "confidence": current_block['confidence'],
                    "bbox": {
                        "x_min": current_block['x_min'],
                        "y_min": current_block['y_min'],
                        "x_max": current_block['x_max'],
                        "y_max": current_block['y_max']
                    },
                    "points": points
                })
                total_confidence += current_block['confidence']

            avg_confidence = total_confidence / len(text_blocks) if text_blocks else 0.0
            
            return {
                "text_detected": len(text_blocks) > 0,
                "text_blocks": text_blocks,
                "total_confidence": float(avg_confidence)
            }

        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
            return {
                "text_detected": False,
                "text_blocks": [],
                "total_confidence": 0.0
            }