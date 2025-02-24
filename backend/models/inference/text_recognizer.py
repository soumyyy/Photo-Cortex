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

    def get_text_summary(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of detected text blocks."""
        if not text_blocks:
            return {
                "total_blocks": 0,
                "total_words": 0,
                "average_confidence": 0,
                "text_content": ""
            }

        total_blocks = len(text_blocks)
        text_content = " ".join(block["text"] for block in text_blocks)
        words = text_content.split()
        total_words = len(words)
        average_confidence = sum(block["confidence"] for block in text_blocks) / total_blocks if total_blocks > 0 else 0

        return {
            "total_blocks": total_blocks,
            "total_words": total_words,
            "average_confidence": average_confidence,
            "text_content": text_content
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better text detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

        # Noise removal
        denoised = cv2.fastNlMeansDenoising(binary)

        return denoised

    def merge_overlapping_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """Merge overlapping text boxes."""
        if not boxes:
            return []

        def boxes_overlap(box1, box2, threshold=0.3):
            # Calculate intersection
            x1 = max(box1['bbox']['x_min'], box2['bbox']['x_min'])
            y1 = max(box1['bbox']['y_min'], box2['bbox']['y_min'])
            x2 = min(box1['bbox']['x_max'], box2['bbox']['x_max'])
            y2 = min(box1['bbox']['y_max'], box2['bbox']['y_max'])

            if x2 < x1 or y2 < y1:
                return False

            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1['bbox']['x_max'] - box1['bbox']['x_min']) * (box1['bbox']['y_max'] - box1['bbox']['y_min'])
            area2 = (box2['bbox']['x_max'] - box2['bbox']['x_min']) * (box2['bbox']['y_max'] - box2['bbox']['y_min'])
            
            overlap = intersection / min(area1, area2)
            return overlap > threshold

        merged = []
        while boxes:
            current = boxes.pop(0)
            to_merge = []

            i = 0
            while i < len(boxes):
                if boxes_overlap(current, boxes[i]):
                    to_merge.append(boxes.pop(i))
                else:
                    i += 1

            if to_merge:
                # Merge boxes
                x_min = min([current['bbox']['x_min']] + [b['bbox']['x_min'] for b in to_merge])
                y_min = min([current['bbox']['y_min']] + [b['bbox']['y_min'] for b in to_merge])
                x_max = max([current['bbox']['x_max']] + [b['bbox']['x_max'] for b in to_merge])
                y_max = max([current['bbox']['y_max']] + [b['bbox']['y_max'] for b in to_merge])

                # Combine text and average confidence
                all_texts = [current['text']] + [b['text'] for b in to_merge]
                all_confidences = [current['confidence']] + [b['confidence'] for b in to_merge]
                
                merged.append({
                    "id": current['id'],
                    "text": " ".join(all_texts),
                    "confidence": sum(all_confidences) / len(all_confidences),
                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    },
                    "points": [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ]
                })
            else:
                merged.append(current)

        return merged

    def detect_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect and recognize text in an image."""
        if not self.initialized:
            return {
                "text_detected": False,
                "text_blocks": [],
                "total_confidence": 0.0
            }

        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(processed_image)

            # Get detailed OCR data with improved parameters
            try:
                ocr_data = pytesseract.image_to_data(
                    pil_image, 
                    output_type=pytesseract.Output.DICT,
                    config='--psm 11 --oem 3'  # Sparse text with OSD, LSTM only
                )
            except Exception as e:
                logger.error(f"OCR processing error: {str(e)}")
                return {
                    "text_detected": False,
                    "text_blocks": [],
                    "total_confidence": 0.0
                }
            
            text_blocks = []
            valid_blocks = 0

            # Process each detected text element
            for idx in range(len(ocr_data.get('text', []))):
                try:
                    text = str(ocr_data['text'][idx]).strip()
                    if not text:
                        continue

                    confidence = float(ocr_data.get('conf', [0])[idx])
                    if confidence <= 0:
                        continue
                    
                    confidence = confidence / 100.0
                    if confidence < 0.4:  # Increased confidence threshold
                        continue

                    # Get bounding box coordinates with padding
                    padding = 5
                    x = max(0, int(ocr_data.get('left', [0])[idx]) - padding)
                    y = max(0, int(ocr_data.get('top', [0])[idx]) - padding)
                    w = int(ocr_data.get('width', [0])[idx]) + (2 * padding)
                    h = int(ocr_data.get('height', [0])[idx]) + (2 * padding)

                    # Create text block
                    text_blocks.append({
                        "id": valid_blocks,
                        "text": text,
                        "confidence": confidence,
                        "bbox": {
                            "x_min": x,
                            "y_min": y,
                            "x_max": x + w,
                            "y_max": y + h
                        },
                        "points": [
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ]
                    })
                    valid_blocks += 1
                except (KeyError, IndexError, ValueError) as e:
                    logger.warning(f"Error processing text block {idx}: {str(e)}")
                    continue

            # Merge overlapping boxes
            if text_blocks:
                text_blocks = self.merge_overlapping_boxes(text_blocks)

            # Calculate average confidence
            total_confidence = sum(block['confidence'] for block in text_blocks)
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