import pytesseract
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List
import os
from pathlib import Path

class TextRecognizer:
    def sanitize_text(self, text: str) -> str:
        """Sanitize text to prevent JSON encoding issues."""
        if not isinstance(text, str):
            return ""
        
        # Replace problematic characters
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        
        # Remove any non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        return text.strip()

    def search_text(self, query: str, text_blocks: List[Dict]) -> List[Dict]:
        """Search for text in the detected text blocks."""
        if not query or not text_blocks:
            return []
        
        matches = []
        query = self.sanitize_text(query).lower()
        
        for block in text_blocks:
            if not block.get('text'):
                continue
                
            text = self.sanitize_text(block['text']).lower()
            if query in text:
                matches.append({
                    'text': self.sanitize_text(block['text']),
                    'confidence': float(block.get('confidence', 0.0)),
                    'bbox': {
                        'x_min': int(block['bbox'].get('x_min', 0)),
                        'y_min': int(block['bbox'].get('y_min', 0)),
                        'x_max': int(block['bbox'].get('x_max', 0)),
                        'y_max': int(block['bbox'].get('y_max', 0))
                    } if block.get('bbox') else None
                })
        
        return matches

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Configure Tesseract parameters for better accuracy
        self.custom_config = r'--oem 3 --psm 11'
        self.initialized = True
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy:
        1. Convert to grayscale
        2. Remove noise
        3. Thresholding to get black and white image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Remove noise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary

    def detect_text(self, image_path: str) -> Dict:
        """
        Detects text in images using Tesseract OCR.
        Returns detailed text blocks with confidence scores and bounding boxes.
        """
        try:
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Get detailed OCR data including bounding boxes
            ocr_data = pytesseract.image_to_data(processed_image, config=self.custom_config, output_type=pytesseract.Output.DICT)
            
            text_blocks = []
            height, width = processed_image.shape
            
            # Process each detected text block
            for i in range(len(ocr_data['text'])):
                # Skip empty text
                if not ocr_data['text'][i].strip():
                    continue
                
                # Calculate confidence
                conf = float(ocr_data['conf'][i])
                if conf == -1:  # Skip entries with invalid confidence
                    continue
                
                # Get bounding box coordinates
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Create text block entry
                text_block = {
                    'id': i,
                    'text': ocr_data['text'][i],
                    'confidence': conf / 100.0,  # Convert to 0-1 range
                    'bbox': {
                        'x_min': x,
                        'y_min': y,
                        'x_max': x + w,
                        'y_max': y + h
                    },
                    'points': [
                        [x, y],           # Top-left
                        [x + w, y],       # Top-right
                        [x + w, y + h],   # Bottom-right
                        [x, y + h]        # Bottom-left
                    ],
                    'block_num': ocr_data['block_num'][i],
                    'line_num': ocr_data['line_num'][i],
                    'word_num': ocr_data['word_num'][i]
                }
                
                text_blocks.append(text_block)
            
            # Group text blocks by lines for better organization
            organized_blocks = self._organize_text_blocks(text_blocks)
            
            # Get text summary
            text_summary = self._get_text_summary(text_blocks)
            
            # Categorize text content
            categories = self._categorize_text(text_blocks)
            
            return {
                'text_detected': len(text_blocks) > 0,
                'text_blocks': organized_blocks,
                'total_confidence': np.mean([block['confidence'] for block in text_blocks]) if text_blocks else 0.0,
                'categories': categories,
                'raw_text': ' '.join([block['text'] for block in text_blocks])
            }

        except Exception as e:
            self.logger.error(f"Error in text detection: {str(e)}")
            return {
                'text_detected': False,
                'text_blocks': [],
                'total_confidence': 0.0,
                'categories': [],
                'raw_text': ''
            }

    def _get_text_summary(self, text_blocks: List[Dict]) -> Dict:
        """Generate a summary of the detected text."""
        if not text_blocks:
            return {
                'total_words': 0,
                'avg_confidence': 0.0,
                'text_length': 0
            }
        
        total_words = len(text_blocks)
        avg_confidence = np.mean([block['confidence'] for block in text_blocks])
        text_length = sum(len(block['text']) for block in text_blocks)
        
        return {
            'total_words': total_words,
            'avg_confidence': avg_confidence,
            'text_length': text_length
        }

    def _organize_text_blocks(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Organize text blocks by lines and paragraphs.
        """
        # Sort blocks by block number, line number, and word number
        sorted_blocks = sorted(
            text_blocks,
            key=lambda x: (x['block_num'], x['line_num'], x['word_num'])
        )
        
        # Group blocks by lines
        current_line = []
        organized_blocks = []
        current_line_num = None
        
        for block in sorted_blocks:
            if current_line_num is None:
                current_line_num = block['line_num']
                
            if block['line_num'] != current_line_num:
                # New line detected, process the current line
                if current_line:
                    line_text = ' '.join([b['text'] for b in current_line])
                    organized_blocks.append({
                        'type': 'line',
                        'text': line_text,
                        'confidence': np.mean([b['confidence'] for b in current_line]),
                        'words': current_line
                    })
                current_line = []
                current_line_num = block['line_num']
            
            current_line.append(block)
        
        # Process the last line
        if current_line:
            line_text = ' '.join([b['text'] for b in current_line])
            organized_blocks.append({
                'type': 'line',
                'text': line_text,
                'confidence': np.mean([b['confidence'] for b in current_line]),
                'words': current_line
            })
        
        return organized_blocks

    def _categorize_text(self, text_blocks: List[Dict]) -> List[str]:
        """
        Categorize text content based on patterns and keywords.
        """
        all_text = ' '.join([block['text'].lower() for block in text_blocks])
        categories = []
        
        # Receipt detection
        receipt_keywords = ['total', 'subtotal', 'tax', 'amount', 'payment', '$', '£', '€']
        if any(keyword in all_text for keyword in receipt_keywords):
            categories.append('receipt')
        
        # Document detection
        document_keywords = ['page', 'chapter', 'section', 'dear', 'sincerely']
        if any(keyword in all_text for keyword in document_keywords):
            categories.append('document')
        
        # Sign detection
        sign_keywords = ['stop', 'exit', 'entrance', 'warning', 'caution']
        if any(keyword in all_text for keyword in sign_keywords):
            categories.append('sign')
        
        # Contact information detection
        contact_patterns = ['@', '.com', '.org', '.net', 'tel:', 'phone:', 'email:']
        if any(pattern in all_text for pattern in contact_patterns):
            categories.append('contact_info')
        
        return categories

    def search_text(self, query: str, text_blocks: List[Dict], threshold: float = 0.6) -> List[Dict]:
        """
        Search for text in the detected text blocks.
        Uses basic string matching with case-insensitive comparison.
        """
        matches = []
        query = query.lower()
        
        for block in text_blocks:
            if isinstance(block, dict) and 'text' in block:
                text = block['text'].lower()
                if query in text:
                    similarity = len(query) / len(text) if len(text) > 0 else 0
                    if similarity >= threshold:
                        matches.append({
                            **block,
                            'similarity_score': similarity
                        })
        
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)