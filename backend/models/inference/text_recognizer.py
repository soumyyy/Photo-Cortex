import pytesseract
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
from fractions import Fraction
from langdetect import detect
import re
from collections import defaultdict
from difflib import SequenceMatcher

class TextRecognizer:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextRecognizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, lang: str = 'eng'):
        if getattr(self, '_initialized', False):
            return
            
        self.logger = logging.getLogger(__name__)
        # Reduced number of configs, focusing on most effective ones
        self.configs = {
            'default': r'--oem 3 --psm 3',  # Default: Fully automatic page segmentation
            'sparse': r'--oem 3 --psm 6',  # Sparse text
        }
        self.tesseract_lang = lang
        self.initialized = True
        self._initialized = True
        self.text_cache = {}
        
        # Common text patterns for verification
        self.text_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'phone': r'\b(?:\+?\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
            'date': r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        }
        
    def enhance_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Enhanced image preprocessing for better text detection."""
        enhanced_images = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Basic enhancement
        enhanced_images.append(gray)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        enhanced_images.append(adaptive_thresh)
        
        # Add contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_contrast = clahe.apply(gray)
        enhanced_images.append(enhanced_contrast)
        
        # Add sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        enhanced_images.append(sharpened)
        
        return enhanced_images

    def detect_text(self, image_path: str) -> Dict:
        """
        Enhanced text detection with multiple preprocessing steps and better block handling.
        """
        try:
            if image_path in self.text_cache:
                return self.text_cache[image_path]
                
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Quick check for text presence using edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            text_likelihood = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            if text_likelihood < 0.02:  # Increased threshold for better filtering
                result = self._empty_result()
                self.text_cache[image_path] = result
                return result
            
            # Get enhanced versions of the image
            enhanced_images = self.enhance_image(image)
            
            all_blocks = []
            for img in enhanced_images:
                try:
                    # Use only the most reliable PSM mode
                    config = '--oem 3 --psm 3'
                    ocr_data = pytesseract.image_to_data(
                        img,
                        lang=self.tesseract_lang,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    blocks = self._process_ocr_data(ocr_data)
                    all_blocks.extend(blocks)
                except Exception as e:
                    self.logger.warning(f"OCR pass failed: {str(e)}")
                    continue
            
            # Merge similar blocks with stricter similarity threshold
            merged_blocks = self._merge_text_blocks(all_blocks)
            
            # Filter blocks with higher confidence threshold
            filtered_blocks = [b for b in merged_blocks if b['confidence'] > 0.6]  # Increased confidence threshold
            
            # Additional filtering for minimum text length and common patterns
            filtered_blocks = [b for b in filtered_blocks if 
                len(b['text'].strip()) >= 3 and  # Minimum text length
                not b['text'].isspace() and  # No whitespace-only blocks
                any(c.isalnum() for c in b['text'])  # Must contain at least one alphanumeric character
            ]
            
            # Sort blocks by vertical position
            sorted_blocks = sorted(filtered_blocks, key=lambda x: x['bbox']['y_min'])
            
            # Detect text language only if we have enough text
            combined_text = ' '.join([b['text'] for b in sorted_blocks])
            try:
                lang = detect(combined_text) if len(combined_text) > 10 else 'unknown'
            except:
                lang = 'unknown'
            
            result = {
                'text_detected': len(sorted_blocks) > 0,
                'text_blocks': sorted_blocks,
                'total_confidence': np.mean([b['confidence'] for b in sorted_blocks]) if sorted_blocks else 0.0,
                'raw_text': combined_text,
                'language': lang
            }
            
            self.text_cache[image_path] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error in text detection: {str(e)}")
            return self._empty_result()

    def _empty_result(self) -> Dict:
        return {
            'text_detected': False,
            'text_blocks': [],
            'total_confidence': 0.0,
            'categories': [],
            'raw_text': '',
            'language': 'unknown',
            'orientation_angle': 0.0
        }

    def _process_ocr_data(self, ocr_data: Dict) -> List[Dict]:
        """Process OCR data into text blocks."""
        blocks = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) < 0:  # Skip entries with negative confidence
                continue
                
            text = ocr_data['text'][i].strip()
            if not text:  # Skip empty text
                continue
                
            blocks.append({
                'text': text,
                'confidence': float(ocr_data['conf'][i]) / 100.0,
                'bbox': {
                    'x_min': ocr_data['left'][i],
                    'y_min': ocr_data['top'][i],
                    'x_max': ocr_data['left'][i] + ocr_data['width'][i],
                    'y_max': ocr_data['top'][i] + ocr_data['height'][i]
                }
            })
            
        return blocks

    def _merge_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Merge similar text blocks to remove duplicates."""
        if not blocks:
            return []
            
        merged = []
        used = set()
        
        for i, block1 in enumerate(blocks):
            if i in used:
                continue
                
            similar_blocks = [block1]
            used.add(i)
            
            for j, block2 in enumerate(blocks[i+1:], i+1):
                if j in used:
                    continue
                    
                # Check for similar text using SequenceMatcher
                similarity = SequenceMatcher(None, block1['text'], block2['text']).ratio()
                if similarity > 0.8:  # Merge if very similar
                    similar_blocks.append(block2)
                    used.add(j)
            
            # Take the block with highest confidence
            best_block = max(similar_blocks, key=lambda x: x['confidence'])
            merged.append(best_block)
            
        return merged

    def _organize_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Organize text blocks by type and position."""
        organized = []
        for block in blocks:
            text_type = 'unknown'
            for pattern_type, pattern in self.text_patterns.items():
                if re.search(pattern, block['text']):
                    text_type = pattern_type
                    break
            
            organized.append({
                'type': text_type,
                'text': block['text'],
                'confidence': block['confidence'],
                'words': [{
                    'text': block['text'],
                    'confidence': block['confidence'],
                    'bbox': block['bbox']
                }]
            })
        
        return organized

    def _categorize_text(self, blocks: List[Dict]) -> List[str]:
        """Enhanced text categorization with more patterns."""
        categories = set()
        all_text = ' '.join([b['text'] for b in blocks])
        
        # Check for emails
        if re.search(self.text_patterns['email'], all_text):
            categories.add('email')
            
        # Check for URLs
        if re.search(self.text_patterns['url'], all_text):
            categories.add('url')
            
        # Check for phone numbers
        if re.search(self.text_patterns['phone'], all_text):
            categories.add('phone')
            
        # Check for dates
        if re.search(self.text_patterns['date'], all_text):
            categories.add('date')
            
        # Check for potential code snippets
        code_indicators = ['def ', 'class ', 'function', 'var ', 'const ', 'let ', '{ }', '[ ]', '// ', '/* */']
        if any(indicator in all_text for indicator in code_indicators):
            categories.add('code')
            
        # Check for addresses
        address_patterns = [
            r'\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)',
            r'[A-Za-z]+,\s*[A-Za-z]+\s*\d{5}'
        ]
        if any(re.search(pattern, all_text, re.IGNORECASE) for pattern in address_patterns):
            categories.add('address')
            
        return list(categories)