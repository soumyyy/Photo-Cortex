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
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import time
from typing import Any, Dict

class PhotoOCR:
    """
    A high-accuracy English OCR class that uses DBNet for text detection and TrOCR for text recognition.
    
    This class performs the following steps:
    1. Uses DBNet to detect text regions in an image
    2. Crops each detected region
    3. Uses TrOCR to recognize text in each region
    4. Combines all recognized text into a single output
    5. Post-processes the text to extract structured entities
    """
    
    def __init__(self, 
                 detection_model_path: Optional[str] = None,
                 recognition_model_name: str = "microsoft/trocr-base-printed",
                 detection_threshold: float = 0.2,  
                 recognition_batch_size: int = 4,
                 device: Optional[str] = None):
        """
        Initialize the PhotoOCR class.
        
        Args:
            detection_model_path: Path to the DBNet model. If None, will use the default model.
            recognition_model_name: Name of the TrOCR model to use.
            detection_threshold: Confidence threshold for text detection (0-1).
            recognition_batch_size: Batch size for text recognition.
            device: Device to run the models on ('cuda' or 'cpu'). If None, will use CUDA if available.
        """
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Text detection settings
        self.detection_threshold = detection_threshold
        self.min_text_height = 8  
        self.max_text_height = 300  
        self.min_confidence = 0.4  
        self.padding_percent = 0.15  
        
        # Initialize models
        self.initialize_detection_model(detection_model_path)
        self.recognition_model_name = recognition_model_name
        self.recognition_batch_size = recognition_batch_size
        self.initialize_recognition_model()
        
        # Enhanced regex patterns for better entity detection
        self.patterns = {
            'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!$&\'()*+,;=:]+)*',
            'phone': r'(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
            'amount': r'\$\s?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?'
        }
        
        self.logger.info("PhotoOCR initialized with enhanced settings")

    def initialize_detection_model(self, model_path: Optional[str] = None):
        """
        Initialize the DBNet text detection model.
        
        Args:
            model_path: Path to the DBNet model. If None, will use OpenCV's EAST text detector.
        """
        try:
            if model_path is None:
                # Use OpenCV's EAST text detector as a fallback
                self.logger.info("No DBNet model path provided, using OpenCV's EAST text detector")
                
                # Use the correct path relative to the current file
                current_dir = Path(__file__).parent.parent.parent
                east_model_path = current_dir / 'models' / 'weights' / 'ocr' / 'frozen_east_text_detection.pb'
                
                self.logger.info(f"Looking for EAST model at: {east_model_path}")
                
                if not east_model_path.exists():
                    # Try to download the model if it doesn't exist
                    try:
                        os.makedirs(east_model_path.parent, exist_ok=True)
                        import urllib.request
                        url = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
                        self.logger.info(f"Downloading EAST model from {url}")
                        urllib.request.urlretrieve(url, east_model_path)
                        self.logger.info("EAST model downloaded successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to download EAST model: {str(e)}")
                        # Fall back to simple text detection
                        self.logger.warning("Falling back to simple text detection")
                        self.detection_model_type = "simple"
                        return
                    
                try:
                    self.detection_model = cv2.dnn.readNet(str(east_model_path))
                    self.detection_model_type = "east"
                except Exception as e:
                    self.logger.error(f"Failed to load EAST model: {str(e)}")
                    # Fall back to simple text detection
                    self.logger.warning("Falling back to simple text detection")
                    self.detection_model_type = "simple"
                    return
            else:
                # Load DBNet model
                self.logger.info(f"Loading DBNet model from {model_path}")
                self.detection_model = cv2.dnn.readNet(model_path)
                self.detection_model_type = "dbnet"
            
            # Set preferred backend and target
            if self.device == 'cuda':
                self.detection_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.detection_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.detection_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.detection_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
            self.logger.info("Text detection model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing text detection model: {str(e)}")
            # Fall back to simple text detection
            self.logger.warning("Falling back to simple text detection")
            self.detection_model_type = "simple"

    def initialize_recognition_model(self):
        """Initialize the TrOCR text recognition model."""
        try:
            self.logger.info(f"Loading TrOCR model: {self.recognition_model_name}")
            
            # Temporarily redirect warnings to prevent the I/O error
            import warnings
            import io
            
            # Save the original showwarning function
            original_showwarning = warnings.showwarning
            
            # Define a custom showwarning function that doesn't write to a file
            def custom_showwarning(message, category, filename, lineno, file=None, line=None):
                self.logger.warning(f"{category.__name__}: {message}")
            
            # Replace the showwarning function
            warnings.showwarning = custom_showwarning
            
            try:
                # Load TrOCR processor and model
                self.processor = TrOCRProcessor.from_pretrained(self.recognition_model_name)
                self.recognition_model = VisionEncoderDecoderModel.from_pretrained(self.recognition_model_name)
                
                # Move model to the appropriate device
                self.recognition_model.to(self.device)
            finally:
                # Restore the original showwarning function
                warnings.showwarning = original_showwarning
            
            self.logger.info("Text recognition model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing text recognition model: {str(e)}")
            raise
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in an image using DBNet or EAST.
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            List of dictionaries containing cropped text regions and their bounding boxes
        """
        try:
            # Make a copy of the image to avoid modifying the original
            original_image = image.copy()
            height, width = image.shape[:2]
            
            # Log the original image dimensions for debugging
            self.logger.info(f"Original image dimensions: {width}x{height}")
            
            # Prepare image for the model
            if self.detection_model_type == "east":
                # EAST requires specific dimensions
                new_width, new_height = (320, 320)
                ratio_width, ratio_height = width / new_width, height / new_height
                
                # Create a blob from the image
                blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), 
                                            (123.68, 116.78, 103.94), swapRB=True, crop=False)
                
                # Set the blob as input and get output layers
                self.detection_model.setInput(blob)
                output_layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
                scores, geometry = self.detection_model.forward(output_layer_names)
                
                # Decode predictions
                rectangles, confidences = self._decode_east_predictions(scores, geometry, self.detection_threshold)
                
                # Apply non-maximum suppression
                if len(rectangles) > 0:
                    indices = cv2.dnn.NMSBoxes(rectangles, confidences, self.detection_threshold, 0.4)
                    
                    # Extract text regions
                    text_regions = []
                    if len(indices) > 0:
                        for i in indices.flatten():
                            # Get coordinates
                            x, y, w, h = rectangles[i]
                            
                            # Scale back to original image size
                            x = int(x * ratio_width)
                            y = int(y * ratio_height)
                            w = int(w * ratio_width)
                            h = int(h * ratio_height)
                            
                            # Add padding around the text region (10% of width/height)
                            padding_x = int(w * 0.1)
                            padding_y = int(h * 0.1)
                            
                            # Ensure coordinates are within image bounds
                            x_start = max(0, x - padding_x)
                            y_start = max(0, y - padding_y)
                            x_end = min(width, x + w + padding_x)
                            y_end = min(height, y + h + padding_y)
                            
                            # Log the coordinates for debugging
                            self.logger.info(f"Text region: [{x_start}, {y_start}, {x_end}, {y_end}]")
                            
                            # Crop the text region
                            text_region = original_image[y_start:y_end, x_start:x_end]
                            
                            # Only add if the region is not empty
                            if text_region.size > 0:
                                text_regions.append({
                                    'image': text_region,
                                    'bbox': [x_start, y_start, x_end, y_end]
                                })
                    
                    return text_regions
                else:
                    self.logger.warning("No text regions detected with EAST")
                    return []
            else:
                # DBNet implementation would go here
                # For now, we'll use a placeholder that returns the same as EAST
                # This should be replaced with actual DBNet implementation
                self.logger.warning("DBNet implementation not available, using EAST instead")
                return []
        except Exception as e:
            self.logger.error(f"Error detecting text regions: {str(e)}")
            return []
    
    def _decode_east_predictions(self, scores, geometry, min_confidence):
        """
        Decode the predictions from the EAST text detector.
        
        Args:
            scores: Score map from the model
            geometry: Geometry map from the model
            min_confidence: Minimum confidence threshold
            
        Returns:
            rectangles: List of detected text rectangles
            confidences: List of confidence scores
        """
        rectangles = []
        confidences = []
        
        # Get dimensions of the score map
        (num_rows, num_cols) = scores.shape[2:4]
        
        # Loop over rows and columns of the score map
        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]
            
            for x in range(num_cols):
                # If the score is below the minimum confidence, ignore it
                if scores_data[x] < min_confidence:
                    continue
                
                # Compute the offset
                offset_x = x * 4.0
                offset_y = y * 4.0
                
                # Extract the rotation angle
                angle = angles_data[x]
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                
                # Calculate dimensions of the bounding box
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]
                
                # Calculate coordinates of the bounding box
                end_x = int(offset_x + (cos_angle * x_data1[x]) + (sin_angle * x_data2[x]))
                end_y = int(offset_y - (sin_angle * x_data1[x]) + (cos_angle * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                
                # Add the bounding box and confidence score
                rectangles.append([start_x, start_y, w, h])
                confidences.append(float(scores_data[x]))
        
        return rectangles, confidences
    
    def recognize_text(self, text_regions: List[Dict]) -> List[Dict]:
        """
        Recognize text in the detected regions using TrOCR.
        
        Args:
            text_regions: List of dictionaries containing text regions and their bounding boxes
            
        Returns:
            List of dictionaries containing recognized text and their bounding boxes
        """
        if not text_regions:
            return []
        
        results = []
        
        # Process in batches to improve efficiency
        for i in range(0, len(text_regions), self.recognition_batch_size):
            batch = text_regions[i:i + self.recognition_batch_size]
            
            # Convert OpenCV BGR images to PIL RGB images
            pil_images = [Image.fromarray(cv2.cvtColor(region['image'], cv2.COLOR_BGR2RGB)) for region in batch]
            
            # Prepare inputs for the model
            pixel_values = self.processor(images=pil_images, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.recognition_model.generate(pixel_values)
                
            # Decode the generated IDs to text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Add results
            for j, text in enumerate(generated_text):
                if text.strip():  # Only add non-empty text
                    results.append({
                        'text': text.strip(),
                        'bbox': batch[j]['bbox']
                    })
        
        # Sort results by vertical position (top to bottom)
        results.sort(key=lambda x: x['bbox'][1])
        
        return results
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract structured entities from text using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted entities by type
        """
        entities = {entity_type: [] for entity_type in self.patterns.keys()}
        
        # Extract entities using regex patterns
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities[entity_type].append(match.group())
        
        return entities
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image through the complete OCR pipeline.
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            Dictionary containing:
            - combined_text: All recognized text combined
            - text_blocks: List of text blocks with their bounding boxes
            - entities: Extracted structured entities
        """
        start_time = time.time()
        
        try:
            # Validate input image
            if image is None:
                raise ValueError("Input image is None")
            
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
            
            if len(image.shape) != 3:
                raise ValueError(f"Expected 3-channel image, got shape {image.shape}")
            
            # Detect text regions
            self.logger.info(f"Detecting text regions in image of shape {image.shape}...")
            
            if self.detection_model_type == "simple":
                text_regions = self.detect_text_regions_simple(image)
            else:
                text_regions = self.detect_text_regions(image)
                
            self.logger.info(f"Found {len(text_regions)} text regions in {time.time() - start_time:.2f}s")
            
            if not text_regions:
                return {
                    'text_blocks': [],
                    'combined_text': '',
                    'entities': {},
                    'processing_time': time.time() - start_time
                }
            
            # Recognize text in detected regions
            text_blocks = self.recognize_text(text_regions)
            
            if not text_blocks:
                return {
                    'text_blocks': [],
                    'combined_text': '',
                    'entities': {},
                    'processing_time': time.time() - start_time
                }
            
            # Combine all text
            combined_text = ' '.join(block['text'] for block in text_blocks)
            
            # Extract entities
            entities = self.extract_entities(combined_text)
            
            return {
                'text_blocks': text_blocks,
                'combined_text': combined_text,
                'entities': entities,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise

    def detect_text_regions_simple(self, image: np.ndarray) -> List[Dict]:
        """Enhanced simple text detection method."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Apply adaptive thresholding with optimized parameters
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 15, 4
            )
            
            # Apply morphological operations to connect text components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            height, width = image.shape[:2]
            min_area = (width * height) * 0.0005  
            max_area = (width * height) * 0.4    
            
            text_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Add padding (15% of width/height)
                    padding_x = int(w * self.padding_percent)
                    padding_y = int(h * self.padding_percent)
                    
                    # Ensure coordinates are within image bounds
                    x_start = max(0, x - padding_x)
                    y_start = max(0, y - padding_y)
                    x_end = min(width, x + w + padding_x)
                    y_end = min(height, y + h + padding_y)
                    
                    # Check aspect ratio and height
                    aspect_ratio = w / float(h)
                    if (0.08 <= aspect_ratio <= 20 and  
                        self.min_text_height <= h <= self.max_text_height):
                        
                        text_region = image[y_start:y_end, x_start:x_end]
                        
                        if text_region.size > 0:
                            text_regions.append({
                                'image': text_region,
                                'bbox': [x_start, y_start, x_end, y_end]
                            })
            
            # Merge overlapping regions
            text_regions = self._merge_overlapping_regions(text_regions)
            
            # Sort regions from top to bottom, left to right
            text_regions.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
            
            return text_regions
            
        except Exception as e:
            self.logger.error(f"Error in simple text detection: {str(e)}")
            return []
    
    def _merge_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """Merge overlapping text regions."""
        if not regions:
            return regions
            
        def overlap_percent(box1, box2):
            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2
            
            x_left = max(x1, x3)
            y_top = max(y1, y3)
            x_right = min(x2, x4)
            y_bottom = min(y2, y4)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
                
            intersection = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x4 - x3) * (y4 - y3)
            
            return intersection / min(box1_area, box2_area)
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
                
            current_box = list(region1['bbox'])
            current_image = region1['image']
            
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                    
                if overlap_percent(current_box, region2['bbox']) > 0.3:  
                    used.add(j)
                    x1 = min(current_box[0], region2['bbox'][0])
                    y1 = min(current_box[1], region2['bbox'][1])
                    x2 = max(current_box[2], region2['bbox'][2])
                    y2 = max(current_box[3], region2['bbox'][3])
                    current_box = [x1, y1, x2, y2]
            
            if i not in used:
                merged.append({
                    'image': current_image,
                    'bbox': current_box
                })
                used.add(i)
        
        return merged

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
            
            if text_likelihood < 0.02:  
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
            filtered_blocks = [b for b in merged_blocks if b['confidence'] > 0.6]  
            
            # Additional filtering for minimum text length and common patterns
            filtered_blocks = [b for b in filtered_blocks if 
                len(b['text'].strip()) >= 3 and  
                not b['text'].isspace() and  
                any(c.isalnum() for c in b['text'])  
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
            if int(ocr_data['conf'][i]) < 0:  
                continue
                
            text = ocr_data['text'][i].strip()
            if not text:  
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
                if similarity > 0.8:  
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