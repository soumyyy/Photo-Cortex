from .face_detector import FaceDetector
from .object_detector import ObjectDetector
from .scene_classifier import SceneClassifier
from .text_recognizer import TextRecognizer
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
import cv2
import os

__all__ = ['ImageAnalyzer', 'TextRecognizer']

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.face_detector = FaceDetector()
        self.object_detector = ObjectDetector()
        self.scene_classifier = SceneClassifier()
        self.text_recognizer = TextRecognizer()
        self._initialized = True
        
        # Add a cache for analyzed images to avoid redundant processing
        self.image_cache = {}
        
        logger.info("All models initialized successfully")

    def analyze_image(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Analyze an image using all models.
        
        Args:
            image: Can be numpy array (BGR) or path to image
        """
        try:
            # Get cache key if input is a path
            cache_key = str(image) if isinstance(image, (str, Path)) else None
            
            # Check cache for prior results if using a file path
            if cache_key and cache_key in self.image_cache:
                return self.image_cache[cache_key]
            
            # Handle different input types
            if isinstance(image, (str, Path)):
                image_array = cv2.imread(str(image))
                if image_array is None:
                    raise ValueError(f"Failed to read image")
            else:
                image_array = image

            # Get face detections
            faces = self.face_detector.detect_faces(image_array)
            
            # Get object detections
            objects = self.object_detector.detect_objects(image_array)
            unique_objects = self.object_detector.get_unique_objects(objects)
            
            # Get scene classification
            scene_info = self.scene_classifier.predict_scene(image_array)
            
            # Get text detections with improved error handling
            text_result = None
            try:
                if isinstance(image, (str, Path)):
                    text_result = self.text_recognizer.detect_text(str(image))
                else:
                    # Save temporary image for OCR if input is numpy array
                    temp_path = "_temp_ocr_image.jpg"
                    cv2.imwrite(temp_path, image_array)
                    text_result = self.text_recognizer.detect_text(temp_path)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error in text recognition: {e}")
                text_result = {
                    'text_detected': False,
                    'text_blocks': [],
                    'total_confidence': 0.0,
                    'categories': [],
                    'raw_text': '',
                    'orientation_angle': 0.0
                }
            
            result = {
                'faces': faces,
                'objects': unique_objects,
                'scene_classification': scene_info,
                'object_detections': objects,
                'metadata': {
                    'text': text_result
                }
            }
            
            # Store in cache if using a file path
            if cache_key:
                self.image_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {
                'faces': [],
                'objects': [],
                'scene_classification': {"scene_type": "unknown", "confidence": 0.0},
                'object_detections': [],
                'metadata': {
                    'text': {
                        'text_detected': False,
                        'text_blocks': [],
                        'total_confidence': 0.0,
                        'categories': [],
                        'raw_text': '',
                        'orientation_angle': 0.0
                    }
                }
            }

    def analyze_directory(self, directory: Union[str, Path], recursive: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all images in a directory.
        
        Args:
            directory: Path to directory containing images
            recursive: Whether to search subdirectories
        """
        results = {}
        directory = Path(directory)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        def process_directory(path: Path):
            for item in path.iterdir():
                if item.is_file() and item.suffix.lower() in image_extensions:
                    try:
                        results[item.name] = self.analyze_image(str(item))
                    except Exception as e:
                        logger.error(f"Failed to analyze {item}: {e}")
                elif item.is_dir() and recursive:
                    process_directory(item)
        
        process_directory(directory)
        return results

    def clear_cache(self):
        """Clear the image analysis cache."""
        self.image_cache.clear()
        self.text_recognizer.text_cache.clear()