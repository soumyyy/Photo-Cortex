from .face_detector import FaceDetector
from .object_detector import ObjectDetector
from .scene_classifier import SceneClassifier
from .text_recognizer import TextRecognizer
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
import cv2

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.object_detector = ObjectDetector()
        self.scene_classifier = SceneClassifier()
        self.text_recognizer = TextRecognizer()
        logger.info("All models initialized successfully")

    def analyze_image(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Analyze an image using all models.
        
        Args:
            image: Can be numpy array (BGR) or path to image
        """
        try:
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
            
            # Get text detections
            texts = self.text_recognizer.detect_text(image_array)
            text_summary = self.text_recognizer.get_text_summary(texts)
            
            return {
                'faces': faces,
                'objects': unique_objects,
                'scene_classification': scene_info,
                'object_detections': objects,
                'text_detections': texts,
                'text_summary': text_summary
            }
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {
                'faces': [],
                'objects': [],
                'scene_classification': {"scene_type": "unknown", "confidence": 0.0},
                'object_detections': [],
                'text_detections': [],
                'text_summary': ""
            }