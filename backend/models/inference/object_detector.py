import os
# Set these environment variables before importing YOLO
os.environ["ULTRALYTICS_HIDE_CONSOLE"] = "1"
os.environ["ULTRALYTICS_HUB_API"] = "false"
os.environ["ULTRALYTICS_LOGGER_VERBOSE"] = "false"
# Add this new environment variable
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
import warnings
import sys

# Filter out PyTorch TypedStorage deprecation warning
warnings.filterwarnings('ignore', message='TypedStorage is deprecated')

logger = logging.getLogger(__name__)

class ObjectDetector:
    """YOLOv8 object detector with singleton pattern to prevent multiple model loads."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObjectDetector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str = "./models/weights/yolo/yolov8s.pt"):
        """Initialize YOLOv8s model."""
        if self._initialized:
            return
            
        try:
            # Create null device to discard output
            null_device = open(os.devnull, 'w')
            
            # Save original stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            # Redirect stdout/stderr to null device
            sys.stdout = null_device
            sys.stderr = null_device
            
            try:
                # Initialize model without verbose parameter
                self.model = YOLO(model_path)
                self._initialized = True
                logger.info("YOLOv8s model initialized successfully")
            finally:
                # Restore original stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                null_device.close()
                
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {str(e)}")
            self.model = None

    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Detect objects in an image using YOLOv8."""
        if self.model is None:
            return []

        try:
            # Use the same approach for inference
            null_device = open(os.devnull, 'w')
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = null_device, null_device
            
            try:
                # Call the model without verbose parameter
                results = self.model(image)
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                null_device.close()

            detections = []
            for result in results:
                boxes = result.boxes
                for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                    class_name = result.names[int(cls)]
                    confidence = float(conf)
                    
                    if confidence > conf_threshold and class_name != 'person':
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': box.tolist()
                        })

            return detections
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []

    def get_unique_objects(self, detections: List[Dict[str, Any]]) -> List[str]:
        """Get a unique list of detected objects."""
        return sorted(list(set(d['class'] for d in detections)))