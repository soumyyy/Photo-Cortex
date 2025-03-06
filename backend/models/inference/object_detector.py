import os
os.environ["ULTRALYTICS_HIDE_CONSOLE"] = "1"
os.environ["ULTRALYTICS_HUB_API"] = "false"
os.environ["ULTRALYTICS_LOGGER_VERBOSE"] = "false"

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
import warnings

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
            # Suppress YOLO initialization messages
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.model = YOLO(model_path)
                self._initialized = True
            logger.info("YOLOv8s model initialized successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {str(e)}")
            self.model = None

    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Detect objects in an image using YOLOv8."""
        if self.model is None:
            return []

        try:
            # Suppress YOLO output during inference
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                results = self.model(image, verbose=False)

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