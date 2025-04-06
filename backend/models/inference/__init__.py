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
import datetime
from geoalchemy2.shape import WKTElement
from database.models import Image, TextDetection, FaceDetection, ObjectDetection, SceneClassification, FaceIdentity

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

    async def analyze_image_with_session(self, image_path: Path, session) -> Dict[str, Any]:
        """
        Analyze an image using various models and save results using provided session.
        This version is used with FastAPI dependency injection.
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
                raise ValueError("Invalid image dimensions")

            # Extract metadata (synchronous operation)
            metadata = self._extract_metadata(image_path)

            # Run all analyses synchronously since they don't have async implementations
            text_analysis = self.text_recognizer.detect_text(image)
            face_analysis = self.face_detector.detect_faces(image)
            object_analysis = self.object_detector.detect_objects(image)
            scene_analysis = self.scene_classifier.classify_scene_combined(image)

            # Create Image record
            db_image = Image(
                filename=image_path.name,
                dimensions=f"{image.shape[1]}x{image.shape[0]}",
                format=image_path.suffix[1:],  # Remove leading dot
                file_size=image_path.stat().st_size / 1024,  # Convert to KB
                date_taken=metadata.get("date_taken"),
                camera_make=metadata.get("camera_make"),
                camera_model=metadata.get("camera_model"),
                focal_length=metadata.get("focal_length"),
                exposure_time=metadata.get("exposure_time"),
                f_number=metadata.get("f_number"),
                iso=metadata.get("iso"),
                embedding=np.zeros(512)  # Default empty embedding until CLIP is implemented
            )

            # Add GPS location if available
            if metadata.get("gps"):
                lat, lon = metadata["gps"]
                db_image.location = WKTElement(f'POINT({lon} {lat})', srid=4326)

            session.add(db_image)
            await session.flush()  # Get the image ID

            # Save text detections
            if text_analysis.get("text_detected", False):
                for text_block in text_analysis.get('text_blocks', []):
                    text_detection = TextDetection(
                        image=db_image,
                        text=text_block['text'],
                        confidence=text_block['confidence'],
                        bounding_box=text_block['bounding_box']  # Now properly formatted as list of lists
                    )
                    session.add(text_detection)

            # Save face detections
            if face_analysis.get("faces"):
                for face in face_analysis["faces"]:
                    face_detection = FaceDetection(
                        image=db_image,
                        bounding_box=face["bbox"],
                        landmarks=face["landmarks"],
                        embedding=np.zeros(512)  # Default empty embedding until face encoder is implemented
                    )
                    session.add(face_detection)

            # Save object detections
            for obj in object_analysis:
                obj_detection = ObjectDetection(
                    image=db_image,
                    label=obj["class"],
                    confidence=obj["confidence"],
                    bounding_box=obj["bbox"]
                )
                session.add(obj_detection)

            # Save scene classification
            if scene_analysis and scene_analysis.get("scene_type") != "Unknown":
                scene_class = SceneClassification(
                    image=db_image,
                    scene_type=scene_analysis["scene_type"],
                    confidence=scene_analysis["confidence"]
                )
                session.add(scene_class)

            await session.commit()
            
            return {
                "filename": image_path.name,
                "metadata": metadata,
                "text_recognition": text_analysis,
                "face_detection": face_analysis,
                "object_detection": object_analysis,
                "scene_classification": scene_analysis
            }

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            await session.rollback()  # Add rollback on error
            return {
                "filename": image_path.name,
                "error": str(e)
            }

    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF metadata from image."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            metadata = {}
            with Image.open(image_path) as img:
                try:
                    exif = img._getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            metadata[tag] = value

                        # Extract common EXIF fields
                        if "DateTimeOriginal" in metadata:
                            metadata["date_taken"] = datetime.datetime.strptime(
                                metadata["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
                            )
                        metadata["camera_make"] = metadata.get("Make")
                        metadata["camera_model"] = metadata.get("Model")
                        metadata["focal_length"] = float(str(metadata.get("FocalLength", "0")).split()[0])
                        metadata["exposure_time"] = str(metadata.get("ExposureTime", ""))
                        metadata["f_number"] = float(str(metadata.get("FNumber", "0")).split()[0])
                        metadata["iso"] = metadata.get("ISOSpeedRatings")

                        # Extract GPS data
                        if "GPSInfo" in metadata:
                            gps = metadata["GPSInfo"]
                            if all(i in gps for i in (1, 2, 3, 4)):
                                lat = self._convert_to_degrees(gps[2], gps[1])
                                lon = self._convert_to_degrees(gps[4], gps[3])
                                metadata["gps"] = (lat, lon)

                except Exception as e:
                    logger.error(f"Error extracting EXIF: {str(e)}")

            return metadata

        except Exception as e:
            logger.error(f"Error opening image for metadata: {str(e)}")
            return {}

    def _convert_to_degrees(self, value, ref) -> float:
        """Convert GPS coordinates to degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        
        degrees = d + (m / 60.0) + (s / 3600.0)
        
        if ref in ['S', 'W']:
            degrees = -degrees
            
        return degrees