import cv2
import numpy as np
from pathlib import Path
import logging
from .text_recognizer import TextRecognizer
from .face_detector import FaceDetector
from .object_detector import ObjectDetector
from .scene_classifier import SceneClassifier
from .clip_encoder import ClipEncoder
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession, async_session
from sqlalchemy.orm import declarative_base, scoped_session
from sqlalchemy.orm.session import sessionmaker
from database.models import Image, TextDetection, SceneClassification, ObjectDetection, FaceDetection, FaceIdentity, ExifMetadata
from database.config import get_db
import datetime
from geoalchemy2.shape import WKTElement
from sqlalchemy import select, delete   

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        """Initialize the image analyzer with various models."""
        self.text_recognizer = TextRecognizer()
        self.face_detector = FaceDetector()
        self.object_detector = ObjectDetector()
        self.scene_classifier = SceneClassifier()
        self.clip_encoder = ClipEncoder()
        logger.info("ImageAnalyzer initialized successfully")

    async def analyze_image_with_session(self, image_path: Path, session: AsyncSession) -> int:
        """
        Analyze an image using various models and save results using provided session.
        This version is used with FastAPI dependency injection.
        """
        try:
            # Ensure image path is absolute
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise ValueError(f"Image file not found: {image_path}")
                
            # Check if image with this filename already exists
            query = select(Image).where(Image.filename == image_path.name)
            result = await session.execute(query)
            existing_image = result.scalar_one_or_none()

            if existing_image:
                # Image already exists, skip analysis
                logger.info(f"Image with filename {image_path.name} already exists. Skipping analysis.")
                return existing_image.id
                
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
            face_analysis = self.face_detector.detect_faces(image, str(image_path.name))
            object_analysis = self.object_detector.detect_objects(image)
            scene_analysis = self.scene_classifier.classify_scene_combined(image)
            clip_embedding = self.clip_encoder.encode_image(image)

            # CRITICAL: Create a clean dictionary with ONLY valid Image model fields
            # This ensures no EXIF fields are passed to the Image constructor
            image_fields = {
                "filename": image_path.name,
                "dimensions": f"{image.shape[1]}x{image.shape[0]}",
                "format": image_path.suffix[1:].lower(),  # Remove leading dot
                "file_size": int(image_path.stat().st_size),  # in bytes
                "date_taken": metadata.get("date_taken"),
                "embedding": clip_embedding
            }

            # Add GPS location if available
            if metadata.get("gps"):
                lat, lon = metadata["gps"]
                image_fields["location"] = WKTElement(f'POINT({lon} {lat})', srid=4326)
                image_fields["latitude"] = lat
                image_fields["longitude"] = lon

            # Create the Image model with only the valid fields
            db_image = Image(**image_fields)
            session.add(db_image)
            await session.flush()  # Get the image ID

            # Create EXIF metadata record as a SEPARATE model
            # These fields should NEVER be passed to the Image model
            exif_data = metadata.get("exif", {})
            if exif_data.get("camera_make") or exif_data.get("camera_model") or exif_data.get("focal_length") or \
               exif_data.get("exposure_time") or exif_data.get("f_number") or exif_data.get("iso"):
                
                exif = ExifMetadata(image_id=db_image.id)
                
                # Only set fields that have values
                if exif_data.get("camera_make"):
                    exif.camera_make = exif_data.get("camera_make")
                if exif_data.get("camera_model"):
                    exif.camera_model = exif_data.get("camera_model")
                if exif_data.get("focal_length"):
                    exif.focal_length = exif_data.get("focal_length")
                if exif_data.get("exposure_time"):
                    exif.exposure_time = exif_data.get("exposure_time")
                if exif_data.get("f_number"):
                    exif.f_number = exif_data.get("f_number")
                if exif_data.get("iso"):
                    exif.iso = exif_data.get("iso")
                
                session.add(exif)

            # Save text detections
            if text_analysis["text_detected"]:
                for block in text_analysis["text_blocks"]:
                    text_detection = TextDetection(
                        image_id=db_image.id,
                        text=block["text"],
                        confidence=block["confidence"],
                        bounding_box=block["bbox"]
                    )
                    session.add(text_detection)

            # Save face detections
            for idx, face in enumerate(face_analysis["faces"]):
                # Ensure we have the corresponding embedding
                if idx < len(face_analysis["embeddings"]):
                    embedding = np.array(face_analysis["embeddings"][idx], dtype=np.float32)
                    
                    # Create face detection record with proper validation
                    try:
                        # Extract attributes from the face data
                        attributes = face.get("attributes", {})
                        
                        face_detection = FaceDetection(
                            image_id=db_image.id,
                            bounding_box=face["bbox"],
                            confidence=face["score"],  # Changed from "confidence" to "score"
                            embedding=embedding,
                            landmarks=face.get("landmarks"),
                            similarity_score=None  # Will be set during identity matching
                        )
                        session.add(face_detection)
                    except Exception as e:
                        logger.error(f"Error creating face detection: {str(e)}")
                        logger.error(f"Face data: {face}")
                        continue

            # Save object detections
            for obj in object_analysis["objects"]:
                object_detection = ObjectDetection(
                    image_id=db_image.id,
                    label=obj["label"],
                    confidence=obj["confidence"],
                    bounding_box=obj["bbox"]
                )
                session.add(object_detection)

            # Save scene classification
            if scene_analysis["scene_type"]:
                scene_classification = SceneClassification(
                    image_id=db_image.id,
                    scene_type=scene_analysis["scene_type"],
                    confidence=scene_analysis["confidence"]
                )
                session.add(scene_classification)

            await session.commit()
            
            # Return the ID of the saved image record
            return db_image.id

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            await session.rollback()  # Rollback the transaction
            
            # Return a dictionary with error information
            if isinstance(image_path, (str, Path)):
                filename = Path(image_path).name
            else:
                filename = "unknown"
                
            return {
                "error": str(e),
                "filename": filename
            }

    async def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Legacy method that creates its own session.
        Use analyze_image_with_session for FastAPI endpoints.
        """
        try:
            # Read the image
            image_path = Path(image_path).resolve()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            # Extract metadata (synchronous operation)
            metadata = self._extract_metadata(image_path)

            # Run all analyses synchronously
            text_analysis = self.text_recognizer.detect_text(image)
            face_analysis = self.face_detector.detect_faces(image)
            object_analysis = self.object_detector.detect_objects(image)
            scene_analysis = self.scene_classifier.classify_scene_combined(image)
            clip_embedding = self.clip_encoder.encode_image(image)
            
            # Return the complete analysis data
            return {
                "faces": face_analysis.get("faces", []),
                "objects": object_analysis.get("objects", []),
                "text_recognition": text_analysis,
                "scene_classification": scene_analysis,
                "embedding": clip_embedding,
                "metadata": {
                    "dimensions": metadata.get("dimensions"),
                    "format": metadata.get("format"),
                    "file_size": metadata.get("file_size"),
                    "date_taken": metadata.get("date_taken"),
                    "gps": metadata.get("gps")
                },
                "exif": metadata.get("exif", {})  # Keep EXIF data separate from metadata
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            # Return empty analysis to avoid breaking the upload flow
            return {
                "faces": [],
                "objects": [],
                "text_recognition": {"text_detected": False, "text_blocks": []},
                "scene_classification": {"scene_type": None, "confidence": 0},
                "embedding": None,
                "metadata": {},
                "exif": {}
            }

    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF metadata from image."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            metadata = {
                "dimensions": None,
                "format": None,
                "file_size": None,
                "date_taken": None,
                "gps": None
            }
            
            exif_metadata = {
                "camera_make": None,
                "camera_model": None,
                "focal_length": None,
                "exposure_time": None,
                "f_number": None,
                "iso": None
            }
            
            with Image.open(image_path) as img:
                # Set basic metadata
                metadata["dimensions"] = f"{img.width}x{img.height}"
                metadata["format"] = image_path.suffix[1:].lower()  # Remove leading dot
                metadata["file_size"] = image_path.stat().st_size / 1024  # KB
                
                try:
                    exif = img._getexif()
                    if exif:
                        raw_exif = {}
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            raw_exif[tag] = value

                        # Extract date taken
                        if "DateTimeOriginal" in raw_exif:
                            metadata["date_taken"] = datetime.datetime.strptime(
                                raw_exif["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
                            )
                            
                        # Extract EXIF data (separate from basic metadata)
                        exif_metadata["camera_make"] = raw_exif.get("Make")
                        exif_metadata["camera_model"] = raw_exif.get("Model")
                        
                        if "FocalLength" in raw_exif:
                            exif_metadata["focal_length"] = float(str(raw_exif["FocalLength"]).split()[0])
                            
                        exif_metadata["exposure_time"] = str(raw_exif.get("ExposureTime", ""))
                        
                        if "FNumber" in raw_exif:
                            exif_metadata["f_number"] = float(str(raw_exif["FNumber"]).split()[0])
                            
                        exif_metadata["iso"] = raw_exif.get("ISOSpeedRatings")

                        # Extract GPS data
                        if "GPSInfo" in raw_exif:
                            gps = raw_exif["GPSInfo"]
                            if all(i in gps for i in (1, 2, 3, 4)):
                                lat = self._convert_to_degrees(gps[2], gps[1])
                                lon = self._convert_to_degrees(gps[4], gps[3])
                                metadata["gps"] = (lat, lon)

                except Exception as e:
                    logger.error(f"Error extracting EXIF: {str(e)}")

            # Combine metadata and exif_metadata
            result = {**metadata}
            result["exif"] = exif_metadata
            
            return result

        except Exception as e:
            logger.error(f"Error opening image for metadata: {str(e)}")
            return {"exif": {}}

    def _convert_to_degrees(self, value, ref) -> float:
        """Convert GPS coordinates to degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        
        degrees = d + (m / 60.0) + (s / 3600.0)
        
        if ref in ['S', 'W']:
            degrees = -degrees
            
        return degrees