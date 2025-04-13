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

            # Run analyses that don't require image_id first
            text_analysis = self.text_recognizer.detect_text(image)
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

            # Create and save the image record first to get the image_id
            image_record = Image(**image_fields)
            session.add(image_record)
            await session.flush()  # This will populate the image_id
            
            # Now we can pass the correct image_id to face_detector
            face_analysis = await self.face_detector.detect_faces(image, image_record.id, session)

            # Save EXIF metadata record as a SEPARATE model
            # These fields should NEVER be passed to the Image model
            exif_data = metadata.get("exif", {})
            if exif_data.get("camera_make") or exif_data.get("camera_model") or exif_data.get("focal_length") or \
               exif_data.get("exposure_time") or exif_data.get("f_number") or exif_data.get("iso"):
                
                exif = ExifMetadata(image_id=image_record.id)
                
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
                        image_id=image_record.id,
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
                        face_detection = FaceDetection(
                            image_id=image_record.id,
                            confidence=float(face.get('score', 0.0)),  # Store detection confidence
                            embedding=embedding.tolist(),
                            bounding_box=face.get('bbox'),
                            landmarks=face.get('landmarks'),
                            similarity_score=0.0  # Will be updated during identity assignment
                        )
                        session.add(face_detection)
                    except Exception as e:
                        logger.error(f"Error saving face detection: {e}")
                        continue

            # Save object detections
            for obj in object_analysis["objects"]:
                object_detection = ObjectDetection(
                    image_id=image_record.id,
                    label=obj["label"],
                    confidence=obj["confidence"],
                    bounding_box=obj["bbox"]
                )
                session.add(object_detection)

            # Save scene classification
            if scene_analysis["scene_type"]:
                scene_classification = SceneClassification(
                    image_id=image_record.id,
                    scene_type=scene_analysis["scene_type"],
                    confidence=scene_analysis["confidence"]
                )
                session.add(scene_classification)

            await session.commit()
            
            # Return the ID of the saved image record
            return image_record.id

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
            # Ensure image path is absolute
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise ValueError(f"Image file not found: {image_path}")
            
            # Create a new session
            async with get_db() as session:
                # Use the new method with session handling
                image_id = await self.analyze_image_with_session(image_path, session)
                
                # Query the database to get the full analysis
                query = select(Image).options(
                    joinedload(Image.face_detections),
                    joinedload(Image.object_detections),
                    joinedload(Image.text_detections),
                    joinedload(Image.scene_classifications),
                    joinedload(Image.exif_metadata)
                ).where(Image.id == image_id)
                
                result = await session.execute(query)
                image = result.unique().scalar_one_or_none()
                
                if not image:
                    logger.error(f"Failed to retrieve image with ID {image_id}")
                    return {
                        "faces": [],
                        "embeddings": [],
                        "objects": [],
                        "text_recognition": {"text_detected": False, "text_blocks": []},
                        "scene_classification": {"scene_type": None, "confidence": 0}
                    }
                
                # Format the results
                faces_data = []
                embeddings_data = []
                
                for face in image.face_detections:
                    face_data = {
                        "bbox": face.bounding_box,
                        "score": face.confidence,
                        "landmarks": face.landmarks
                    }
                    faces_data.append(face_data)
                    
                    if face.embedding is not None:
                        embeddings_data.append(face.embedding)
                
                # Build metadata dictionary
                metadata = {
                    "dimensions": image.dimensions,
                    "format": image.format,
                    "file_size": str(image.file_size),
                    "date_taken": image.date_taken.isoformat() if image.date_taken else None,
                    "gps": {
                        "latitude": float(image.latitude),
                        "longitude": float(image.longitude)
                    } if image.latitude is not None and image.longitude is not None else None
                }
                
                # Keep EXIF data separate from metadata
                exif_data = {}
                if image.exif_metadata:
                    exif_data = {
                        "camera_make": image.exif_metadata.camera_make,
                        "camera_model": image.exif_metadata.camera_model,
                        "focal_length": image.exif_metadata.focal_length,
                        "exposure_time": image.exif_metadata.exposure_time,
                        "f_number": image.exif_metadata.f_number,
                        "iso": image.exif_metadata.iso
                    }
                
                # Build the response
                return {
                    "filename": image.filename,
                    "metadata": metadata,
                    "exif": exif_data,
                    "faces": faces_data,
                    "embeddings": embeddings_data,
                    "objects": [obj.label for obj in image.object_detections],
                    "text_recognition": {
                        "text_detected": len(image.text_detections) > 0,
                        "text_blocks": [
                            {
                                "text": text.text,
                                "confidence": text.confidence,
                                "bbox": text.bounding_box
                            } for text in image.text_detections
                        ]
                    },
                    "scene_classification": next(
                        (
                            {
                                "scene_type": scene.scene_type,
                                "confidence": scene.confidence
                            } for scene in image.scene_classifications
                        ),
                        {"scene_type": None, "confidence": 0}
                    )
                }
                
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "faces": [],
                "embeddings": [],
                "objects": [],
                "text_recognition": {"text_detected": False, "text_blocks": []},
                "scene_classification": {"scene_type": None, "confidence": 0},
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